import ROOT
import ROOT.RDataFrame as RDF
import itertools
from collections import deque
from enum import Enum
from time import perf_counter
from pathlib import Path

import util.branch_selection as bs
from util import region_selections as reg_sel
from util.config import con, setup_logger, get_sample_name
from util.saved_original_branch import original_list, original_kl_list, original_sys_list

# ROOT.EnableImplicitMT()
verbosity = ROOT.Experimental.RLogScopedVerbosity(ROOT.Detail.RDF.RDFLogChannel(), ROOT.Experimental.ELogLevel.kInfo)


class WType(Enum):
    loose = 0
    tight = 1
    none = 2
    muon_loose = 3
    muon_none = 4

def get_id(idx: int, w_type: WType):
    is_e = f'abs(lep_ID_{idx}) == 11'
    is_m = f'abs(lep_ID_{idx}) == 13'

    if w_type == WType.muon_loose:
        return f'(({is_e}) * lep_isLooseLH_{idx} + ({is_m}) * lep_isLoose_{idx})'
    if w_type == WType.muon_none:
        return f'(({is_e}) * lep_isLooseLH_{idx})'
    if w_type == WType.none:
        return 'true'
    if w_type == WType.loose:
        return f'(({is_e}) * lep_isLooseLH_{idx} + ({is_m}) * lep_isLoose_{idx})'
    if w_type == WType.tight:
        return f'(({is_e}) * lep_isTightLH_{idx} + ({is_m}) * lep_isMedium_{idx})'
    return None


def gen_selection(select: list, result: str = ""):
    """
    Generate the expression of selections: c1 * ( c1 + c2 * ( c2 + c3 * (...) ) )
    :param select: list of cuts
    :param result: initial string
    :return: expr of selections
    """
    if len(select) == 0:
        return result
    if len(select) > 0:
        s = select.pop(0)
        result += f'({s}) * ( ({s}) {"+" if len(select) > 0 else ""} '
        result = gen_selection(select, result)
        result += ')'
    return result


class Preprocessor:
    def __init__(self, root_file: ROOT.TFile, output_dir: str = "", is_data: bool = False):
        self.logger = setup_logger('Analyzer')

        self.is_data = is_data

        self.file = root_file
        self.total_entries = -1
        self.rdf = None

        self.cols = ROOT.std.vector('std::string')()
        self.total_sum_weights = -1
        self.total_sum_events = -1

        self.output_dir = output_dir

        self.sum_info = self.get_weight()
        ROOT.gInterpreter.Declare(self.sum_info['mc_gen_str'])
        self._load_macros()
        
    def _load_macros(self):
        ROOT.gInterpreter.Declare(con.macros)

    def _de(self, var: str, expr: str, store: bool = True):
        self.rdf = self.rdf.Define(var, expr)
        if store:
            self.cols.push_back(var)
            
    def _filter(self, rdf, cur_sum: list, selection: str, sel_name: str = ""):
        filtered_rdf = rdf.Filter(selection, sel_name)
        cur_sum += filtered_rdf.Sum("weight")

        return filtered_rdf
        
    def get_weight(self, weight_tree: str = 'sumWeights'):
        sum_info = {
            "weighted": 0.,
            "total_event": 0.,
            "dsid": 0
        }

        root_parent_dir = Path(self.file.GetName()).parent

        for r_dir in root_parent_dir.glob('*.root'):
            try:
                rdf_w = RDF(weight_tree, str(r_dir))
            except:
                self.logger.error(f"Weight Tree: {weight_tree} does not exist")
                exit(128)

            if rdf_w is not None:
                sum_weight = rdf_w.Sum("totalEventsWeighted")
                sum_event = rdf_w.Sum("totalEvents")
                dsid = rdf_w.Max('dsid')

                sum_info['weighted'] += sum_weight.GetValue()
                sum_info['total_event'] += sum_event.GetValue()
                sum_info['dsid'] = int(dsid.GetValue())

                if not self.is_data:
                    name_branch = rdf_w.AsNumpy(columns=['names_mc_generator_weights'])
                    names_mc_generator_weights = name_branch['names_mc_generator_weights'][0]
                    mc_gen_names = ','.join([f'"{v}"' for v in names_mc_generator_weights])
                    mc_gen_str = f"""
                    std::vector<std::string> names_mc_generator_weights = {{{mc_gen_names}}};
                    """
                else:
                    mc_gen_str = f"""
                    std::vector<std::string> names_mc_generator_weights = {{}};
                    """
                sum_info['mc_gen_str'] = mc_gen_str

        return sum_info

    def apply(self, tree_name: str = "nominal", prefix: str = ""):     
        self.tree_name = tree_name  
        self.rdf = RDF(self.tree_name, self.file)
        self.total_entries = self.file.Get(self.tree_name).GetEntries()
        self.cols = ROOT.std.vector('std::string')()

        sample_name = 'Unknown'
        if self.is_data:
            sample_name = 'data'
        else:
            # read tree first to get sample name
            t_tmp = self.file.Get(self.tree_name)
            nb = t_tmp.GetEntry(0)
            if nb > 0:
                sample_dsid = str(t_tmp.mcChannelNumber)
                sample_name = get_sample_name(sample_dsid)

        sum_info = self.sum_info

        start = perf_counter()
        self.logger.info(f'({prefix}) [{sample_name}] -- Branch: {self.tree_name}')
        self.logger.info(
            f'({prefix}) [{self.tree_name}] - original total events: {sum_info["total_event"]:0.0f}, weighted sum: {sum_info["weighted"]:0.2f} ')
        self.logger.info(f'({prefix}) [{self.tree_name}] - Start to loop for {self.total_entries:0.0f} events')

        report = self.rdf.Report()
        # define weight
        weight_str = f"""
        (36646.74 * (RunYear == 2015 or RunYear == 2016) + 
        44630.6 * (RunYear == 2017) + 
        58791.6 * (RunYear == 2018)) *
        weight_mc * 
        weight_pileup * 
        jvtSF_customOR * 
        bTagSF_weight_DL1r_77 * 
        mc_rawXSection * 
        mc_kFactor *
        custTrigSF_LooseID_FCLooseIso_SLTorDLT * 
        
        lep_SF_El_Reco_AT_0 * lep_SF_El_ChargeMisID_LooseAndBLayerLH_FCLoose_AT_0 *
        lep_SF_El_ID_LooseAndBLayerLH_AT_0 * lep_SF_El_PLVLoose_0 *
        lep_SF_Mu_ID_Loose_AT_0 * lep_SF_Mu_TTVA_AT_0 * lep_SF_Mu_PLVLoose_0 *

        lep_SF_El_Reco_AT_1 * lep_SF_El_ChargeMisID_LooseAndBLayerLH_FCLoose_AT_1 *
        lep_SF_El_ID_TightLH_AT_1 * lep_SF_El_PLVTight_1 *
        lep_SF_Mu_ID_Medium_AT_1 * lep_SF_Mu_TTVA_AT_1 * lep_SF_Mu_PLVTight_1 *

        lep_SF_El_Reco_AT_2 * lep_SF_El_ChargeMisID_LooseAndBLayerLH_FCLoose_AT_2 *
        lep_SF_El_ID_TightLH_AT_2 * lep_SF_El_PLVTight_2 *
        lep_SF_Mu_ID_Medium_AT_2 * lep_SF_Mu_TTVA_AT_2 * lep_SF_Mu_PLVTight_2 /
        {sum_info['weighted']}
        """

        # self._de('weight', weight_str)
        # # define sample name using DSID
        # self._de("Sample_Name", "map_dsid(mcChannelNumber)")

        if self.is_data:
            self._de('weight', "1")
            self._de("Sample_Name", "TString(\"data\")")
        else:
            self._de('weight', weight_str)
            # define sample name using DSID
            self._de("Sample_Name", "map_dsid(mcChannelNumber)")
            

        self._de("EvtNum", "eventNumber % 100")
        # define flavor category (flavor_categorization.cpp)
        self._de("FlavorCat", "flavor_category(lep_ID_0, lep_ID_1, lep_ID_2)")
        self._de('Evt_SFOS', 'sfos_map(FlavorCat)')


        '''
        Event Quality Selection:
            1/True: pass
            0/False: fail
        '''
        lep_loop = [0, 1, 2]
        jet_loop = (['lead', 'sublead'], [0, 1])


        # define 4-vector
        for i in lep_loop:
            self._de(
                f"lep_lvec_{i}",
                f"ROOT::Math::PtEtaPhiEVector(lep_Pt_{i},lep_Eta_{i}, lep_Phi_{i}, lep_E_{i});",
                store=False,
            )
        for i, j in zip(*jet_loop):
            self._de(
                f"jet_lvec_{j}",
                f"ROOT::Math::PtEtaPhiEVector({i}_jetPt,{i}_jetEta, {i}_jetPhi, {i}_jetE);",
                store=False,
            )
        # invariant mass / distance
        # only lepton
        # for i, j in zip([0, 0, 1], [1, 2, 2]):
        #     self._de(f'M_l{i}l{j}', f'(lep_lvec_{i} + lep_lvec_{j}).mass()')
        #     self._de(f'dR_l{i}l{j}', f'ROOT::Math::VectorUtil::DeltaR(lep_lvec_{i}, lep_lvec_{j})')

        # for i, j in itertools.product(lep_loop, lep_loop):
        for i in lep_loop:
            # mass & charge of lepton
            self._de(f"lep_Mass_{i}", f"get_mass_from_pdg(lep_ID_{i})")
            self._de(f"lep_Charge_{i}", f"lep_ID_{i}>0 ? -1 : 1")
            for j in lep_loop[i:]:
                self._de(f'M_l{i}l{j}', f'(lep_lvec_{i} + lep_lvec_{j}).mass()')
                self._de(f'dR_l{i}l{j}', f'ROOT::Math::VectorUtil::DeltaR(lep_lvec_{i}, lep_lvec_{j})')

        # lepton + jet
        for i, j in itertools.product(lep_loop, jet_loop[1]):
            self._de(f'M_l{i}j{j}', f'(lep_lvec_{i} + jet_lvec_{j}).mass()')
            self._de(f'dR_l{i}j{j}', f'ROOT::Math::VectorUtil::DeltaR(lep_lvec_{i}, jet_lvec_{j})')

        # lepton + closest jet
        for i in lep_loop:
            self._de(f'M_l{i}j', f' ( dR_l{i}j0 <= dR_l{i}j1 ) ? M_l{i}j0 : M_l{i}j1 ', store=False)
            self._de(f'dR_l{i}j', f' ( dR_l{i}j0 <= dR_l{i}j1 ) ? dR_l{i}j0 : dR_l{i}j1 ', store=False)

        # others
        self._de(f'M_lll', f'({"+".join([f"lep_lvec_{i}" for i in lep_loop])}).mass()')
        self._de(
            f'M_llljj',
            f'({"+".join([f"lep_lvec_{i}" for i in lep_loop] + [f"jet_lvec_{i}" for i in jet_loop[1]])}).mass()'
        )
        
        # self._de(f'lep_tmp_ID', 'std::vector<Float_t>({lep_ID_0, lep_ID_1, lep_ID_2})', store=True)
        self._de(
            'Evt_AuthorCut',
            '&&'.join([f'((fabs(lep_ID_{i}) == 11) ? lep_ambiguityType_{i} == 0 : true)' for i in lep_loop]),
            store=False
        )
        self._de(
            'Evt_ChargeIDBDT',
            '&&'.join([f'((fabs(lep_ID_{i}) == 11) ? lep_chargeIDBDTLoose_{i} == 1 : true)' for i in lep_loop]), 
            store=False
        )
        # Low Mass Veto (kinematics.cpp)
        self._de('Evt_LowMassVeto', 'low_mass_cut(Evt_SFOS, M_l0l1, M_l0l2)', store=False)
        # Z Mass Veto (kinematics.cpp)
        self._de('Evt_ZMassVeto', 'z_mass_cut(Evt_SFOS, M_l0l1, M_l0l2)', store=False)
        # M-lll Veto
        self._de('Evt_MlllVeto', 'm_lll_cut(M_lll)', store=False)
        # Event ID and Isolation
        self._de('Evt_ID_SR', f'{get_id(0, WType.loose)} && {get_id(1, WType.tight)} && {get_id(2, WType.tight)}')
        self._de('Evt_ISO_SR', f'lep_plvWP_Loose_0 && lep_plvWP_Tight_1 && lep_plvWP_Tight_2')

        '''
        Region Selection
        '''
        total_weighted_sum = self.rdf.Sum("weight")

        # Template Regions
        #   Internal/Material Conversion:
        lep_Mtrktrk_atConvV_CO_cut = "(lep_Mtrktrk_atConvV_CO_{0} < 0.1 && lep_Mtrktrk_atConvV_CO_{0} > 0)"
        lep_IntConv_atPV_CO_cut = "(lep_Mtrktrk_atPV_CO_{0} < 0.1 && lep_Mtrktrk_atPV_CO_{0} > 0)"
        material_conv_cut = """
                (abs(lep_ID_{0}) == 11 && (lep_RadiusCO_{0} > 20 && ({1}))) || abs(lep_ID_{0}) == 13
            """
        ele_cut = """
                (abs(lep_ID_{0}) == 13) ||
                (abs(lep_ID_{0}) == 11 &&
                (!(({2}) && !(lep_RadiusCO_{0} > 20 && {1})) &&
                !(lep_RadiusCO_{0} > 20 && ({1}))))
            """
        mat_conv = [material_conv_cut.format(i, lep_Mtrktrk_atConvV_CO_cut.format(i)) for i in [1, 2]]
        ele = [ele_cut.format(
            i,
            lep_Mtrktrk_atConvV_CO_cut.format(i),
            lep_IntConv_atPV_CO_cut.format(i)
        ) for i in [1, 2]]

        self._de('material_con_1', mat_conv[0], store=False)
        self._de('material_con_2', mat_conv[1], store=False)
        self._de('ele_1', ele[0], store=False)
        self._de('ele_2', ele[1], store=False)

        #   Heavy Flavor electron/muon:
        self._de(
            'Evt_ID_HF', f'{get_id(0, WType.loose)} && {get_id(1, WType.loose)} && {get_id(2, WType.loose)}',
            store=False
        )



        # convert to Hist1D
        region_names = [
            'SR', 'WZ',
            'ExtConv', 'HF_e', 'HF_m',
            'ExtConv_sys', 'HF_e_sys', 'HF_m_sys'
        ]
        region_selections = [
            reg_sel.sr, reg_sel.wz,
            reg_sel.ext_conv, reg_sel.hf_e, reg_sel.hf_m,
            reg_sel.ext_conv_sys, reg_sel.hf_e_sys, reg_sel.hf_m_sys
        ]

        # hist_count = {
        #     "weighted": [],
        #     "raw": []
        # }
        for r, s in zip(region_names, region_selections):
            self._de(f'Evt_{r}_count', gen_selection(select=s.copy()), store=True)
            self._de(f'Evt_{r}', f'Evt_{r}_count == {len(s)}')
            # hist_count['weighted'] += [self.rdf.Histo1D(
            #     (f"Evt_{r}_count_weighted", ";Selection;Yields", len(s) + 1, -0.5, len(s) + 0.5),
            #     f"Evt_{r}_count", "weight"
            # )]
            # hist_count['raw'] += [self.rdf.Histo1D(
            #     (f"Evt_{r}_count_raw", ";Selection;Yields", len(s) + 1, -0.5, len(s) + 0.5),
            #     f"Evt_{r}_count"
            # )]

        '''
        Sample Selection
        '''
        # Prompt Events
        self._de(
            'Evt_Prompt',  # Truth Prompt Check
            '&&'.join([f'lep_isPrompt_{i}' for i in lep_loop])
        )
        brem_election_cut = " (lep_truthParentPdgId_{0} == (int) lep_ID_{0} && lep_truthParentType_{0} == 2) "
        prompt_e_cut = "(abs((int) lep_ID_{0}) == 11 && lep_truthOrigin_{0} == 5 && {1})"
        prompt_m_cut = "(abs((int) lep_ID_{0}) == 13 && lep_truthOrigin_{0} == 0)"

        prompt_e_1_cut = prompt_e_cut.format(1, brem_election_cut.format(1))
        prompt_e_2_cut = prompt_e_cut.format(2, brem_election_cut.format(2))
        prompt_m_1_cut = prompt_m_cut.format(1)
        prompt_m_2_cut = prompt_m_cut.format(2)

        self._de(
            'sample_prompt_1',
            f'lep_isPrompt_1 || ({prompt_e_1_cut}) || ({prompt_m_1_cut})',
            store=False,
        )
        self._de(
            'sample_prompt_2',
            f'lep_isPrompt_2 || ({prompt_e_2_cut}) || ({prompt_m_2_cut})',
            store=False,
        )
        # prompt
        self._de('Sample_Prompt', 'sample_prompt_1 && sample_prompt_2')
        # qmisID
        self._de('Sample_QMisID', 'lep_isQMisID_0 > 0 || lep_isQMisID_1 > 0 || lep_isQMisID_2 > 0')

        self._de(
            'sample_conv',
            f"""(lep_truthOrigin_1 == 5 && !{brem_election_cut.format(1)}) || 
                    (lep_truthOrigin_2 == 5 && !{brem_election_cut.format(2)})""",
            store=False
        )
        self._de(
            'sample_qed',
            f"""(lep_truthParentType_1 == 21 && lep_truthParentOrigin_1 == 0) || 
                    (lep_truthParentType_2 == 21 && lep_truthParentOrigin_2 == 0)""",
            store=False
        )
        # ext/int conversion
        self._de('Sample_ExtConv', '!(Sample_Prompt && Sample_QMisID) && (sample_conv && !sample_qed)')
        self._de('Sample_intConv', '!(Sample_Prompt && Sample_QMisID) && (sample_conv && sample_qed)')

        hf_cut = """
            (lep_truthOrigin_{0} >= 25 && lep_truthOrigin_{0} <= 29) || 
            lep_truthOrigin_{0} == 32 ||
            lep_truthOrigin_{0} == 33
            """
        hf_e = f"""
            ((({hf_cut.format(1)}) && abs(lep_ID_1) == 11) || (({hf_cut.format(2)}) && abs(lep_ID_2) == 11)) &&
            (   (!({prompt_e_1_cut} || lep_isPrompt_1) && abs(lep_ID_1) == 11) ||
                (!({prompt_e_2_cut} || lep_isPrompt_2) && abs(lep_ID_2) == 11) ) &&
            !sample_conv
            """
        hf_m = f"""
            ((({hf_cut.format(1)}) && abs(lep_ID_1) == 13) || (({hf_cut.format(2)}) && abs(lep_ID_2) == 13)) &&
            (   (!({prompt_m_1_cut} || lep_isPrompt_1) && abs(lep_ID_1) == 13) ||
                (!({prompt_m_2_cut} || lep_isPrompt_2) && abs(lep_ID_2) == 13) ) &&
            !sample_conv
            """
        lf_e = f"""
            !((({hf_cut.format(1)}) && abs(lep_ID_1) == 11) || (({hf_cut.format(2)}) && abs(lep_ID_2) == 11)) &&
            (   (!({prompt_e_1_cut} || lep_isPrompt_1) && abs(lep_ID_1) == 11) ||
                (!({prompt_e_2_cut} || lep_isPrompt_2) && abs(lep_ID_2) == 11) ) &&
            !sample_conv
            """
        lf_m = f"""
            !((({hf_cut.format(1)}) && abs(lep_ID_1) == 13) || (({hf_cut.format(2)}) && abs(lep_ID_2) == 13)) &&
            (   (!({prompt_m_1_cut} || lep_isPrompt_1) && abs(lep_ID_1) == 13) ||
                (!({prompt_m_2_cut} || lep_isPrompt_2) && abs(lep_ID_2) == 13) ) &&
            !sample_conv
            """

        # heavy-flavor electron/muon
        self._de('Sample_HF_e', hf_e)
        self._de('Sample_HF_m', hf_m)
        self._de('Sample_LF_e', lf_e)
        self._de('Sample_LF_m', lf_m)

        self.rdf = self.rdf.Filter('||'.join([f'Evt_{r}' for r in region_names]), 'final output')

        for branch in bs.branches:
            self.cols.push_back(branch)

        snap_option = ROOT.RDF.RSnapshotOptions()
        snap_option.fMode = "UPDATE"
        snap_option.fOverwriteIfExists = True

        original_list_inter = list(set(original_list) & set(self.rdf.GetColumnNames()))
        original_sys_list_inter = list(set(original_sys_list) & set(self.rdf.GetColumnNames()))

        for b in original_list_inter:
            self.cols.push_back(b)
        # if self.kl:
        #     for b in original_kl_list:
        #         self.cols.push_back(b)
        if not self.is_data:
            for b in original_sys_list_inter:
                self.cols.push_back(b)
                

        self.rdf.Snapshot(self.tree_name, self.output_dir, self.cols, snap_option)

        end = perf_counter()
        self.logger.info(f'[{self.file.GetName()} {self.tree_name}] - Done in {end - start:0.2f} sec')

        # report.Print()
        return report
