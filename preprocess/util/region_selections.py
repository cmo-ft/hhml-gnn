# Signal Region
sr = [
    "1",
    "total_leptons == 3",
    "abs(total_charge) == 1",
    "GlobalTrigDecision && custTrigMatch_LooseID_FCLooseIso_SLTorDLT",
    "nTaus_OR_Pt25_RNN == 0",
    "Evt_ChargeIDBDT > 0",
    "Evt_AuthorCut > 0",
    "lep_Pt_0 > 10.0* GeV && lep_Pt_1 > 15.0* GeV && lep_Pt_2 > 15.0*GeV",
    "nJets_OR >= 1",
    "nJets_OR_DL1r_77 == 0",
    "Evt_LowMassVeto > 0",
    "Evt_ZMassVeto > 0",
    "Evt_MlllVeto > 0",
    "Evt_ID_SR > 0 && Evt_ISO_SR > 0",
]

# WZ CR
wz = [
    "1",
    "total_leptons == 3",
    "abs(total_charge) == 1",
    "GlobalTrigDecision && custTrigMatch_LooseID_FCLooseIso_SLTorDLT",
    "nTaus_OR_Pt25_RNN == 0",
    "Evt_ChargeIDBDT > 0",
    "Evt_AuthorCut > 0",
    "lep_Pt_0 > 10.0* GeV && lep_Pt_1 > 15.0* GeV && lep_Pt_2 > 15.0*GeV",
    "nJets_OR >= 1",
    "nJets_OR_DL1r_77 == 0",
    "Evt_LowMassVeto > 0",
    "Evt_ZMassVeto == 0",  # require Z-mass window
    "Evt_MlllVeto > 0",
    "Evt_ID_SR > 0 && Evt_ISO_SR > 0",
    # "met_met >= 30 * GeV",  # require missing energy for W
]

# External/Material Conversion
ext_conv = [
    "1",
    "total_leptons == 3",
    "abs(total_charge) == 1",
    "GlobalTrigDecision && custTrigMatch_LooseID_FCLooseIso_SLTorDLT",
    "nTaus_OR_Pt25_RNN == 0",
    "Evt_ChargeIDBDT > 0",
    "Evt_AuthorCut > 0",
    "lep_Pt_0 > 10.0* GeV && lep_Pt_1 > 15.0* GeV && lep_Pt_2 > 15.0*GeV",
    # "nJets_OR >= 1",  # newly added
    "nJets_OR_DL1r_77 == 0",
    "Evt_LowMassVeto > 0",
    # "Evt_ZMassVeto > 0",
    "Evt_MlllVeto == 0",
    "Evt_ID_SR > 0 && Evt_ISO_SR > 0",
    "(material_con_1 && material_con_2) && !(ele_1 && ele_2)",  # specify for material conversion
]

ext_conv_sys = [
    "1",
    "total_leptons == 3",
    "abs(total_charge) == 1",
    "GlobalTrigDecision && custTrigMatch_LooseID_FCLooseIso_SLTorDLT",
    "nTaus_OR_Pt25_RNN == 0",
    "Evt_ChargeIDBDT > 0",
    "Evt_AuthorCut == 0",  # sys
    # "nJets_OR >= 1", # newly added
    "lep_Pt_0 > 10.0* GeV && lep_Pt_1 > 15.0* GeV && lep_Pt_2 > 15.0*GeV",
    "nJets_OR_DL1r_77 == 0",
    "Evt_LowMassVeto > 0",
    "Evt_MlllVeto == 0",
    "Evt_ID_SR > 0 && Evt_ISO_SR == 0",  # sys
    "(material_con_1 && material_con_2) && !(ele_1 && ele_2)",  # specify for material conversion
]

hf_common = [
    "1",
    "total_leptons == 3",
    "abs(total_charge) == 1",
    "GlobalTrigDecision && custTrigMatch_LooseID_FCLooseIso_SLTorDLT",
    "nTaus_OR_Pt25_RNN == 0",
    "Evt_ChargeIDBDT > 0",
    "Evt_AuthorCut > 0",
    "(lep_Pt_0 > 10.0 * GeV) && (lep_Pt_1 > 15.0* GeV) && (lep_Pt_2 > 15.0*GeV)",
    "nJets_OR >= 1",  # high jet multiplicities
    "nJets_OR_DL1r_77 >= 1",  # high jet multiplicities
    "Evt_LowMassVeto > 0",
    "Evt_ZMassVeto > 0",
    "Evt_MlllVeto > 0",
]

hf_e = [
    *hf_common,
    "Evt_ID_SR > 0",  # no isolation requirement, and tight id
    "(abs(lep_ID_1) == 11) && (abs(lep_ID_2) == 11)"  # two electrons
]

hf_e_sys = [
    *hf_common,
    # "Evt_ID_SR == 0",  # sys anti-tight id
    "Evt_ID_HF > 0 && Evt_ID_SR == 0",  # sys anti-tight id
    "abs(lep_ID_1) == 11 && abs(lep_ID_2) == 11"  # two electrons
]

hf_m = [
    *hf_common,
    "Evt_ID_SR > 0",  # no isolation requirement, and tight id
    "abs(lep_ID_1) == 13 && abs(lep_ID_2) == 13"  # two muons
]

hf_m_sys = [
    *hf_common,
    "Evt_ID_HF > 0 && Evt_ID_SR == 0",  # sys anti-tight id
    # "Evt_ID_SR == 0",  # sys anti-tight id
    "abs(lep_ID_1) == 13 && abs(lep_ID_2) == 13"  # two muons
]
