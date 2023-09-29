import logging
import os

from util.submit_dsid import datasets_df


def setup_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s: [ %(name)s: %(levelname)s ] => %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.propagate = False

    return logger


logger = setup_logger(__name__)


class Config:

    def __init__(self):
        self.root_dir = os.path.dirname(__file__)

        self.data_dir = os.path.join(self.root_dir, '../data')
        self.bdtg_xml = os.path.join(self.data_dir, 'mva', 'hh3l_BDTG_fold_{0}.xml')
        self.macro_dir = os.path.join(self.root_dir, '../macros')
        self.macros = self._load_macros()


    def _load_macros(self):
        func = ""
        for filename in os.listdir(self.macro_dir):
            if not filename.endswith(".cpp"):
                continue
            logger.info(f'loading macro {filename}')

            with open(os.path.join(self.macro_dir, filename), "r") as f:
                for l in f:
                    func += l
                pass

        func += self._generate_dsid_map()
        func += """\
            float get_mass_from_pdg(float pdg){
                float mass = 0;
                if( fabs(pdg)==11 ){
                        mass = 0.5;
                } else if( fabs(pdg)==13 ){
                        mass = 105.6;
                } else{
                    std::cout<<"error: pdg=="<<pdg<<std::endl;
                }
                return mass;
            }
        """

        # print(func)

        return func

    def _generate_dsid_map(self):
        func = "\nTString map_dsid(int dsid){\n"
        for index, row in datasets_df.iterrows():
            func += f"    if( dsid == {row['DSID']}) return TString(\"{row['Sample']}\");\n"

        func += "    return TString(\"Unknown\"); \n}"

        return func


con = Config()


def load_config():
    global con


def get_sample_name(dsid):
    series = datasets_df.loc[datasets_df['DSID'] == f'{dsid}', 'Sample'].values
    if len(series) >= 1:
        return series[0]
    else:
        return 'Unknown'


if __name__ == '__main__':
    load_config()

    # con.generate_dsid_map()

    pass
