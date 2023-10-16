import ROOT
import ROOT.RDataFrame as RDF
import argparse
import logging
import os
from pathlib import Path
from time import perf_counter

from util.config import load_config, setup_logger
from preprocessor import Preprocessor

logger = setup_logger('runner')


def main(input: str, output: str = "", is_data: bool = True, only_nominal: bool = True):
    logger.info('Start')

    run_all = not is_data and not only_nominal
    logger.info(f'Input: {input}')
    logger.info(f'Output: {output}')
    logger.info(f'Is DATA: {is_data}')
    logger.info(f'Run only nominal: {not run_all}')
    start = perf_counter()

    load_config()

    tree_list = ['nominal']
    else_list = []

    f = ROOT.TFile(input)

    if run_all:
        for obj in f.GetListOfKeys():
            name = obj.GetName()
            tree = f.Get(name)
            is_tree = tree.IsA().InheritsFrom(ROOT.TTree.Class())
            if is_tree and ('__1up' in name or '__1down' in name or 'MET_SoftTrk' in name):
                tree_list += [name]
            else:
                else_list += [name]

    logger.info(f'Loop over {len(tree_list)} trees')
    
    prep = Preprocessor(f, output, is_data)

    for idx, tree in enumerate(tree_list, start=1):
        prep.apply(tree, f'{idx} / {len(tree_list)}')

    end = perf_counter()
    logger.info(f'Done in {end - start:0.2f} sec')



if __name__ == '__main__':
    # setup logger
    logging.root.setLevel(logging.NOTSET)

    parser = argparse.ArgumentParser(description='Analysis Framework for HH->Multilepton-> 3 leptons')
    parser.add_argument('-i', '--input', required=True, type=str, help='input root file')
    parser.add_argument('-o', '--output', type=str, help='output root file [directory]')

    parser.add_argument('--data', action=argparse.BooleanOptionalAction, default=False, help='flag for DATA')
    parser.add_argument('--only_nominal', action=argparse.BooleanOptionalAction, default=False,
                        help='flag for only process nominal tree')

    args = parser.parse_args()

    input = Path(args.input)
    output = f'{input.stem}_output{input.suffix}'
    if args.output is not None:
        if not os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)
        output = os.path.join(args.output, output)

    main(args.input, output, args.data, args.only_nominal)
