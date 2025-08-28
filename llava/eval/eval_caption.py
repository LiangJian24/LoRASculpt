import os
import argparse
import json
import re
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import tempfile





class COCOEvaler(object):
    def __init__(self, annfile):
        super(COCOEvaler, self).__init__()
        self.coco = COCO(annfile)
        args = get_args()
        if not os.path.exists(f'{args.output_dir}/tmp'):
            os.mkdir(f'{args.output_dir}/tmp')

    def eval(self, result):
        args = get_args()
        in_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=f'{args.output_dir}/tmp')
        json.dump(result, in_file)
        in_file.close()

        cocoRes = self.coco.loadRes(in_file.name)
        cocoEval = COCOEvalCap(self.coco, cocoRes)
        cocoEval.evaluate()
        os.remove(in_file.name)
        return cocoEval.eval
def jsonl2json(pred):
    data=[]
    for pred_ in pred:
        data.append(pred_)
    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file',default='', type=str)
    parser.add_argument('--result-file',default='', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--summary-output-dir', type=str, default=None)
    return parser.parse_args()

def main() -> None:
    args = get_args()
    evaler = COCOEvaler(args.annotation_file)
    preds= [json.loads(line) for line in open(args.result_file)]
    preds=jsonl2json(preds)
    json.dump(preds,open(f'{args.output_dir}/preds,json','w'))
    res=evaler.eval(json.load(open(f'{args.output_dir}/preds,json')))
    print(res)


    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'result-coco.txt')
        with open(output_file, 'w') as f_sum:
            f_sum.write('CIDEr on COCO: {:.4f}\n'.format(res['CIDEr']))

    if args.summary_output_dir is not None: 
        with open(args.summary_output_dir, 'a') as f_sum:
            f_sum.write('CIDEr on COCO: {:.4f}\n'.format(res['CIDEr']))

if __name__ == "__main__":
    main()
