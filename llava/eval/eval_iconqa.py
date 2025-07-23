import os
import argparse
import json



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default=None)
    parser.add_argument('--result-file', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--summary-output-dir', type=str, default=None)
    return parser.parse_args()



def eval_single(annotation_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    # annotations = json.load(open(annotation_file))
    annotations = [json.loads(line) for line in open(annotation_file)]
    annotations = {data['question_id']: data for data in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    total = len(results)
    right = 0
    for result in results:
        annotation = annotations[result['question_id']]
        ground_truth = annotation['answer']
        pred_list.append({"pred_answer": result['text']})

        if result['text'].lower() == ground_truth.lower():
            right += 1


    acc = 100. * right / total
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), acc))


    ### 将结果写到文件中 ###
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'result-iconqa.txt')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), acc))
    

    ### 将计算结果统一输出到一个汇总的txt文件中 ###
    if args.summary_output_dir is not None: 
        with open(args.summary_output_dir, 'a') as f_sum:
            f_sum.write('\nSamples: {}\nAccuracy on IconQA: {:.2f}%\n'.format(len(pred_list), acc))
    



if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)
