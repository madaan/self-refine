import json
import tokenize

from tqdm import tqdm
from io import BytesIO
from argparse import ArgumentParser


def count_comments(code):
    comment_count = 0
    total_lines = len([l for l in code.splitlines() if l.strip()])
    
    tokens = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
    for token in tokens:
        if token.type == tokenize.COMMENT:
            comment_count += 1
    return comment_count, comment_count / total_lines

def main():
    parser = ArgumentParser()
    parser.add_argument('--file', type=str)
    args = parser.parse_args()
    
    output = args.file[:-6] + '_comment_count.jsonl'

    score_counter = None
            
    with open(output, 'w') as fout:
        input_lines = open(args.file).readlines()
        for line in tqdm(input_lines, total=len(input_lines)):
            data = json.loads(line)
            original_code = data['original_code']
            updated_codes = [x['updated_code'] for x in data['updates']]
            
            if score_counter is None:
                score_counter = [[] for _ in range(len(updated_codes) + 1)]
                
            data['update_comment_count'] = []
            data['update_comment_ratio'] = []
        
            for i, code in enumerate([original_code] + updated_codes):
                num_comments, ratio = -1, -1
                
                if code:
                    try:
                        num_comments, ratio = count_comments(code)
                    except:
                        pass
                
                data['update_comment_count'].append(num_comments)
                data['update_comment_ratio'].append(ratio)
                
                score_counter[i].append(max(0, ratio))
            fout.write(json.dumps(data) + '\n')
    
    for i, scores in enumerate(score_counter):
        print(f'Update {i}: avg comment ratio {sum(scores) / len(scores)}')

if __name__ == '__main__':
    main()
            
            