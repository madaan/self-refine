import json
import ast

from argparse import ArgumentParser
from tqdm import tqdm

def count_functions(code):
    tree = ast.parse(code)
    num_functions = sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
    return num_functions

def main():
    parser = ArgumentParser()
    parser.add_argument('--file', type=str)
    args = parser.parse_args()
    
    output = args.file[:-6] + '_func_count.jsonl'

    score_counter = None

    with open(output, 'w') as fout:
        input_lines = open(args.file).readlines()
        for line in tqdm(input_lines, total=len(input_lines)):
            data = json.loads(line)
            original_code = data['original_code']
            updated_codes = [x['updated_code'] for x in data['updates']]
            
            if score_counter is None:
                score_counter = [[] for _ in range(len(updated_codes) + 1)]
                
            data['update_func_count'] = []
            
            for i, code in enumerate([original_code] + updated_codes):
                num_functions = -1
                
                if code:
                    try:
                        num_functions = count_functions(code)
                    except:
                        pass
                
                data['update_func_count'].append(num_functions)
                score_counter[i].append(max(0, num_functions))
            fout.write(json.dumps(data) + '\n')

    for i, scores in enumerate(score_counter):
        print(f'Update {i}: avg function number {sum(scores) / len(scores)}')

if __name__ == '__main__':
    main()