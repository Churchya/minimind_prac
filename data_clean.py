
import json
import re

def is_chinese_english(text):

    pattern = pattern = r'^[\u4e00-\u9fff\u3000-\u303fa-zA-Z0-9\s\.,\?!，%。？！:：""''\'\";\-\(\)（）《》/\[\]]+$'
    return bool(re.match(pattern, text))

def clean_data(input_file, output_file):
    cleaned_data = []
    filtered_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                conversations = data['conversations']
                
                valid = True
                for conv in conversations:
                    if not is_chinese_english(conv['content']):
                        valid = False
                        break
                
                if valid:
                    cleaned_data.append(data)

            except json.JSONDecodeError:
                continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in cleaned_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    # with open(filtered_output_file, 'w', encoding='utf-8') as f:
    #     for data in filtered_data:
    #         f.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    input_file = 'dataset/sft_512_trans.jsonl'  #
    output_file = 'dataset/sft_512_trans_clean.jsonl'  #
    #filtered_output_file = 'dataset/sft_512_trans2_filtered.jsonl'  # 
    clean_data(input_file, output_file)