#coding=gbk
import json
import re
from pathlib import Path

def clean_special_symbols(text):
    """ͳһ����������źͿո�"""
    # �滻����ת���ַ�
    replacements = {
        r'&quot;': '"',
        r'&apos;': "'",
        r'\u2028': ' ',  # �滻Unicode�зָ���
        r'\s+': ' '      # �ϲ�����ո�
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text.strip()

def process_item(item):
    """����������������Ի��ṹ"""
    try:
        # ��ϴ�����ı�
        en = clean_special_symbols(item["english"])
        zh = clean_special_symbols(item["chinese"])
        
        # �����Ի���ʽ
        return {
            "conversations": [
                {
                    "role": "user",
                    "content": f"���ҷ������Ӣ�ģ�{en}"
                },
                {
                    "role": "assistant", 
                    "content": zh
                }
            ]
        }
    except KeyError as e:
        print(f"������ȱ�ٱ�Ҫ�ֶΣ�{e}")
        return None
    except Exception as e:
        print(f"����������ʱ��������{str(e)}")
        return None

def stream_process_json(input_path, output_path, batch_size=1000, max_samples=None):
    """
    ��ʽ�����JSON�ļ�
    Args:
        input_path: �����ļ�·��
        output_path: ����ļ�·��
        batch_size: �������С
        max_samples: �������������None��ʾ������������
    """
    output = Path(output_path)
    
    # ��ջ򴴽�����ļ�
    with output.open('w', encoding='utf-8') as f:
        pass
    
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            # ���Խ���ΪJSON����
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("������Ч��JSON����")
                
            # ���������ʽ
            with output.open('a', encoding='utf-8') as out_f:
                batch = []
                processed_count = 0
                
                for item in data:
                    if max_samples and processed_count >= max_samples:
                        break
                        
                    processed = process_item(item)
                    if processed:
                        batch.append(json.dumps(processed, ensure_ascii=False))
                        processed_count += 1
                        
                        # ����д��
                        if len(batch) >= batch_size:
                            out_f.write('\n'.join(batch) + '\n')
                            batch = []
                
                # д��ʣ������
                if batch:
                    out_f.write('\n'.join(batch) + '\n')
                    
        except json.JSONDecodeError:
            # ���˴���JSONL��ʽ
            f.seek(0)
            with output.open('a', encoding='utf-8') as out_f:
                batch = []
                processed_count = 0
                
                for line in f:
                    if max_samples and processed_count >= max_samples:
                        break
                        
                    try:
                        item = json.loads(line)
                        processed = process_item(item)
                        if processed:
                            batch.append(json.dumps(processed, ensure_ascii=False))
                            processed_count += 1
                            
                            if len(batch) >= batch_size:
                                out_f.write('\n'.join(batch) + '\n')
                                batch = []
                    except json.JSONDecodeError:
                        print(f"������Ч��JSON�У�{line[:50]}...")
                        continue
                
                if batch:
                    out_f.write('\n'.join(batch) + '\n')

if __name__ == "__main__":
    input_file = "translation2019zh_train.json"
    output_file = "dataset/sft_512_trans.jsonl"
    
    stream_process_json(input_file, output_file)
    
    print(f"ת����ɣ�����ļ���{output_file}")