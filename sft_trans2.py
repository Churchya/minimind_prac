#coding=gbk
import json
from tqdm import tqdm
import sys
import os

def convert_json_to_dialog(input_file, output_file):

    try:
        total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
        processed = 0
        errors = 0
        
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            
            try:
                data = json.load(f_in)
                if not isinstance(data, list):
                    raise ValueError("����JSON����")
                
                with tqdm(total=len(data), desc="����JSON����") as pbar:
                    for item in data:
                        try:
                            dialog = process_item(item)
                            f_out.write(json.dumps(dialog, ensure_ascii=False) + '\n')
                            processed += 1
                        except Exception as e:
                            errors += 1
                            print(f"\n������Ŀ��{item.get('id','δ֪ID')} - {str(e)}")
                        finally:
                            pbar.update(1)
                            
            except json.JSONDecodeError:
                
                f_in.seek(0)
                with tqdm(total=total_lines, desc="����JSONL�ļ�") as pbar:
                    for line in f_in:
                        try:
                            item = json.loads(line.strip())
                            dialog = process_item(item)
                            f_out.write(json.dumps(dialog, ensure_ascii=False) + '\n')
                            processed += 1
                        except Exception as e:
                            errors += 1
                            print(f"\n���������ݣ�{line[:50]}... - {str(e)}")
                        finally:
                            pbar.update(1)

        print(f"\nת����ɣ��ɹ�����{processed}����ʧ�ܣ�{errors}��")
        print(f"����ļ���{os.path.abspath(output_file)}")

    except Exception as e:
        print(f"��������ʧ�ܣ�{str(e)}")
        sys.exit(1)

def process_item(item):
    """������JSON��Ŀ"""
    
    if 'translation' not in item:
        raise KeyError("ȱ��translation�ֶ�")
    if 'en' not in item['translation'] or 'zh' not in item['translation']:
        raise KeyError("translation�ֶ�ȱ��en/zh���ֶ�")
    
    return {
        "conversations": [
            {
                "role": "user",
                "content": f"���ҷ������Ӣ�ģ�{item['translation']['en']}"
            },
            {
                "role": "assistant",
                "content": item['translation']['zh']
            }
        ]
    }

if __name__ == "__main__":
    input_json = "trans_data2.json"  
    output_jsonl = "dataset/sft_512_trans2.jsonl"  
    
    convert_json_to_dialog(input_json, output_jsonl)