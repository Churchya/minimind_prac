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
                    raise ValueError("不是JSON数组")
                
                with tqdm(total=len(data), desc="处理JSON数组") as pbar:
                    for item in data:
                        try:
                            dialog = process_item(item)
                            f_out.write(json.dumps(dialog, ensure_ascii=False) + '\n')
                            processed += 1
                        except Exception as e:
                            errors += 1
                            print(f"\n错误条目：{item.get('id','未知ID')} - {str(e)}")
                        finally:
                            pbar.update(1)
                            
            except json.JSONDecodeError:
                
                f_in.seek(0)
                with tqdm(total=total_lines, desc="处理JSONL文件") as pbar:
                    for line in f_in:
                        try:
                            item = json.loads(line.strip())
                            dialog = process_item(item)
                            f_out.write(json.dumps(dialog, ensure_ascii=False) + '\n')
                            processed += 1
                        except Exception as e:
                            errors += 1
                            print(f"\n错误行内容：{line[:50]}... - {str(e)}")
                        finally:
                            pbar.update(1)

        print(f"\n转换完成！成功处理：{processed}条，失败：{errors}条")
        print(f"输出文件：{os.path.abspath(output_file)}")

    except Exception as e:
        print(f"程序运行失败：{str(e)}")
        sys.exit(1)

def process_item(item):
    """处理单个JSON条目"""
    
    if 'translation' not in item:
        raise KeyError("缺少translation字段")
    if 'en' not in item['translation'] or 'zh' not in item['translation']:
        raise KeyError("translation字段缺少en/zh子字段")
    
    return {
        "conversations": [
            {
                "role": "user",
                "content": f"帮我翻译这段英文：{item['translation']['en']}"
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