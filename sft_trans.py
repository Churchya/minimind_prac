#coding=gbk
import json
import re
from pathlib import Path

def clean_special_symbols(text):
    """统一处理特殊符号和空格"""
    # 替换常见转义字符
    replacements = {
        r'&quot;': '"',
        r'&apos;': "'",
        r'\u2028': ' ',  # 替换Unicode行分隔符
        r'\s+': ' '      # 合并多个空格
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text.strip()

def process_item(item):
    """处理单个数据项并构建对话结构"""
    try:
        # 清洗输入文本
        en = clean_special_symbols(item["english"])
        zh = clean_special_symbols(item["chinese"])
        
        # 构建对话格式
        return {
            "conversations": [
                {
                    "role": "user",
                    "content": f"帮我翻译这段英文：{en}"
                },
                {
                    "role": "assistant", 
                    "content": zh
                }
            ]
        }
    except KeyError as e:
        print(f"数据项缺少必要字段：{e}")
        return None
    except Exception as e:
        print(f"处理数据项时发生错误：{str(e)}")
        return None

def stream_process_json(input_path, output_path, batch_size=1000, max_samples=None):
    """
    流式处理大JSON文件
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        batch_size: 批处理大小
        max_samples: 最大处理样本数，None表示处理所有数据
    """
    output = Path(output_path)
    
    # 清空或创建输出文件
    with output.open('w', encoding='utf-8') as f:
        pass
    
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            # 尝试解析为JSON数组
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("不是有效的JSON数组")
                
            # 处理数组格式
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
                        
                        # 批量写入
                        if len(batch) >= batch_size:
                            out_f.write('\n'.join(batch) + '\n')
                            batch = []
                
                # 写入剩余数据
                if batch:
                    out_f.write('\n'.join(batch) + '\n')
                    
        except json.JSONDecodeError:
            # 回退处理JSONL格式
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
                        print(f"跳过无效的JSON行：{line[:50]}...")
                        continue
                
                if batch:
                    out_f.write('\n'.join(batch) + '\n')

if __name__ == "__main__":
    input_file = "translation2019zh_train.json"
    output_file = "dataset/sft_512_trans.jsonl"
    
    stream_process_json(input_file, output_file)
    
    print(f"转换完成！输出文件：{output_file}")