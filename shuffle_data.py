#coding=gbk
import json
import random

def merge_jsonl_files(file1_path, file2_path, output_path, ratio1=1, ratio2=1):
    #
    data1 = []
    with open(file1_path, 'r', encoding='utf-8') as f:
        for line in f:
            data1.append(json.loads(line))
            
    #
    data2 = []
    with open(file2_path, 'r', encoding='utf-8') as f:
        for line in f:
            data2.append(json.loads(line))
            
    # 
    sample_size1 = int(len(data1) * ratio1)
    sampled_data1 = random.sample(data1, sample_size1)
    sample_size2 = int(len(data2) * ratio2)
    sampled_data2 = random.sample(data2, sample_size2)
    
    # 
    merged_data = sampled_data1 + sampled_data2
    
    # 
    random.shuffle(merged_data)
    
    #
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in merged_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
            
    print(f'合并完成!')
    print(f'从第一个数据集选取了 {len(sampled_data1)} 条数据')
    print(f'从第二个数据集选取了 {len(sampled_data2)} 条数据')
    print(f'合并后共 {len(merged_data)} 条数据')

if __name__ == '__main__':
    file1 = 'dataset/sft_512_mixed_shuffled.jsonl'  # 
    file2 = 'dataset/sft_512_trans_final.jsonl'  #  
    output = 'dataset/sft_512_trans_mixed_lora.jsonl'  # 
    merge_jsonl_files(file1, file2, output, ratio1=0.2, ratio2 = 0.1)


