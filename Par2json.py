
import pyarrow.parquet as pq
import pandas as pd

table = pq.read_table('dataset/en_zh.parquet')
df = table.to_pandas()

with open('trans_data2.json', 'w', encoding='utf-8') as f:

    df.to_json(f, 
              orient='records', 
              lines=True, 
              force_ascii=False) #