# minimind实践记录与部分扩展
## 1.pretrain和sft加入检查点功能
- 由于pretrain和sft的训练时间较长，所以添加了save_checkpoint和load_checkpoint功能，单卡训练和多卡训练均支持。文件名后缀中有"self"的则是添加了检查点功能
## 2.添加数据处理的脚本
- 期望实现一个能中英对话且可以中译英的slm，所以增加了一些将翻译数据处理为sft所需格式的脚本，以及数据清洗,混合脚本
- [将.parquet转化为.json格式](https://github.com/Churchya/minimind_prac/blob/master/data_clean.py)
- [对sft所需数据集清洗](https://github.com/Churchya/minimind_prac/blob/master/data_clean.py)
- [转化为sft所需格式](https://github.com/Churchya/minimind_prac/blob/master/sft_trans.py)
- [将两个数据集按比例混合](https://github.com/Churchya/minimind_prac/blob/master/shuffle_data.py)
## 3.中英pretrain&sft数据集来源
- [](https://huggingface.co/datasets/jiarui1/Minimind_train_dataset)
## 4.翻译数据集来源
- [数据集1](https://aistudio.baidu.com/datasetdetail/209041)格式如下：
  ```
  {"english": "This strong degree of metallic yarn , and traction ability.", "chinese": "这样的金银丝纱线牢固度好，牵引能力强。"}
  ```
- [news_commentry](https://huggingface.co/datasets/Helsinki-NLP/news_commentary/viewer/en-zh/train)
## 5.效果展示
