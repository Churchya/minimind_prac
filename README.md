# minimind实践记录与部分扩展
## 1.pretrain和sft加入检查点功能
- 由于pretrain和sft的训练时间较长，所以添加了save_checkpoint和load_checkpoint功能，单卡训练和多卡训练均支持。文件名后缀中有"self"的则是添加了检查点功能
## 2.添加数据处理的脚本
- 期望实现一个能中英对话且可以中译英的slm，所以增加了一些将翻译数据处理为sft所需格式的脚本，以及数据清洗,混合脚本
1.[将.parquet转化为.json格式](https://github.com/Churchya/minimind_prac/blob/master/data_clean.py)
2.[对sft所需数据集清洗](https://github.com/Churchya/minimind_prac/blob/master/data_clean.py)
3.[转化为sft所需格式](https://github.com/Churchya/minimind_prac/blob/master/sft_trans.py)
4.[将两个数据集按比例混合](https://github.com/Churchya/minimind_prac/blob/master/shuffle_data.py)
## 3.中英pretrain&sft数据集来源
- [jiarui1](https://huggingface.co/datasets/jiarui1/Minimind_train_dataset)
## 4.翻译数据集来源
- [数据集1](https://aistudio.baidu.com/datasetdetail/209041)格式如下：
  ```
  {"english": "This strong degree of metallic yarn , and traction ability.", "chinese": "这样的金银丝纱线牢固度好，牵引能力强。"}
  ```
- [news_commentry](https://huggingface.co/datasets/Helsinki-NLP/news_commentary/viewer/en-zh/train)
## 5.效果展示
![image](https://github.com/Churchya/minimind_prac/blob/master/images/1.png)
![image](https://github.com/Churchya/minimind_prac/blob/master/images/3.png)
![image](https://github.com/Churchya/minimind_prac/blob/master/images/4.png)
模型参数量为104M，pretrain使用中英混合的数据集，sft使用中英对话与翻译混合的数据集，比例约2:1, 之后lora微调使用更小量的数据集，同时手动添加一些日常短句的翻译文本，因为发现模型在翻译短句与一些日常对话语句时效果较差

