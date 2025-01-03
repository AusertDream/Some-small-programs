自己写的transformer模型，数据集来源WMT14，EN-DE

train大概450w条数据，validation和test都是3000条左右。

注意：

1. 选择device的时候，不仅main.py里面要改，transformer.py里面也要改，在embedding层，有三个创建的torch。要手动改的，别忘了，这里有点史，但是我懒得改了（
2. 训练好的参数文件就不放了，太大了，一直传不上去（我太菜了
3. 想直接跑的话，记得把相关文件和文件夹路径都准备好，linux下，如果中间文件夹不存在会直接报错，我没做存在性检查。
4. config.json文件夹里面有部分参数是没用的，注意甄别。
5. tokenizer我选择了预训练好的llama3.2的tokenizer，这里没给，要参数的去huggingface上下载或者魔塔上下载也行。

