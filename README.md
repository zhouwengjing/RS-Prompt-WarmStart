## 环境搭建

基础环境准备（需要先下载好 anaconda3）

在终端创建名为 multimodal-env 虚拟环境并切换到这环境上

```
conda create -n multimodal-env python=3.10
conda activate multimodal-env
```

我们的实验都是在 GPU 上运行的

```
# 检查是否有 NVIDIA GPU
nvidia-smi
```

我们下载的是 PyTorch 12.6 版本

访问 PyTorch 的官方网站  https://pytorch.org/get-started/locally/

在刚才打开的终端下载 requirements.txt 中的库文件

```
pip install -r requirements.txt
```

环境配置完整！

## 数据下载

所有下载好的数据均需放入根目录里的 data 文件夹下，详细结构如下如下 

![image-20260117110956250](C:\Users\WenJing\Desktop\image-20260117110956250.png)

**EuroSAT RGB version Dataset（eursoat）**

可在 GitHub、keggle、Zenodo 等网站上下载

仅提供 GitHub 下载地址：

下载链接： https://github.com/phelber/eurosathttps://www.kaggle.com/datasets/apollo2506/eurosat-dataset

或直接利用代码通过Hugging Face Datasets直接加载

```py
from datasets import load_dataset
dataset = load_dataset("eurosat")
```




**UC Merced Land Use Dataset（UCMereced_LandUse）**

直接从官方网站下载

下载链接： http://weegee.vision.ucmerced.edu/datasets/landuse.html

**NWPU-RESISC45**

![image-20260117103047428](C:\Users\WenJing\AppData\Roaming\Typora\typora-user-images\image-20260117103047428.png)
