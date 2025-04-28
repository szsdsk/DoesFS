# 基于单对样例的可控人脸风格化变形

## 开始训练
训练前确保环境正常配置，通过以下指令配置环境 `pip install -r requirements.txt`

同时确保各个预训练模型正确下载且放置到 `./checkpoints`文件夹中。
需要的预训练模型为：[alex.pth](https://drive.google.com/drive/folders/1niTspxYSQi62Vgqai5QEgBUcdLRwy9lw?usp=drive_link)、[alexnet-owt-7be5be79.pth](https://download.pytorch.org/models/alexnet-owt-7be5be79.pth)、[e4e_ffhq_encode.pt](https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view?usp=sharing)、[stylegan2-ffhq-config-f.pt](https://drive.usercontent.google.com/download?id=1Yr7KuD959btpmcKGAUsbAk5rPjX2MytK&authuser=0)、[shape_predictor_68_face_landmarks.dat](https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)


然后就可以开始训练，训练前确保训练的面部图片已经进行过面部对齐，如果没有对齐的话，可以使用 `fase_align.py`对图片进行对齐：
```bash
python face_align.py --path=[YOUR_IMAGE_PATH] --output=[PATH_TO_SAVE]
```
对齐后将对齐的文件放到 `./data/style_images_aligned`文件夹中。

然后就可以通过 `train.py`进行模型的训练，训练好的模型会被放到`./outputs/models`文件夹：
```bash
python train.py --style=[STYLE_NAME] --source=[REAL_IMAGE_PATH] --target=[TARGET_IMAGE_PATH] 
```
例如：
```bash
python train.py --style=style1 --source=source1.png --target=target1.png
```

也可以使用 `--use_adv`、 `--use_cons`、 `--use_ca`、`--use_ca`、`--use_stn`、 `--use_rtstn` 来进行消融实验相关模型的训练，来选择模型训练是是否使用各种损失及颜色对齐和空间变换模块。

例如使用以下命令行来训练不适用空间变换器的模型：
```bash
python train.py --style=style1 --source=source1.png --target=target1.png --use_stn=False --use_rtstn=False
```

## 推理
可以使用 `inference.py`来进行模型推理，使用前确保将训练好的模型放置到 `./checkpoints`文件夹中。
也可以使用 `--alpha` 来控制变形程度。
```bash
python inference.py --style=[STYLE_NAME] --input_image=[IMAGE_PATH] --alpha=0.8
```
结果图片会被放置到 `./outputs/inference`文件夹中。

## 变形控制
可以使用 `deformation_control.py` 来探究 `alpha`对结果的影响，指定 `alpha` 在 `alpha0` 到 `alpha1`之间变化。
```bash
python deformation.py --alpha0=-1.0 --alpha1=1.5 --style=[STYLE_NAME]--input_image=[IMAGE_PATH]
```

## DINO特征可视化
可以使用 `self_sim_pca.py` 来对DINO中间特征的自相似矩阵进行pca可视化：
```bash
python self_sim_pca.py --image_path=[IMAGE_PATH] --save_path=[SAVE_PATH]
```
