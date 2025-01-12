# DreamBooth 深度学习项目

## 概述

本项目是一个使用 DreamBooth 方法进行肖战图像修复训练微调。它允许用户通过提供实例图像和类图像来训练一个文本到图像的生成模型。项目使用了 Hugging Face 的 stabilityai/stable-diffusion-2-inpainting预训练模型。

## 前置条件

- Python 3.10
- PyTorch 2.1.0
- CUDA 11.8
- Transformers 4.28.0
- Accelerate 0.18.0
- Diffusers 0.16.0
- bitsandbytes 0.45.0

## 运行方法

1. 下载本项目到本地。
2. 确保已安装所有前置条件。
3. 下载数据集到合适位置
4. 运行 `dreambooth_train.ipynb` Jupyter Notebook 文件。

## 一些参数：

- `--pretrained_model_name_or_path`：预训练模型的路径或标识符。
- `--pretrained_vae_name_or_path`：预训练VAE的路径或标识符。
- `--instance_data_dir`：实例图像的数据目录。
- `--class_data_dir`：类图像的数据目录（如果启用了先验保持）。
- `--instance_prompt`：实例图像的提示。
- `--class_prompt`：类图像的提示（如果启用了先验保持）。
- `--output_dir`：输出目录，用于保存训练好的模型。
- `--train_batch_size`：训练批次大小。
- `--num_train_epochs`：训练的轮数。
- `--learning_rate`：学习率。
- `--adam_beta1`、`--adam_beta2`、`--adam_weight_decay`、`--adam_epsilon`：Adam优化器的超参数。
- `--max_grad_norm`：最大梯度范数。
- `--push_to_hub`：是否将模型推送到Hugging Face Hub。

## 注意

- 代码基于https://github.com/ShivamShrirao/diffusers/blob/main/examples/dreambooth/train_inpainting_dreambooth.py和https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py进行修改
- 本项目的运行环境为 Google Colab，如果您在其他环境中运行，可能需要调整代码以适应不同的环境配置。

