# 使用的简短说明

- 创建虚拟环境 virtualenv ~/.venv/pytorch, 进入环境 . ~/.venv/pytorch/bin/activate
- 安装依赖 pip install -r \<requireument.txt\>

	```pip
	ffmpeg-python==0.2.0
	torch==2.2.2
	torchaudio==2.2.2
	torchvision==0.17.2
	tqdm==4.66.2
	```

- 手动下载 [realesr-animevideov3](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth) 模型

- 运行 python video.py --help
