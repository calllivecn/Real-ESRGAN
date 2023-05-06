import os
import sys
import time
import math
import heapq
import argparse
import mimetypes
import fractions
import threading
from pathlib import Path


import cv2
import torch
import numpy as np
from torch import multiprocessing as torch_mp
from torch.multiprocessing import Queue as torch_Queue

from torch.utils.data import (
    Dataset,
    IterableDataset,
    DataLoader,
)

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from torch.nn import functional as F
from tqdm import tqdm

# from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

try:
    import ffmpeg
except ImportError:
    import pip
    pip.main(['install', '--user', 'ffmpeg-python'])
    import ffmpeg


# =================================================


# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR_utils = Path(__file__).parent

print(f"""{torch_mp.get_start_method("spawn")=}""")
# torch_mp.set_start_method("spawn")


class RealESRGANer():
    """A helper class for upsampling images with RealESRGAN.

    Args:
        scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
        model_path (str): The path to the pretrained model. It can be urls (will first download it automatically).
        model (nn.Module): The defined network. Default: None.
        tile (int): As too large images result in the out of GPU memory issue, so this tile option will first crop
            input images into tiles, and then process each of them. Finally, they will be merged into one image.
            0 denotes for do not use tile. Default: 0.
        tile_pad (int): The pad size for each tile, to remove border artifacts. Default: 10.
        pre_pad (int): Pad the input images to avoid border artifacts. Default: 10.
        half (float): Whether to use half precision during inference. Default: False.
    """

    def __init__(self,
                 scale,
                 model_path,
                 dni_weight=None,
                 model=None,
                 tile=0,
                 tile_pad=10,
                 pre_pad=10,
                 half=False,
                 device=None,
                 gpu_id=None):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half


        self.img_mode = 'L'
        self.max_range = None
        self.w_input = None
        self.h_input = None

        # initialize model
        if gpu_id:
            self.device = torch.device(
                f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu') if device is None else device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        if isinstance(model_path, list):
            # dni
            assert len(model_path) == len(dni_weight), 'model_path and dni_weight should have the save length.'
            loadnet = self.dni(model_path[0], model_path[1], dni_weight)
        else:
            # if the model_path starts with https, it will first download models to the folder: weights
            if str(model_path).startswith('https://'):
                model_path = load_file_from_url(
                    url=model_path, model_dir=ROOT_DIR_utils / 'weights', progress=True, file_name=None)
            loadnet = torch.load(model_path, map_location=torch.device('cpu'))

        # prefer to use params_ema
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)

        model.eval()
        self.model = model.to(self.device)
        if self.half:
            self.model = self.model.half()

    def dni(self, net_a, net_b, dni_weight, key='params', loc='cpu'):
        """Deep network interpolation.

        ``Paper: Deep Network Interpolation for Continuous Imagery Effect Transition``
        """
        net_a = torch.load(net_a, map_location=torch.device(loc))
        net_b = torch.load(net_b, map_location=torch.device(loc))
        for k, v_a in net_a[key].items():
            net_a[key][k] = dni_weight[0] * v_a + dni_weight[1] * net_b[key][k]
        return net_a

    def pre_process(self, img):
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible
        """
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        if self.half:
            self.img = self.img.half()

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        # mod pad for divisible borders
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')


    @torch.no_grad()
    def process(self, imgs):
        # model inference
        # self.output = self.model(self.img)
        return self.model(imgs)

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    with torch.no_grad():
                        output_tile = self.model(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self, output):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = output.size()
            output = output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = output.size()
            output = output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return output


    # @torch.no_grad()
    # def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
    def enhance_dataloader_pre(self, img):

        self.alpha_upsampler = 'realesrgan'

        self.h_input, self.w_input = img.shape[0:2]
        # img: numpy
        img = img.astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            self.max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            self.max_range = 255

        img = img / self.max_range
        if len(img.shape) == 2:  # gray image
            self.img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            self.img_mode = 'RGBA'
            self.alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.alpha_upsampler == 'realesrgan':
                self.alpha = cv2.cvtColor(self.alpha, cv2.COLOR_GRAY2RGB)
        else:
            self.img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ------------------- process image (without the alpha channel) ------------------- #
        self.pre_process(img)

        return self.img

    #
    def gpu2cpu(self, img):
        output_img = self.post_process(img)
        # 在这里把数据从gpu 拿回cpu 的, 从GPU拿回数据，这个操作要在同dataloader 进程中。
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        return output_img


    def enhance_dataloader_post(self, output_img, outscale=None):
        """
        这里拿到的 img 是 self.post_process()之后， 从gpu 拿回来之后的
        """

        """
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
        """

        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        if self.img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------- process the alpha channel if necessary ------------------- #
        if self.img_mode == 'RGBA':
            if self.alpha_upsampler == 'realesrgan':
                self.pre_process(self.alpha)

                """
                # 这一步先入着,以后在处理
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                """

                output_alpha = output_img
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:  # use the cv2 resize for alpha channel
                h, w = self.alpha.shape[0:2]
                output_alpha = cv2.resize(self.alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)

            # merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # ------------------------------ return ------------------------------ #
        if self.max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output, (
                    int(self.w_input * outscale),
                    int(self.h_input * outscale),
                ), interpolation=cv2.INTER_LANCZOS4)

        return output, self.img_mode


# =================================================

VIDEO_CONTAINER = (".mp4", ".mkv")


def get_video_meta_info(video_path):
    ret = {}
    filename = Path(video_path)
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    ret['fps'] = float(fractions.Fraction(video_streams[0]['avg_frame_rate']))

    ret["suffix"] = filename.suffix

    if not filename.suffix in VIDEO_CONTAINER:
        print(f"目前只支持 {VIDEO_CONTAINER} 格式, 可以先自行使用ffmpeg工具转换格式。")
        sys.exit(1)

    if filename.suffix.lower() == ".mp4":
        ret["duration"] = float(video_streams[0]["duration"])
        ret["nb_frames"] = int(video_streams[0].get('nb_frames'))

    elif filename.suffix.lower() == ".mkv":
        ret["duration"] = float(probe["format"]["duration"])
        ret['nb_frames'] = round(ret["duration"] * ret["fps"])

    return ret


class Reader:

    #def __init__(self, args, total_workers=1, worker_idx=0):
    def __init__(self, args, video_path):
        self.args = args
        input_type = mimetypes.guess_type(args.input)[0]
        self.input_type = 'folder' if input_type is None else input_type
        self.paths = []  # for image&folder type
        self.audio = None
        self.input_fps = None
        if self.input_type.startswith('video'):
            self.stream_reader = (
                ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='bgr24',
                                                loglevel='error').run_async(
                                                    pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
            meta = get_video_meta_info(video_path)
            self.width = meta['width']
            self.height = meta['height']
            self.input_fps = meta['fps']
            self.audio = meta['audio']
            self.nb_frames = meta['nb_frames']


    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        return self.nb_frames

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img


    def get_frame(self):
        return self.get_frame_from_stream()

    def close(self):
        if self.input_type.startswith('video'):
            self.stream_reader.stdin.close()
            self.stream_reader.wait()


class Writer:

    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width * args.outscale), int(height * args.outscale)
        if out_height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')

        # encoder = "libx265"
        encoder = args.encoder

        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 audio,
                                 video_save_path,
                                 pix_fmt='yuv420p',
                                 #vcodec='libx264',
                                 vcodec=encoder,
                                 loglevel='error',
                                 acodec='copy').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 video_save_path, pix_fmt='yuv420p', vcodec=encoder,
                                 loglevel='error').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))

    def write_frame(self, frame):
        frame = frame.astype(np.uint8).tobytes()
        self.stream_writer.stdin.write(frame)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()


# 2023-05-06
class StreamData(IterableDataset):

    def __init__(self, reader, inference):

        # self.reader = Reader(args, video_path)
        self.reader = reader

        self.inference = inference

    def __iter__(self):

        while (img := self.reader.get_frame()) is not None:
            img = self.inference.enhance_dataloader_pre(img)[0]
            yield img #.to(torch.device("cuda"))


# 后处理在单核性能不行， 使用 多进程处理。
class ModelPostPorcess:

    def __init__(self, func=None, pool=4, queue_size=16):
        """
        func: 必须是 func(task)
        """

        self._alived = True

        self.func = func

        cpu_count = os.cpu_count()
        if pool > cpu_count:
            pool = cpu_count
        else:
            self.pool = pool

        self.pools = []

        self.queue_size = queue_size
        # self.in_queue = mp.Queue(pool)
        # self.out_queue = mp.Queue(pool)
        self.in_queue = torch_Queue(self.queue_size)
        self.out_queue = torch_Queue(self.queue_size)
        self.pull_queue = torch_Queue(self.queue_size)

        self.in_seq = 0
        self.out_seq = 0
        self.pull_seq = 0
        self.stash = []

        if self.func is not None:
            self.submit_func(self.func)


    def submit_func(self, func):
        if self.func is not None:
            raise ValueError("函数已定义")
        self.func = func

        for i in range(self.pool):
            p = torch_mp.Process(target=self.__proc, name="后处理池", daemon=True)
            p.start()
            self.pools.append(p)

        th = threading.Thread(target=self.next_result, daemon=True)
        th.start()

        self.pools.append(th)

    def push(self, task):
        # print(f"push() ： {self.in_seq=}")
        if task is None:
            self.in_queue.put(None)
            return

        self.in_queue.put((self.in_seq, task))
        self.in_seq += 1

    def pull(self):
        result = self.pull_queue.get()
        if result is None:
            self._alived = False

        print(f"pull() ： {self.pull_seq=}")
        self.pull_seq += 1

        return result


    def next_result(self):
        """
        用来保证，任务是有序且连续的进来，和出去的。 self.stash 暂存区
        seq_frames: (seq, task)
        """
        while True:
            seq_frame = self.out_queue.get()

            if seq_frame is None:
                break

            heapq.heappush(self.stash, seq_frame)

            frames = []
            while len(self.stash) > 0 and self.out_seq == self.stash[0][0]:
                _, frame = heapq.heappop(self.stash)
                frames.append(frame)
                self.out_seq += 1

            [self.pull_queue.put(frame) for frame in frames]


    def join(self):
        for th in self.pools:
            th.join()


    def __proc(self):
        while True:
            seq_task = self.in_queue.get()

            if seq_task is None:
                self.out_queue.put(None)
                break

            seq, task = seq_task

            result = self.func(task)

            self.out_queue.put((seq, result))



    def is_alived(self):
        return self._alived


# def inference_video(args, put_queue, get_queue, device=None):
def inference_video(args, reader, get_queue, device=None):
    # ---------------------- determine models according to model names ---------------------- #
    args.model_name = args.model_name.split('.pth')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # ---------------------- determine model paths ---------------------- #
    # model_path = os.path.join('weights', args.model_name + '.pth')
    weights = Path('weights')
    model_path = weights / (args.model_name + '.pth')
    if not model_path.is_file():
        ROOT_DIR = Path(__file__).parent
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(url=url, model_dir=weights, progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        device=device,
    )

    if 'anime' in args.model_name and args.face_enhance:
        print('face_enhance is not supported in anime models, we turned this option off for you. '
              'if you insist on turning it on, please manually comment the relevant lines of code.')
        args.face_enhance = False

    if args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)  # TODO support custom device
    else:
        face_enhancer = None



    my_dataset = StreamData(reader, upsampler)
    video_frame_loader = DataLoader(
        my_dataset,
        batch_size=args.batch_size,
        # num_workers=4,
        # pin_memory=True,
    )

    print(f"inference_video() pid: {os.getpid()} 启动")

    INIT=True

    for imgs in video_frame_loader:
        # print(f"{imgs.size()=}")
        if imgs is None:
            # put_queue.put(None)
            print(f"inference_video() pid: {os.getpid()} 退出")
            break

        # seq, img = img

        imgs.to(upsampler.device)


        # 视频长宽信息，需要在预处理之后 在启动进程，才会有
        if INIT:
            get_queue.submit_func(lambda img: upsampler.enhance_dataloader_post(img, args.outscale))
            INIT = False

        try:
            # 这个先没支持 args.face_enhance 是 False
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(imgs, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                # output, _ = upsampler.enhance(img, outscale=args.outscale)
                outputs = upsampler.process(imgs)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            # raise error
            sys.exit(1)
        else:
            # print(f"后处理时 outputs 的位置：{outputs.size()=} {len(outputs)=} {outputs.device=}")
            l = len(outputs)
            for i in range(l):
                # 后处理
                img = upsampler.gpu2cpu(outputs[i])

                # 这里需要传到一个队列，进行多进程处理在返回。
                # output, _ = upsampler.enhance_dataloader_post(outputs[img_i], outscale=args.outscale)
                get_queue.push(img)

        # torch.cuda.synchronize(device)



    post_process_multi = get_queue
    post_process_multi.push(None)
    post_process_multi.join()
    # put_queue.close()
    # get_queue.close()


def run(args):

    # torch_mp.set_start_method('spawn')

    meta = get_video_meta_info(args.input)
    video_suffix = meta["suffix"]

    output = Path(args.output)

    input_path = Path(args.input)

    video_save_path = output / f"{os.path.splitext(input_path.name)[0]}_{args.suffix}{video_suffix}"

    # put_queue = torch_Queue(8)
    get_queue = torch_Queue(8)

    num_gpus = torch.cuda.device_count()

    total_process = num_gpus * args.num_process_per_gpu

    reader = Reader(args, input_path)

    # 启动GPU 进程
    process = []

    # 使用和dataloader batch_size 相同的进程数
    post_process_multi = ModelPostPorcess(pool=8)
    # post_process_multi = ModelPostPorcess(pool=args.batch_size)


    """
    for i in range(total_process):
        p = torch_mp.Process(target=inference_video, args=(args, put_queue, get_queue, torch.device(i % num_gpus)))
        p.start()
        process.append(p)

    print("model Pool start")
    """

    p = torch_mp.Process(target=inference_video, args=(args, reader, post_process_multi), name="CUDA连接处理器") # torch.device(i % num_gpus)))
    p.start()
    process.append(p)

    # 开始 main()
    # reader = Reader(args, input_path) # zx

    audio = reader.get_audio()
    height, width = reader.get_resolution()
    fps = reader.get_fps()
    writer = Writer(args, audio, height, width, str(video_save_path), fps)

    # p1 = torch_mp.Process(target=put2inference, args=(put_queue, reader))
    # p1.start()

    # p2 = torch_mp.Process(target=get4inference, args=(get_queue, writer))
    # p2.start()

    c = 0
    t1 = time.time()
    while (frame := post_process_multi.pull()) is not None:
        frame, img_mode = frame
        writer.write_frame(frame)
        t2 = time.time()
        t = t2 - t1
        c += 1
        if t >= 1:
            print(f"当前处理速度： {round(c/t, 1)} frame/s")
            c = 0
            t1 = t2

    # 等待GPU Pool 退出
    [ p.join() for p in process ]

    # 给 get4inference() 退出信号
    get_queue.put(None)

    # p1.join()
    # p2.join()

    print("已经inference_vide() 处理完了")

    # finally cleanup
    reader.close()
    writer.close()


def main():
    """Inference demo for Real-ESRGAN.
    It mainly for restoring anime videos.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input video, image or folder')
    parser.add_argument('-n', '--model_name', type=str, default='realesr-animevideov3',
        help=('Model names: realesr-animevideov3 | RealESRGAN_x4plus_anime_6B | RealESRGAN_x4plus | RealESRNet_x4plus |'
              ' RealESRGAN_x2plus | realesr-general-x4v3'
              'Default:realesr-animevideov3'))

    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')

    parser.add_argument('-dn', '--denoise_strength', type=float, default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))

    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored video')

    # 使用 dataloader 这个先不支持
    # parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')

    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')

    # 使用 dataloader 这个先不支持
    # parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')

    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument('--fps', type=float, default=None, help='FPS of the output video')
    parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg', help='The path to ffmpeg')
    parser.add_argument('--encoder', type=str, default='libx265', help='默认使用CPU(慢), 可以使用GPU(hevc_nvenc)')
    # parser.add_argument('--extract_frame_first', action='store_true')
    parser.add_argument('--num_process_per_gpu', type=int, default=1, help="每个 GPU 的数量进程")

    parser.add_argument('--alpha_upsampler', type=str, default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')

    parser.add_argument('--ext', type=str, default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')

    parser.add_argument("--batch-size", type=int, default=4, help="dataset batch size")

    args = parser.parse_args()

    # args.input = args.input.rstrip('/').rstrip('\\')
    # os.makedirs(args.output, exist_ok=True)
    args.input = Path(args.input)
    args.output = Path(args.output)
    args.output.mkdir(exist_ok=True)

    args.face_enhance = False
    args.tile = 0

    run(args)

if __name__ == '__main__':
    main()
