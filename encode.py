#!/bin/env python3
from utils import util

cfg = util.Config('config.json', factor='crf')
project = f'ffmpeg_crf_12videos_60s'
yuv_input = f'..{cfg.sl}yuv-full'

cfg.videos_list = {"ball": {},
                   "elephants": {},
                   "lions": {},
                   "manhattan": {},
                   "om_nom": {},
                   "pluto": {},
                   "ski": {},
                   "super_mario": {}}

server = True
if server:
    gpds = f'{cfg.sl}mnt{cfg.sl}ssd{cfg.sl}henrique{cfg.sl}'
else:
    gpds = ''

output = f'{gpds}results{cfg.sl}{project}'


def main():
    encode()


def encode():
    # Create video object and your main folders
    video = util.VideoParams(config=cfg,
                             yuv=yuv_input)

    # Set basic configuration
    video.encoder = 'ffmpeg'
    video.project = output

    # iterate over 3 factors: video (complexity), tiles format, quality
    for video.name in cfg.videos_list:
        for video.tile_format in cfg.tile_list:
            for video.quality in getattr(cfg, f'{video.factor}_list'):
                util.encode(video)
                # util.encode(video)
                # util.encapsule(video)
                # util.extract_tile(video)
                # util.make_segments(video)


if __name__ == '__main__':
    main()
