#!/bin/env python3
from utils import util


def main():
    encode()


def encode():
    # Configure objetcts
    config = util.Config('config.json', factor='qp')
    sl = util.check_system()['sl']

    # Create video object and your main folders
    video = util.VideoParams(config=config,
                             yuv=f'{sl}mnt{sl}nas{sl}henrique{sl}yuv-full',
                             hevc_base='hevc',
                             mp4_base='mp4',
                             segment_base='segment',
                             dectime_base='dectime')

    # Set basic configuration
    video.encoder = 'ffmpeg'
    video.project = f'{sl}mnt{sl}ssd{sl}henrique{sl}results{sl}ffmpeg_qp_12videos_60s'

    # iterate over 3 factors: video (complexity), tiles format, quality
    for video.name in config.videos_list:
        for video.tile_format in config.tile_list:
            for video.quality in getattr(config, f'{video.factor}_list'):
                util.encode(video)
                # util.encapsule(video)
                # util.extract_tile(video)
                util.make_segments(video)


if __name__ == '__main__':
    main()
