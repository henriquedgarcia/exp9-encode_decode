import subprocess

from utils import util

config = util.Config('config.json')
video = util.VideoParams(config)

original = f'../original'
scale = config.scale
fps = config.fps
yuv_forders_60s = f'../yuv-full'

util.makedir(f'{yuv_forders_60s}')

for name in config.videos_list:
    start_time = config.videos_list[name]['time']

    in_name = f'{original}/{name}.mp4'
    out_name = f'{name}_{config.scale}_{config.fps}.yuv'

    par_in = f'-y -hide_banner -v quiet -ss {start_time} -i {in_name}'
    par_out_60s = f'-t 60 -r {config.fps} -vf scale={config.scale} -map 0:0 ../yuv-full/{out_name}'

    command = f'ffmpeg {par_in} {par_out_60s}'
    print(command)
    subprocess.run(command, shell=True, stderr=subprocess.STDOUT)

