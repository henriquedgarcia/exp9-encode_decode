import json
import os
import subprocess


def main():
    config = Config('config.json')

    folder_in = f'../original'
    folder_out = f'../yuv-full'
    
    os.makedirs(f'{folder_out}', exist_ok=True)
    
    for name in config.videos_list:
        start_time = config.videos_list[name]['time']
    
        in_name = f'{folder_in}/{name}.mp4'
        out_name = f'{name}_{config.scale}_{config.fps}.yuv'
    
        par_in = f'-y -hide_banner -v quiet -ss {start_time} -i {in_name}'
        par_out_60s = f'-t 60 -r {config.fps} -vf scale={config.scale} -map 0:0 {folder_out}/{out_name}'
    
        command = f'ffmpeg {par_in} {par_out_60s}'
        print(f'Processando {name}.')
        subprocess.run(command, shell=True, stderr=subprocess.STDOUT)


class Config:
    def __init__(self, filename: str = ''):
        self.filename = filename
        self.scale = None
        self.fps = None
        self.gop = None
        self.duration = None
        self.qp_list = None
        self.rate_list = None
        self.tile_list = None
        self.videos_list = None
        if filename:
            self._load_config(filename)

    def _load_config(self, filename: str):
        with open(filename, 'r') as f:
            config_data = json.load(f)

        for key in config_data:
            setattr(self, key, config_data[key])


if __name__ == '__main__':
    main()
