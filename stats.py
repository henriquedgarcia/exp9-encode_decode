#!/bin/python3
# coding=utf-8
import os
import pickle
from itertools import product as it

import fitter
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib import colors

from utils import util

sl = util.check_system()['sl']
project = 'ffmpeg_crf_12videos_60s'
config = util.Config('config.json', factor='scale')
dectime_name = f'times_{project}'
if os.path.isfile(f'{dectime_name}_multikey.json'):
    dectime_multi = util.load_json(f'{dectime_name}_multikey.json')
if os.path.isfile(f'{dectime_name}_single.json'):
    dectime_flat = util.load_json(f'{dectime_name}_single.json')

color_list = ['blue', 'orange', 'green', 'red']
bins = 'auto'

c_dist = {
    'burr12': 'yellow',  # 2, 4
    'invgauss': 'black',  # 1, 2, 3, 4
    'lognorm': 'red',  # 1, 2, 3, 4
    'fatiguelife': 'yellow',  # 3
    'genpareto': 'yellow'
    }  # 1

a = ("#f5793a",
     "#a95aa1",
     "#85c0f9",
     "#0f2080",
     "#9c9eb5",
     "#c9bd9e")


def main():
    # psnr()  # Processar psnr também. à fazer
    # stats()
    global bins
    bin_types1 = ['auto']
    bin_types2 = np.linspace(15, 80, 14, dtype=int)
    bin_types3 = ['fd', 'rice', 'sturges', 'sqrt', 'doane', 'scott']
    bin_types4 = ['fd', 'rice', 'sturges', 'sqrt', 'doane', 'scott',
                  'auto', 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
                  80]
    for bins in bin_types4:
        pass
        # bins = int(bins)

        histogram_fmt('histogram_fmt', force_fit=False)
        histogram_fmt_quality('histogram_fmt_quality',
                              force_fit=True, join_quality=True)
        histogram_fmt_quality_2('histogram_fmt_quality_2',
                                force_fit=True, join_quality=True)
        histogram_fmt_group('histogram_fmt_group',
                            force_fit=False,
                            join_quality=True)
        histogram_fmt_group_2('histogram_fmt_group_2',
                              force_fit=False,
                              join_quality=True)

    # graph0(graph_folder='0_graph0-tudo-fmts_x_chunks')
    # graph0_sum_ts(graph_folder='0_graph0-tudo-fmt-sumtiles_x_chunks')
    # graph0_sum_s(graph_folder='0_graph0-tudo-fmt-sumtiles_x_chunks')
    # graph1(graph_folder='0_graph0-group-fmts_x_chunks')
    # graph1(graph_folder='1_graph1-tiles-chunks_x_dec_time')
    # graph2(graph_folder='2_graph2-quality-chunks_x_dec_time')
    # graph3(graph_folder='3_graph3_heatmap')
    # histogram_name_fmt('histogram_name-fmt')
    # heatmap_fmt_quality_2('heatmap_fmt-quality_time')
    # hist1samefig(graph_folder="hist1samefig")
    # hist1sameplt(graph_folder="hist1sameplt")
    # hist2samefig(graph_folder="hist2samefig")
    # hist2sameplt(graph_folder="hist2sameplt")
    # graph4()   # Por grupo
    # graph5b()  # Geralzão por tile
    # graph5c()  # Geralzão por qualidade
    # graph5d()  # Geralzão por fmt
    # graph5e()  # Geralzão por name
    pass


def json2pandas(json_filename):
    dec = util.load_json(json_filename)
    # 12 videos x (1+6+24+92) tiles x 4 qualidades = 5.904 listas de 60 chunks
    # x 2 11.808
    df = pd.DataFrame()
    for name in config.videos_list:
        for fmt in config.tile_list:
            m, n = list(map(int, fmt.split('x')))
            for quality in config.quality_list:
                for tile in range(1, m * n + 1):
                    col_name = (f'{config.videos_list[name]["group"]}_{name}_'
                                f'{fmt}_{config.factor}{quality}_tile{tile}')
                    time, size, _ = get_data_chunks(name, fmt, quality, tile,
                                                    dec=dec)
                    df[f'{col_name}_time'] = time
                    df[f'{col_name}_size'] = size
    util.save_json(df.to_dict(), f'{json_filename}_singlekey.json')


def stats():
    # Base object
    video_seg = util.VideoStats(config=config,
                                project=f'results{sl}{project}')

    df = {}

    for (video_seg.name,
         video_seg.fmt,
         video_seg.quality) in it(config.videos_list,
                                  config.tile_list,
                                  config.quality_list):
        for video_seg.tile in range(1, video_seg.num_tiles + 1):
            col_name = (f'{config.videos_list[video_seg.name]["group"]}_'
                        f'{video_seg.name}_'
                        f'{video_seg.fmt}_'
                        f'{config.factor}{video_seg.quality}_'
                        f'tile{video_seg.tile}')

            # Coletando taxa Salvando em bps
            chunks_rates = []
            for video_seg.chunk in range(1, video_seg.chunks + 1):
                file = f'{video_seg.segment_path}.mp4'
                if os.path.isfile(file):
                    # O tamanho do chunk só pode ser considerado taxa porque o
                    # chunk tem 1 segndo
                    chunk_size_b = os.path.getsize(file) * 8
                    chunk_dur = config.gop / config.fps
                    rate = chunk_size_b / chunk_dur
                    chunks_rates.append(rate)
                    video_seg.size = rate
            df[f'{col_name}_rate'] = chunks_rates

            # Processando tempos
            times_list = []
            for video_seg.chunk in range(1, video_seg.chunks + 1):
                file = f'{video_seg.log_path}.txt'
                if os.path.isfile(file):
                    dec_time = get_time(file, video_seg)
                    times_list.append(dec_time)
                    video_seg.times = dec_time
            df[f'{col_name}_time'] = times_list

    # Save singlekey
    name = f'{dectime_name}_single.json'
    util.save_json(df, name)

    # Save multikey
    out_name = f'times_{project}_multikey.json'
    util.save_json(dict(video_seg.dectime), f'{out_name}')
    # json2pandas(out_name)


# 21/09/2019
def graph0_sum_ts(graph_folder):
    """
    Já estou usando o novo json com chaves simples
    :param graph_folder:
    :return:
    """
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(f'{dirname}{sl}data', exist_ok=True)
    fig = plt.figure(figsize=(7, 7), dpi=200)
    ax1: matplotlib.axes.Axes = fig.add_subplot(211)
    ax2: matplotlib.axes.Axes = fig.add_subplot(212)

    for fmt in config.tile_list:
        df1 = get_data(tile_list=[fmt], metrics='time')
        df2 = get_data(tile_list=[fmt], metrics='size')
        time_chunks = df1.sum(axis=1) / (12 * 4)
        rate_chunks = df2.sum(axis=1) / (12 * 4)
        leg1 = (f'{fmt}\n'
                f'avg={time_chunks.mean(): .3f} s\n'
                f'std={time_chunks.std(): .3f} s')
        leg2 = (f'{fmt}\n'
                f'avg={rate_chunks.mean() / (1000000): .2f} Mbps\n'
                f'std={rate_chunks.std() / (1000000): .2f} Mbps')
        ax1.plot(time_chunks, label=leg1)
        ax2.plot(rate_chunks, label=leg2)

    ax1.set_title(f'Decoding Time x chunk')
    ax1.set_xlabel('Chunk')
    ax1.set_ylabel('Decoding Time (s)')
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
    ax2.set_title(f'Bitrate x Time')
    ax2.set_xlabel('Chunk')
    ax2.set_ylabel('Bitrate (bps)')
    # ax2.set_ylim(bottom=0)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(6, 6))
    ax2.legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))

    fig.tight_layout()
    fig.savefig(f'{dirname}{sl}graph_fmt-bitrate_x_time_ts')
    # plt.show()
    print('')


def graph1(graph_folder):
    """
    Já estou usando o novo json com chaves simples
    :param graph_folder:
    :return:
    """
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(f'{dirname}{sl}data', exist_ok=True)
    fig = plt.figure(figsize=(7, 7), dpi=200)
    ax1: matplotlib.axes.Axes = fig.add_subplot(211)
    ax2: matplotlib.axes.Axes = fig.add_subplot(212)

    for fmt in config.tile_list:
        df1 = get_data(tile_list=[fmt], metrics='time')
        df2 = get_data(tile_list=[fmt], metrics='size')
        time_chunks = df1.sum(axis=1) / (12 * 4)
        rate_chunks = df2.sum(axis=1) / (12 * 4)
        leg1 = (f'{fmt}\n'
                f'avg={time_chunks.mean(): .3f} s\n'
                f'std={time_chunks.std(): .3f} s')
        leg2 = (f'{fmt}\n'
                f'avg={rate_chunks.mean() / (1000000): .2f} Mbps\n'
                f'std={rate_chunks.std() / (1000000): .2f} Mbps')
        ax1.plot(time_chunks, label=leg1)
        ax2.plot(rate_chunks, label=leg2)

    ax1.set_title(f'Decoding Time x chunk')
    ax1.set_xlabel('Chunk')
    ax1.set_ylabel('Decoding Time (s)')
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
    ax2.set_title(f'Bitrate x Time')
    ax2.set_xlabel('Chunk')
    ax2.set_ylabel('Bitrate (bps)')
    # ax2.set_ylim(bottom=0)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(6, 6))
    ax2.legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(f'{dirname}{sl}graph_fmt-bitrate_x_time_ts')
    # plt.show()
    print('')


def graph0(graph_folder):
    """
    Já estou usando o novo json com
    :param graph_folder:
    :return:
    """
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(f'{dirname}{sl}data', exist_ok=True)
    fig = plt.figure(figsize=(22.5, 4.5), dpi=200)

    # plot1
    ax: matplotlib.axes.Axes = fig.add_subplot(1, 4, 1)
    for n, fmt in zip([4, 3, 2, 1], config.tile_list):
        df = get_data(tile_list=[fmt])
        time_chunks = df.mean(axis=1)
        avg = time_chunks.mean()
        std = time_chunks.std()

        leg = (f'{fmt}\n'
               f'avg={avg: .3f} s\n'
               f'std={std: .3f} s')
        ax.plot(time_chunks, label=leg)
    df = get_data()
    df = df.mean(axis=1)
    avg = df.mean()
    std = df.std()

    leg = (f'Average\n'
           f'avg={avg: .3f} s\n'
           f'std={std: .3f} s')
    ax.plot(df, label=leg)

    ax.set_title(f'Decoder time/tile x Chunks')
    ax.set_xlabel('Chunk')
    ax.set_ylabel('Second')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))

    # plot2 - Histograma
    bins = 'auto'
    ax: matplotlib.axes.Axes = fig.add_subplot(1, 4, 2)
    for fmt in config.tile_list:
        tridata = get_data_tudo_fmt(fmt)
        time, size, corr = tridata
        data_stats_t = [avg, std, corr] = [np.average(time), np.std(time), corr]

        f_t_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_time_{bins}bins_tudo_{fmt}_{config.factor}'
                    f'.pickle')
        f_t = make_fit(data=time, bins=bins, out_file=f_t_name)

        errors = f_t.df_errors
        errors_sorted = errors.sort_values(by="sumsquare_error")
        short_sse = errors_sorted.index[0:1]
        label = (f'{fmt}\n'
                 f'dectime_avg={avg:.03f} s\n'
                 f'dectime_std={std:.03f} s\n'
                 f'corr={corr:.03f}\n'
                 f'best={short_sse[0]}')

        ax = plota_hist(f_t, ax, bins, data_stats_t, 'time', 'pdf', fmt=fmt,
                        label=label)

        ax.set_title(f'PDF Decode Time')
        ax.set_ylim(top=2)
        ax.set_xlabel('Decode Time')

    fig: matplotlib.figure.Figure
    fig.tight_layout()
    # fig.savefig('Dectime')
    plt.show()
    print('')


def graph1(graph_folder):
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(dirname, exist_ok=True)
    df2 = pd.DataFrame()
    fig = plt.figure(figsize=(22.5, 4.5), dpi=200)
    ax = None
    for n1, group in zip([4, 3, 2, 1], ['3', '2', '1', '0']):
        if ax is None:
            ax = fig.add_subplot(1, 4, n1)
        else:
            ax = fig.add_subplot(1, 4, n1, sharey=ax)

        for n, fmt in enumerate(config.tile_list, 1):
            df = get_data(groups=[group], tile_list=[fmt])
            df2[fmt] = df.mean(axis=1)
            avg = df2[fmt].mean()
            std = df2[fmt].std()

            leg = (f'{fmt}\n'
                   f'avg={avg: .3f} s\n'
                   f'std={std: .3f} s')
            ax.plot(df2[fmt], label=leg)
            ax.set_title(f'Decoder time/tile - Group {group} - {fmt}')
            ax.set_xlabel('Chunk')
            ax.set_ylabel('second')
            ax.set_ylim(bottom=0)
            # ax.legend()
            ax.legend(loc='upper right')
            # ax.legend(loc='upper left', ncol=1,
            #           bbox_to_anchor=(1.01, 1.0))
    fig: matplotlib.figure.Figure
    fig.tight_layout()
    fig.savefig('Dectime')
    plt.show()
    print('')


def histogram_name_fmt(graph_folder):
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(dirname + f'{sl}data', exist_ok=True)

    for bins in ['auto', 25, 50, 75, 100, 125, 150, 200]:
        for name in config.videos_list:
            for fmt in config.tile_list:
                # Coleta dados
                tridata = get_data_quality(name, fmt)

                # Faz o plot
                fig = make_hist(tridata, dirname, bins, group=None, name=name,
                                fmt=fmt, quality=None, tile=None, chunk=None)

                # Salva
                fig.savefig(f'{dirname}{sl}'
                            f'hist_groups_bins{bins}_{name}_{fmt}')
                # fig.show()
                print(f'hist {bins} bins, {name}_{fmt}')


def histogram_fmt(graph_folder, force_fit=False, join_quality=True):
    """Usado no SVR e Eletronic Imaging"""
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(dirname + f'{sl}data', exist_ok=True)

    plt.close()
    fig = plt.figure(figsize=(13, 7.2), dpi=220, tight_layout=True,
                     facecolor='0.9')
    # fig = plt.figure(figsize=(7.5, 5), dpi=220, tight_layout=True)
    fig_relative = plt.figure(figsize=(13, 7.2), dpi=220, tight_layout=True,
                              facecolor='0.5')
    ax = None

    df_columns = {
        'Format': [],
        'Statistics': [],
        'Distribution': [],
        'SSE': [],
        'Parameters': []
        }

    for n, fmt in enumerate(config.tile_list, 1):
        if fmt in "9x8": continue

        print(f'processando {fmt}')

        # Prepara plot
        if join_quality:
            if ax is None:
                ax = fig.add_subplot(2, 3, n)
            else:
                ax = fig.add_subplot(2, 3, n, sharex=ax)
            ax_relative = fig_relative.add_subplot(2, 3, n)
        else:
            fig = plt.figure(figsize=(13, 7.2), dpi=220,
                             facecolor='0.5')
            ax = fig.add_subplot(1, 1, 1)
            fig_relative = plt.figure(figsize=(13, 7.2), dpi=220,
                                      tight_layout=True,
                                      facecolor='0.5')
            ax_relative = fig_relative.add_subplot(2, 3, n)

        # Coleta dados
        print(f'Coletando dados')

        df_t, df_r, corr = get_data(tile_list=[fmt])
        time = df_t.stack().tolist()
        stats_t = {
            'avg': np.average(time),
            'std': np.std(time),
            'corr': corr
            }
        st = (f'Average {stats_t["avg"]}, '
              f'Standard Deviation {stats_t["std"]}, '
              f'Correlation {stats_t["corr"]}')

        f_t_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_time_{bins}bins_tudo_{fmt}_{config.factor}'
                    f'.pickle')

        # Faz o fit e salva distribuições em dataframe
        print(f'Fazendo o fit e criando tabela de resultados')
        f_t = make_fit(data=time, bins=bins, out_file=f_t_name,
                       overwrite=force_fit)
        make_df_params_from_fit(f_t, df_columns, fmt=fmt, st=st)

        # Verifica se a PDF e a frequencia relativa estão coerentes
        # Calcula a frequencia relativa
        rel_t = scipy.stats.relfreq(time, numbins=len(f_t.y))
        # soma a área de cada barra do histograma de densidade
        a1 = np.sum([y * (f_t.x[1] - f_t.x[0]) for y in f_t.y])
        # soma todas as frequencias relativas
        a2 = np.sum(rel_t.frequency)

        # Faz o plot da frequência relativa
        ax_relative.bar(f_t.x, rel_t.frequency, width=f_t.x[1] - f_t.x[0])
        ax_relative.legend([f'sum PDF={a1: .3f}\n'
                            f'sum freq_rel={a2: .3f}\n'
                            f'numbins={len(f_t.x)}'])
        ax_relative.set_title(f'{fmt}')
        ax_relative.set_xlabel('Decoding Time')
        ax_relative.set_ylabel('Relative Frequency')

        # Faz o plot
        label = f'Empirical'
        ax = plota_hist(f_t, ax, bins, stats_t, 'time', 'pdf', label=label)

        # infos
        ax.legend(loc='best')
        ax.set_title(f'{fmt}')
        ax.set_xlabel('Decoding Time')

    print(f'Salvando a figura')
    fig.set_tight_layout(True)
    name = f'{dirname}{sl}hist_{bins}bins_tudo_{config.factor}'
    fig.savefig(f'{name}', facecolor='0.9')
    fig_relative.savefig(f'{name}_relative', facecolor='0.9')

    # Salva Dataframe
    print(f'Salvando a tabela')
    name = f'{dirname}{sl}hist_{bins}bins_tudo_{config.factor}'
    pd.DataFrame(df_columns).to_csv(f'{name}.csv', index='Format')

    # fig.show()
    print(f'hist bins {bins}, tudo_{config.factor}')


def histogram_fmt_quality(graph_folder, force_fit=False, join_quality=True):
    """Usado no Eletronic Imaging
    """
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(dirname + f'{sl}data', exist_ok=True)
    df_columns = {
        'Format': [],
        'Quality': [],
        'Statistics': [],
        'Distribution': [],
        'SSE': [],
        'Parameters': []
        }

    for fmt in config.tile_list:
        if fmt in "9x8": continue

        plt.close()
        fig = plt.figure(figsize=(13, 7.2), dpi=220, tight_layout=True,
                         facecolor='0.9')
        fig_relative = plt.figure(figsize=(13, 7.2), dpi=220, tight_layout=True,
                                  facecolor='0.9')
        ax = None

        for n, quality in enumerate(config.quality_list, 1):
            print(f'processando {fmt}-{quality}')

            # Prepara plot
            if join_quality:
                if ax is None:
                    ax = fig.add_subplot(2, 3, n)
                else:
                    ax = fig.add_subplot(2, 3, n, sharex=ax)
                ax_relative = fig_relative.add_subplot(2, 3, n)
            else:
                fig = plt.figure(figsize=(13, 7.2), dpi=220,
                                 facecolor='0.9')
                ax = fig.add_subplot(1, 1, 1)
                fig_relative = plt.figure(figsize=(13, 7.2), dpi=220,
                                          tight_layout=True,
                                          facecolor='0.9')
                ax_relative = fig_relative.add_subplot(2, 3, n)

            # Coleta dados
            print(f'Coletando dados')
            df_t, df_r, corr = get_data(tile_list=[fmt],
                                        quality_list=[quality])
            time = df_t.stack().tolist()
            stats_t = {
                'avg': np.average(time),
                'std': np.std(time),
                'corr': corr
                }
            st = (f'Average {stats_t["avg"]}, '
                  f'Standard Deviation {stats_t["std"]}, '
                  f'Correlation {stats_t["corr"]}')
            f_t_name = (f'{dirname}{sl}data{sl}'
                        f'fitter_time_{bins}bins_tudo_{fmt}_{config.factor}'
                        f'.pickle')
            # Faz o fit
            print(f'Fazendo o fit e criando tabela de resultados')
            f_t = make_fit(data=time, bins=bins, out_file=f_t_name,
                           overwrite=force_fit)
            make_df_params_from_fit(f_t, df_columns, fmt=fmt, quality=quality,
                                    st=st)

            # verifica se a PDF e a frequencia relativa estão coerentes
            # Calcula a frequencia relativa
            rel_t = scipy.stats.relfreq(time, numbins=len(f_t.y))
            # soma a área de cada barra do histograma de densidade
            a1 = np.sum([y * (f_t.x[1] - f_t.x[0]) for y in f_t.y])
            # soma todas as frequencias relativas
            a2 = np.sum(rel_t.frequency)

            # Faz o plot da frequência relativa
            ax_relative.bar(f_t.x, rel_t.frequency, width=f_t.x[1] - f_t.x[0])
            ax_relative.legend([f'sum PDF={a1: .3f}\n'
                                f'sum freq_rel={a2: .3f}\n'
                                f'numbins={len(f_t.x)}\n'
                                f'dectime_avg={stats_t["avg"]:.03f} s\n'
                                f'dectime_std={stats_t["std"]:.03f} s\n'
                                f'rate_corr={stats_t["corr"]:.03f}'],
                               loc='upper right')
            ax_relative.set_title(f'{fmt} - {config.factor} {quality}')
            ax_relative.set_xlabel('Decoding Time')
            ax_relative.set_ylabel('Relative Frequency')

            # Faz o plot
            # label = None
            label = f'Empirical'
            ax = plota_hist(f=f_t, ax=ax, bins=bins, data_stats=stats_t,
                            metric='time', func='pdf', label=label)

            # infos
            ax.legend(loc='upper right')
            ax.set_title(f'{fmt} - {config.factor} {quality}')
            ax.set_xlabel('Decoding Time')

        print(f'Salvando a figura')
        fig.set_tight_layout(True)
        name = f'{dirname}{sl}hist_tudo_{fmt}_{config.factor}_{bins}bins'
        fig.savefig(f'{name}', facecolor='0.9')
        fig_relative.savefig(f'{name}_relative', facecolor='0.9')

    # Salva Dataframe
    print(f'Salvando a tabela')
    name = f'{dirname}{sl}hist_{bins}bins_fmt-quality_{config.factor}'
    pd.DataFrame(df_columns).to_csv(f'{name}.csv', index='Format')

    print(f'hist bins {bins}, tudo_{config.factor}')


def histogram_fmt_quality_2(graph_folder, force_fit=False, join_quality=True):
    """Usado no Eletronic Imaging
    Esse gráfico plota para cada qualidade todos os fmts
    """
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(dirname + f'{sl}data', exist_ok=True)
    df_columns = {
        'Format': [],
        'Quality': [],
        'Statistics': [],
        'Distribution': [],
        'SSE': [],
        'Parameters': []
        }

    for quality in config.quality_list:
        plt.close()
        fig = plt.figure(figsize=(13, 7.2), dpi=220, tight_layout=True,
                         facecolor='0.9')
        fig_relative = plt.figure(figsize=(13, 7.2), dpi=220, tight_layout=True,
                                  facecolor='0.9')
        ax = None

        for n, fmt in enumerate(config.tile_list, 1):
            if fmt in "9x8": continue
            print(f'processando {quality}-{fmt}')

            # Prepara plot
            if join_quality:
                if ax is None:
                    ax = fig.add_subplot(2, 3, n)
                else:
                    ax = fig.add_subplot(2, 3, n, sharex=ax)
                ax_relative = fig_relative.add_subplot(2, 3, n)
            else:
                fig = plt.figure(figsize=(13, 7.2), dpi=220,
                                 facecolor='0.9')
                ax = fig.add_subplot(1, 1, 1)
                fig_relative = plt.figure(figsize=(13, 7.2), dpi=220,
                                          tight_layout=True,
                                          facecolor='0.9')
                ax_relative = fig_relative.add_subplot(2, 3, n)

            # Coleta dados
            print(f'Coletando dados')
            df_t, df_r, corr = get_data(tile_list=[fmt],
                                        quality_list=[quality])
            time = df_t.stack().tolist()
            stats_t = {
                'avg': np.average(time),
                'std': np.std(time),
                'corr': corr
                }
            st = (f'Average {stats_t["avg"]}, '
                  f'Standard Deviation {stats_t["std"]}, '
                  f'Correlation {stats_t["corr"]}')
            f_t_name = (f'{dirname}{sl}data{sl}'
                        f'fitter_time_{bins}bins_tudo_{fmt}_{config.factor}'
                        f'.pickle')
            # Faz o fit
            print(f'Fazendo o fit e criando tabela de resultados')
            f_t = make_fit(data=time, bins=bins, out_file=f_t_name,
                           overwrite=force_fit)
            make_df_params_from_fit(f_t, df_columns, fmt=fmt, quality=quality,
                                    st=st)

            # verifica se a PDF e a frequencia relativa estão coerentes
            # Calcula a frequencia relativa
            rel_t = scipy.stats.relfreq(time, numbins=len(f_t.y))
            # soma a área de cada barra do histograma de densidade
            a1 = np.sum([y * (f_t.x[1] - f_t.x[0]) for y in f_t.y])
            # soma todas as frequencias relativas
            a2 = np.sum(rel_t.frequency)

            # Faz o plot da frequência relativa
            ax_relative.bar(f_t.x, rel_t.frequency, width=f_t.x[1] - f_t.x[0])
            ax_relative.legend([f'sum PDF={a1: .3f}\n'
                                f'sum freq_rel={a2: .3f}\n'
                                f'numbins={len(f_t.x)}\n'
                                f'dectime_avg={stats_t["avg"]:.03f} s\n'
                                f'dectime_std={stats_t["std"]:.03f} s\n'
                                f'rate_corr={stats_t["corr"]:.03f}'],
                               loc='upper right')
            ax_relative.set_title(f'{config.factor} {quality} - {fmt}')
            ax_relative.set_xlabel('Decoding Time')
            ax_relative.set_ylabel('Relative Frequency')

            # Faz o plot
            # label = None
            label = f'Empirical'
            ax = plota_hist(f=f_t, ax=ax, bins=bins, data_stats=stats_t,
                            metric='time', func='pdf', label=label)

            # infos
            ax.legend(loc='upper right')
            ax.set_title(f'{config.factor} {quality} - {fmt}')
            ax.set_xlabel('Decoding Time')

        print(f'Salvando a figura')
        fig.set_tight_layout(True)
        name = f'{dirname}{sl}hist_{config.factor}{quality}_{bins}bins'
        fig.savefig(f'{name}', facecolor='0.9')
        fig_relative.savefig(f'{name}_relative', facecolor='0.9')

    # Salva Dataframe
    print(f'Salvando a tabela')
    name = f'{dirname}{sl}hist_{bins}bins_quality_{config.factor}-fmt'
    pd.DataFrame(df_columns).to_csv(f'{name}.csv', index='Format')

    print(f'hist bins {bins}, tudo_{config.factor}')


def histogram_fmt_group(graph_folder, force_fit=True,
                        join_quality=True):
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(dirname + f'{sl}data', exist_ok=True)
    df_columns = {
        'Format': [],
        'Group': [],
        'Statistics': [],
        'Distribution': [],
        'SSE': [],
        'Parameters': []
        }

    for fmt in config.tile_list:
        if fmt in "9x8": continue

        plt.close()
        fig = plt.figure(figsize=(13, 7.2), dpi=220, tight_layout=True,
                         facecolor='0.9')
        fig_relative = plt.figure(figsize=(13, 7.2), dpi=220, tight_layout=True,
                                  facecolor='0.9')
        ax = None

        for n, group in enumerate(['0', '1', '2', '3'], 1):
            print(f'processando {fmt}-group{group}')

            # Prepara plot
            if join_quality:
                if ax is None:
                    ax = fig.add_subplot(2, 3, n)
                else:
                    ax = fig.add_subplot(2, 3, n, sharex=ax)
                ax_relative = fig_relative.add_subplot(2, 3, n)
            else:
                fig = plt.figure(figsize=(13, 7.2), dpi=220,
                                 facecolor='0.9')
                ax = fig.add_subplot(1, 1, 1)
                fig_relative = plt.figure(figsize=(13, 7.2), dpi=220,
                                          tight_layout=True,
                                          facecolor='0.9')
                ax_relative = fig_relative.add_subplot(2, 3, n)

            # Coleta dados
            print(f'Coletando dados')
            df_t, df_r, corr = get_data(tile_list=[fmt],
                                        groups=[group])
            time = df_t.stack().tolist()

            stats_t = {
                'avg': np.average(time),
                'std': np.std(time),
                'corr': corr
                }
            st = (f'Average {stats_t["avg"]}, '
                  f'Standard Deviation {stats_t["std"]}, '
                  f'Correlation {stats_t["corr"]}')
            f_t_name = (f'{dirname}{sl}data{sl}'
                        f'fitter_time_{bins}bins_tudo_{fmt}_group{group}'
                        f'{config.factor}.pickle')

            # Faz o fit
            print(f'Fazendo o fit e criando tabela de resultados')
            f_t = make_fit(data=time, bins=bins, out_file=f_t_name,
                           overwrite=force_fit)
            make_df_params_from_fit(f_t, df_columns, fmt=fmt, group=group,
                                    st=st)

            # verifica se a PDF e a frequencia relativa estão coerentes
            # Calcula a frequencia relativa
            res = scipy.stats.relfreq(time, numbins=len(f_t.y))
            # soma a área de cada barra do histograma de densidade
            a1 = np.sum([y * (f_t.x[1] - f_t.x[0]) for y in f_t.y])
            # soma todas as frequencias relativas
            a2 = np.sum(res.frequency)
            # Faz o plot da frequência relativa
            ax_relative.bar(f_t.x, res.frequency, width=f_t.x[1] - f_t.x[0])
            ax_relative.legend([f'sum PDF={a1: .3f}\n'
                                f'sum freq_rel={a2: .3f}\n'
                                f'numbins={len(f_t.x)}\n'
                                f'dectime_avg={stats_t["avg"]:.03f} s\n'
                                f'dectime_std={stats_t["std"]:.03f} s\n'
                                f'rate_corr={stats_t["corr"]:.03f}'],
                               loc='upper right')
            ax_relative.set_title(f'{fmt} - {config.factor} group {group}')
            ax_relative.set_xlabel('Decoding Time')
            ax_relative.set_ylabel('Relative Frequency')

            # Faz o plot
            # label = None
            label = f'Empirical'
            ax = plota_hist(f=f_t, ax=ax, bins=bins, data_stats=stats_t,
                            metric='time', func='pdf', label=label)

            # infos
            ax.legend(loc='upper right')
            ax.set_title(f'{fmt} - {config.factor} group{group}')
            ax.set_xlabel('Decoding Time')

        print(f'Salvando a figura')
        fig.set_tight_layout(True)
        name = f'{dirname}{sl}hist_groups_{fmt}_{config.factor}_{bins}bins'
        fig.savefig(f'{name}')
        fig_relative.savefig(f'{name}_relative')

    # Salva Dataframe
    print(f'Salvando a tabela')
    name = f'{dirname}{sl}hist_{bins}bins_fmt-group_{config.factor}'
    pd.DataFrame(df_columns).to_csv(f'{name}.csv', index='Format')

    print(f'hist bins {bins}, tudo_{config.factor}')


def histogram_fmt_group_2(graph_folder, force_fit=True,
                          join_quality=True):
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(dirname + f'{sl}data', exist_ok=True)
    df_columns = {
        'Format': [],
        'Group': [],
        'Statistics': [],
        'Distribution': [],
        'SSE': [],
        'Parameters': []
        }

    for fmt in config.tile_list:
        for group in ['0', '1', '2', '3']:
            plt.close()
            fig = plt.figure(figsize=(13, 7.2), dpi=220, tight_layout=True,
                             facecolor='0.9')
            fig_relative = plt.figure(figsize=(13, 7.2), dpi=220,
                                      tight_layout=True,
                                      facecolor='0.9')
            ax = None

            for n, fmt in enumerate(config.tile_list, 1):
                if fmt in "9x8": continue
                print(f'processando {fmt}-group{group}')

                # Prepara plot
                if join_quality:
                    if ax is None:
                        ax = fig.add_subplot(2, 3, n)
                    else:
                        ax = fig.add_subplot(2, 3, n, sharex=ax)
                    ax_relative = fig_relative.add_subplot(2, 3, n)
                else:
                    fig = plt.figure(figsize=(13, 7.2), dpi=220,
                                     facecolor='0.9')
                    ax = fig.add_subplot(1, 1, 1)
                    fig_relative = plt.figure(figsize=(13, 7.2), dpi=220,
                                              tight_layout=True,
                                              facecolor='0.9')
                    ax_relative = fig_relative.add_subplot(2, 3, n)

                # Coleta dados
                print(f'Coletando dados')
                df_t, df_r, corr = get_data(tile_list=[fmt],
                                            groups=[group])
                time = df_t.stack().tolist()

                stats_t = {
                    'avg': np.average(time),
                    'std': np.std(time),
                    'corr': corr
                    }
                st = (f'Average {stats_t["avg"]}, '
                      f'Standard Deviation {stats_t["std"]}, '
                      f'Correlation {stats_t["corr"]}')
                f_t_name = (f'{dirname}{sl}data{sl}'
                            f'fitter_time_{bins}bins_tudo_{fmt}_group{group}'
                            f'{config.factor}.pickle')

                # Faz o fit
                print(f'Fazendo o fit e criando tabela de resultados')
                f_t = make_fit(data=time, bins=bins, out_file=f_t_name,
                               overwrite=force_fit)
                make_df_params_from_fit(f_t, df_columns, fmt=fmt, group=group,
                                        st=st)

                # verifica se a PDF e a frequencia relativa estão coerentes
                # Calcula a frequencia relativa
                res = scipy.stats.relfreq(time, numbins=len(f_t.y))
                # soma a área de cada barra do histograma de densidade
                a1 = np.sum([y * (f_t.x[1] - f_t.x[0]) for y in f_t.y])
                # soma todas as frequencias relativas
                a2 = np.sum(res.frequency)

                # Faz o plot da frequência relativa
                ax_relative.bar(f_t.x, res.frequency, width=f_t.x[1] - f_t.x[0])
                ax_relative.legend([f'sum PDF={a1: .3f}\n'
                                    f'sum freq_rel={a2: .3f}\n'
                                    f'numbins={len(f_t.x)}\n'
                                    f'dectime_avg={stats_t["avg"]:.03f} s\n'
                                    f'dectime_std={stats_t["std"]:.03f} s\n'
                                    f'rate_corr={stats_t["corr"]:.03f}'],
                                   loc='upper right')
                ax_relative.set_title(f'{fmt} - {config.factor} group {group}')
                ax_relative.set_xlabel('Decoding Time')
                ax_relative.set_ylabel('Relative Frequency')

                # Faz o plot
                # label = None
                label = f'Empirical'
                ax = plota_hist(f=f_t, ax=ax, bins=bins, data_stats=stats_t,
                                metric='time', func='pdf', label=label)

                # infos
                ax.legend(loc='upper right')
                ax.set_title(f'group{group} - {config.factor} {fmt}')
                ax.set_xlabel('Decoding Time')

            print(f'Salvando a figura')
            fig.set_tight_layout(True)
            name = f'{dirname}{sl}hist_groups_{group}_{config.factor}_{bins}bins'
            fig.savefig(f'{name}')
            fig_relative.savefig(f'{name}_relative')

    # Salva Dataframe
    print(f'Salvando a tabela')
    name = f'{dirname}{sl}hist_{bins}bins_group-fmt_{config.factor}'
    pd.DataFrame(df_columns).to_csv(f'{name}.csv', index='Format')

    print(f'hist bins {bins}, tudo_{config.factor}')


def heatmap_fmt_quality(graph_folder):
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(f'{dirname}{sl}data', exist_ok=True)

    for fmt in config.tile_list:
        m, n = list(map(int, fmt.split('x')))

        fig, ax = plt.subplots(2, 4, figsize=(12, 5), dpi=150)

        c1, c2 = [], []
        for count, quality in enumerate(config.quality_list):
            # Cria e preenche o heatmap
            heatmap = Heatmap(m, n)
            for tile, (x, y) in zip(range(1, m * n + 1),
                                    it(range(1, m + 1), range(1, n + 1))):
                time, size, corr = get_data_name_chunk(fmt, tile, quality)

                time = np.average(time)
                size = np.average(size)

                heatmap.average_time.append(time)
                heatmap.average_size.append(size)
                heatmap.std_time.append(np.std(time))
                heatmap.std_size.append(np.std(size))

                heatmap.frame_time[y - 1, x - 1] = time
                heatmap.frame_size[y - 1, x - 1] = size

            # Plota um Pcolor (Heatmap) e pega sua collection
            c1.append(ax[0][count].pcolor(heatmap.frame_time,
                                          cmap='jet'))
            c2.append(ax[1][count].pcolor(heatmap.frame_size,
                                          cmap='jet'))

            # Configura os eixos do heatmap da qualidade atual
            ax[0][count].set_title(f'Dectime Heatmap {fmt}')
            ax[1][count].set_xlabel(f'crf={quality}')
            ax[0][count].set_xticklabels([])
            ax[0][count].set_yticklabels([])

            ax[1][count].set_title(f'Bitrate Heatmap {fmt}')
            ax[1][count].set_xlabel(f'crf={quality}')
            ax[1][count].set_xticklabels([])
            ax[1][count].set_yticklabels([])

        # Normaliza heatmap
        vmin1 = min(collection.get_array().min() for collection in c1)
        vmax1 = max(collection.get_array().max() for collection in c1)
        vmin2 = min(collection.get_array().min() for collection in c2)
        vmax2 = max(collection.get_array().max() for collection in c2)

        norm1 = colors.Normalize(vmin=vmin1, vmax=vmax1)
        norm2 = colors.Normalize(vmin=vmin2, vmax=vmax2)
        for collection1, collection2 in zip(c1, c2):
            collection1.set_norm(norm1)
            collection2.set_norm(norm2)

        # Colorbar
        # fig.colorbar(c1[-1], ax=ax[0],
        #              orientation='horizontal')
        # fig.colorbar(c2[-1], ax=ax[1],
        #              orientation='horizontal')
        # plt.legend()
        # fig.suptitle(f'Heatmap dectime and bitrate {fmt}.', y=1.05, fontsize=16)
        plt.tight_layout()

        # Finaliza
        print(f'Salvando {dirname}{sl}{fmt}_{config.factor}.')
        fig.savefig(f'{dirname}{sl}{fmt}_{config.factor}')
        # plt.show()
        print('')


def hist1samefig(graph_folder):
    """ Compara os histogramas. Para cada video-fmt plota todas as qualidades
    (agrega tiles e chunks)
    :return:
    """
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(f'{dirname}', exist_ok=True)

    best_dist_df = util.AutoDict()
    times = util.AutoDict()
    sizes = util.AutoDict()

    # Coleta dados
    iterator = it(config.videos_list, config.tile_list, config.quality_list)
    for name, fmt, quality in list(iterator):
        m, n = list(map(int, fmt.split('x')))
        times[name][fmt][quality] = []
        sizes[name][fmt][quality] = []

        for tile, chunk in list(it(range(1, m * n + 1),
                                   range(1, config.duration + 1))):
            if name in 'ninja_turtles' and chunk > 58: continue
            t = dectime_multi[name][fmt][str(quality)][str(tile)][str(chunk)][
                'times']
            s = dectime_multi[name][fmt][str(quality)][str(tile)][str(chunk)][
                'size']
            times[name][fmt][quality].append(t)
            sizes[name][fmt][quality].append(s)

    # Calcular melhores fit e plota
    for name, fmt in list(it(config.videos_list, config.tile_list)):
        # Faz um fit por name-fmt-qualidade

        fitter_dict = {}
        for quality in config.quality_list:
            data = times[name][fmt][quality]
            col_label = f'{fmt}-{quality}'

            # Faz o fit
            f = fitter.fitter.Fitter(data,
                                     bins='auto',
                                     distributions=config.dists,
                                     verbose=False)
            fitter_dict[quality] = f
            fitter_dict[quality].fit()

            # Procura os menores SSE
            short_sse = fitter_dict[quality]
            short_sse = short_sse.df_errors
            short_sse = short_sse.sort_values(by="sumsquare_error")
            short_sse = short_sse.index[0:5]

            # Melhor fit
            e0 = fitter_dict[quality].df_errors["sumsquare_error"][short_sse[0]]
            e1 = fitter_dict[quality].df_errors["sumsquare_error"][short_sse[1]]
            e2 = fitter_dict[quality].df_errors["sumsquare_error"][short_sse[2]]
            e3 = fitter_dict[quality].df_errors["sumsquare_error"][short_sse[3]]
            e4 = fitter_dict[quality].df_errors["sumsquare_error"][short_sse[4]]
            best_dist_df[name][col_label] = (f'{short_sse[0]}-SSE={e0: .1f},'
                                             f'{short_sse[1]}-SSE={e1: .1f},'
                                             f'{short_sse[2]}-SSE={e2: .1f},'
                                             f'{short_sse[3]}-SSE={e3: .1f},'
                                             f'{short_sse[4]}-SSE={e4: .1f}')

        # Para cada video e fmt cria uma figura. cuidado. se mudar o formato do
        # gráfico vai ter que mudar tudo dentro da função.
        plt.close()
        fig = plt.figure(figsize=(13, 9), dpi=150)

        color_it = iter(color_list)
        m, n = list(map(int, fmt.split('x')))
        ax_rate = fig.add_subplot(3, 2, 5)
        ax_cdf = fig.add_subplot(3, 2, 6)

        for idx, quality in enumerate(config.quality_list, 1):
            color = next(color_it)

            ###############################################################
            # plota o histograma desta qualidade
            t_avg = np.average(times[name][fmt][quality])
            t_std = np.std(times[name][fmt][quality])
            t_ct = np.average(times[name][fmt][quality]) * (m * n)

            label = (f'dectime_avg={t_avg:.03f} s\n'
                     f'dectime_std={t_std:.03f} s\n'
                     f'time/chunk/tile={t_ct:.03f} s')

            ax = fig.add_subplot(3, 2, idx, sharex=ax_cdf)
            ax.hist(times[name][fmt][quality], color=color, bins='auto',
                    histtype='bar',
                    density=True, label=label)
            ax.set_title(f'{name}, {fmt}, crf {quality}')
            ax.set_ylabel("PDF")
            ax.set_xlabel('Decoder Times')

            ###############################################################
            # plota os 3 best fit para esta qualidade
            best_dists = fitter_dict[quality].df_errors
            best_dists = best_dists.sort_values(by="sumsquare_error")
            best_dists = best_dists.index[0:5]

            for dist_name in best_dists:
                sse = fitter_dict[quality]
                sse = sse.df_errors["sumsquare_error"][dist_name]
                label = f'{dist_name},\nSSE = {sse: .3f}'
                ax.plot(fitter_dict[quality].x,
                        fitter_dict[quality].fitted_pdf[dist_name],
                        label=label)
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

            ###############################################################
            # plota o bar da taxa para esta qualidade
            s_avg = np.average(sizes[name][fmt][quality]) * 8
            s_std = np.std(sizes[name][fmt][quality]) * 8
            s_ct = np.average(sizes[name][fmt][quality]) * 8
            label = (f'rate_avg={s_avg / 1000:,.0f} Kbps\n'
                     f'rate_std={s_std / 1000:,.0f} Kbps\n'
                     f'rate/tile={s_ct / (1000 * m * n):.03f} kbps')
            ax_rate.bar(idx - 1, s_avg, color=color, yerr=s_std, label=label)
            ax_rate.set_title('Rate')
            ax_rate.set_ylabel("Rate (10³ bps)")
            ax_rate.set_xlabel("Decoder Time (s)")
            ax_rate.ticklabel_format(axis='y', style='scientific',
                                     scilimits=(3, 3))
            ax_rate.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

            ###############################################################
            # plota a CDF para esta qualidade
            label = f'{name}_{fmt}_crf{quality}'
            ax_cdf.hist(times[name][fmt][quality], color=color, bins='auto',
                        density=True, cumulative=True, histtype='step',
                        label=label)
            ax_cdf.set_title('CDF')
            ax_cdf.set_ylabel("CDF")
            ax_cdf.set_xlabel("Decoder Time (s)")
            ax_cdf.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

        ax_rate.set_xticklabels(['0'] + config.quality_list)
        plt.tight_layout()
        group = config.videos_list[name]["group"]
        fig.savefig(f'{dirname}{sl}{group}-hist_{name}_{fmt}')
        # plt.show()
        print(f'hist_{name}_{fmt}')

    # Salva melhores fits em um csv
    best_dist_df = pd.DataFrame(best_dist_df)
    best_dist_df.to_csv(f'{dirname}{config.sl}best_dist.csv')


def hist1sameplt(graph_folder):
    """ Compara os histogramas. Para cada video-fmt plota todas as qualidades
    (agrega tiles e chunks)
    :return:
    """
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    times = util.AutoDict()
    sizes = util.AutoDict()

    os.makedirs(f'{dirname}', exist_ok=True)

    # Coleta dados
    for name, fmt, quality in it(config.videos_list, config.tile_list,
                                 config.quality_list):
        m, n = list(map(int, fmt.split('x')))
        times[name][fmt][quality] = []
        sizes[name][fmt][quality] = []

        for tile, chunk in it(range(1, m * n + 1),
                              range(1, config.duration + 1)):
            if name in 'ninja_turtles' and chunk > 58: continue
            t = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['times']
            s = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['size']
            times[name][fmt][quality].append(t)
            sizes[name][fmt][quality].append(s)

    # Calcular melhores fit e plota
    for name, fmt in it(config.videos_list, config.tile_list):
        # Para cada video e fmt cria uma figura. cuidado. se mudar o formato do
        # gráfico vai ter que mudar tudo dentro da função.
        plt.close()
        fig = plt.figure(figsize=(13, 9), dpi=150)

        color_it = iter(color_list)
        m, n = list(map(int, fmt.split('x')))
        ax_cdf = fig.add_subplot(2, 1, 2)
        ax = fig.add_subplot(2, 1, 1, sharex=ax_cdf)

        for idx, quality in enumerate(config.quality_list, 1):
            color = next(color_it)

            ###############################################################
            # plota o histograma desta qualidade
            t_avg = np.average(times[name][fmt][quality])
            t_std = np.std(times[name][fmt][quality])
            t_ct = np.average(times[name][fmt][quality]) * (m * n)

            label = (f'dectime_avg={t_avg:.03f} s\n'
                     f'dectime_std={t_std:.03f} s\n'
                     f'time/chunk/tile={t_ct:.03f} s')

            ax.hist(times[name][fmt][quality], color=color, bins='auto',
                    histtype='bar', density=True, label=label)
            ax.set_title(f'{name}, {fmt}, crf {quality}')
            ax.set_ylabel("PDF")
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

            ###############################################################
            # plota a CDF para esta qualidade
            label = f'{name}_{fmt}_crf{quality}'
            ax_cdf.hist(times[name][fmt][quality], color=color, bins='auto',
                        density=True, cumulative=True, histtype='step',
                        label=label)
            ax_cdf.set_title('CDF')
            ax_cdf.set_ylabel("CDF")
            ax_cdf.set_xlabel("Decoder Time (s)")
            ax_cdf.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

        plt.tight_layout()
        grupo = config.videos_list[name]["group"]
        fig.savefig(f'{dirname}{sl}{grupo}-hist_{name}_{fmt}')
        # plt.show()
        print(f'hist_{name}_{fmt}')


def hist2samefig(graph_folder):
    """ Compara os histogramas. Para cada video-fmt plota todas as qualidades
    (agrega tiles e chunks)
    :return:
    """
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(f'{dirname}', exist_ok=True)

    best_dist_df = util.AutoDict()
    times = util.AutoDict()
    sizes = util.AutoDict()

    # Coleta dados
    for name, fmt, quality in it(config.videos_list, config.tile_list,
                                 config.quality_list):
        m, n = list(map(int, fmt.split('x')))
        times[name][fmt][quality] = []
        sizes[name][fmt][quality] = []

        for tile, chunk in it(range(1, m * n + 1),
                              range(1, config.duration + 1)):
            if name in 'ninja_turtles' and chunk > 58: continue
            t = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['times']
            s = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['size']
            times[name][fmt][quality].append(t)
            sizes[name][fmt][quality].append(s)

    # Calcular melhores fit e plota
    for name, quality in it(config.videos_list, config.quality_list):
        # Faz um fit por name-fmt-qualidade
        fitter_dict = {}
        for fmt in config.tile_list:
            data = times[name][fmt][quality]
            col_label = f'{fmt}-{quality}'

            # Faz o fit
            f = fitter.fitter.Fitter(data, bins='auto',
                                     distributions=config.dists,
                                     verbose=False)
            fitter_dict[fmt] = f
            fitter_dict[fmt].fit()

            # Procura os menores SSE
            short_sse = fitter_dict[fmt]
            short_sse = short_sse.df_errors
            short_sse = short_sse.sort_values(by="sumsquare_error")
            short_sse = short_sse.index[0]

            # Melhor fit
            sse = fitter_dict[fmt].df_errors["sumsquare_error"][short_sse]
            best_dist_df[name][col_label] = f'{short_sse}, SSE={sse: .1f}'

        # Para cada video e fmt cria uma figura. cuidado. se mudar o formato do
        # gráfico vai ter que mudar tudo dentro da função.
        plt.close()
        fig = plt.figure(figsize=(13, 9), dpi=150)

        color_it = iter(color_list)
        ax_rate = fig.add_subplot(3, 2, 5)
        ax_cdf = fig.add_subplot(3, 2, 6)

        for idx, fmt in enumerate(config.tile_list, 1):
            color = next(color_it)
            m, n = list(map(int, fmt.split('x')))

            ###############################################################
            # plota o histograma desta qualidade
            t_avg = np.average(times[name][fmt][quality])
            t_std = np.std(times[name][fmt][quality])
            t_ct = np.average(times[name][fmt][quality]) * (m * n)

            label = (f'dectime_avg={t_avg:.03f} s\n'
                     f'dectime_std={t_std:.03f} s\n'
                     f'time/chunk/tile={t_ct:.03f} s')

            ax = fig.add_subplot(3, 2, idx, sharex=ax_cdf)
            ax.hist(times[name][fmt][quality], color=color, bins='auto',
                    histtype='bar',
                    density=True, label=label)
            ax.set_title(f'{name}, crf {quality}, {fmt}')
            ax.set_ylabel("PDF")
            ax.set_xlabel('Decoder Times')

            ###############################################################
            # plota os 3 best fit para esta qualidade
            best_dists = fitter_dict[fmt].df_errors
            best_dists = best_dists.sort_values(by="sumsquare_error")
            best_dists = best_dists.index[0:5]

            for dist_name in best_dists:
                sse = fitter_dict[fmt]
                sse = sse.df_errors["sumsquare_error"][dist_name]
                label = f'{dist_name},\nSSE = {sse: .3f}'
                ax.plot(fitter_dict[fmt].x,
                        fitter_dict[fmt].fitted_pdf[dist_name],
                        label=label)
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

            ###############################################################
            # plota o bar da taxa para esta qualidade
            s_avg = np.average(sizes[name][fmt][quality]) * 8
            s_std = np.std(sizes[name][fmt][quality]) * 8
            s_ct = np.average(sizes[name][fmt][quality]) * 8
            label = (f'rate_avg={s_avg / 1000:,.0f} Kbps\n'
                     f'rate_std={s_std / 1000:,.0f} Kbps\n'
                     f'rate/tile={s_ct / (1000 * m * n):.03f} kbps')
            ax_rate.bar(idx, s_avg, color=color, yerr=s_std, label=label)
            ax_rate.set_title('Rate')
            ax_rate.set_ylabel("Rate (10³ bps)")
            ax_rate.set_xlabel("Decoder Time (s)")
            ax_rate.ticklabel_format(axis='y', style='scientific',
                                     scilimits=(3, 3))
            ax_rate.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

            ###############################################################
            # plota a CDF para esta qualidade
            label = f'{name}_crf{quality}_{fmt}'
            ax_cdf.hist(times[name][fmt][quality], color=color, bins='auto',
                        density=True, cumulative=True, histtype='step',
                        label=label)
            ax_cdf.set_title('CDF')
            ax_cdf.set_ylabel("CDF")
            ax_cdf.set_xlabel("Decoder Time (s)")
            ax_cdf.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

        ax_rate.set_xticklabels(['0'] + config.tile_list)
        plt.tight_layout()
        grupo = config.videos_list[name]["group"]
        fig.savefig(f'{dirname}{sl}{grupo}-hist_{name}_{quality}')
        # plt.show()
        print(f'hist_{name}_{quality}')

    # Salva melhores fits em um csv
    # best_dist_df = pd.DataFrame(best_dist_df)
    # best_dist_df.to_csv(f'{dirname}{config.sl}best_dist.csv')


def hist2sameplt(graph_folder):
    """ Compara os histogramas. Para cada video-fmt plota todas as qualidades
    (agrega tiles e chunks)
    :return:
    """
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    times = util.AutoDict()
    sizes = util.AutoDict()

    os.makedirs(f'{dirname}', exist_ok=True)

    # Coleta dados
    for name, fmt, quality in it(config.videos_list, config.tile_list,
                                 config.quality_list):
        m, n = list(map(int, fmt.split('x')))
        times[name][fmt][quality] = []
        sizes[name][fmt][quality] = []

        for tile, chunk in it(range(1, m * n + 1),
                              range(1, config.duration + 1)):
            if name in 'ninja_turtles' and chunk > 58: continue
            t = dectime_multi[name][fmt][str(quality)][str(tile)][str(chunk)][
                'times']
            s = dectime_multi[name][fmt][str(quality)][str(tile)][str(chunk)][
                'size']
            times[name][fmt][quality].append(t)
            sizes[name][fmt][quality].append(s)

    # Calcular melhores fit e plota
    for name, quality in it(config.videos_list, config.quality_list):
        # Para cada video e fmt cria uma figura. cuidado. se mudar o formato do
        # gráfico vai ter que mudar tudo dentro da função.
        plt.close()
        fig = plt.figure(figsize=(13, 9), dpi=150)

        color_it = iter(color_list)
        ax_cdf = fig.add_subplot(2, 1, 2)
        ax = fig.add_subplot(2, 1, 1, sharex=ax_cdf)

        for idx, fmt in enumerate(config.tile_list, 1):
            color = next(color_it)
            m, n = list(map(int, fmt.split('x')))

            ###############################################################
            # plota o histograma desta qualidade
            t_avg = np.average(times[name][fmt][quality])
            t_std = np.std(times[name][fmt][quality])
            t_ct = np.average(times[name][fmt][quality]) * (m * n)

            label = (f'dectime_avg={t_avg:.03f} s\n'
                     f'dectime_std={t_std:.03f} s\n'
                     f'time/chunk/tile={t_ct:.03f} s')

            ax.hist(times[name][fmt][quality], color=color, bins='auto',
                    histtype='bar', density=True, label=label)
            ax.set_title(f'{name}, crf {quality}, {fmt}')
            ax.set_ylabel("PDF")
            ax.set_xlabel('Decoder Times')
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

            ###############################################################
            # plota a CDF para esta qualidade
            label = f'{name}_crf{quality}_{fmt}'
            ax_cdf.hist(times[name][fmt][quality], color=color, bins='auto',
                        density=True, cumulative=True, histtype='step',
                        label=label)
            ax_cdf.set_title('CDF')
            ax_cdf.set_ylabel("CDF")
            ax_cdf.set_xlabel("Decoder Time (s)")
            ax_cdf.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

        plt.tight_layout()
        grupo = config.videos_list[name]["group"]
        fig.savefig(f'{dirname}{sl}{grupo}-hist_{name}_{quality}')
        # plt.show()
        print(f'hist_{name}_{quality}')


def make_hist(tridata, dirname, bins, group=None, name=None, fmt=None,
              quality=None, tile=None, chunk=None, plot=None, fig=None,
              ax=None, x=4, y=1, new_fit=False):
    m_, n_ = list(map(int, fmt.split('x')))
    time, size, corr = tridata
    data_stats_t = [np.average(time), np.std(time), corr]
    data_stats_s = [np.average(size), np.std(size), corr]

    # Persistência
    print('Calculando o fit.')
    f_t_name = ''
    f_s_name = ''
    title = ''
    if group is not None \
            and fmt is not None:
        """ Com grupo e fmt """
        f_t_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_time_{bins}bins_group{group}_{fmt}.pickle')
        f_s_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_rate_{bins}bins_group{group}_{fmt}.pickle')
        title = f'{bins}bins_group{group}_{fmt}'
    if name is not None \
            and fmt is not None:
        """ Com nome e fmt """
        f_t_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_time_{bins}bins_{name}_{fmt}.pickle')
        f_s_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_rate_{bins}bins_{name}_{fmt}.pickle')
        title = f'{bins}bins_{name}_{fmt}'
    if fmt is not None \
            and quality is not None \
            and tile is not None:
        """ Com fmt, quality e tile """
        f_t_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_time_{bins}bins_{fmt}_{config.factor}{quality}_'
                    f'tile{tile}.pickle')
        f_s_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_rate_{bins}bins_{fmt}_{config.factor}{quality}_'
                    f'tile{tile}.pickle')
        title = f'{bins}bins_{name}_{fmt}'
    if group is None \
            and name is None \
            and fmt is not None \
            and quality is None \
            and tile is None:
        """ Com fmt apenas """
        f_t_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_time_{bins}bins_tudo_{fmt}_{config.factor}'
                    f'.pickle')
        f_s_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_rate_{bins}bins_tudo_{fmt}_{config.factor}'
                    f'.pickle')
        title = f'{fmt}'

    f_t = make_fit(data=time, bins=bins, out_file=f_t_name, overwrite=new_fit)
    f_s = make_fit(data=size, bins=bins, out_file=f_s_name, overwrite=new_fit)

    # Faz histograma
    if plot is None:
        plot = ['pdf_dectime', 'pdf_rate', 'cdf_dectime', 'cdf_rate']

    if ax is None:
        fig, ax = plt.subplots(x, y, figsize=(14, 11), dpi=150)

    for k in plot:
        if k in 'pdf_dectime':
            plota_hist(f_t, ax, bins, data_stats_t, 'time', 'pdf')
            ax.set_title(f'PDF Dectime - {title}')
            ax.set_xlabel('Decoder Time')
        elif k in 'pdf_rate':
            plota_hist(f_s, ax, bins, data_stats_s, 'rate', 'pdf')
            ax.set_title(f'PDF Bitrate - {title}')
            ax.set_xlabel('Bitrate')
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        elif k in 'cdf_dectime':
            plota_hist(f_t, ax, bins, data_stats_t, 'time', 'cdf')
            ax.set_title(f'CDF Dectime - {title}')
            ax.set_xlabel('Decoder Time')
        elif k in 'cdf_rate':
            plota_hist(f_s, ax, bins, data_stats_s, 'rate', 'cdf')
            ax.set_title(f'CDF Bitrate - {title}')
            ax.set_xlabel('Bitrate')
    return fig, ax


def make_fit(data, out_file, bins, overwrite=False):
    if os.path.exists(out_file) and not overwrite:
        # Se o pickle existe carregar o arquivo
        print(f'Carregando {out_file}.')
        with open(out_file, 'rb') as f1:
            f = pickle.load(f1)
    else:
        # Caso contrário calcule o fit e salve.
        print('Calculando o fit.')
        f = fitter.fitter.Fitter(data, bins=bins,
                                 distributions=config.dists,
                                 verbose=False,
                                 timeout=30)
        f.fit()

        print(f'Salvando picle em {out_file}.')
        with open(out_file, 'wb') as f1:
            pickle.dump(f, f1, pickle.HIGHEST_PROTOCOL)
    return f


def make_df_params_from_fit(f, df_dict: dict, st, fmt=None, quality=None,
                            group=None):
    errors = f.df_errors
    errors_sorted = errors.sort_values(by="sumsquare_error")
    shorted_dist = errors_sorted.index

    for dist in shorted_dist:
        try:
            params = f.fitted_param[dist]
        except KeyError:
            continue

        p = ''
        if dist in 'burr12':
            p = (f'c={params[0]}, d={params[1]}, loc={params[2]}, '
                 f'scale={params[3]}')
        elif dist in 'fatiguelife':
            p = f'c={params[0]}, loc={params[1]}, scale={params[2]}'
        elif dist in 'gamma':
            p = f'a={params[0]}, loc={params[1]}, scale={params[2]}'
        elif dist in 'invgauss':
            p = f'mu={params[0]}, loc={params[1]}, scale={params[2]}'
        elif dist in 'rayleigh':
            p = f'loc={params[0]}, scale={params[1]}'
        elif dist in 'lognorm':
            p = f's={params[0]}, loc={params[1]}, scale={params[2]}'
        elif dist in 'genpareto':
            p = f'c={params[0]}, loc={params[1]}, scale={params[2]}'
        elif dist in 'pareto':
            p = f'b={params[0]}, loc={params[1]}, scale={params[2]}'
        elif dist in 'halfnorm':
            p = f'loc={params[0]}, scale={params[1]}'
        elif dist in 'expon':
            p = f'loc={params[0]}, scale={params[1]}'

        df_dict['Distribution'].append(dist)
        df_dict['Parameters'].append(p)
        df_dict['Statistics'].append(st)
        df_dict['SSE'].append(f.df_errors["sumsquare_error"][dist])
        if fmt:
            df_dict['Format'].append(fmt)
        if quality:
            df_dict['Quality'].append(quality)
        if group:
            df_dict['Group'].append(group)


def plota_hist(f, ax: matplotlib.axes.Axes, bins, data_stats, metric, func,
               label=None, fmt='') -> plt.Axes:
    if isinstance(data_stats, dict):
        avg = data_stats['avg']
        std = data_stats['std']
        corr = data_stats['corr']
    elif isinstance(data_stats, list):
        [avg, std, corr] = data_stats

    errors = f.df_errors
    errors_sorted = errors.sort_values(by="sumsquare_error")
    short_sse = errors_sorted.index[0:3]

    if func in 'pdf':
        ax.set_ylabel("Probability Density")
        if metric is 'time':
            if label is None:
                label = (f'dectime_avg={avg:.03f} s\n'
                         f'dectime_std={std:.03f} s\n'
                         f'rate_corr={corr:.03f}')
            ax.ticklabel_format(axis='y', style='scientific')
        elif metric is 'rate':
            label = (f'rate_avg={avg:.03f} bps\n'
                     f'rate_std={std:.03f} bps\n'
                     f'time_corr={corr:.03f}')

        # ax.hist(f._data,
        #         bins=bins,
        #         histtype='bar',
        #         density=True,
        #         label=label)
        ax.bar(f.x, f.y, width=f.x[1] - f.x[0], label=label)
        # hist, bin_edges = np.histogram(f._data, bins='auto', density=True)
        # ax.bar(f.x, f.y)

        # plota os 3 melhores fits
        for dist_name in short_sse:
            # for dist_name in reversed(short_sse):
            sse = f.df_errors["sumsquare_error"][dist_name]
            label = f'{dist_name}'
            # label = f'{dist_name},\nSSE = {sse: .3E}'
            ax.plot(f.x, f.fitted_pdf[dist_name], label=label)
            # ax.plot(f.x, f.fitted_pdf[dist_name], label=label,
            #         color=c_dist[dist_name])

    if func in 'cdf':
        ax.set_ylabel("Cumulative Distribution")
        if metric is 'time':
            label = (f'dectime_avg={avg:.03f} s\n'
                     f'dectime_std={std:.03f} s\n'
                     f'corr={corr:.03f}')
        elif metric is 'rate':
            label = (f'rate_avg={avg:.03f} bps\n'
                     f'rate_std={std:.03f} bps\n'
                     f'corr={corr:.03f}')
        ax.hist(f._data,
                bins=bins,
                density=True,
                cumulative=True,
                histtype='step',
                label=label)
        # ax.bar(f.x, f.y, width=f.x[1] - f.x[0], label=label)

    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
    return ax


def get_data(groups=(0, 1, 2, 3),
             videos_list=config.videos_list,
             tile_list=config.tile_list,
             quality_list=config.quality_list,
             tiles=None,
             single=True):
    df_t = pd.DataFrame()
    df_r = pd.DataFrame()
    corr = []

    for group in groups:
        for name in videos_list:
            if config.videos_list[name]['group'] not in str(group):
                continue

            for fmt in tile_list:
                m, n = list(map(int, fmt.split('x')))

                if tiles is None:
                    tiles = range(1, m * n + 1)

                for quality in quality_list:
                    quality = str(quality)
                    for tile in tiles:
                        col_t = (f'{group}_{name}_{fmt}_'
                                 f'{config.factor}{quality}_'
                                 f'tile{tile}_'
                                 f'time')
                        col_r = (f'{group}_{name}_{fmt}_'
                                 f'{config.factor}{quality}_'
                                 f'tile{tile}_'
                                 f'rate')

                        time = []
                        rate = []
                        if single:
                            # As taxas do json singlekey já estão em bps.
                            d = dectime_flat
                            time = d[col_t]
                            rate = d[col_r]
                            c = np.corrcoef((time, rate))[1][0]
                            corr.append(c)
                        else:
                            pass
                            # Dúvida... as taxas dos dados multikey estão em
                            # bits ou bytes?
                            # d = dectime_multi
                            # for chunk in range(1, config.duration + 1):
                            #     if name in 'ninja_turtles' and chunk > 58:
                            #         time.append(0)
                            #         rate.append(0)
                            #     d = d[fmt][str(quality)][str(tile)][str(chunk)]
                            #     time.append(float(d['times']))
                            #     rate.append(float(d['size']) * 8)

                        df_t[col_t] = time
                        df_r[col_r] = rate

    return df_t, df_r, np.average(corr)


def get_data_tudo_fmt(fmt):
    size = []
    time = []

    for name in config.videos_list:
        t, s, _ = get_data_quality(name, fmt)
        time.extend(t)
        size.extend(s)
    corr = np.corrcoef((time, size))[1][0]

    return time, size, corr


def get_data_name_tile(fmt, quality):
    size = []
    time = []

    for name in config.videos_list:
        t, s, _ = get_data_tiles(name, fmt, quality)
        time.extend(t)
        size.extend(s)
    corr = np.corrcoef((time, size))[1][0]

    return time, size, corr


def get_data_tudo_fmt_quality(fmt, quality):
    size = []
    time = []
    corr = []
    for name in config.videos_list:
        t, s, _ = get_data_tiles(name, fmt, quality)
        time.extend(t)
        size.extend(s)
        corr.append(np.corrcoef((t, s))[1][0])
    corr = np.average(corr)

    return time, size, corr


def get_data_group_fmt(group, fmt):
    size = []
    time = []
    for name in config.videos_list:
        if config.videos_list[name]['group'] in group:
            t, s, _ = get_data_quality(name, fmt)
            time.extend(t)
            size.extend(s)
    corr = np.corrcoef((time, size))[1][0]

    return time, size, corr


def get_data_name():
    size = []
    time = []

    for name in config.videos_list:
        t, s, _ = get_data_fmt(name)
        time.extend(t)
        size.extend(s)

    corr = np.corrcoef((time, size))[1][0]
    return time, size, corr


def get_data_fmt(name):
    size = []
    time = []

    for fmt in config.tile_list:
        t, s, corr = get_data_quality(name, fmt)
        time.extend(t)
        size.extend(s)

    corr = np.corrcoef((time, size))[1][0]
    return time, size, corr


def get_data_quality(name, fmt):
    size = []
    time = []

    for quality in config.quality_list:
        t, s, corr = get_data_tiles(name, fmt, quality)
        time.extend(t)
        size.extend(s)

    corr = np.corrcoef((time, size))[1][0]
    return time, size, corr


def get_data_name_chunk(fmt, tile, quality):
    size = []
    time = []

    for name in config.videos_list:
        t, s, corr = get_data_chunks(name, fmt, quality, tile)
        time.extend(t)
        size.extend(s)

    corr = np.corrcoef((time, size))[1][0]
    return time, size, corr


def get_data_tiles(name, fmt, quality):
    size = []
    time = []
    m_, n_ = list(map(int, fmt.split('x')))

    for tile in range(1, m_ * n_ + 1):
        t, s, corr = get_data_chunks(name, fmt, quality, tile)
        time.extend(t)
        size.extend(s)

    corr = np.corrcoef((time, size))[1][0]
    return time, size, corr


def get_data_chunks(name, fmt, quality, tile):
    # dec = dectime_multi
    size = []
    time = []

    # Lista todos os chunks
    for chunk in range(1, config.duration + 1):
        if name in 'ninja_turtles' and chunk > 58:
            size.append(0)
            time.append(0)
        dec = dec[fmt]
        dec = dec[str(quality)]
        dec = dec[str(tile)]
        dec = dec[str(chunk)]
        s = dec['size']
        t = dec['times']
        size.append(float(s) * 8)
        time.append(float(t))

    # Plota os dados
    corr = np.corrcoef((time, size))[1][0]
    return time, size, corr


def get_time(file, video_seg):
    print(f'Processando {file}')
    try:
        f = open(file, 'r', encoding='utf-8')
    except FileNotFoundError:
        print(f'Arquivo {file} não encontrado.')
        return 0
    video_seg.bench_stamp = 'utime='
    chunks_times = []
    for line in f:
        if line.find(video_seg.bench_stamp) >= 0:
            # Pharse da antiga decodificação
            if video_seg.factor in 'crf':
                line2 = line.replace('bench: ', ' ').replace('s', '').strip()
                chunks_times.append(float(line2.split('=')[-1]))
            elif video_seg.factor in ['scale', 'qp']:
                line = line.strip()
                line = line.split(' ')[1]
                line = line.split('=')[1]
                chunks_times.append(float(line[:-1]))
    f.close()
    return np.average(chunks_times)


class Heatmap:
    def __init__(self, m, n):
        self.average_time = []
        self.average_size = []
        self.std_time = []
        self.std_size = []
        self.frame_time = np.zeros((n, m))
        self.frame_size = np.zeros((n, m))


def make_heatmap(name, fmt, quality):
    m, n = list(map(int, fmt.split('x')))
    heatmap = Heatmap(m, n)

    for tile, (x, y) in zip(range(1, m * n + 1),
                            it(range(1, m + 1),
                               range(1, n + 1))):
        # Coleta dados de todos os chunks
        time, size, _ = get_data_chunks(name, fmt, quality, tile)

        time = np.average(time)
        size = np.average(size)

        heatmap.average_time.append(time)
        heatmap.average_size.append(size)
        heatmap.std_time.append(np.std(time))
        heatmap.std_size.append(np.std(size))

        # Preenche a matriz do heatmap
        heatmap.frame_time[y - 1, x - 1] = time
        heatmap.frame_size[y - 1, x - 1] = size

    return heatmap


if __name__ == "__main__":
    main()
