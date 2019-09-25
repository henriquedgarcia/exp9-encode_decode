#!/bin/python3
# coding=utf-8
import os
import pickle
from itertools import product as it

import fitter
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

from utils import util

sl = util.check_system()['sl']

project = 'times_12videos_crf'
config = util.Config('Config.json', factor='crf')
dectime = util.load_json('times_12videos_crf.json')
color_list = ['blue', 'orange', 'green', 'red']
# bins = 100
dists = ['burr12', 'expon', 'fatiguelife', 'gamma', 'genpareto', 'halfnorm',
         'invgauss', 'rayleigh', 'rice', 't']


def main():
    json2pandas()
    # psnr()  # Processar psnr também. à fazer
    # stats()
    # graph1(graph_folder='1_graph1-tiles-chunks_x_dec_time')
    # graph2(graph_folder='2_graph2-quality-chunks_x_dec_time')
    # graph3(graph_folder='3_graph3_heatmap')
    # histogram_name_fmt('histogram_name-fmt')
    # histogram_group_fmt('histogram_group-fmt')
    # heatmap_fmt_quality('heatmap_fmt-quality')
    # hist1samefig(graph_folder="hist1samefig")
    # hist1sameplt(graph_folder="hist1sameplt")
    # hist2samefig(graph_folder="hist2samefig")
    # hist2sameplt(graph_folder="hist2sameplt")
    # graph4()   # Por grupo
    # graph5b()  # Geralzão por tile
    # graph5c()  # Geralzão por qualidade
    # graph5d()  # Geralzão por fmt
    # graph5e()  # Geralzão por name


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
    video_seg = util.Video(config=config)
    video_seg.project = f'results{sl}ffmpeg_scale_12videos_60s'
    video_seg.factor = f'scale'
    video_seg.segment_base = f'segment'
    video_seg.dectime_base = f'dectime_ffmpeg'
    video_seg.bench_stamp = f'bench: utime'
    video_seg.multithread = f'single'

    video_seg.quality_list = config.quality_list

    for video_seg.name in config.videos_list:
        for video_seg.fmt in config.tile_list:
            for video_seg.quality in config.quality_list:
                print(f'Processing {video_seg.basename}.txt')

                for video_seg.tile in range(1, video_seg.num_tiles + 1):
                    chunks = video_seg.duration * video_seg.fps

                    for video_seg.chunk in range(1, chunks + 1):
                        # Processing segment size (bytes)
                        file = video_seg.segment_path + '.mp4'
                        if os.path.isfile(file):
                            video_seg.size = os.path.getsize(file)

                        # Processing decoding time
                        file = video_seg.log_path + '.txt'
                        if os.path.isfile(file):
                            times = []
                            f = open(file, 'r', encoding='utf-8')

                            for line in f:
                                if line.find(video_seg.bench_stamp) >= 0:
                                    # Pharse da antiga decodificação
                                    if video_seg.factor in 'crf':
                                        line = line.replace('bench: ', ' ')
                                        line = line.replace('s ', ' ')
                                        line = line.strip()[:-1]
                                        line = line.split(' ')
                                        for i in range(0, len(line), 3):
                                            times.append(float(line[i][6:]))

                                    elif video_seg.factor in ['scale', 'qp']:
                                        line = line.split(' ')[1]
                                        line = line.split('=')[1]
                                        times.append(float(line[:-1]))
                            f.close()

                            video_seg.times = np.average(times)
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
        df1 = get_data_tudo(tile_list=[fmt], metrics='time')
        df2 = get_data_tudo(tile_list=[fmt], metrics='size')
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
        df = get_data_tudo(tile_list=[fmt])
        time_chunks = df.mean(axis=1)
        avg = time_chunks.mean()
        std = time_chunks.std()

        leg = (f'{fmt}\n'
               f'avg={avg: .3f} s\n'
               f'std={std: .3f} s')
        ax.plot(time_chunks, label=leg)
    df = get_data_tudo()
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
            df = get_data_tudo(groups=[group], tile_list=[fmt])
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


# hist0
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


# hist1
def histogram_group_fmt(graph_folder):
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(dirname + f'{sl}data', exist_ok=True)

    for bins in ['auto', 25, 50, 75, 100, 125, 150, 200]:
        for group in ['0', '1', '2', '3']:
            for fmt in config.tile_list:
                # Coleta dados
                tridata = get_data_group_fmt(group, fmt)

                # Faz o plot
                fig = make_hist(tridata, dirname, bins=bins, group=group,
                                name=None, fmt=fmt, quality=None, tile=None,
                                chunk=None)

                # Salva
                fig.savefig(f'{dirname}{sl}'
                            f'hist_groups_bins{bins}_group{group}_{fmt}')
                # fig.show()
                print(f'hist bins {bins}, group{group}_{fmt}')


def histogram_tudo_fmt(graph_folder):
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(dirname + f'{sl}data', exist_ok=True)

    for bins in ['auto']:
        plt.close()
        fig = plt.figure(figsize=(7.5, 6), dpi=200)
        for n, fmt in enumerate(config.tile_list, 1):
            ax = fig.add_subplot(2, 2, n)

            # Coleta dados
            time, _, corr = get_data_tudo_fmt(fmt)
            data_stats_t = [np.average(time), np.std(time), corr]
            f_t_name = (f'{dirname}{sl}data{sl}'
                        f'fitter_time_{bins}bins_tudo_{fmt}_{config.factor}'
                        f'.pickle')

            # Faz o fit
            f_t = make_fit(data=time, bins=bins, out_file=f_t_name)
            # Faz o plot
            ax = plota_hist(f_t, ax, bins, data_stats_t, 'time', 'pdf')
            # infos
            ax.legend(loc='best')
            ax.set_title(f'PDF Decoding Time - {fmt}')
            ax.set_xlabel('Decoding Time')

        # Salva
        plt.tight_layout()

        # fig.savefig(f'{dirname}{sl}hist_groups_bins{bins}_tudo_'
        #             f'{config.factor}')
        fig.show()
        print(f'hist bins {bins}, tudo_{config.factor}')


def heatmap_fmt_quality(graph_folder):
    dirname = f'results{sl}{project}{sl}{graph_folder}'
    os.makedirs(f'{dirname}{sl}data', exist_ok=True)
    bins = 'auto'

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
            t = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['times']
            s = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['size']
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
            f = fitter.fitter.Fitter(data, bins='auto', distributions=dists,
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
            f = fitter.fitter.Fitter(data, bins='auto', distributions=dists,
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
            t = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['times']
            s = dectime[name][fmt][str(quality)][str(tile)][str(chunk)]['size']
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
              quality=None, tile=None, chunk=None):
    m_, n_ = list(map(int, fmt.split('x')))
    time, size, corr = tridata
    data_stats_t = [np.average(time), np.std(time), corr]
    data_stats_s = [np.average(size), np.std(size), corr]

    # Persistência
    print('Calculando o fit.')
    f_t_name = ''
    f_s_name = ''
    if group is not None \
            and fmt is not None:
        f_t_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_time_{bins}bins_group{group}_{fmt}.pickle')
        f_s_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_rate_{bins}bins_group{group}_{fmt}.pickle')
        tile = f'{bins}bins_group{group}_{fmt}'
    if name is not None \
            and fmt is not None:
        f_t_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_time_{bins}bins_{name}_{fmt}.pickle')
        f_s_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_rate_{bins}bins_{name}_{fmt}.pickle')
        tile = f'{bins}bins_{name}_{fmt}'
    if fmt is not None \
            and quality is not None \
            and tile is not None:
        f_t_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_time_{bins}bins_{fmt}_{config.factor}{quality}_'
                    f'tile{tile}.pickle')
        f_s_name = (f'{dirname}{sl}data{sl}'
                    f'fitter_rate_{bins}bins_{fmt}_{config.factor}{quality}_'
                    f'tile{tile}.pickle')
        tile = f'{bins}bins_{name}_{fmt}'

    f_t = make_fit(data=time, bins=bins, out_file=f_t_name)
    f_s = make_fit(data=size, bins=bins, out_file=f_s_name)

    # Faz histograma
    plt.close()
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), dpi=150)
    for ax, k in zip(axes, ['pdf_dectime', 'pdf_rate',
                            'cdf_dectime', 'cdf_rate']):
        if k in 'pdf_dectime':
            ax = plota_hist(f_t, ax, bins, data_stats_t, 'time', 'pdf')
            ax.set_title(f'PDF Dectime - {tile}')
            ax.set_xlabel('Decoder Time')
        elif k in 'pdf_rate':
            ax = plota_hist(f_s, ax, bins, data_stats_s, 'rate', 'pdf')
            ax.set_title(f'PDF Bitrate - {tile}')
            ax.set_xlabel('Bitrate')
        elif k in 'cdf_dectime':
            ax = plota_hist(f_t, ax, bins, data_stats_t, 'time', 'cdf')
            ax.set_title(f'CDF Dectime - {tile}')
            ax.set_xlabel('Decoder Time')
        elif k in 'cdf_rate':
            ax = plota_hist(f_s, ax, bins, data_stats_s, 'rate', 'cdf')
            ax.set_title(f'CDF Bitrate - {tile}')
            ax.set_xlabel('Bitrate')

    plt.tight_layout()
    return fig


def make_fit(data, out_file, bins, overwrite=False):
    if os.path.exists(out_file) and not overwrite:
        print(f'Carregando {out_file}.')
        with open(out_file, 'rb') as f1:
            f = pickle.load(f1)
    else:
        print('Calculando o fit do tempo.')
        f = fitter.fitter.Fitter(data, bins=bins,
                                 distributions=dists,
                                 verbose=False)
        f.fit()

        print(f'Salvando {out_file}.')
        with open(out_file, 'wb') as f1:
            pickle.dump(f, f1, pickle.HIGHEST_PROTOCOL)

    return f


def plota_hist(f, ax: matplotlib.axes.Axes, bins, data_stats, metric, func,
               label='') -> plt.Axes:
    [avg, std, corr] = data_stats
    errors = f.df_errors
    errors_sorted = errors.sort_values(by="sumsquare_error")
    short_sse = errors_sorted.index[0:5]

    if func in 'pdf':
        ax.set_ylabel("Probability Density")
        if metric is 'time':
            label = (f'dectime_avg={avg:.03f} s\n'
                     f'dectime_std={std:.03f} s\n'
                     f'corr={corr:.03f}')
            ax.ticklabel_format(axis='y', style='scientific')
        elif metric is 'rate':
            label = (f'rate_avg={avg:.03f} bps\n'
                     f'rate_std={std:.03f} bps\n'
                     f'corr={corr:.03f}')
        ax.hist(f._data,
                bins=bins,
                histtype='bar',
                density=True,
                label=label)
        # plota os 3 melhores fits
        for dist_name in short_sse:
            sse = f.df_errors["sumsquare_error"][dist_name]
            label = f'{dist_name},\nSSE = {sse: .3E}'
            ax.plot(f.x, f.fitted_pdf[dist_name], label=label)

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

    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
    return ax


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
    size = []
    time = []

    # Lista todos os chunks
    for chunk in range(1, config.duration + 1):
        if name in 'ninja_turtles' and chunk > 58:
            size.append(0)
            time.append(0)
        dec = dectime[name]
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
