for i in *.yuv; do
    mkdir ${i%.yuv};
    ffmpeg -s 4320x2160 -r 30 -f rawvideo -i ${i} -s 600x300 ${i%.yuv}/${i%.yuv}%04d.jpg;
done