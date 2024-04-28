

#python video.py --suffix AIx2 -s 2 --num_process_per_gpu 4 \
#python inference_realesrgan_video.py --suffix AIx4 -s 4 \
#python video-dataloader.py --suffix AIx4 -s 4 --encoder hevc_nvenc --batch-size 4 \
python video.py --suffix AIx4 -s 4 --encoder hevc_nvenc --num_process_per_gpu 6 \
		-i ~/samba/data/net_disk/media/魔天战神/魔天战神-日语无字/魔天战神02-大灾难降临.mp4 \
		-o ~/samba/data/net_disk/media/魔天战神/魔天战神-日语无字-AIx4/
		#-i ~/samba/data/net_disk/media/宫崎骏/千与千寻/千与千寻.2001.1080p.h265.crf18.mp4 -o ~/samba/data/AI放大-test/


exit 0

for i in $(seq -w 9 12)
do
	#echo "i: $i"
	#batch-task2.py -- python video.py --suffix AIx2 -s 2 --num_process_per_gpu 4 \
	python video.py --suffix AIx2 -s 2 --num_process_per_gpu 4 \
		-i ~/data/samba/ro/net_disk/media/一拳超人/S2_libx265/S02EP${i}.mp4 -o ~/data/samba/rw/aliyun-webdav/net_disk/media/一拳超人/S2-AIx2/
done
