

#python video.py --suffix AIx2 -s 2 --num_process_per_gpu 4 \
#python video.py --suffix AIx4 -s 4 --encoder hevc_nvenc --num_process_per_gpu 8 \
#python video-dataloader.py --suffix AIx4 --encoder hevc_nvenc -s 4 --batch-size 16 \
python video-dataloader.py --suffix AIx2 -s 2 --fp32 --batch-size 16 \
		-i ~/samba/data/net_disk/media/魔天战神/魔天战神-粤语国语/魔天战神02-大灾难降临.mp4 -o ~/samba/data/net_disk/media/魔天战神/魔天战神-粤语国语-AI放大/


exit 0

for i in $(seq -w 9 12)
do
	#echo "i: $i"
	#batch-task2.py -- python video.py --suffix AIx2 -s 2 --num_process_per_gpu 4 \
	python video.py --suffix AIx2 -s 2 --num_process_per_gpu 4 \
		-i ~/data/samba/ro/net_disk/media/一拳超人/S2_libx265/S02EP${i}.mp4 -o ~/data/samba/rw/aliyun-webdav/net_disk/media/一拳超人/S2-AIx2/
done
