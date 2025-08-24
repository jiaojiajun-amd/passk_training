docker run -it \
	--device /dev/dri \
	--device /dev/kfd \
	-p 8265:8265 \
	--group-add video \
	--cap-add SYS_PTRACE \
	--security-opt seccomp=unconfined \
	--privileged \
    -v /group/ossdphi_algo_scratch_14/jiajjiao/code/passk_training/:/code/passk_training \
	--shm-size 128G \
	-w /code \
	passk-verl
