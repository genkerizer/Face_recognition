docker run --rm -it \
			--net=host \
			--name=fake_jpp \
			-v `pwd`/.:/hopny \
			quhu_gpu:2.2.0 bash