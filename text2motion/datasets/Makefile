ROOT = $(shell git rev-parse --show-toplevel)/text2motion

mesh:
	echo "make sure to run on hpc dtu gui with graphics support and that you use 'vglrun' before python3 call!"
	cd $(ROOT) && vglrun python3 -m datasets.smplx_mesher
