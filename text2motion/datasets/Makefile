ROOT = $(shell git rev-parse --show-toplevel)/text2motion

mesh:
	echo "make sure to run on hpc dtu gui with graphics support and that you use 'vglrun' before python3 call!"
	cd $(ROOT) && python3 -m vglrun datasets.smplx_mesher

disp_data:
	cd $(ROOT) && python3 -m datasets.motionx_loader
