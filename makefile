all:
	pip install . --user

representation:
	python3 directory_tree.py

clean:
	rm -rf build
	rm -rf src/opencllib.egg-info
	rm -rf sample_opencl_lib.egg-info
	rm -rf dist
	pip uninstall -y sample_opencl_lib

test:
	python3 -m unittest test.py