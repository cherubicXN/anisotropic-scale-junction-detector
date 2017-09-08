# ASJ: Anisotropic Scale Junction Detection and Matching

[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

ASJ is a novel compact geometric feature for characterising images. 

If you use this code for research please cite:

	@article{XXBZS17,
		author = {Xue, Nan and Xia, Gui-Song and Bai, Xiang and Zhang, Liangpei and Shen, Weiming},
		journal = {IEEE Transactions on Image Processing},		
		title = {{Anisotropic-Scale Junction Detection and Matching for Indoor Images.}},
		url = {https://arxiv.org/pdf/1703.05630.pdf},
		year = {2017}
	}

## Requirements

Building and using requires the following libraries and programs

    OpenCV

The code has been tested on Ubuntu 14.04 and Windows 10.


## Build instructions

```
git clone git@github.com:cherubicXN/anisotropic-scale-junction-detector.git
cd anisotropic-scale-junction-detector
mkdir build
cd build
cmake ..
make
```
## Useage

Linux:
```
./ASJDetector ../example_image/im1.jpg
```

Windows:
```
ASJDetector.exe ../example_image/im1.jpg
```

### Output file format

coming soon.
