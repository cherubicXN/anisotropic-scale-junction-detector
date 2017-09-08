/*----------------------------------------------------------------------
  Anisotropic Scale Junction Detector
  Copyright (C) 2017  Nan Xue, Gui-Song Xia

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
  ---------------------------------------------------------------------
*/
#include <iostream>
#include <fstream>
#include <ctime>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <list>

#include "acj.h"
#include "aacj.h"
#include "matcher.h"
#include "helper.h"
#include "lsdcall.h"


int main(int argc, char **argv)
{

	ACJParameters par;
	par.nfaApproximateOn = false;
	par.maxNumPixel = 120;
	par.nsMax = 30;
	par.rdMax = 10;

	par.scaleFixed = 7;
	par.deltaRLocal = 2.5f;
	par.noiseACJ = 0.008;
	par.noiseASJ = 0;

	switch (argc)
	{
	case 2:
	{
	std::string filename1 = argv[1];
	AACJDetection aacj1(filename1, 2.f, 500.f, 1, &par);
	break;
	}
	default:
	std::cout<<"Error usage!\nExample:\n\t  (Linux) ./ASJDetector your_image.jpg  \n\t (Windows) ASJDetector.exe your_image.jpg"<<std::endl;
	break;
	}

	return 0;
}
