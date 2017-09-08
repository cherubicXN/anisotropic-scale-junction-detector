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
#include "lsdcall.h"
#include "lsd.h"
#include "regionsupport.h"

void lsdcall(const cv::Mat &imgray, std::vector<cv::Vec4f> &result)
{
	//double *image;
	image_double image;
	int X = imgray.cols;
	int Y = imgray.rows;
	image = new_image_double(X, Y);
	//image = (double *)malloc(X*Y*sizeof(double));
	
	typedef float T;
	if (imgray.depth() == CV_8UC1)
		typedef char T;
	for (int x = 0; x < image->xsize; ++x)
		for (int y = 0; y < image->ysize; ++y)
		{
			double temp = (double)imgray.at<T>(y, x);
			image->data[x + y*image->xsize] = (double)imgray.at<T>(y, x);
		}
	int n;
	//double *out;
	ntuple_list out = lsd(image);
	//out = lsd(&n, image, X, Y);

	for (int i = 0; i < out->size; ++i)
		//result.push_back(cv::Vec4f((float)out[7 * i], (float)out[7 * i + 1], (float)out[7 * i + 2], (float)out[7 * i + 3]));
		result.push_back(cv::Vec4f(out->values[i*out->dim + 0], out->values[i*out->dim + 1], out->values[i*out->dim + 2], out->values[i*out->dim + 3]));
}