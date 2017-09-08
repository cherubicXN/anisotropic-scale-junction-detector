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
#include <limits>
#include <vector>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv2/opencv.hpp>
#include "helper.h"

template <typename T>
int checkTypeOfInputImage(const T &im)
{
	if (im.type() == CV_8UC1)
		return 0;
	if (im.type() == CV_8UC3)
		return 1;
	return 2;
}

template <typename T>
void getMinMax(const T &im,
	float &minValue,
	float &maxValue)
{
	assert(im.channels() == 1);
	float min = std::numeric_limits<float>::max();
	float max = std::numeric_limits<float>::min();
	float *data = (float *)im.data;
	for (int i = 0; i < im.cols*im.rows; ++i)
	{
		if (min >data[i])
			min = data[i];
		if (max < data[i])
			max = data[i];
	}
}

void convolution1D(const std::vector<double> &a, const std::vector<double> &kernel,
	std::vector<double> &b, int optionSame)
{
	b = std::vector<double>(a.size() + kernel.size() - 1, 0);
	for (int j = 0; j < a.size(); ++j)
	{
		for (int k = 0; k < kernel.size(); ++k)// k = i-j+1 
		{
			if (j + k < b.size())
				b[j + k] += a[j] * kernel[k];
		}
	}

	if (optionSame)
	{
		int sz = (kernel.size() - 1) / 2;
		b.erase(b.begin(), b.begin() + sz);
		b.erase(b.end() - sz, b.end());
	}
}

void imCrop(const cv::Mat &ImageIn,
	int x, int y, int radius,
	cv::Mat &imageCroppedVec)
{
#ifdef _MSC_VER
	assert(ImageIn.depth() != 0, "The type of image must be CV_32FC1, please check");
#else
	assert(ImageIn.depth() != 0);
#endif
	const size_t rows = ImageIn.rows;
	const size_t cols = ImageIn.cols;

	//imageCroppedVec.resize((2 * radius + 1)*(2 * radius + 1), 0.0);
	imageCroppedVec = cv::Mat(2 * radius + 1, 2 * radius + 1, CV_32FC1);

	for (int j = y - radius, jj = 0; j <= y + radius; ++j, ++jj)
	{
		for (int i = x - radius, ii = 0; i <= x + radius; ++i, ++ii)
		{
			if (i < 0 || i >= cols || j < 0 || j >= rows)
				imageCroppedVec.at<float>(jj, ii) = 0.0f;
			else
				imageCroppedVec.at<float>(jj, ii) =
				ImageIn.at<float>(j, i);
		}
	}
}

cv::Mat im2float(const cv::Mat& image)
{
	cv::Mat imageFloat(image.rows, image.cols, CV_MAKETYPE(CV_32F, image.channels()), cv::Scalar(0));
	float *imageFloatData = imageFloat.ptr<float>(0);
	const unsigned char *imageData = image.ptr<uchar>(0);

	int numPixel = image.cols*image.rows*image.channels();
	for (int i = 0; i < numPixel; ++i)
	{
		//*imageFloatData = (float)*imageData/255.0f;
		*imageFloatData = (float)*imageData;
		++imageFloatData;
		++imageData;
	}
	//	for (int i = 0; i < image.cols*image.rows; ++i)
	//	{
	//		imageFloatData[i] = (float) imageData[i];
	//	}
	return imageFloat;
}

cv::Mat im2uchar(const cv::Mat& image)
{
	//cv::Mat imageFloat(image.rows, image.cols, CV_32FC1, cv::Scalar(0));
	cv::Mat imageUchar(image.rows, image.cols, CV_MAKETYPE(CV_8U, image.channels()), cv::Scalar(0));
	unsigned char *imageUcharData = imageUchar.ptr<unsigned char>(0);
	const float *imageData = image.ptr<float>(0);

	float max_value = std::numeric_limits<float>::min(),
		min_value = std::numeric_limits<float>::max();
	int npixels = image.cols*image.rows*image.channels();

	imageData = image.ptr<float>(0);
	for (int i = 0; i < npixels; ++i)
	{
		float value = *imageData;

		*imageUcharData = (unsigned char)(value * 255);
		++imageUcharData;
		++imageData;
	}
	//	for (int i = 0; i < image.cols*image.rows; ++i)
	//	{
	//		imageFloatData[i] = (float) imageData[i];
	//	}
	return imageUchar;
}

template <typename T>
void gaussianBlurInplace(T &inplace, float sigma)
{
	int size = (int)(2.0*3.0*sigma + 1.0);
	if (size % 2 == 0)
		++size;
	cv::GaussianBlur(inplace, inplace, cv::Size(size, size),
		sigma, sigma, cv::BORDER_REPLICATE);
}

template <typename T>
T gaussianBlur(const T input, float sigma)
{
	T ret(input.rows, input.cols, input.type());
	int size = (int)(2.0 * 3.0 * sigma + 1.0); if (size % 2 == 0) size++;
	cv::GaussianBlur(input, ret, cv::Size(size, size), sigma, sigma, cv::BORDER_REPLICATE);
	return ret;
}

template <typename T> void swap(T *a, T *b)
{
	T tmp = *a; *a = *b; *b = tmp;
}

void solveLinear3x3(float *A, float *b)
{
	// find pivot of first column
	int i = 0;
	float *pr = A;
	float vp = abs(A[0]);
	float tmp = abs(A[3]);
	if (tmp > vp)
	{
		// pivot is in 1st row
		pr = A + 3;
		i = 1;
		vp = tmp;
	}
	if (abs(A[6]) > vp)
	{
		// pivot is in 2nd row
		pr = A + 6;
		i = 2;
	}

	// swap pivot row with first row
	if (pr != A) { swap(pr, A); swap(pr + 1, A + 1); swap(pr + 2, A + 2); swap(b + i, b); }

	// fixup elements 3,4,5,b[1]
	vp = A[3] / A[0]; A[4] -= vp*A[1]; A[5] -= vp*A[2]; b[1] -= vp*b[0];

	// fixup elements 6,7,8,b[2]]
	vp = A[6] / A[0]; A[7] -= vp*A[1]; A[8] -= vp*A[2]; b[2] -= vp*b[0];

	// find pivot in second column
	if (abs(A[4]) < abs(A[7])) { swap(A + 7, A + 4); swap(A + 8, A + 5); swap(b + 2, b + 1); }

	// fixup elements 7,8,b[2]
	vp = A[7] / A[4];
	A[8] -= vp*A[5];
	b[2] -= vp*b[1];

	// solve b by back-substitution
	b[2] = (b[2]) / A[8];
	b[1] = (b[1] - A[5] * b[2]) / A[4];
	b[0] = (b[0] - A[2] * b[2] - A[1] * b[1]) / A[0];
}

void meanStd2(const cv::Mat& image, float &mean, float &std)
{
	mean = 0.0;
	std = 0.0;

	float *data = (float *)image.data;
	int n = image.cols*image.rows;
	for (int i = 0; i < n; ++i)
		mean += data[i];
	mean /= (float)n;

	for (int i = 0; i < n; ++i)
		std += (data[i] - mean)*(data[i] - mean);
	std /= (float)(n);
	std = sqrt(std);
}

void conv2(const cv::Mat &img, const cv::Mat& kernel, ConvolutionType type, cv::Mat& dest) {
	cv::Mat source = img;
	if (CONVOLUTION_FULL == type) {
		source = cv::Mat();
		const int additionalRows = kernel.rows - 1, additionalCols = kernel.cols - 1;
		copyMakeBorder(img, source, (additionalRows + 1) / 2, additionalRows / 2, (additionalCols + 1) / 2, additionalCols / 2, cv::BORDER_CONSTANT, cv::Scalar(0));
	}

	cv::Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);
	int borderMode = cv::BORDER_CONSTANT;

	filter2D(source, dest, img.depth(), kernel, anchor, 0, borderMode);

	if (CONVOLUTION_VALID == type) {
		dest = dest.colRange((kernel.cols - 1) / 2, dest.cols - kernel.cols / 2)
			.rowRange((kernel.rows - 1) / 2, dest.rows - kernel.rows / 2);
	}
}

