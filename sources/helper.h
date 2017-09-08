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
#ifndef __HELPER__
#define __HELPER__
#include "constant.hpp"
#include <vector>

#define PI_DOUBLE      6.28318530
#define PI_DVI_TWO     1.57079633
#define PI_DVI_FOUR	   0.78539816
#define PI_DVI_EIG     0.78539816
#define FOUR_DVI_PI    1.27323954
#define FOUR_DVI_PI_PI 0.40528473
enum ConvolutionType {
	/* Return the full convolution, including border */
	CONVOLUTION_FULL,

	/* Return only the part that corresponds to the original image */
	CONVOLUTION_SAME,

	/* Return only the submatrix containing elements that were not influenced by the border */
	CONVOLUTION_VALID
};
template <typename T>
int checkTypeOfInputImage(const T &);

template <typename T>
void getMinMax(const T &im,
	float &minValue,
	float &maxValue);
template <typename T>
T atSub(const cv::Mat &im, const cv::Point2f &pts)
{
	CV_DbgAssert(dims <= 2);
	CV_DbgAssert(im.channels() == 1);
	CV_DbgAssert(data);
	CV_DbgAssert(DataType<_Tp>::channels == 1);
	CV_DbgAssert((unsigned)pt.y < (unsigned)im.size.p[0]);
	CV_DbgAssert((unsigned)(pt.x) < (unsigned)(im.size.p[1]));
	//CV_DbgAssert((unsigned)(pt.x * DataType<_Tp>::channels) < (unsigned)(im.size.p[1] * channels()));
	//CV_DbgAssert(CV_ELEM_SIZE1(DataType<_Tp>::depth) == elemSize1());


	int x = (int)pts.x;
	int y = (int)pts.y;
	int x0 = cv::borderInterpolate(x, im.cols, cv::BORDER_REFLECT_101);
	int x1 = cv::borderInterpolate(x + 1, im.cols, cv::BORDER_REFLECT_101);
	int y0 = cv::borderInterpolate(y, im.rows, cv::BORDER_REFLECT_101);
	int y1 = cv::borderInterpolate(y + 1, im.rows, cv::BORDER_REFLECT_101);

	CV_DbgAssert((unsigned)y0 < (unsigned)size.p[0]);
	CV_DbgAssert((unsigned)x0 < (unsigned)(size.p[1]));
	CV_DbgAssert((unsigned)y1 < (unsigned)size.p[0]);
	CV_DbgAssert((unsigned)x1 < (unsigned)(size.p[1]));

	float a = pts.x - (float)x;
	float c = pts.y - (float)y;

	float val00, val01, val10, val11;
	if (im.type() == CV_8UC1)
	{
		val00 = (float)im.at<uchar>(y0, x0);
		val01 = (float)im.at<uchar>(y1, x0);
		val10 = (float)im.at<uchar>(y0, x1);
		val11 = (float)im.at<uchar>(y1, x1);
	}
	else
	{
		val00 = im.at<float>(y0, x0);
		val01 = im.at<float >(y1, x0);
		val10 = im.at<float>(y0, x1);
		val11 = im.at<float>(y1, x1);
	}

	float val = ((val00 * (1.f - a) + val10 * a) * (1.f - c)
		+ (val01* (1.f - a) + val11* a) * c);

	return (T)val;
}

template <typename T>
T atSub(const cv::Mat &im, const cv::Point2i &pts)
{
	CV_DbgAssert(dims <= 2);
	CV_DbgAssert(im.channels() == 1);
	CV_DbgAssert(data);
	CV_DbgAssert(DataType<_Tp>::channels == 1);
	CV_DbgAssert((unsigned)pt.y < (unsigned)im.size.p[0]);
	CV_DbgAssert((unsigned)(pt.x) < (unsigned)(im.size.p[1]));
	//CV_DbgAssert((unsigned)(pt.x * DataType<_Tp>::channels) < (unsigned)(im.size.p[1] * channels()));
	//CV_DbgAssert(CV_ELEM_SIZE1(DataType<_Tp>::depth) == elemSize1());


	int x = (int)pts.x;
	int y = (int)pts.y;

	float val;
	if (im.type() == CV_8UC1)
	{
		val = (float)im.at<uchar>(pts);
	}
	else
	{
		val = (float)im.at<float>(pts);
	}

	return (T)val;
}

/*----------------------------------------------------------------------------*/
/** Absolute value angle difference.
*/
/*
double angle_diff(double a, double b)
{
	a -= b;
	while (a <= -PI) a += 2 * PI;
	while (a >   PI) a -= 2 * PI;
	if (a < 0.0) a = -a;
	return a;
}

double angle_diff_signed(double a, double b)
{
	a -= b;
	while (a <= -PI) a += PI * 2;
	while (a > PI) a -= PI * 2;
	return a;
}
*/
void imCrop(const cv::Mat &ImageIn,
	int x, int y, int radius,
	cv::Mat &imageCroppedVec);

//template <typename T>
//void convolution1D(
//	const std::vector<T> &a,
//	const std::vector<T> &kernel,
//	std::vector<T> &b,
//	int optionSame = 1);
void convolution1D(const std::vector<double> &a, const std::vector<double> &kernel,
				   std::vector<double> &b, int optionSame);

cv::Mat im2float(const cv::Mat& image);

void meanStd2(const cv::Mat& image, float &mean, float &std);

void conv2(const cv::Mat &img, const cv::Mat& kernel, ConvolutionType type, cv::Mat& dest);

cv::Mat im2uchar(const cv::Mat& image);

template<typename T>
void gaussianBlurInplace(T &inplace, float sigma);

template<typename T>
T gaussianBlur(const T input, float sigma);

template <typename T> void swap(T *a, T *b);

void solveLinear3x3(float *A, float *b);
inline float round_pi(float x)
{
	while (x < -PI) x += 2 * PI;
	while (x > PI) x -= 2 * PI;
	return x;
	//	return (x <-PI) ? (x + PI*2) : ((x > PI) ? (x - PI*2) : x);
}

inline float diff_circular(const float a, const float b, const float C)
{
	float temp = ABS(a - b);
	return (MIN(temp, C - temp));
}

inline float pow2(float x)
{
	return x*x;
}

inline float fast_log2(float val)
{
	int* const    exp_ptr = reinterpret_cast<int*>(&val);
	int            x = *exp_ptr;
	const int      log_2 = ((x >> 23) & 255) - 128;
	x &= ~(255 << 23);
	x += 127 << 23;
	*exp_ptr = x;

	val = ((-1.0f / 3) * val + 2) * val - 2.0f / 3;   // (1)

	return (val + log_2);
}

inline float fast_log10(const float val)
{
	return (fast_log2(val) * 0.30103000f);
}

inline float in_angle(const float x, const float y)
{
	float temp = ABS(x - y);
	return (float)((temp<PI) ? temp : (PI_DOUBLE - temp));
}

inline float up_pi(const float x)
{
	return (float)((x > PI) ? (x - 2 * PI) : x);
}

// lower -PI suppresion for sin and cos
inline float low_pi(const float x)
{
	return (float)((x < -PI) ? (x + 2 * PI) : x);
}

inline float fastsin(const float x)
{
	return (float)(FOUR_DVI_PI*x - FOUR_DVI_PI_PI*x*ABS(x));
}
// fast cos in [-PI, PI]
inline float fastcos(float x)
{
	x += (float)(PI_DVI_TWO);
	x -= (float)((x > PI) ? PI_DOUBLE : 0);

	return (float)(FOUR_DVI_PI*x - FOUR_DVI_PI_PI*x*ABS(x));
}
// fast sin in [-PI, PI] with EXTRA_PRECISION
inline float fastsin_ex(const float x)
{
	float y = (float)(FOUR_DVI_PI*x - FOUR_DVI_PI_PI*x*ABS(x));

#ifdef EXTRA_PRECISION
	y = P * (y * ABS(y) - y) + y;   // Q * y + P * y * abs(y)
#endif            
	return y;
}
// fast cos in [-PI, PI] with EXTRA_PRECISION
inline float fastcos_ex(float x)
{
	x += (float)PI_DVI_TWO;
	x -= (float)((x > PI) ? PI_DOUBLE : 0);

	float y = float(FOUR_DVI_PI*x - FOUR_DVI_PI_PI*x*ABS(x));

#ifdef EXTRA_PRECISION
	y = P * (y * ABS(y) - y) + y;   // Q * y + P * y * abs(y)
#endif            
	return y;
}

#endif

