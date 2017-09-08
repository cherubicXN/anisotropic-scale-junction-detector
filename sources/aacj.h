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
#ifndef __AACJ__
#define __AACJ__
#include "acj.h"
#include "junctionStructure.hpp"
#include "generateSector.hpp"
#include "helper.h"


struct DiffSector
{
	DiffSector(float R_,
		float theta_,
		float widthAngleUnit_,
		const cv::Point2i location_,
		const cv::Mat &gradPhase, const cv::Mat &gradMagnitude,
		float radiusThreshold_ = 100.0f,
		int level_ = 1)
		:R(R_),
		theta(theta_),
		widthAngleUnit(widthAngleUnit_),
		location(location_),
		radiusThreshold(radiusThreshold_)
	{
		size_t rows = gradPhase.rows;
		size_t cols = gradPhase.cols;

		if (R > radiusThreshold)
			angleWidth = widthAngleUnit / radiusThreshold;
		else
			angleWidth = widthAngleUnit / R;
		float AL = theta;
		float AL1 = theta - angleWidth;
		float AL2 = theta + angleWidth;
		if (!pixelArc.empty())
			pixelArc.clear();

		float COS_AL1 = cos(AL1);
		float SIN_AL1 = sin(AL1);
		float COS_AL2 = cos(AL2);
		float SIN_AL2 = sin(AL2);

		level = level_;
		float tempR = R;
		for (int tempLevel = 0; tempLevel < level; ++tempLevel)
		{
			int xmin = (int)MIN(MIN(0, round(R*COS_AL1)), round(R*COS_AL2));
			int xmax = (int)MAX(MAX(0, round(R*COS_AL2)), round(R*COS_AL1));
			int ymin = (int)MIN(MIN(0, round(R*SIN_AL2)), round(R*SIN_AL1));
			int ymax = (int)MAX(MAX(0, round(R*SIN_AL1)), round(R*SIN_AL2));
			for (int x = xmin; x <= xmax; ++x)
				for (int y = ymin; y <= ymax; ++y)
				{
					if ((y*y + x*x <= R*R) && (y*y + x*x >(R - 1)*(R - 1))
						&& (y*COS_AL1 - x*SIN_AL1 - EPS >= 0)
						&& (y*COS_AL2 - x*SIN_AL2 + EPS <= 0)
						)
					{
						if (x + location.x<0 || x + location.x >= cols || y + location.y<0 || y + location.y >= rows)
							continue;
						pixelArc.push_back(cv::Point2i(x + location.x, y + location.y));
					}
				}
			if (pixelArc.empty())
				break;
			R -= 1.0f;
		}
		R = tempR;
	}

	float R;
	float theta;
	float angleWidth;
	std::vector<cv::Point2i> pixelArc;
	cv::Point2i location;
	float widthAngleUnit;
	float radiusThreshold;
	float dDelta;
	int level;
};


class AACJDetection : public ACJDetection
{
public:
	AACJDetection(const std::string & filename,
		float widthAngleUnit_,
		float radiusThreshold_,
		int level_,
		ACJParameters *par_ = NULL) :
		ACJDetection(filename, par_),
		widthAngleUnit(widthAngleUnit_),
		radiusThreshold(radiusThreshold_),
		level(level_)
	{
		//sketchProposalFixed(967, 643);
		numOfTest = (double)(image.cols*image.rows);
		numOfTest = std::sqrt(numOfTest);
		//if (pdfs == NULL)
		//pdfs = new ProbabilityDF((char *)pdffile.c_str(), 100);
		pdfs = new ProbabilityDF(100);
		bDecomposed = false;

		std::string aacjpath = filename + ".asj";
		FILE *file = fopen(aacjpath.c_str(), "r");
		if (file == NULL)
		{
			image = image_clean.clone();

			if (image.channels() == 3)
				cv::cvtColor(image, image, CV_BGR2GRAY);

			if (image.depth() == 0)
				image = im2float(image);

			cv::Mat rand(image.rows, image.cols, CV_32FC1, cv::Scalar(0.f));
			cv::randn(rand, 0.f, par.noiseASJ);
			image = image / 255.f;
			image += rand;
			image = image*255.f;
			for (int i = 0; i < image.cols; ++i)
				for (int j = 0; j < image.rows; ++j)
				{
					image.at<float>(j, i) = round(image.at<float>(j, i));
					if (image.at<float>(j, i) < 0.0)
						image.at<float>(j, i) = 0.0;
					if (image.at<float>(j, i) > 255.0)
						image.at<float>(j, i) = 255.0;
				}
			cv::Mat fx, fy;
			imGradient(image, fx, fy);
			cv::Mat gradNorm = fx.mul(fx) + fy.mul(fy);
			cv::sqrt(gradNorm, gradNorm);

			gradPhase = cv::Mat(image.rows, image.cols, CV_32FC1);

			for (int i = 0; i < image.rows; ++i)
				for (int j = 0; j < image.cols; ++j)
					gradPhase.at<float>(i, j)
					= atan2(fx.at<float>(i, j), -fy.at<float>(i, j));

			cv::Mat smoothKernel(7, 7, CV_32FC1, cv::Scalar(1.0f / 49.0f));
			//cv::Mat smoothKernel(5, 5, CV_32FC1, cv::Scalar(1.0f / 25.0f));
			cv::Mat M;
			conv2(gradNorm, smoothKernel, CONVOLUTION_SAME, M);

			//fx = fx / (M + std::numeric_limits<float>::epsilon());
			fx = fx / (M + 2.2204e-16);
			fy = fy / (M + 2.2204e-16);
			//fy = fy / (M + std::numeric_limits<float>::epsilon());
			float meanValueX, meanValueY;
			float stdValueX, stdValueY;


			meanStd2(fx, meanValueX, stdValueX);
			meanStd2(fy, meanValueY, stdValueY);
			cv::Mat gx, gy;
			gx = (fx - meanValueX) / stdValueX;
			gy = (fy - meanValueY) / stdValueY;

			gradMagnitude = cv::Mat(image.rows, image.cols, CV_32FC1);
			for (int i = 0; i < image.rows; ++i)
				for (int j = 0; j < image.cols; ++j)
					gradMagnitude.at<float>(i, j) = sqrt(gx.at<float>(i, j)*gx.at<float>(i, j) + gy.at<float>(i, j)*gy.at<float>(i, j));

			sketchProposalFixed(par.scaleFixed);
			anisoDetection();
			exportAACJ((filename + ".asj").c_str());
		}
		else
		{
			std::cout << "ASJ have been detected." << std::endl;
		}
	}
	~AACJDetection()
	{
		delete pdfs;
	}
	void read(FILE *file);

	void output(std::vector<Junction> &results);

	void exportAACJ(const char *filename);

	void anisoDetection(Junction &junctRet);

	void anisoDetection()
	{
		for (int i = 0; i < Junctions.size(); ++i)
			anisoDetection(Junctions[i]);
		std::vector<Junction> results;
		output(results);
		Junctions = results;
		//results.clear();
	}

private:
	void generateSectorForAngle(const float &theta, const int &R,
		const cv::Point2i &location,
		const float &angleWidth, Sector &sector);

	void computeStrengthForSector(const Sector &sector, const cv::Point2i &location,
		float &strength, float &strengthMean);
	void accurateSector(const float &theta0,
		const cv::Point2i &location,
		const int &scale, Sector &sector);
	double computeProbability(const float &strength, int numOfPixel);
	double detectAlongTheta(float R,
		const cv::Point2i &loc, float theta, float &dStrength);
	void detectAlongTheta(const cv::Point2f &loc,
		float theta,
		float Rlower,
		std::vector<float> &vecR,
		std::vector<double> &vecNFA,
		std::vector<double> &vecDStrength);

	float calcScale(const std::vector<float> &R,
		const std::vector<double> &NFA);
	float calcScale(const cv::Point2i &location, float theta, float Rlower);
	float calcScale(const cv::Point2i &location, float theta, float Rlower,
		std::vector<float> &vecR,
		std::vector<double> &vecStrengthArc,
		std::vector<double> &vecNFA,
		std::vector<double> &vecDStrength);

	bool localTheta(int x, int y, float theta, float &orient, float &strength);

	float calcStrength(const SectorFast<int> &sec, size_t cols, size_t rows);

	bool isOriented(float theta1, float theta2);

	double MU = 3.304946062926473e-01;
	double SIGMA = 5.041364327678440e-01;
	ProbabilityDF *pdfs;
	float widthAngleUnit;
	float radiusThreshold;
	double numOfTest;
	int level;
	bool bDecomposed;
};
#endif