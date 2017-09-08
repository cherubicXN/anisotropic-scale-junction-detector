#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv2/opencv.hpp>
#include <cassert>
#include "acj.h"
#include "constant.hpp"
#include "junctionStructure.hpp"
#include "helper.h"

double ACJDetection::computeNFACLT(int numBranch, int numPixel, float branchStrength,
	int rmin, int rmax, float widthAngle, int imageSize)
{
	float EPSILON = 0.0f;
	//assert(numPixel < logCDF.size(), "logCDF is not enough");
	for (int r = rmin; r <= rmax; ++r)
	{
		float epsilon = 1.0f;
		for (int i = 0; i < numBranch; ++i)
		{
			if ((1 - 2 * i*widthAngle / PI) <= 0)
				break;
			epsilon *= 1 * floor(2 * PI*r)*(1 - 2 * i*widthAngle / PI) / (i + 1);
		}
		EPSILON += epsilon;
	}
	EPSILON *= imageSize;

	//float logNFA = log10(EPSILON) + logCDF[numPixel - 1][(int)floor(branchStrength / par.CDF_INTERVAL)];
	double probability;
	double MU = 3.304946062926473e-01;
	double SIGMA = 5.041364327678440e-01;
	MU *= (double)numPixel;
	SIGMA *= sqrt((double)numPixel);

	probability = 1.0f - 0.5f*(1 + erf((branchStrength - MU) / SIGMA / sqrt(2.0f) + EPS));
	//	if (probability < EPS)
	//		probability = EPS;
	//float logNFA = log10(EPSILON) + log10(probability);
	float NFA = EPSILON*probability;

	return NFA;
}


float ACJDetection::computeLogNFA(int numBranch, int numPixel, float branchStrength,
	int rmin, int rmax, float widthAngle, int imageSize,
	int lengthUnitCDF)
{
	float EPSILON = 0.0f;
#ifdef _MSC_VER
	assert(numPixel < logCDF.size(), "logCDF is not enough");
#else
	assert(numPixel < logCDF.size());
#endif
	for (int r = rmin; r <= rmax; ++r)
	{
		float epsilon = 1.0f;
		for (int i = 0; i < numBranch; ++i)
		{
			if ((1 - 2 * i*widthAngle / PI) <= 0)
				break;
			epsilon *= 1 * floor(2 * PI*r)*(1 - 2 * i*widthAngle / PI) / (i + 1);
		}
		EPSILON += epsilon;
	}
	EPSILON *= imageSize;

	//float logNFA = log10(EPSILON) +
	//	logCDF[numPixel - 1][(int)floor(branchStrength / par.CDF_INTERVAL)];
	int index = (int)floor(branchStrength / par.CDF_INTERVAL);
	if (index >= logCDF[numPixel - 1].size())
		index = logCDF[numPixel - 1].size() - 1;
	float logNFA = fast_log10(EPSILON) + (float)numBranch*logCDF[numPixel - 1][index];

	return logNFA;
}

void ACJDetection::generatelogCDFAll()
{
	//if (!logCDF.empty())
	//	logCDF.clear();
	size_t nbPixel = par.maxNumPixel;

	std::vector<double> pdf(1, 1);
	std::vector<double> tempPdf;
	for (int i = 0; i < nbPixel; ++i)
	{
		convolution1D(pdf, strengthPDF, tempPdf, 0);
		pdf = tempPdf;
		std::vector<double> cdfTemp;
		//			cumSum(pdf, cdfTemp);
		cvtpdf2log10CDF(pdf, cdfTemp);
		logCDF.push_back(cdfTemp);
	}
}


template<typename T>
void ACJDetection::cvtpdf2log10CDF(const std::vector<T> &pdf,
	std::vector<T> &log10CDF)
{
	size_t nb = pdf.size();
	log10CDF.resize(nb);
	std::vector<T> cdf_con(nb, 0);
	cdf_con[nb - 1] = pdf[nb - 1];
	//		for (int i = 1; i < nb; ++i)		
	//			cdf_con[i] = cdf_con[i - 1] + pdf[i];
	for (int i = nb - 2; i >= 0; --i)
		cdf_con[i] = cdf_con[i + 1] + pdf[i];
	for (int i = 0; i < nb; ++i)
		log10CDF[i] = log10(cdf_con[i]);
}

float ACJDetection::computeLogNFAScaleFixed(int scale, int numBranch, int numPixel, float branchStrength,
	float widthAngle, int imageSize, int lengthUnitCDF)
{
#ifdef _MSC_VER
	assert(numPixel < logCDF.size(), "logCDF is not enough");
#else
	assert(numPixel < logCDF.size());
#endif
	float EPSILON = 0.0f;
	//for (int r = rmin; r <= rmax; ++r)
	float epsilon = 1.0f;
	int r = scale;
	{

		for (int i = 0; i < numBranch; ++i)
		{
			if ((1 - 2 * i*widthAngle / PI) <= 0)
				break;
			epsilon *= 1 * floor(2 * PI*r)*(1 - 2 * i*widthAngle / PI) / (i + 1);
		}
		EPSILON += epsilon;
	}
	//EPSILON *= imageSize;

	float logNFA =
		(float)numBranch*logCDF[numPixel - 1][(int)floor(branchStrength / par.CDF_INTERVAL)];
	logNFA += fast_log10(EPSILON);

	return logNFA;
}