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
#include <cmath>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

#include "aacj.h"
#include "generateSector.hpp"
#include "helper.h"


//using std::min;
//using std::max;
//using std::round;
#ifndef EPS
#define EPS 0.0000000000001
#endif

void AACJDetection::generateSectorForAngle(const float &theta, const int &R,
	const cv::Point2i &location,
	const float &angleWidth, Sector &sector)
{
	sector.numberOfPoint = 0;
	sector.Orientation = theta;
	if (!sector.sectPointInd_A.empty())
		sector.sectPointInd_A.clear();
	if (!sector.sectPointInd_C.empty())
		sector.sectPointInd_C.clear();
	if (!sector.sectPointInd_M.empty())
		sector.sectPointInd_M.clear();

	float AL = theta;
	float AL1 = theta - angleWidth;
	float AL2 = theta + angleWidth;

	float COS_AL1 = cos(AL1);
	float SIN_AL1 = sin(AL1);
	float COS_AL2 = cos(AL2);
	float SIN_AL2 = sin(AL2);


	int xmin = (int)MIN(MIN(0, round(R*COS_AL1)), round(R*COS_AL2));
	int xmax = (int)MAX(MAX(0, round(R*COS_AL2)), round(R*COS_AL1));
	int ymin = (int)MIN(MIN(0, round(R*SIN_AL2)), round(R*SIN_AL1));
	int ymax = (int)MAX(MAX(0, round(R*SIN_AL1)), round(R*SIN_AL2));

	for (int x = xmin; x <= xmax; ++x)
		for (int y = ymin; y <= ymax; ++y)
		{
			if ((y*y + x*x <= R*R)
				&& (y*COS_AL1 - x*SIN_AL1 - EPS >= 0)
				&& (y*COS_AL2 - x*SIN_AL2 + EPS <= 0)
				&& (x + location.x >= 0) && (x + location.x<image.cols)
				&& (y + location.y >= 0) && (y + location.y<image.rows))
				sector.sectPointInd_C.push_back(cv::Point2i(x + location.x, y + location.y));
		}

}

double AACJDetection::computeProbability(const float &strength, int numOfPixel)
{
	double probability = pdfs->logProbability(strength, numOfPixel);

	return probability;
}

double AACJDetection::detectAlongTheta(float R,
	const cv::Point2i &loc, float theta, float &dStrength)
{
	float widthAngle = widthAngleUnit;
	DiffSector dsec(R, theta, widthAngle, loc,
		gradPhase, gradMagnitude, radiusThreshold, level);
	//std::cout << dsec.pixelArc.size() << std::endl;
	//DiffSectorInterp dsec(R, theta, widthAngle, loc,
	//	gradPhase, gradMagnitude,radiusThreshold,level);
	const std::vector<cv::Point2i> &refArc = dsec.pixelArc;

	if (refArc.empty())
		return 1.0;

	double probArc;
	double strength = 0.0;
	int numP = 0;
	for (std::vector<cv::Point2i>::const_iterator iter = refArc.begin();
		iter != refArc.end(); ++iter)
	{
		float Strength, orient;
		double SIN = -1;

		orient = atan2((float)(iter->y - loc.y), (float)(iter->x - loc.x));

		Strength = 0.0;
		if (!localTheta(iter->x, iter->y, theta, orient, Strength))
			continue;
		SIN = abs(cos(orient - theta)) - abs(sin(orient - theta));
		if (SIN <= 0.0)
			continue;
		Strength *= SIN;
		//SIN = fastsin(theta - orient);
		//Strength = Strength*pow2(pow2(1.f-pow2(SIN)));
		strength += Strength;
	}
	probArc = computeProbability(strength, refArc.size());
	dStrength = strength;

	return probArc;
}

void AACJDetection::detectAlongTheta(const cv::Point2f &loc,
	float theta,
	float Rlower,
	std::vector<float> &vecR,
	std::vector<double> &vecNFA,
	std::vector<double> &vecDStrength)
{
	if (!vecR.empty())
		vecR.clear();
	if (!vecNFA.empty())
		vecNFA.clear();
	if (!vecDStrength.empty())
		vecDStrength.clear();
	std::vector<double> vecProbability;
	std::vector<double> logProb;
	cv::Point2f dir(cos(theta), sin(theta));

	int imgSz = image.cols*image.rows;
	for (float R = Rlower; R <= sqrt(image.cols*image.rows); R += 1.0)
	{
		cv::Point2f tempLoc = loc + R*dir;
		int tempLocX = tempLoc.x;
		int tempLocY = tempLoc.y;
		if (tempLocX<0 || tempLocX >= image.cols || tempLocY<0 || tempLocY >= image.rows)
			break;

		double probability;
		float dStrength;
		probability = detectAlongTheta(R, loc, theta, dStrength);

		vecDStrength.push_back(dStrength);
		vecProbability.push_back(probability);
		vecR.push_back(R);
	}

	vecNFA.push_back(0.0);
	double Ns = sqrt(image.cols*image.rows);
	for (int j = 1; j < vecProbability.size(); ++j)
	{
		double NFA;
		NFA = vecProbability[j] + log10((double)Ns);
		vecNFA.push_back(NFA);
	}

}
float AACJDetection::calcScale(const std::vector<float> &R,
	const std::vector<double> &NFA)
{
	int index = 0;
	for (int i = 0; i < NFA.size(); ++i)
	{
		if (NFA[i]>1.0)
			break;
		index = i;
	}
	return R[index];
}

float AACJDetection::calcScale(const cv::Point2i &location, float theta, float Rlower)
{
	cv::Point2f dir(cos(theta), sin(theta));
	cv::Point2f loc((float)location.x, (float)location.y);

	double Ns = 0.0;
	double NFA;
	float ret = 0;
	for (float R = Rlower; R <= sqrt(image.cols*image.rows); R += 1.0)
	{
		cv::Point2f tempLoc = loc + R*dir;
		int tempLocX = tempLoc.x;
		int tempLocY = tempLoc.y;
		if (tempLocX<0 || tempLocX >= image.cols || tempLocY<0 || tempLocY >= image.rows)
			break;
		Ns += 1.0;
	}

	for (float R = Rlower; R <= sqrt(image.cols*image.rows); R += 1.0)
	{
		cv::Point2f tempLoc = loc + R*dir;
		int tempLocX = tempLoc.x;
		int tempLocY = tempLoc.y;
		if (tempLocX<0 || tempLocX >= image.cols || tempLocY<0 || tempLocY >= image.rows)
			break;
		double probability;
		float dStrength;
		probability = detectAlongTheta(R, loc, theta, dStrength);

		NFA = probability + log10((double)Ns);

		if (NFA <= 0.0)
		{
			ret = R;
			continue;
		}
		break;
	}
	return MAX(ret, Rlower);
}

float AACJDetection::calcScale(const cv::Point2i &location, float theta, float Rlower,
	std::vector<float> &vecR,
	std::vector<double> &vecStrengthArc,
	std::vector<double> &vecNFA,
	std::vector<double> &vecDStrength)
{
	detectAlongTheta(location, theta, Rlower, vecR, vecNFA, vecDStrength);
	float scale = calcScale(vecR, vecNFA);

	return scale;
}


bool AACJDetection::localTheta(int x, int y, float theta, float &orient, float &strength)
{
	int index = sketchProposalFixed(x, y);
	if (index == -1)
		return false;
	Junction &local = localJunctions[index];
	float minDis = 1e6;
	theta = round_pi(theta);

	float ret = 0.f;
	bool flag = false;
	for (int i = 0; i < local.branch.size(); ++i)
	{
		if (local.branch[i].logNFA && local.branch[i].logNFAall> 0)
			break;
		flag = true; 
		float temp = round_pi(local.branch[i].branch);
		float dis = in_angle(temp, theta);
		if (dis < minDis)
		{
			minDis = dis;
			ret = temp;
			//strength = local.branch[i].branchStrength;
			float n = (float)local.branch[i].numberOfPoint;
			strength = 1 / SIGMA*(local.branch[i].branchStrength / sqrt(n) - sqrt(n)*MU);
		}
	}
	orient = ret;
	return flag;
}

float AACJDetection::calcStrength(const SectorFast<int> &sec, size_t cols, size_t rows)
{

	std::vector<cv::Point_<int>>::const_iterator iter;
	float ret = 0.f;
	float theta = round_pi(sec.theta);
	for (iter = sec.vecPoints.begin(); iter != sec.vecPoints.end(); ++iter)
	{
		if (iter->x < 0 || iter->x >= cols || iter->y < 0 || iter->y >= rows)
			continue;
		float orient = atan2((float)(iter->y - sec.location.y), (float)(iter->x - sec.location.x));
		float strength, SIN;
		if (!localTheta(iter->x, iter->y, theta, orient, strength))
			continue;
		/*
		SIN = abs(cos(orient - theta)) - abs(sin(orient - theta));
		if (SIN <= 0.0)
			continue;
		*/
		SIN = fastsin(round_pi(orient - theta));
		strength *= pow2(pow2(1 - pow2(SIN)));
		ret += strength;
	}
	return ret;
}

void AACJDetection::anisoDetection(Junction &junctRet)
{
	int x = junctRet.location.x;
	int y = junctRet.location.y;
	std::string color = "rgbcky";
	//FILE *file = fopen("..\\..\\debug.m","w");
	//fprintf(file, "close all;");

	int pownum = 1;

	//std::vector<cv::Point2f> locations;
	//std::vector<float> angles;
	for (int i = 0; i < junctRet.junctionClass; ++i)
	{
		std::vector<double> vecStrengthArc;
		std::vector<double> vecNFA;
		std::vector<float> vecR;
		std::vector<double> vecDStrength;
		Sector sector;
		float theta = junctRet.branch[i].branch;
		float Rlower = junctRet.scale;
		float scale = junctRet.scale;

		float maxStrength = -1;
		float maxScale = -1;
		float maxTheta = -1;
		int iter = 0;

		//for (float theta0 = theta - 0.15; theta0 <= theta + 0.15; theta0 += 0.003)
		for (float theta0 = theta - 0.1; theta0 <= theta + 0.1; theta0 += 0.001)
		{
			++iter;

			float tempscale = calcScale(junctRet.location, theta0, Rlower);
			if (tempscale < Rlower)
				continue;

			SectorFast<int> sec(junctRet.location, tempscale, 2.5f, theta0);
			float tempstrength = calcStrength(sec, image.cols, image.rows);
			float s = sec.calcStrength(gradMagnitude, gradPhase);
			float NFA = computeNFACLT(1, sec.vecPoints.size(), s, 1, tempscale, 2.5f, image.cols*image.rows);
			if (NFA > 1.f)
				continue;
			if (maxStrength < tempstrength)
			{
				maxStrength = tempstrength;
				maxScale = tempscale;
				maxTheta = theta0;
			}
		}

		if (maxStrength < -EPS)
		{
			junctRet.branch[i].branchScale = -1;
			continue;
		}

		junctRet.branch[i].branchScale = maxScale;
		junctRet.branch[i].branch = maxTheta;
	}

}