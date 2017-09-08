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
#include <opencv2/opencv.hpp>
#include <omp.h>
#include "acj.h"
#include "constant.hpp"
#include "junctionStructure.hpp"
#include "helper.h"
#include "nms.h"
#include "lsdcall.h"


void ACJDetection::displayJunction(const cv::Mat &im, cv::Mat &imOut, const Junction &junct)
{
#ifdef _MSC_VER
	assert(im.depth() == CV_8UC1 || im.depth() == CV_8UC3,
		"Error in displayJunction(): \n the image type must be CV_8UC1 or CV_8UC3");
#else
	assert(im.depth() == CV_8UC1 || im.depth() == CV_8UC3);
#endif

	if (im.channels() == 1)
		cv::cvtColor(im, imOut, CV_GRAY2BGR);
	else
		imOut = im.clone();

	int nfa = (int)pow(10.0f, junct.logNFA)*255.0f;
	cv::Scalar color(abs(255 - nfa), abs(127 - nfa), abs(0 - nfa));
	//cv::Scalar color(0,255,0);
	//cv::circle(imOut, junct.location, junct.scale, color);
	std::vector<cv::Scalar> rgbcky;
	rgbcky.push_back(cv::Scalar(0, 0, 255.0));
	rgbcky.push_back(cv::Scalar(0, 255.0, 0));
	rgbcky.push_back(cv::Scalar(255.0, 0, 0));
	rgbcky.push_back(cv::Scalar(255.0, 255.0, 0.0));
	rgbcky.push_back(cv::Scalar(255.0, 0, 255.0));
	for (int i = 0; i < junct.junctionClass; ++i)
	{
		float theta = junct.branch[i].branch;
		float scale = junct.branch[i].branchScale;
		//if (scale < 0)
		//	continue;
		if (scale <= junct.scale)
			scale = junct.scale;
		cv::Point2f branchEND = (float)scale*cv::Point2f(cos(theta), sin(theta));
		branchEND.x += (float)junct.location.x;
		branchEND.y += (float)junct.location.y;
		cv::line(imOut, junct.location, branchEND, rgbcky[i], 2);
	}
}

void normalize(cv::Mat mat)
{
	int cols = mat.cols;
	int rows = mat.rows;
	float maxval = std::numeric_limits<float>::min();
	float minval = std::numeric_limits<float>::max();
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			float val = mat.at<float>(i, j);
			if (val > maxval)
				maxval = val;
			if (val < minval)
				minval = val;
		}
	}
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			float val = mat.at<float>(i, j);
			mat.at<float>(i, j) = (val - minval) / (maxval - minval);
		}
	}
}
void ACJDetection::initialization()
{
//	cv::Mat tmp;
	image = image_clean.clone();

	if (image.channels() == 3)
		cv::cvtColor(image, image, CV_BGR2GRAY);

	if (image.depth() == 0)
		image = im2float(image);
	lsdcall(image, lsdResults);

	cv::Mat rand(image.rows, image.cols, CV_32FC1,cv::Scalar(0.f));
	cv::randn(rand, 0.f, par.noiseACJ);
	//cv::randn(rand,0.f,0.0089);
	//cv::RNG rng(888);
	//rng.fill(rand, 1, 0.f, par.noiseACJ);
	image = image / 255.f;
	image += rand;
	image = image*255.f;

	for (int i = 0; i < image.cols; ++i)
		for (int j = 0; j < image.rows; ++j) {
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

	//cv::Mat smoothKernel(7, 7, CV_32FC1, cv::Scalar(1.0f / 49.0f));
	cv::Mat smoothKernel(5, 5, CV_32FC1, cv::Scalar(1.0f / 25.0f));
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
	//gradMagnitude = fx.mul(fx) + fy.mul(fy);
	//cv::sqrt(gradMagnitude, gradMagnitude);

	OrientationPatch = cv::Mat(par.nsMax * 2 + 1, par.nsMax * 2 + 1, CV_32FC1);
	for (int x = 0; x < 2 * par.nsMax + 1; ++x)
		for (int y = 0; y < 2 * par.nsMax + 1; ++y)
			OrientationPatch.at<float>(y, x) = atan2((float)y - par.nsMax, (float)x - par.nsMax);

	std::vector<Sector> sectorList;
	for (int ns = par.nsMin;
		ns <= par.nsMax; ++ns)
	{
		if (!sectorList.empty())
			sectorList.clear();

		int nbSector = (int)floor(2 * PI*ns);
		float widthSector = par.deltaR / ns;

		generateSector(ns, nbSector, widthSector, sectorList);
		sectorVecVec.push_back(sectorList);
	}

	{
		strengthPDF.clear();
		size_t nbOfzInterval = (int)par.CDF_MAXBASE / par.CDF_INTERVAL + 1;
		double z = 0.0f;
		double normcoef = 0;
		for (int i = 0; i < nbOfzInterval; ++i)
		{
			double val = 1 / sqrt(PI)*exp(-z * z / 4)*erfc(z / 2);
			strengthPDF.push_back(val);
			z += par.CDF_INTERVAL;
			if (i > 0)
				normcoef += val;
		}
		for (int i = 1; i < nbOfzInterval; ++i)
		{
			strengthPDF[i] /= normcoef;
			strengthPDF[i] /= 2.0f;
		}
		strengthPDF[0] = 0.5;
	}
	if (par.nfaApproximateOn)
		par.maxNumPixel = 15;
	if (logCDF.size()<par.maxNumPixel)
		generatelogCDFAll();
	nfaImage = cv::Mat(image.rows, image.cols, CV_32SC1, cv::Scalar(-2));
}

template<typename TMat>
void ACJDetection::computeStrengthMax(const TMat &GradMagPath,
	const TMat &PhasePatch,
	TMat &StrengthMaxVec)
{
	float SIN;
	size_t cols = GradMagPath.cols;
	size_t rows = GradMagPath.rows;

	typedef float FLOAT;
	StrengthMaxVec = cv::Mat(rows, cols, CV_32FC1);
	for (int i = 0; i < cols; ++i) {
		for (int j = 0; j < rows; ++j) {
			SIN = fastsin(round_pi(PhasePatch. template at< FLOAT >(j, i) - OrientationPatch.template at< FLOAT >(j, i)));
			StrengthMaxVec. template at< FLOAT >(j, i) = GradMagPath. template at< FLOAT >(j, i) * pow2(pow2(1 - pow2(SIN)));
		}
	}
}

template<typename TMat>
void ACJDetection::computeStrengthMaxLocal(const TMat &GradMagPath,
	const TMat &PhasePatch,
	TMat &StrengthMaxVec)
{
	float SIN;
	size_t cols = GradMagPath.cols;
	size_t rows = GradMagPath.rows;

	StrengthMaxVec = cv::Mat(rows, cols, CV_32FC1);

	for (int i = 0; i < cols; ++i)
		for (int j = 0; j < rows; ++j)
		{
			SIN = std::sin(PhasePatch. template at<float>(j, i) - OrientationPatchLocal.template at<float>(j, i));
			SIN = fabs(cos(SIN)) - fabs(sin(SIN));
			SIN = MAX(SIN, 0.0);
			SIN *= GradMagPath. template at<float>(j, i);
			StrengthMaxVec. template at<float>(j, i) = SIN;
			//SIN = fastsin(round_pi(PhasePatch.at<float>(j, i) - OrientationPatchLocal.at<float>(j, i)));
			//StrengthMaxVec.at<float>(j, i) = GradMagPath.at<float>(j, i)*pow2(pow2(1 - pow2(SIN)));
		}
}

template <typename TMat>
void ACJDetection::computeStrength(const TMat &strengthMax,
	const std::vector<Sector> &sectorList,
	std::vector<float> &strength)
{
#ifdef _MSC_VER
	assert(strengthMax.depth() != 0, "strengthMax must be CV_32FC1 TYPE!");
#else
assert(strengthMax.depth() != 0);
#endif
	if (!strength.empty())
		strength.clear();
	float strengthTemp;
	float minval = std::numeric_limits<float>::max();
	float maxval = std::numeric_limits<float>::min();

	for (std::vector<Sector>::const_iterator
		sectorIter = sectorList.begin();
		sectorIter != sectorList.end();		++sectorIter)
	{
		strengthTemp = 0.0f;
		for (std::list<cv::Point2i>::const_iterator pointIter = sectorIter->sectPointInd_C.begin();
			pointIter != sectorIter->sectPointInd_C.end();	++pointIter)
			strengthTemp += strengthMax.template at<float>(*pointIter);
		strength.push_back(strengthTemp);
	}
}

bool ACJDetection::detectJunction(int x, int y, std::list<Junction> &jlist)
{
	cv::Mat gradPatch, phasePatch;
	imCrop(gradMagnitude, x, y, par.nsMax, gradPatch);
	imCrop(gradPhase, x, y, par.nsMax, phasePatch);

	cv::Mat strengthMax;
	computeStrengthMax(gradPatch, phasePatch, strengthMax);

	std::vector<float> strengthVec, strengthVecNMS;
	std::vector<std::vector<Sector>>::const_iterator sectorListIter;
	std::vector<Branch> proposalBranches;

	std::vector<std::vector<int>>	rdListValue, rdListState;
	rdListState.assign(par.maxBranch, std::vector<int>(par.nsMax, 0));
	rdListValue.assign(par.maxBranch, std::vector<int>(par.nsMax, 0));

	int nsIter;
	for (sectorListIter = sectorVecVec.begin(), nsIter = par.nsMin;
		sectorListIter != sectorVecVec.end(); ++nsIter, ++sectorListIter)
	{
		computeStrength(strengthMax, *sectorListIter, strengthVec);
		if (x<nsIter || y<nsIter || x + nsIter>image.cols || y + nsIter>image.rows)
			break;
		int nbSector = (int)floor(2 * PI*nsIter);
		//float widthAngle = (float)2.5f / nsIter;
		float widthAngle = (float)par.deltaR / nsIter;
		float widthSector = (float)par.deltaR / nsIter;
		int nNeighbor = (int)floor(max(2 * widthAngle, 2 * widthSector) / (2.0f*PI / nbSector));

		// NMS strength
		nms1d_cir(strengthVec, strengthVec.size(), nNeighbor, strengthVecNMS);

		// Get proposal branches
		int maxNumP = 0;
		if (!proposalBranches.empty())
			proposalBranches.clear();
		int i;
		std::vector<Sector>::const_iterator sectorIter;
		Branch tempBranch;
		for (i = 0, sectorIter = sectorListIter->begin();
			sectorIter != sectorListIter->end();
			++i, ++sectorIter)
		{
			if (strengthVecNMS[i] <= sectorIter->numberOfPoint*0.5)
				continue;
			tempBranch.index = i;
			tempBranch.branch = sectorIter->Orientation;
			tempBranch.branchStrength = strengthVecNMS[i];
			tempBranch.numberOfPoint = sectorIter->numberOfPoint;
			tempBranch.branchScale = nsIter;
			maxNumP = maxNumP > sectorIter->numberOfPoint ?
			maxNumP : sectorIter->numberOfPoint;

			proposalBranches.push_back(tempBranch);
		}
		sort(proposalBranches.begin(), proposalBranches.end(),
			compareBranchByStrength);
		reverse(proposalBranches.begin(), proposalBranches.end());

		size_t branchSize = proposalBranches.size();

		int tm;
		std::vector<Branch>::const_iterator branchIter;
		for (tm = 0, branchIter = proposalBranches.begin();
			tm < branchSize; ++branchIter, ++tm)
		{
			if (tm + 1 > par.maxBranch)
				break;

			if (tm < 1)
				continue;

			float logNFA;
			if (!par.nfaApproximateOn)
			{
				logNFA = computeLogNFA(tm + 1, maxNumP, branchIter->branchStrength,
					par.nsMin, nsIter, widthAngle, image.cols*image.rows, logCDF[0].size());

			}
			else
			{
				logNFA = computeNFACLT(tm + 1, maxNumP, branchIter->branchStrength,
					par.nsMin, nsIter, widthAngle, image.cols*image.rows);
			}


			if ((logNFA >= par.epsilon && par.nfaApproximateOn) || (logNFA >= log10(par.epsilon) && !par.nfaApproximateOn))
			//if ((logNFA >= par.epsilon && par.nfaApproximateOn) || (logNFA >= log10(10.f) && !par.nfaApproximateOn))
				continue;
			if (rdListValue[tm][nsIter - 2] > par.rdMax)
				goto FINISH;
			Junction junction;
			junction.junctionClass = tm + 1;
			junction.location.x = x;
			junction.location.y = y;
			junction.scale = nsIter;
			junction.branch = proposalBranches;

			if (junctionRefinement(widthAngle, strengthMax, phasePatch, sectorVecVec[nsIter - par.nsMin], junction))
			{
				rdListState[tm][nsIter - 1] = 1;
				if (nsIter >= 3 && rdListState[tm][nsIter - 2] + rdListState[tm][nsIter - 3] >= 1)
					rdListValue[tm][nsIter - 1] = max(rdListValue[tm][nsIter - 2], rdListValue[tm][nsIter - 3]);
				else
					rdListValue[tm][nsIter - 1] = nsIter;
				junction.r_d = rdListValue[tm][nsIter - 1];
				junction.logNFA = logNFA;
				jlist.push_back(junction);
			}

		}
	}
FINISH:{; }

	if (jlist.size() == 0)
		return false;
	return true;
}

void ACJDetection::detectJunction(std::vector<Junction> &junctVec)
{
	std::cout<<"Detecting isotropic scale junctions"<<std::endl;
	if (!junctVec.empty())
		junctVec.clear();
	int rows = image.rows;
	int cols = image.cols;
	cv::Mat lsdCandidate(rows, cols, CV_8UC1, cv::Scalar(0));

	int cntlsd = 0;
	if (lsdResults.size() != 0)
	{
#pragma omp for
		for (int i = 0; i < lsdResults.size(); ++i)
		{
			int x0 = (int)lsdResults[i][0];
			int y0 = (int)lsdResults[i][1];
			int x1 = (int)lsdResults[i][2];
			int y1 = (int)lsdResults[i][3];
			if (x0 >= 0 && x0<cols&&y0 >= 0 && y0<rows)
				lsdCandidate.at<uchar>(y0, x0) = 255;
			if (x1 >= 0 && x1<cols&&y1 >= 0 && y1<rows)
				lsdCandidate.at<uchar>(y1, x1) = 255;
		}

		cv::Mat strel(7, 7, CV_8UC1, cv::Scalar(0));
		
		
		strel.at<uchar>(0, 2) = strel.at<uchar>(0, 3) = strel.at<uchar>(0, 4) = 1;
		strel.at<uchar>(1, 1) = strel.at<uchar>(1, 2) = strel.at<uchar>(1, 3) = strel.at<uchar>(1, 4) = strel.at<uchar>(1, 5) = 1;
		for (int i = 2; i < 5; ++i)
			for (int j = 0; j < 7; ++j)
				strel.at<uchar>(i, j) = 1;		
		strel.at<uchar>(5, 1) = strel.at<uchar>(5, 2) = strel.at<uchar>(5, 3) = strel.at<uchar>(5, 4) = strel.at<uchar>(5, 5) = 1;
		strel.at<uchar>(6, 2) = strel.at<uchar>(6, 3) = strel.at<uchar>(6, 4) = 1;
		
		//cv::Mat strel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7),cv::Point(3,3));

		//strel = cv::getStructuringElement(1, cv::Size(7,7));
		cv::dilate(lsdCandidate, lsdCandidate, strel);
		
		/*
		for (int j = 0; j < lsdCandidate.rows; ++j)
			for (int i = 0; i < lsdCandidate.cols; ++i)
				cntlsd += (lsdCandidate.at<uchar>(j,i) != 0);
		cv::imshow("lsd", lsdCandidate);
		cv::waitKey(0);		
		std::cout << "LSDCANDIDATE = " << cntlsd << std::endl;
		*/
		
	}
	std::list<Junction> jlist;
	clock_t start, finished;
	TicToc timer;

	timer.tic();
	if (lsdResults.size() != 0)
	{
#pragma omp for
		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j)
			{
				if (lsdCandidate.at<uchar>(i, j) == 0)
					continue;
//				std::cout<<i<<","<<j<<std::endl;
				detectJunction(j, i, jlist);
			}
	}
	else
	{
		for (int i = 0; i < rows; ++i)
			for (int j = 0; j < cols; ++j)
			{
				detectJunction(j, i, jlist);
			}
	}
	timer.toc();
	std::cout << "Ellapse " << timer << "seconds." << std::endl;
	std::cout << "Detected " << jlist.size() << " Junctions!" << std::endl;
	junctionStability(jlist);
	std::cout << "Find " << jlist.size() << " stable Junctions!" << std::endl;
	junctionLocation(jlist, 0.5f);
	std::cout << "Refined " << jlist.size() << " Junctions by locations!" << std::endl;
	junctionClassify(jlist, 0.5f);
	std::cout << "Refined " << jlist.size() << " Junctions by classification!" << std::endl;
	std::list<Junction>::const_iterator iter;
	for (iter = jlist.begin(); iter != jlist.end(); ++iter)
		junctVec.push_back(*iter);
	sort(junctVec.begin(), junctVec.end(), compareJunctionByLocation);
}

template<typename TMat>
bool ACJDetection::junctionRefinement(float widthAngle,
	const TMat &strengthMax,
	const TMat &phasePatch,
	const std::vector<Sector> &sectorVec,
	Junction &junction)
{
	size_t scale = junction.scale;
	size_t junctionClass = junction.junctionClass;
	for (int i = 0; i < junctionClass; ++i)
	{
		float Sx = 0.0f, Sy = 0.0f;
		size_t t = junction.branch[i].index;
		float theta = junction.branch[i].branch;
		theta = round_pi(theta);

		for (std::list<cv::Point2i>::const_iterator
			pointIter = sectorVec[t].sectPointInd_C.begin();
			pointIter != sectorVec[t].sectPointInd_C.end();
		++pointIter)
		{
			float tempPhase = phasePatch. template at<float>(*pointIter);
			tempPhase = (diff_circular(tempPhase, theta, 2 * PI) > PI / 2) ? (tempPhase + PI + EPS) : tempPhase;

			float weight = strengthMax. template at<float>(*pointIter);

			Sx += cos(tempPhase)*weight;
			Sy += sin(tempPhase)*weight;
		}
		junction.branch[i].branch = atan2(Sy, Sx);
	}
	if (junctionClass == 1)
		return true;
	if (junctionClass == 2 &&
		abs(
		diff_circular(junction.branch[0].branch,
		junction.branch[1].branch, PI)
		) <= widthAngle*2.0f)
		return false;
	for (int i = 0; i < junctionClass; ++i)
		for (int j = i + 1; j < junctionClass; ++j)
			if (abs(
				diff_circular(junction.branch[i].branch,
				junction.branch[j].branch, 2 * PI)
				) <= widthAngle*2.0f)
				return false;
	return true;
}

void ACJDetection::junctionStability(std::list<Junction> &junctionList)
{
	std::list<Junction>::iterator
		jIterator;
	for (jIterator = junctionList.begin();
		jIterator != junctionList.end(); ++jIterator)
	{
		size_t r = jIterator->scale;
		size_t rd = jIterator->r_d;
		size_t m = jIterator->junctionClass;
		if (abs((int)(r - rd)) <= 1 || rd > par.rdMax)
		{
			jIterator = junctionList.erase(jIterator);
			jIterator--;
			continue;
		}
	}
}

void ACJDetection::junctionLocation(std::list<Junction> &jlist, float sigma = 0.5f)
{
	std::vector<Junction> jlistvec;
	for (std::list<Junction>::const_iterator iter = jlist.begin();
		iter != jlist.end(); ++iter)
	{
		jlistvec.push_back(*iter);
	}
	sort(jlistvec.begin(), jlistvec.end());
	jlist.clear();
	for (std::vector<Junction>::const_iterator iter = jlistvec.begin();
		iter != jlistvec.end(); ++iter)
	{
		jlist.push_back(*iter);
	}
	jlistvec.clear();

	std::list<Junction>::iterator jlistIter1, jlistIter2;

	for (jlistIter1 = jlist.begin(); jlistIter1 != jlist.end(); ++jlistIter1)
	{
		for (jlistIter2 = jlistIter1, jlistIter2++; jlistIter2 != jlist.end(); ++jlistIter2)
		{
			float rd = (float)max(jlistIter1->r_d, jlistIter2->r_d);
			float distance = cv::norm(jlistIter1->location - jlistIter2->location);
			if (distance*distance > (sigma*rd*sigma*rd))
				continue;
			if ((jlistIter2->junctionClass == jlistIter1->junctionClass)
				&& (jlistIter2->logNFA <= jlistIter1->logNFA))
			{
				jlistIter1 = jlist.erase(jlistIter1);
				jlistIter1--;
				break;
			}
		}
	}
}

void ACJDetection::junctionClassify(std::list<Junction> &jlist, float sigma)
{
	std::list<Junction>::iterator iter1, iter2;
	for (iter1 = jlist.begin(); iter1 != jlist.end(); ++iter1)
	{
		for (iter2 = jlist.begin(); iter2 != jlist.end(); ++iter2)
		{
			size_t rd = max(iter1->r_d, iter2->r_d);
			float distance = cv::norm(iter1->location - iter2->location);
			if (distance*distance > (sigma*rd*sigma*rd))
				continue;
			if (iter2->junctionClass < iter1->junctionClass)
			{
				iter2 = jlist.erase(iter2);
				iter2--;
				continue;
			}
		}
	}
}


void ACJDetection::junctionRefinementSimple(float widthAngle,
	const cv::Mat &strengthMax,
	const cv::Mat &phasePatch,
	const std::vector<Sector> &sectorVec,
	Junction &junction)
{
	unsigned int btype, t, i, j;
	btype = junction.branch.size();
	float Sx, Sy, tempPha, theta, thetaTemp, weight;
	int PointInd;
	cv::Point2i tempPoint;

	btype = junction.branch.size();

	// refine sketch proposal branches
	for (i = 0; i<btype; i++)
	{
		Sx = 0.0;  Sy = 0.0;
		t = (junction.branch[i]).index;
		theta = (junction.branch[i]).branch;
		theta = up_pi(theta); // branch original theta

		std::list<cv::Point2i>::const_iterator it = sectorVec[t].sectPointInd_C.begin();
		for (PointInd = 0; PointInd < sectorVec[t].sectPointInd_C.size(); PointInd++, it++)
		{
			tempPoint = *it;
			tempPha = (phasePatch.at<float>(tempPoint));
			tempPha = (in_angle(tempPha, theta) > PI / 2) ? (tempPha + PI + 1e-5) : tempPha;

			thetaTemp = round_pi(tempPha);
			weight = strengthMax.at<float>(tempPoint);

			Sx += (fastcos(thetaTemp))*weight;
			Sy += (fastsin(thetaTemp))*weight;
		}
	}

	for (i = 0; i < junction.branch.size(); i++)
	{
		for (j = i + 1; j < junction.branch.size(); j++)
		{
			if (abs(diff_circular(junction.branch[i].branch, junction.branch[j].branch, 2 * PI)) <= 2.*widthAngle)
			{
				junction.branch.erase(junction.branch.begin() + j);
				j--;
			}
		}
	}
	return;
}

int ACJDetection::sketchProposalFixed(int x, int y)
{
	int scale = par.scaleFixed;
	if (nfaImage.at<int>(y, x) >= -1)
		return nfaImage.at<int>(y, x);
	OrientationPatchLocal = cv::Mat(scale * 2 + 1, scale * 2 + 1, CV_32FC1);
	for (int xx = 0; xx < 2 * scale + 1; ++xx)
		for (int yy = 0; yy < 2 * scale + 1; ++yy)
			OrientationPatchLocal.at<float>(yy, xx) = atan2((float)yy - scale, (float)xx - scale);

	std::vector<Sector> sectors;
	int nbSector = (int)floor(2 * PI*scale);
	float widthSector = (float) par.deltaRLocal/ scale;/// important
	if (!sectors.empty())
		sectors.clear();
	generateSector(scale, nbSector, widthSector, sectors, true);

	int cols = image.cols;
	int rows = image.rows;
	int nNeighbor = (int)floor(2 * widthSector / (par.deltaRLocal *PI / nbSector));

	cv::Mat gradPatch, phasePatch;
	cv::Mat strengthMax;

	std::vector<float> strengthVec, strengthVecNMS;

	int cnt = 0;
	float process = 0.0;
	float dprocess = 0.0;

	Junction proposal;
	Branch tempBranch;

	strengthVec.clear();
	strengthVecNMS.clear();
	imCrop(gradMagnitude, x, y, scale, gradPatch);
	imCrop(gradPhase, x, y, scale, phasePatch);
	computeStrengthMaxLocal(gradPatch, phasePatch, strengthMax);
	computeStrength(strengthMax, sectors, strengthVec);//strengthVec different
	nms1d_cir(strengthVec, strengthVec.size(), nNeighbor, strengthVecNMS);

	size_t maxNumP = 0;
	if (!proposal.branch.empty())
		proposal.branch.clear();

	for (int i = 0; i < sectors.size(); ++i)
	{
		if (strengthVecNMS[i] < EPS)
			continue;
		tempBranch.index = i;
		tempBranch.branch = sectors[i].Orientation;
		tempBranch.branchStrength = strengthVecNMS[i];
		tempBranch.numberOfPoint = sectors[i].numberOfPoint;
		maxNumP = std::max(maxNumP, sectors[i].numberOfPoint);
		proposal.branch.push_back(tempBranch);
	}

	if (!proposal.branch.empty())
	{
		sort(proposal.branch.begin(), proposal.branch.end(), compareBranchByStrength);
		reverse(proposal.branch.begin(), proposal.branch.end());
		junctionRefinementSimple(widthSector, strengthMax, phasePatch, sectors, proposal);
		int branchSize = proposal.branch.size();
		bool flag = false;
		for (int i = 0; i < branchSize; ++i)
		{
			proposal.branch[i].logNFA = computeLogNFAScaleFixed(scale, 1, maxNumP,
				proposal.branch[i].branchStrength, widthSector, image.cols*image.rows, par.CDF_INTERVAL);
			proposal.branch[i].logNFAall = computeLogNFAScaleFixed(scale, i + 1, maxNumP,
				proposal.branch[i].branchStrength, widthSector, image.cols*image.rows, par.CDF_INTERVAL);
			if (proposal.branch[i].logNFA <= 0.0 && proposal.branch[i].logNFAall <= 0.0)
				flag = true;
		}
		if (flag)
		{
			proposal.index = x*rows + y;
			proposal.location = cv::Point2i(x, y);
			localJunctions.push_back(proposal);
			nfaImage.at<int>(y, x) = localJunctions.size() - 1;
		}
		else
		{
			nfaImage.at<int>(y, x) = -1;
		}
	}
	return nfaImage.at<int>(y, x);
}

void ACJDetection::sketchProposalFixed(int scale)
{
	OrientationPatchLocal = cv::Mat(scale * 2 + 1, scale * 2 + 1, CV_32FC1);
	for (int x = 0; x < 2 * scale + 1; ++x)
		for (int y = 0; y < 2 * scale + 1; ++y)
			OrientationPatchLocal.at<float>(y, x) = atan2((float)y - scale, (float)x - scale);

	std::vector<Sector> sectors;
	int nbSector = (int)floor(2 * PI*scale);
	float widthSector = (float) par.deltaRLocal / scale;
	if (!sectors.empty())
		sectors.clear();
	generateSector(scale, nbSector, widthSector, sectors, true);

	int cols = image.cols;
	int rows = image.rows;
	int nNeighbor = (int)floor(2 * widthSector / (par.deltaRLocal*PI / nbSector));

	cv::Mat gradPatch, phasePatch;
	cv::Mat strengthMax;

	std::vector<float> strengthVec, strengthVecNMS;

	int cnt = 0;
	float process = 0.0;
	float dprocess = 0.0;
	//nfaImage = cv::Mat(image.rows, image.cols, CV_32SC1,cv::Scalar(-2));	
	//int x = 587;
	//int y = 664;
	for (int y = 0; y < rows; ++y)
	{
		for (int x = 0; x < cols; ++x)
		{
			Junction proposal;
			Branch tempBranch;

			strengthVec.clear();
			strengthVecNMS.clear();
			imCrop(gradMagnitude, x, y, scale, gradPatch);
			imCrop(gradPhase, x, y, scale, phasePatch);
			computeStrengthMaxLocal(gradPatch, phasePatch, strengthMax);
			computeStrength(strengthMax, sectors, strengthVec);//strengthVec different
			nms1d_cir(strengthVec, strengthVec.size(), nNeighbor, strengthVecNMS);

			size_t maxNumP = 0;
			if (!proposal.branch.empty())
				proposal.branch.clear();

			for (int i = 0; i < sectors.size(); ++i)
			{
				//if(strengthVecNMS[i] <= sectors[i].numberOfPoint*0.5)
				//	continue;
				if (strengthVecNMS[i] < EPS)
					continue;
				tempBranch.index = i;
				tempBranch.branch = sectors[i].Orientation;
				tempBranch.branchStrength = strengthVecNMS[i];
				tempBranch.numberOfPoint = sectors[i].numberOfPoint;
				maxNumP = std::max(maxNumP, sectors[i].numberOfPoint);
				proposal.branch.push_back(tempBranch);
			}

			if (!proposal.branch.empty())
			{
				sort(proposal.branch.begin(), proposal.branch.end(), compareBranchByStrength);
				reverse(proposal.branch.begin(), proposal.branch.end());
				junctionRefinementSimple(widthSector, strengthMax, phasePatch, sectors, proposal);
				int branchSize = proposal.branch.size();
				bool flag = false;
				for (int i = 0; i < branchSize; ++i)
				{
					proposal.branch[i].logNFA = computeLogNFAScaleFixed(scale, 1, maxNumP,
						proposal.branch[i].branchStrength, widthSector, image.cols*image.rows, par.CDF_INTERVAL);
					proposal.branch[i].logNFAall = computeLogNFAScaleFixed(scale, i + 1, maxNumP,
						proposal.branch[i].branchStrength, widthSector, image.cols*image.rows, par.CDF_INTERVAL);
					if (proposal.branch[i].logNFA <= 0.0 && proposal.branch[i].logNFAall <= 0.0)
						flag = true;
				}

				if (flag)
				{
					proposal.index = x*rows + y;
					proposal.location = cv::Point2i(x, y);
					localJunctions.push_back(proposal);
					nfaImage.at<int>(y, x) = cnt++;
				}
				//std::cout << nfaImage.at<int>(y, x) << std::endl;
			}
			//std::cout << "Orz\n";
			dprocess += 1;
			float dratio = (float)dprocess / (image.cols*image.rows);
			if (dratio >= 0.05)
			{
				process += dratio;
				dprocess = 0.0;
//				printf("%.2f %%\r\n", process * 100);
				printf("%.2f %%\n", process * 100);
			}

		}
	}
	std::cout<<"\n";
//	std::cout << cnt << std::endl;

}

void ACJDetection::readGradientPhaseFromFile(const std::string &filename)
{
	std::ifstream file(filename);
	//file.open(, std::ios::in);
	int cols, rows;
	file >> rows >> cols;
	gradPhase = cv::Mat(rows, cols, CV_32FC1);
	float *data = (float *)gradPhase.data;
	for (int i = 0; i < rows*cols; ++i)
		file >> data[i];
	file.close();
}

void ACJDetection::readGradientMagFromFile(const std::string &filename)
{
	std::ifstream file(filename);
	//file.open(filename, std::ios::in);
	int cols, rows;
	file >> rows >> cols;
	gradMagnitude = cv::Mat(rows, cols, CV_32FC1);
	float *data = (float *)gradMagnitude.data;
	for (int i = 0; i < rows*cols; ++i)
		file >> data[i];
	file.close();
}