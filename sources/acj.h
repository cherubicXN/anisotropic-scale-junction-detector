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
#ifndef __ACJ__
#define __ACJ__
#include <opencv2/opencv.hpp>
#include <fstream>
#include <ctime>
#include "junctionStructure.hpp"
#include "constant.hpp"
#include "tictoc.h"
#include "probtool.h"


struct ACJParameters
{
	ACJParameters(float nsMin_ = 3.0f,
		float nsMax_ = 30.0f,
		float CDF_INTERVAL_ = 0.05,
		float CDF_MAXBASE_ = 6.0f,
		float deltaR_ = 2.5f,
		float epsilon_ = 1.0f,
		size_t maxBranch_ = 5,
		int maxNumPixel_ = 120,
		float rdMax_ = 10.0f,
		float deltaRLocal = 2.5f,
		float scaleFixed_ = 7.0f,
		float noiseACJ_ = 0.0089,
		float noiseASJ_ = 0.00000008) :
		nsMin(nsMin_),
		nsMax(nsMax_),
		deltaR(deltaR_),
		CDF_INTERVAL(CDF_INTERVAL_),
		CDF_MAXBASE(CDF_MAXBASE_),
		epsilon(epsilon_),
		maxBranch(maxBranch_),
		rdMax(rdMax_),
		nfaApproximateOn(true),
		maxNumPixel(maxNumPixel_),
		scaleFixed(scaleFixed_),
		noiseACJ(noiseACJ_),
		noiseASJ(noiseASJ_)
	{
		if (maxNumPixel < 4 * nsMax)
			maxNumPixel = 4 * nsMax;
	}
public:
	float nsMin, nsMax;
	float deltaR;
	float CDF_INTERVAL, CDF_MAXBASE;
	float epsilon;
	float rdMax;
	float scaleFixed;
	float noiseACJ;
	float noiseASJ;
	float deltaRLocal;
	int maxNumPixel;
	size_t maxBranch;
	bool nfaApproximateOn;
};

template<typename T>
class SectorFast
{
public:
	/*
	SectorFast(float scale_,
	float widthAngle_, float theta_ = 0.0) :
	scale(scale_), theta(theta_), width(widthAngle_)
	{
	float AL1 = -width / scale;
	float AL2 = width / scale;

	float COS_AL1 = cos(AL1);
	float SIN_AL1 = sin(AL1);
	float COS_AL2 = cos(AL2);
	float SIN_AL2 = sin(AL2);
	float xmin = (int)MIN(MIN(0, round(scale*COS_AL1)), round(scale*COS_AL2));
	float xmax = (int)MAX(MAX(0, round(scale*COS_AL2)), round(scale*COS_AL1));
	float ymin = (int)MIN(MIN(0, round(scale*SIN_AL2)), round(scale*SIN_AL1));
	float ymax = (int)MAX(MAX(0, round(scale*SIN_AL1)), round(scale*SIN_AL2));

	float cs = cos(theta), ss = sin(theta);

	for (int x = xmin; x <= xmax; ++x)
	for (int y = ymin; y <= ymax; ++y)
	{
	if ((y*y + x*x <= scale*scale)
	&& (y*COS_AL1 - x*SIN_AL1 - EPS >= 0)
	&& (y*COS_AL2 - x*SIN_AL2 + EPS <= 0)
	)
	{
	cv::Point2f pts((float)x, (float)y);
	float xx = pts.x*cs - pts.y*ss;
	float yy = pts.x*ss + pts.y*cs;
	vecPoints.push_back(cv::Point_<T>((T)xx, (T)yy));
	}
	}
	}
	*/
	SectorFast(const cv::Point_<T> &loc, float scale_,
		float widthAngle_, float theta_ = 0.0) :
		scale(scale_), theta(theta_), width(widthAngle_), location(loc)
	{
		float AL1 = theta - width / scale;
		float AL2 = theta + width / scale;

		float COS_AL1 = cos(AL1);
		float SIN_AL1 = sin(AL1);
		float COS_AL2 = cos(AL2);
		float SIN_AL2 = sin(AL2);
		float xmin = (int)MIN(MIN(0, round(scale*COS_AL1)), round(scale*COS_AL2));
		float xmax = (int)MAX(MAX(0, round(scale*COS_AL2)), round(scale*COS_AL1));
		float ymin = (int)MIN(MIN(0, round(scale*SIN_AL2)), round(scale*SIN_AL1));
		float ymax = (int)MAX(MAX(0, round(scale*SIN_AL1)), round(scale*SIN_AL2));

		float cs = cos(theta), ss = sin(theta);

		for (int x = xmin; x <= xmax; ++x)
			for (int y = ymin; y <= ymax; ++y)
			{
				if ((y*y + x*x <= scale*scale)
					&& (y*COS_AL1 - x*SIN_AL1 - EPS >= 0)
					&& (y*COS_AL2 - x*SIN_AL2 + EPS <= 0)
					)
				{
					//	cv::Point2f pts((float)x, (float)y);
					//	float xx = pts.x*cs - pts.y*ss;
					//	float yy = pts.x*ss + pts.y*cs;
					vecPoints.push_back(cv::Point_<T>((T)x + loc.x, (T)y + loc.y));
					Orientations.push_back(atan2((float)y, (float)x));
				}
			}
		location.x = (T)0;
		location.y = (T)0;
	}
	~SectorFast()
	{

	}

	SectorFast operator +(const cv::Point_<T> &p)
	{
		this->location += p;
		for (int i = 0; i < numOfPoints; ++i)
			this->vecPoints[i] += p;
		return *this;
	}
	SectorFast operator +=(const cv::Point_<T> &p)
	{
		this->location += p;
		for (int i = 0; i < numOfPoints; ++i)
			this->vecPoints[i] += p;
		return *this;
	}

//#ifndef _MSC_VER
//#endif
	float calcStrength(const cv::Mat &gradMag, const cv::Mat &gradPhase)
	{
		assert(gradMag.cols == gradPhase.cols && gradMag.rows == gradPhase.rows);

		size_t cols = gradMag.cols, rows = gradPhase.rows;

#ifdef _MSC_VER
		std::vector<cv::Point_<T>>::const_iterator iter;
#else
		typename std::vector<cv::Point_<T>>::const_iterator iter;
#endif
		float ret = 0.f;
		int cnt;
		for (iter = vecPoints.begin(), cnt = 0; iter != vecPoints.end(); ++iter, ++cnt)
		{
			if (iter->x < (T)0 || iter->x >= (T)cols || iter->y < (T)0 || iter->y >= (T)rows)
				continue;
			/*
			float alpha = Orientations[cnt];
			float phi = gradPhase.at<float>(*iter);
			float gamma = gradMag.at<float>(*iter);
			float SIN = phi - alpha;
			SIN = fabs(cos(SIN)) - fabs(sin(SIN));			
			if (SIN < 0.0f)
				SIN = 0.0f;
			*/			
			float SIN = fastsin(round_pi(gradPhase.at<float>(*iter)-Orientations[cnt]));
			float gamma = gradMag.at<float>(*iter);

			ret += gamma*pow2(pow2(1-pow2(SIN)));
		}
		return ret;
	}

	float calcStrength(bool *LocalTheta(int, int, float, float&, float&), float theta, size_t rows, size_t cols)
	{
#ifdef _MSC_VER
		std::vector<cv::Point_<T>>::const_iterator iter;
#else
		typename 	std::vector<cv::Point_<T>>::const_iterator iter;
#endif
		float ret = 0.f;
		int cnt;
		float phi, gamma;
		for (iter = vecPoints.begin(), cnt = 0; iter != vecPoints.end(); ++iter, ++cnt)
		{
			if (iter->x < (T)0 || iter->x >= (T)cols || iter->y < (T)0 || iter->y >= (T)rows)
				continue;

			float alpha = Orientations[cnt];
			LocalTheta(iter->x, iter->y, theta, phi, gamma);
			float SIN = phi - alpha;
			SIN = fabs(cos(SIN)) - fabs(sin(SIN));
			if (SIN < 0.0f)
				SIN = 0.0f;

			ret += gamma;
		}
		return ret;
	}
	cv::Mat tm;
	float theta;
	float scale, width;
	size_t numOfPoints;
	std::vector<cv::Point_<T>> vecPoints;
	std::vector<float> Orientations;
	cv::Point_<T> location;
private:
};

class ACJDetection
{
public:

	ACJDetection(const cv::Mat &image_,
		ACJParameters *par_ = NULL)
		: image(image_)
	{
		if (par_ != NULL)
			par = *par_;
		initialization();
	}

	ACJDetection(const std::string &filename_,
		ACJParameters *par_ = NULL)
		:filename(filename_)
	{
		if (par_ != NULL)
			par = *par_;
		std::string filenameACJ = filename + ".acj";
		image_clean = cv::imread(filename, 0);
		std::fstream _file;
		_file.open(filenameACJ, std::ios::in);
		if (!_file)
		{
			//clock_t start_pre = clock();				
			tictoc.tic();
			std::cout << "Preprocessing....." << std::endl;
			initialization();
			tictoc.toc();
			std::cout << "Preprocessing ellapses " << tictoc << " seconds!." << std::endl;
			detectJunction(Junctions);
			std::cout << "Writing junctions to disk....." << std::endl;
			tictoc.tic();
			_file.open(filenameACJ, std::ios::out);
			writeJunctions(_file);
			tictoc.toc();
			_file.close();
			std::cout << "Writing junctions to disk ellapses " << tictoc << " seconds!." << std::endl;
		}
		else
		{
			std::cout << "Junctions have been detected....." << std::endl;
			std::cout << "Reading junctions and parameters from disk....." << std::endl;
			//clock_t start_read = clock();			
			tictoc.tic();
			readJunctions(_file);
			_file.close();
			initialization();
			tictoc.toc();
			//clock_t finishi_read = clock();
			std::cout << "Reading junctions to disk ellapses " << tictoc << " seconds!." << std::endl;
		}
	}
	virtual ~ACJDetection()
	{
	}

	void writeJunctions(std::fstream &out)
	{
		/*
		Format example:
		location_x location_y
		junctionClass scale rd logNFA
		branch_0 strength_0 scale_0
		branch_1 strength_1 scale_1
		...
		branch_{junctionClass} strength_{junctionClass} logNFA_{junctionClass}
		*/
		std::vector<Junction>::const_iterator iter;
		out << Junctions.size() << std::endl;

		for (iter = Junctions.begin(); iter != Junctions.end(); ++iter)
		{
			out << iter->location.x << " " << iter->location.y << std::endl;
			out << iter->junctionClass << " " << iter->scale
				<< " " << iter->r_d
				<< " " << iter->logNFA << std::endl;
			for (int i = 0; i < iter->junctionClass; ++i)
				out << iter->branch[i].branch << " " << iter->branch[i].branchStrength << " " << iter->branch[i].branchScale << std::endl;
		}
	}

	void readJunctions(std::fstream &in)
	{
		size_t sz;
		in >> sz;

		while (sz--)
		{
			Junction tmp;
			Branch btmp;
			in >> tmp.location.x >> tmp.location.y;
			in >> tmp.junctionClass >> tmp.scale >> tmp.r_d >> tmp.logNFA;
			tmp.branch.clear();
			for (int i = 0; i < tmp.junctionClass; ++i)
			{
				in >> btmp.branch >> btmp.branchStrength >> btmp.branchScale;
				tmp.branch.push_back(btmp);
			}
			Junctions.push_back(tmp);
		}
	}

	void readGradientPhaseFromFile(const std::string &filename);
	void readGradientMagFromFile(const std::string &filename);
	void imGradient(const cv::Mat &image,
		cv::Mat &fx,
		cv::Mat &fy);//compute Normalized gradient

	void sketchProposalFixed(int scale);
	int sketchProposalFixed(int x, int y);

	bool detectJunction(int x, int y,
		std::list<Junction> &jlist);//detect junction for given coordinate

	virtual void detectJunction(std::vector<Junction> &junctVec);

	virtual void displayJunction(const cv::Mat &im, cv::Mat &imOut,
		const Junction &junct);

	std::vector<cv::Vec4f> lsdResults;

	ACJParameters par;
	cv::Mat image_clean;
	cv::Mat image;	
	cv::Mat nfaImage;
	std::vector<Junction> Junctions;
	std::vector<Junction> localJunctions;
protected:
	//data fields
	cv::Mat gradMagnitude, gradPhase;
	cv::Mat grad;
	cv::Mat lsdCandidate;
	cv::Mat OrientationPatch;
	cv::Mat OrientationPatchLocal;
	std::vector<std::vector<Sector>> sectorVecVec;
	std::vector<double> strengthPDF;
	std::vector<std::vector<double>> logCDF;
	std::string filename;
	TicToc tictoc;
	//private:
	//function fields
	void initialization();

	void generatelogCDFAll();

	template<typename T> void cvtpdf2log10CDF(const std::vector<T> &pdf,
		std::vector<T> &log10CDF);

	void generateSector(int Ns, int nbSector, float widthSector,
		std::vector<Sector> &sectorList, bool local = false);

	template<typename TMat>
	void computeStrengthMax(const TMat &GradMagPath, const TMat &PhasePatch,
		TMat &StrengthMaxVec);

	template<typename TMat>
	void computeStrengthMaxLocal(const TMat &GradMagPath, const TMat &PhasePatch,
		TMat &StrengthMaxVec);

	template<typename TMat>
	void computeStrength(const TMat &strengthMax,
		const std::vector<Sector> &sectorList, std::vector<float> &strength);

	float computeLogNFA(int numBranch, int numPixel, float branchStrength,
		int rmin, int rmax, float widthAngle, int imageSize,
		int lengthUnitCDF);

	double computeNFACLT(int numBranch, int numPixel, float branchStrength,
		int rmin, int rmax, float widthAngle, int imageSize);

	float computeLogNFAScaleFixed(int scale, int numBranch, int numPixel, float branchStrength,
		float widthAngle, int imageSize,
		int lengthUnitCDF);

	template<typename TMat>
	bool junctionRefinement(float widthAngle,
		const TMat &strengthMax,
		const TMat &phasePatch,
		const std::vector<Sector> &sectorVec,
		Junction &junction);

	void junctionRefinementSimple(float widthAngle,
		const cv::Mat &strengthMax,
		const cv::Mat &phasePatch,
		const std::vector<Sector> &sectorVec,
		Junction &junction);

	void junctionStability(std::list<Junction> &junctionList);

	void junctionLocation(std::list<Junction> &jlist, float sigma);

	void junctionClassify(std::list<Junction> &jlist, float sigma);

};
#endif