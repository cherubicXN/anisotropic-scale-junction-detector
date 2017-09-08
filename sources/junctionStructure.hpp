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
#ifndef __JUNCTIONSTRUCTURE__
#define __JUNCTIONSTRUCTURE__
#include <opencv2/opencv.hpp>
#include <list>
#include <vector>

struct Sector
{
	float Orientation;
	size_t numberOfPoint;
	// points to be kept
	std::list<cv::Point2i> sectPointInd_C;
	// points to be removed
	std::list<cv::Point2i> sectPointInd_M;
	// points to be added
	std::list<cv::Point2i> sectPointInd_A;
};

struct Branch
{
	float branch;
	float branchStrength;
	float branchScale;
	float logNFA;
	float logNFAall;
	unsigned int numberOfPoint;
	unsigned int index;
};

struct Junction
{
	double logNFA;
	cv::Point2f location;
	size_t junctionClass;
	size_t index;
	size_t scale;
	size_t r_d;
	size_t removed;
	std::vector<Branch> branch;
	std::vector<float> StrengthAll;
	friend bool operator<(const Junction &t1, const Junction &t2)
	{
		return  (t1.logNFA > t2.logNFA);
	}
};

inline bool compareBranchByStrength(const Branch &t1, const Branch &t2)
{
	return t1.branchStrength < t2.branchStrength;
}

inline bool compareJunctionByNFAdescent(const Junction &t1, const Junction &t2)
{
	return t1.logNFA > t2.logNFA;
}

inline bool compareJunctionByLocation(const Junction &t1, const Junction &t2)
{
	if (t1.location.x == t2.location.x)
		return t1.location.y < t2.location.y;
	return t1.location.x < t2.location.x;
}

#endif