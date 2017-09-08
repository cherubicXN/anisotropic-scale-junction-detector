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
#include "generateSector.hpp"

void generatesector_index(int Ns_max,
	int Ns,
	int nbSECTOR,
	float widthSECTOR,
	std::vector<Sector> &sectorList)
{
	Sector sector;
	cv::Point2i tempPoint;
	int tempIndex;

	int n, xmin, xmax, ymin, ymax;
	float AL, AL1, AL2, COS_AL1, SIN_AL1, COS_AL2, SIN_AL2;

	std::vector<Sector>::iterator sector_Iter;
	std::list<cv::Point2i>::iterator point_Iter;
	cv::Mat bVisited(2 * Ns_max + 1, 2 * Ns_max + 1, CV_8UC1, cv::Scalar(0));

	if (!sectorList.empty())
		sectorList.clear();

	for (n = 0; n < nbSECTOR; ++n)
	{
		AL = (float)2 * PI / (nbSECTOR)*n;
		AL1 = AL - widthSECTOR;
		AL2 = AL + widthSECTOR;

		sector.Orientation = AL;
		sector.sectPointInd_C.clear();
		sector.sectPointInd_M.clear();
		sector.sectPointInd_A.clear();

		COS_AL1 = cos(AL1);  SIN_AL1 = sin(AL1);
		COS_AL2 = cos(AL2);  SIN_AL2 = sin(AL2);
		xmin = (int)MIN(MIN(0, round(Ns*COS_AL1)), round(Ns*COS_AL2));
		xmax = (int)MAX(MAX(0, round(Ns*COS_AL2)), round(Ns*COS_AL1));
		ymin = (int)MIN(MIN(0, round(Ns*SIN_AL2)), round(Ns*SIN_AL1));
		ymax = (int)MAX(MAX(0, round(Ns*SIN_AL1)), round(Ns*SIN_AL2));

		if (n == 0)
		{
			for (int x = xmin; x <= xmax; ++x)
				for (int y = ymin; y <= ymax; ++y)
				{
					if ((y*y + x*x <= Ns*Ns)
						&& (y*COS_AL1 - x*SIN_AL1 - EPS >= 0)
						&& (y*COS_AL2 - x*SIN_AL2 + EPS <= 0)
						)
					{
						sector.sectPointInd_C.push_back(cv::Point2i(x + Ns_max, y + Ns_max));
						bVisited.at<uchar>(y + Ns_max, x + Ns_max) = 1;
					}
				}
		}
		else
		{
			for (int x = xmin; x <= xmax; x++)
				for (int y = ymin; y <= ymax; y++)
				{
					if ((y*y + x*x <= Ns*Ns)
						&& (y*COS_AL1 - x*SIN_AL1 - EPS >= 0)
						&& (y*COS_AL2 - x*SIN_AL2 + EPS <= 0)
						&& (bVisited.at<uchar>(y + Ns_max, x + Ns_max) == 0))
					{
						//tempIndex = (x + Ns_max)*(2 * Ns_max + 1) + (y + Ns_max);
						sector.sectPointInd_A.push_back(cv::Point2i(x + Ns_max, y + Ns_max));
						sector.sectPointInd_C.push_back(cv::Point2i(x + Ns_max, y + Ns_max));
						bVisited.at<uchar>(y + Ns_max, x + Ns_max) = 1;
					}
				}
			sector_Iter = sectorList.end();
			sector_Iter--;
			for (point_Iter = sector_Iter->sectPointInd_C.begin();
				point_Iter != sector_Iter->sectPointInd_C.end();
				++point_Iter)
			{
				int x = point_Iter->x - Ns_max;
				int y = point_Iter->y - Ns_max;

				if ((y*COS_AL1 - x*SIN_AL1 - EPS >= 0) && (y*COS_AL2 - x*SIN_AL2 + EPS <= 0))
					sector.sectPointInd_C.push_back(*point_Iter);
				else
				{
					sector.sectPointInd_M.push_back(*point_Iter);
					bVisited.at<uchar>(*point_Iter) = 0;
				}
			}
		}
		sector.numberOfPoint = sector.sectPointInd_C.size();
		sectorList.push_back(sector);
	}
}