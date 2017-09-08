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
#ifndef __PROBTOOL__
#define __PROBTOOL__

#include <stdio.h>
#include <vector>
#include "helper.h"
#include "pdf.h"

class ProbabilityDF
{
public:
	ProbabilityDF(size_t convTimes_) :convTimes(convTimes_)
	{
		int n = pdf_num;
		double logval;

		interval = pdf_interval;
		lowerBound = pdf_lowerBound;
		upperBound = pdf_upperBound;

		for (int i = 0; i < n; ++i)
		{
			logval = pdf_logvals[i];
			baseprob.push_back(pow(10.0, logval));
		}

		for (int i = 0; i < convTimes; ++i)
		{
			std::vector<double> temppdf;
			if (i == 0)
				temppdf = baseprob;
			else
				convolution1D(convprob[i - 1], baseprob, temppdf, 0);
			std::vector<double> tempcdf(temppdf.size(), 0.0);
			tempcdf[0] = temppdf[0];
			for (int j = 1; j < temppdf.size(); ++j)
				tempcdf[j] = tempcdf[j - 1] + temppdf[j];
			for (int j = 0; j < temppdf.size(); ++j)
			{
				tempcdf[j] = log10(1.0 - tempcdf[j]);
			}
			convprob.push_back(temppdf);
			logCDF.push_back(tempcdf);
		}

	}

	double logProbability(double strength, int numOfPixel) const
	{
		if (numOfPixel >= convTimes)
			fprintf(stderr, "Error in ProbabilityDF::logProbability, the number of pixel is larger than expceted");
		double lb = lowerBound*numOfPixel;
		double ub = upperBound*numOfPixel;
		const std::vector<double> &cdf = logCDF[numOfPixel];
		if (strength < lb)
			return cdf[0];
		if (strength>ub)
			return cdf[cdf.size() - 1];
		int index = (int)round((strength - lb) / interval);
		if (index >= cdf.size())
			fprintf(stderr, "Error in ProbabilityDF::logProbability; Check logic for function ");
		return cdf[index];
	}
private:
	std::vector<double> baseprob;
	std::vector<std::vector<double>> convprob;
	std::vector<std::vector<double>> logCDF;
	double interval;
	double lowerBound, upperBound;
	size_t convTimes;
};

#endif