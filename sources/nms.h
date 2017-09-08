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
#ifndef __NMS__
#define __NMS__
#include <stdio.h>
#include <string.h>
#include <list>
#include <vector>
#include <stdlib.h>
#include <math.h>

using namespace std;

/*---------------------------------------------
*- sub-function of Non-maximum Suppression(NMS)
*- Function: compute partial maximum, pmax, of
*- the sequence {I[from],,...,I[to]} as follows:
*- pmax[i] = max{I[i],I[i+1],I[to-1],...,I[to]}
* ----------------------------------------------
*/
inline int CompPartialMax(float* I, int from, int to, float* pmax)
{
	int best;
	pmax[to] = I[to];
	best = to;

	while (to>from)
	{
		to = to - 1;
		if (I[to] <= I[best])
			pmax[to] = I[best];
		else
		{
			pmax[to] = I[to];
			best = to;
		}
	}
	return best;
}

/*- ---------------------------------------------
*- 1D circular Non-maximum Suppression(NMS)
*- The algorithm is detailed in the paper
*- "Efficient Non-Maximum Suppression" by
*- Alexander Neubeck and Luc Van Gool.
*- NOTE: there was a bug in that paper, which is
*- fixed here.
* ----------------------------------------------
*/
void nms1d_cir(std::vector<float> &J, int lenJ, int n, std::vector<float> &MAXIMUMAT)
{
	int i, j, k, iter, lenI, chkpt;
	lenI = lenJ + 3 * n;
	float* I = (float*)malloc(sizeof(float)*lenI);
	float* MaximumAt = (float*)malloc(sizeof(float)*lenI);
	float* pmax = (float*)malloc(sizeof(float)*lenI);
	//std::vector<float> MAXIMUMAT;
	MAXIMUMAT.assign(lenJ, 0.0);

	/*copy J to I for circular computation*/
	for (j = 0; j<n; j++)
		I[j] = J[lenJ - n + j];
	for (j = n; j<n + lenJ; j++)
		I[j] = J[j - n];
	for (i = n + lenJ; j<2 * n + lenJ; j++)
		I[j] = J[j - n - lenJ];
	for (i = 2 * n + lenJ; j<lenI; j++)
		I[j] = -100000.0;

	i = n;
	for (j = 0; j<lenI; j++)
	{
		pmax[j] = 0.0;
		MaximumAt[j] = pmax[j];
	}
	CompPartialMax(I, 0, i, pmax);
	chkpt = -1;

	while (i < lenI - 2 * n)
	{
		j = CompPartialMax(I, i, i + n, pmax);
		k = CompPartialMax(I, i + n, j + n, pmax);
		if ((i == j) || (I[j] > I[k]))
		{
			if (((chkpt <= j - n) || (I[j] >= pmax[chkpt])) && ((j == n + i) || (I[j] >= pmax[j - n])))
				MaximumAt[j] = I[j];
			if (i<j)
				chkpt = i + n;
			i = j + n + 1;
		}
		else
		{
			i = k;
			chkpt = j + n;
			while (i< lenI - 2 * n)
			{
				j = CompPartialMax(I, chkpt - 1, i + n, pmax);
				if (I[i] > I[j])
				{
					MaximumAt[i] = I[i];
					i = i + n + 1;
					break;
				}
				else
				{
					chkpt = i + n;
					i = j;
				}
			}
		}
	}

	/*copy results*/
	for (iter = 0; iter<lenJ; iter++)
		MAXIMUMAT[iter] = MaximumAt[iter + n];

	free(I);
	free(MaximumAt);
	free(pmax);
}

void nms1d_cir_list(std::vector<float> &J, int lenJ, int n, std::list<float> &MAXIMUMAT)
{
	int i, j, k, iter, lenI, chkpt;
	lenI = lenJ + 3 * n;
	float* I = (float*)malloc(sizeof(float)*lenI);
	float* MaximumAt = (float*)malloc(sizeof(float)*lenI);
	float* pmax = (float*)malloc(sizeof(float)*lenI);
	//std::list<float> MAXIMUMAT;

	/*copy J to I for circular computation*/
	for (j = 0; j<n; j++)
		I[j] = J[lenJ - n + j];
	for (j = n; j<n + lenJ; j++)
		I[j] = J[j - n];
	for (i = n + lenJ; j<2 * n + lenJ; j++)
		I[j] = J[j - n - lenJ];
	for (i = 2 * n + lenJ; j<lenI; j++)
		I[j] = -100000.0;

	i = n;
	for (j = 0; j<lenI; j++)
	{
		pmax[j] = 0.0;
		MaximumAt[j] = 0.0;
	}
	CompPartialMax(I, 0, i, pmax);
	chkpt = -1;

	while (i < lenI - 2 * n)
	{
		j = CompPartialMax(I, i, i + n, pmax);
		k = CompPartialMax(I, i + n, j + n, pmax);
		if ((i == j) || (I[j] > I[k]))
		{
			if (((chkpt <= j - n) || (I[j] >= pmax[chkpt])) && ((j == n + i) || (I[j] >= pmax[j - n])))
				MaximumAt[j] = I[j];
			if (i<j)
				chkpt = i + n;
			i = j + n + 1;
		}
		else
		{
			i = k;
			chkpt = j + n;
			while (i< lenI - 2 * n)
			{
				j = CompPartialMax(I, chkpt - 1, i + n, pmax);
				if (I[i] > I[j])
				{
					MaximumAt[i] = I[i];
					i = i + n + 1;
					break;
				}
				else
				{
					chkpt = i + n;
					i = j;
				}
			}
		}
	}

	/*copy results*/
	for (iter = 0; iter<lenJ; iter++)
		if (MaximumAt[iter + n] != 0.0)
			MAXIMUMAT.push_back(iter);

	free(I);
	free(MaximumAt);
	free(pmax);
}
#endif