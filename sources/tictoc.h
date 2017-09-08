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
#ifndef __TICTOC__
#define __TICTOC__
#include <time.h>
#include <ctime>
#include <fstream>

class TicToc
{
private:
	clock_t start, finish;
	double seconds;
public:
	TicToc() :start(0), finish(0)
	{
	}
	~TicToc()
	{
	}
	void tic()
	{
		start = clock();
	}
	void toc()
	{
		finish = clock();
		seconds = (double)(finish - start) / CLOCKS_PER_SEC;
	}
private:
	clock_t getic()
	{
		return start;
	}
	clock_t getoc()
	{
		return finish;
	}

	friend std::ostream &operator<<(std::ostream &os, const TicToc &tictoc)
	{
		os << tictoc.seconds;
		return os;
	}
};
#endif