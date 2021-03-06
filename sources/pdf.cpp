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
#include "pdf.h"

double pdf_interval = 0.050000;
double pdf_lowerBound = -5;
double pdf_upperBound = 5;

int pdf_num = 201;

double pdf_logvals[] = {
	-14.270735,
	-14.046392,
	-13.824144,
	-13.603991,
	-13.385929,
	-13.169959,
	-12.956079,
	-12.744286,
	-12.534580,
	-12.326959,
	-12.121422,
	-11.917966,
	-11.716590,
	-11.517292,
	-11.320070,
	-11.124923,
	-10.931848,
	-10.740844,
	-10.551908,
	-10.365038,
	-10.180232,
	-9.997489,
	-9.816805,
	-9.638178,
	-9.461605,
	-9.287085,
	-9.114614,
	-8.944191,
	-8.775811,
	-8.609472,
	-8.445172,
	-8.282906,
	-8.122673,
	-7.964468,
	-7.808288,
	-7.654130,
	-7.501989,
	-7.351863,
	-7.203747,
	-7.057638,
	-6.913530, -6.771420, -6.631302, -6.493173, -6.357027, -6.222860, -6.090665, -5.960438, -5.832172, -5.705861, -5.581500, -5.459081, -5.338598, -5.220044, -5.103410, -4.988690, -4.875874, -4.764955, -4.655923, -4.548768, -4.443481, -4.340050, -4.238466, -4.138714, -4.040785, -3.944663, -3.850334, -3.757785, -3.666998, -3.577957, -3.490643, -3.405037, -3.321117, -3.238860, -3.158243, -3.079237, -3.001814, -2.925942, -2.851585, -2.778704, -2.707256, -2.637191, -2.568456, -2.500987, -2.434714, -2.369554, -2.305410, -2.242167, -2.179689, -2.117807, -2.056311, -1.994934, -1.933326, -1.871012, -1.807322, -1.741254, -1.671199, -1.594287, -1.504448, -1.383934, -0.301030, -1.383934, -1.504448, -1.594287, -1.671199, -1.741254, -1.807322, -1.871012, -1.933326, -1.994934, -2.056311, -2.117807, -2.179689, -2.242167, -2.305410, -2.369554, -2.434714, -2.500987, -2.568456, -2.637191, -2.707256, -2.778704, -2.851585, -2.925942, -3.001814, -3.079237, -3.158243, -3.238860, -3.321117, -3.405037, -3.490643, -3.577957, -3.666998, -3.757785, -3.850334, -3.944663, -4.040785, -4.138714, -4.238466, -4.340050, -4.443481, -4.548768, -4.655923, -4.764955, -4.875874, -4.988690, -5.103410, -5.220044, -5.338598, -5.459081, -5.581500, -5.705861, -5.832172, -5.960438, -6.090665, -6.222860, -6.357027, -6.493173, -6.631302, -6.771420, -6.913530, -7.057638, -7.203747, -7.351863, -7.501989, -7.654130, -7.808288, -7.964468, -8.122673, -8.282906, -8.445172, -8.609472, -8.775811, -8.944191, -9.114614, -9.287085, -9.461605, -9.638178, -9.816805, -9.997489, -10.180232, -10.365038, -10.551908, -10.740844, -10.931848, -11.124923, -11.320070, -11.517292, -11.716590, -11.917966, -12.121422, -12.326959, -12.534580, -12.744286, -12.956079, -13.169959, -13.385929, -13.603991, -13.824144, -14.046392, -14.270735
};