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
#include "aacj.h"

void AACJDetection::output(std::vector<Junction> &results)
{
	if (!results.empty())
		results.clear();

	int cnt = 0;
	for (int i = 0; i < Junctions.size(); ++i)
	{
		Junction junct = Junctions[i];

		std::vector<Branch> branches;
		branches.clear();
		for (int j = 0; j < junct.junctionClass; ++j)
		{
			if (junct.branch[j].branchScale >= junct.scale)
				branches.push_back(junct.branch[j]);
		}
		//std::cout << junct.branch.size() << " " << branches.size() << std::endl;
		if (branches.size() < 2)
			continue;

		for (int j = 0; j < branches.size(); ++j)
			for (int k = j + 1; k < branches.size(); ++k)
			{
				Junction jnew;
				jnew.location = junct.location;
				jnew.junctionClass = 2;
				jnew.scale = junct.scale;
				jnew.logNFA = junct.logNFA;
				jnew.r_d = junct.r_d;
				jnew.branch.clear();

				if (in_angle(round_pi(branches[j].branch + PI), round_pi(branches[k].branch)) < 0.05)
					continue;

				if (isOriented(branches[j].branch, branches[k].branch))
				{
					jnew.branch.push_back(branches[j]);
					jnew.branch.push_back(branches[k]);
				}
				else
				{
					jnew.branch.push_back(branches[k]);
					jnew.branch.push_back(branches[j]);
				}
				++cnt;
				results.push_back(jnew);
			}
	}
	bDecomposed = true;
}

void AACJDetection::exportAACJ(const char *filename)
{
	std::vector<Junction> results;
	if (!bDecomposed)
	{
		output(results);
		Junctions = results;
	}

	/*
	Format example:
	location_x location_y
	junctionClass scale rd logNFA
	branch_0 strength_0 scale_0
	branch_1 strength_1 scale_1
	...
	branch_{junctionClass} strength_{junctionClass} logNFA_{junctionClass}
	*/
	FILE *file = fopen(filename, "w");
	fprintf(file, "%d\n", Junctions.size());
	for (int i = 0; i < Junctions.size(); ++i)
	{
		fprintf(file, "%f %f\n", Junctions[i].location.x, Junctions[i].location.y);
		fprintf(file, "%d %d %d %f\n", Junctions[i].junctionClass, Junctions[i].scale, Junctions[i].r_d, Junctions[i].logNFA);
		for (int j = 0; j < Junctions[i].junctionClass; ++j)
			fprintf(file, "%f %f %f\n", Junctions[i].branch[j].branch, Junctions[i].branch[j].branchStrength, Junctions[i].branch[j].branchScale);
	}
	fclose(file);
}

bool AACJDetection::isOriented(float theta1, float theta2)
{
	float x1 = cos(theta1), x2 = cos(theta2);
	float y1 = sin(theta1), y2 = sin(theta2);
	float ang = round_pi(atan2(sin(theta2 - theta1), cos(theta2 - theta1)));
	float ang2 = round_pi(acos(x1*x2 + y1*y2));

	return (in_angle(ang, ang2) < 0.01);

}

void AACJDetection::read(FILE *file)
{
	int n;
	//FILE *file = fopen(filename, "r");
	fscanf(file, "%d", &n);
	if (!Junctions.empty())
		Junctions.clear();
	Junction junct;
	Branch branch;
	for (int i = 0; i < n; ++i)
	{
		fscanf(file, "%f %f\n", &junct.location.x, &junct.location.y);
		int junctionClass, scale, r_d;
		float logNFA;

		fscanf(file, "%d %d %d %f\n", &junctionClass, &scale, &r_d, &logNFA);
		junct.junctionClass = (size_t)junctionClass;
		junct.scale = (size_t)scale;
		junct.r_d = (size_t)r_d;
		junct.logNFA = (size_t)logNFA;
		for (int j = 0; j < junct.junctionClass; ++j)
		{
			fscanf(file, "%f %f %f\n", &branch.branch, &branch.branchStrength, &branch.branchScale);
			junct.branch.push_back(branch);
		}
		Junctions.push_back(junct);
	}
	fclose(file);
}