#pragma once
#include <iostream>
#include <vector>
#include "distance.h"


namespace claim {

void
kMeansPP(const float* pdata, size_t nb, size_t dim, size_t num_clusters,
         std::vector<std::vector<float>>& centroids, p_dis& disf,
         std::vector<std::vector<std::pair<int, float>>>& distances);

} // namespace claim

