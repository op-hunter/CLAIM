#include <random>
#include <vector>
#include <cassert>
#include "kmeans.h"




namespace claim {

void
kMeansPP(const float* pdata, const size_t nb, const size_t dim, const size_t num_clusters,
         std::vector<std::vector<float>>& centroids, p_dis& disf,
         std::vector<std::vector<std::pair<int, float>>>& distances) {


    std::vector<size_t> cluster_ids;
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    auto         x = rd();
    std::mt19937 generator(x);  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<long> distribution(0, nb - 1);
//    std::uniform_real_distribution<double> distribution(0, nb - 1);

    auto init_id = (size_t)distribution(generator);
    std::vector<float> dist(nb, std::numeric_limits<float>::max());
//    double sumdx2 = 0.0;

//    for (auto i = 0; i < nb; i ++) {
//        dist[i] = disf(pdata + i * dim, pdata + init_id * dim, &dim);
//        sumdx2 += dist[i];
//    }

    centroids.resize(num_clusters);
    centroids[0].resize(dim);
    distances.resize(nb);
    for (auto i = 0; i < dim; i ++) {
        centroids[0][i] = pdata[init_id * dim + i];
    }
    cluster_ids.push_back(init_id);

    for (auto i = 1; i <= num_clusters; i ++) {
        double sumdx2 = 0.0;
#pragma omp parallel for schedule(static, 4096) reduction(+:sumdx2)
        for (auto j = 0; j < nb; j ++) {
            // todo: advance continue
            auto dis_cj = disf(pdata + j * dim, centroids[i - 1].data(), &dim);
            if (std::isnan(dis_cj)) {
                std::cout << "dist between " << j << " and centroid " << i - 1 << " is " << dis_cj << std::endl;
            }
            assert(!std::isnan(dis_cj));
            dist[j] = std::min(dist[j], dis_cj);
            sumdx2 += dist[j];
            if (distances[j].size() > 0) {
                if (distances[j][0].second > dis_cj) {
                    distances[j][0].second = dis_cj;
                    distances[j][0].first = i - 1;
                }
            } else {
                distances[j].push_back(std::pair<int, float>(i - 1, dis_cj));
            }
        }
        if (i == num_clusters)
            break;
        centroids[i].resize(dim);
        std::uniform_real_distribution<double> distribution1(0, sumdx2);
        auto prob = distribution1(generator);
        for (auto j = 0; j < nb; j ++) {
            if (prob <= 0) {
                for (auto k = 0; k < dim; k ++) {
                    centroids[i][k] = pdata[j * dim + k];
                }
                cluster_ids.push_back((size_t)j);
                break;
            }
            prob -= dist[j];
        }
        if (i % 100 == 0)
            std::cout << "has got " << i << " centroids." << std::endl;
    }

    std::cout << "show centroids id of kmeans++: " << std::endl;
    for (auto i = 0; i < cluster_ids.size(); i ++) {
        std::cout << cluster_ids[i] << " ";
    }
    std::cout << std::endl;
}


} // namespace claim

