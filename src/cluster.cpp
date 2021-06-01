
#include <iostream>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <vector>
#include <cstring>
#include <chrono>
#include <unordered_map>
#include <bits/unordered_map.h>
#include <cmath>
#include <random>
#include "cluster.h"
#include "heap.h"




namespace claim {

#define EXPANSION_FACTOR 1.2

Cluster::Cluster(const size_t nb, const size_t dim, const size_t cs, const size_t max_iter_time)
: nb_(nb), dim_(dim), cs_(cs), iter_(max_iter_time), data_(nullptr) {
    nc_ = (nb - 1) / cs + 1;
    centroids_.reserve((size_t)(nc_ * EXPANSION_FACTOR));
    invertlist_.reserve((size_t)(nc_ * EXPANSION_FACTOR));
    data_ = (char*) malloc(nb * dim * sizeof(float));
    assert(data_ != nullptr);
    disf_ = L2Square;
    scale_coefficient_ = 2.5;
    max_hot_pts_ = 3;
    top_hot_cls_ = 3;
}

/*
 * load data from file *source_data_path* in binary format.
 */
void
Cluster::Load(std::string& source_data_path) {
    auto tstart = std::chrono::high_resolution_clock::now();
    std::ifstream fin(source_data_path.c_str(), std::ios::binary);

    size_t npts, ndim;
    unsigned rdnb, rddim;
    fin.read((char*)&rdnb, 4);
    fin.read((char*)&rddim, 4);
    npts = rdnb;
    ndim = rddim;
    assert(npts == nb_);
    assert(ndim == dim_);
    assert(data_ != nullptr);
    fin.read(data_, npts * ndim * sizeof(float));

    fin.close();
    auto tend = std::chrono::high_resolution_clock::now();
    std::cout << "read data from " << source_data_path
              << " done in " << std::chrono::duration_cast<std::chrono::milliseconds>( tend - tstart ).count()
              << " milliseconds, nb = " << npts << ", dim = " << ndim << std::endl;
    auto pp = (float*)data_;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "show the 1000000th raw data:" << std::endl;
    for (auto i = 0; i < dim_; i ++)
        std::cout << pp[999999 * dim_ + i] << " ";
    std::cout << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
}

Cluster::~Cluster() {
    if (data_)
        free(data_);
}

int
Cluster::split(size_t max_iter_times) {
    std::vector<int> to_split;
    for (auto i = 0; i < nc_; i ++) {
        if (invertlist_[i].size() > cs_)
            to_split.push_back(i);
    }
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    auto         x = rd();
    std::mt19937 generator(x);  // Standard mersenne_twister_engine seeded with rd()
    auto pd = (float*)data_;
    for (auto &ci : to_split) {
        auto cluster_nums = (invertlist_[ci].size() - 1) / cs_ + 1;
        std::cout << "start to split cluster " << ci << " into " << cluster_nums
                  << " clusters whose size is " << invertlist_[ci].size() << std::endl;
        std::uniform_int_distribution<int> distribution(0, (int)invertlist_[ci].size() - 1);
        std::vector<bool> flag(invertlist_[ci].size(), false);
        std::vector<std::vector<float>> tmp_centroids(cluster_nums, std::vector<float>(dim_, 0.0));
        std::vector<std::vector<size_t>> tmp_ivls(cluster_nums);
        std::vector<std::pair<int, float>> tmp_distances(invertlist_[ci].size(), std::pair<int, float>(-1, std::numeric_limits<float>::max()));
        for (auto i = 0; i < cluster_nums; i ++) {
            auto cid = distribution(generator);
            if (flag[cid]) {
                i --;
                continue;
            }
            flag[cid] = true;
            memcpy(tmp_centroids[i].data(), pd + invertlist_[ci][cid] * dim_, dim_ * sizeof(float));
        }
        for (auto it = 0; it < max_iter_times; it ++) {
            float err = 0.0;
            for (auto ii = 0; ii < cluster_nums; ii ++) {
                tmp_ivls[ii].clear();
            }
            for (auto ii = 0; ii < invertlist_[ci].size(); ii ++)
                tmp_distances[ii].second = std::numeric_limits<float>::max();
            for (auto ii = 0; ii < cluster_nums; ii ++) {
                for (auto jj = 0; jj < invertlist_[ci].size(); jj ++) {
                    auto dist_iijj = disf_(tmp_centroids[ii].data(), pd + invertlist_[ci][jj] * dim_, &dim_);
                    if (dist_iijj < tmp_distances[jj].second) {
                        tmp_distances[jj].second = dist_iijj;
                        tmp_distances[jj].first = ii;
                    }
                }
            }
            for (auto ii = 0; ii < tmp_distances.size(); ii ++) {
                tmp_ivls[tmp_distances[ii].first].push_back(invertlist_[ci][ii]);
                err += tmp_distances[ii].second;
            }
            for (auto ii = 0; ii < tmp_centroids.size(); ii ++) {
                for (auto jj = 0; jj < tmp_centroids[ii].size(); jj ++)
                    tmp_centroids[ii][jj] = 0.0;
                for (auto jj = 0; jj < tmp_ivls[ii].size(); jj ++) {
                    auto ppp = pd + tmp_ivls[ii][jj] * dim_;
                    for (auto kk = 0; kk < dim_; kk ++) {
                        tmp_centroids[ii][kk] += ppp[kk];
                    }
                }
                for (auto jj = 0; jj < dim_; jj ++)
                    tmp_centroids[ii][jj] /= tmp_ivls[ii].size();
            }
            std::cout << "it = " << it << ", err = " << err << std::endl;
        }
        centroids_[ci].swap(tmp_centroids[0]);
        invertlist_[ci].swap(tmp_ivls[0]);
        nc_ += cluster_nums - 1;
        int mx_sz = tmp_ivls[0].size();
        int mn_sz = tmp_ivls[0].size();
        for (auto ii = 1; ii < tmp_centroids.size(); ii ++) {
            centroids_.push_back(tmp_centroids[ii]);
            invertlist_.push_back(tmp_ivls[ii]);
            if (mx_sz < tmp_ivls[ii].size())
                mx_sz = tmp_ivls[ii].size();
            if (mn_sz > tmp_ivls[ii].size())
                mn_sz = tmp_ivls[ii].size();
        }
        std::cout << "split cluster " << ci << " into " << cluster_nums << " clusters, "
                  << " min size = " << mn_sz
                  << " max size = " << mx_sz
                  << std::endl;
        std::cout << "current nc_ = " << nc_ << ", invlist.size = " << invertlist_.size()
                  << ", centroids.size = " << centroids_.size()
                  << std::endl;
    }
    return (int)to_split.size();
}

void
Cluster::Kmeans(size_t max_iter_times) {
    std::cout << "Kmeans.max_iter_times = " << max_iter_times << std::endl;
    std::vector<std::vector<std::pair<int, float>>> distances;
    auto t0 = std::chrono::high_resolution_clock::now();
    kMeansPP((float *)data_, nb_, dim_, nc_, centroids_, disf_, distances);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Init::kMeansPP done in "
              << std::chrono::duration_cast<std::chrono::milliseconds>( t1 - t0 ).count()
              << " milliseconds." << std::endl;
    assert(centroids_.size() == nc_);
    std::vector<std::vector<size_t>> ivl;
    ivl.resize(nc_);
    t0 = std::chrono::high_resolution_clock::now();
    for (auto it = 0; it < max_iter_times; it ++) {
//        float best_dis;
//        int best_cls;
        float err = 0.0;
        std::vector<std::pair<int, float>> avg_dst(nc_);
        for (auto i = 0; i < nc_; i ++)
            avg_dst[i].first = i, avg_dst[i].second = 0.0;
        // assign all points to the nearest cluster
        // todo: parallelize
//#pragma omp parallel for schedule(static, 4096) reduction(+:err)
        for (auto i = 0; i < nb_; i ++) {
            auto best_cls = distances[i][0].first;
            auto best_dis = distances[i][0].second;
            assert(best_cls < nc_);
            for (auto j = 1; j < distances[i].size(); j ++) {
                if (best_dis > distances[i][j].second) {
                    best_dis = distances[i][j].second;
                    best_cls = distances[i][j].first;
                }
            }
            avg_dst[best_cls].second += best_dis;
            ivl[best_cls].push_back((size_t)i);
            distances[i].clear();
//            distances[i].push_back(std::pair<int, float>(-1, std::numeric_limits<float>::max()));
            err += best_dis; // reduction
        }
//        for (auto i = 0; i < nb_; i ++) {
//            assert(distances[i][0].first >= 0);
//            assert(distances[i][0].first < nc_);
//            assert(!std::isnan(distances[i][0].second));
//            avg_dst[distances[i][0].first].second += distances[i][0].second;
//        }
        std::cout << "total err = " << err << " after " << it << " iteration." << std::endl;
        float smsm = 0.0;
        std::sort(avg_dst.begin(), avg_dst.end(), [](const auto &l, const auto &r) {
            return l.second < r.second;
        });
//        std::cout << "err of each cluster:" << std::endl;
//        for (auto &pr : avg_dst) {
//            std::cout << "(" << pr.first << ", " << pr.second << ") ";
//            smsm += pr.second;
//        }
//        std::cout << std::endl;
        std::cout << "avg_dist.min = " << avg_dst[0].second << ", avg_dist.mid = " << avg_dst[avg_dst.size() / 2].second
                  << ", avg_dist.avg = " << smsm / nc_ << ", avg_dist.max = " << avg_dst[avg_dst.size() - 1].second
                  << std::endl;
        if (it == max_iter_times - 1) {
            invertlist_.resize(ivl.size());
            for (auto ili = 0; ili < ivl.size(); ili ++)
                invertlist_[ili].swap(ivl[ili]);
            std::cout << "ShowCSHistogram after " << max_iter_times << " times of iteration." << std::endl;
            std::cout << "-------------------------------------------------------------------------" << std::endl;
            ShowCSHistogram();
            std::cout << "-------------------------------------------------------------------------" << std::endl;
            break;
        }
        // re-calculate the new centroids
        clean_empty_cluster(ivl);
        calculate_centroids(ivl, true);
        // re-calculate the nearest cluster
//        calculate_dist2centroids(distances, it == max_iter_times - 1);
        calculate_dist2centroids(distances, false);
    }
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Init::kmeans of " << max_iter_times << " times iteration costs "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " milliseconds." << std::endl;
}

/*
 * run k-means++ algorithm to get initial clusters and ensure the size of each cluster not more than *cs_*.
 */
void
Cluster::Init() {
    std::vector<std::vector<std::pair<int, float>>> distances;
    auto t0 = std::chrono::high_resolution_clock::now();
    kMeansPP((float *)data_, nb_, dim_, nc_, centroids_, disf_, distances);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Init::kMeansPP done in "
              << std::chrono::duration_cast<std::chrono::milliseconds>( t1 - t0 ).count()
              << " milliseconds." << std::endl;
    // todo: iter several(3?) times to wait centroids become stable
    std::vector<std::vector<size_t>> ivl;
    ivl.resize(nc_);
    t0 = std::chrono::high_resolution_clock::now();
    for (auto it = 0; it <= 3; it ++) {
        float best_dis;
        int best_cls;
        float err = 0.0;
        // assign all points to the nearest cluster
        // todo: parallelize
//#pragma omp parallel for schedule(static, 4096) reduction(+:err)
        for (auto i = 0; i < nb_; i ++) {
            best_cls = distances[i][0].first;
            best_dis = distances[i][0].second;
            assert(best_cls < nc_);
            assert(!std::isnan(best_dis));
            for (auto j = 1; j < distances[i].size(); j ++) {
                if (best_dis > distances[i][j].second) {
                    assert(!std::isnan(distances[i][j].second));
                    best_dis = distances[i][j].second;
                    best_cls = distances[i][j].first;
                }
            }
            ivl[best_cls].push_back((size_t)i);
            distances[i].clear();
//            distances[i].push_back(std::pair<int, float>(-1, std::numeric_limits<float>::max()));
            err += best_dis; // reduction
        }
        std::cout << "after the " << it << "th iteration, err = " << err << std::endl;
        if (it == 3) {
            std::cout << "ShowCSHistogram after 3 times of iteration." << std::endl;
            std::cout << "-------------------------------------------------------------------------" << std::endl;
            int mx_sz = 0;
            std::unordered_map<int, int> hist_map;
            for (auto &cls : ivl) {
                if (hist_map.find(cls.size()) != hist_map.end())
                    hist_map[cls.size()] ++;
                else
                    hist_map[cls.size()] = 1;
                mx_sz = std::max(mx_sz, (int)cls.size());
            }
            size_t sssums = 0;
            std::vector<int> histogram(mx_sz + 1, 0);
            int summmm1 = 0;
            for (auto &pr : hist_map) {
                histogram[pr.first] = pr.second;
                summmm1 += pr.first * pr.second;
//                if (histogram[pr.first] > 0) {
//                    sssums += pr.first * pr.second;
//                    std::cout << "there are " << pr.second << " clusters whose size is " << pr.first << std::endl;
//                }
            }
            for (int kkkkk = (int)histogram.size() - 1; kkkkk >= 0; kkkkk --) {
                if (histogram[kkkkk] > 0) {
                    sssums += histogram[kkkkk] * kkkkk;
                    std::cout << "there are " << histogram[kkkkk] << " clusters whose size is " << kkkkk << std::endl;
                }
            }
            std::cout << "ShowCSHistogram: sums = " << sssums << ", summm = " << summmm1 << ", nb_ = " << nb_ << std::endl;
            assert(sssums == nb_);
            std::cout << "-------------------------------------------------------------------------" << std::endl;
            break;
        }
        // re-calculate the new centroids
        clean_empty_cluster(ivl);
        calculate_centroids(ivl, true);
        // re-calculate the nearest cluster
        calculate_dist2centroids(distances, it == 2);
    }
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Init::kmeans of 3 times iteration costs "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " milliseconds." << std::endl;
    std::vector<std::pair<int, float>> avg_dists(nc_);
    t0 = std::chrono::high_resolution_clock::now();
    split_into_limited_cluster(avg_dists, distances);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Init::the1stsplit_into_limited_cluster totally costs "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " milliseconds." << std::endl;
    std::sort(avg_dists.begin(), avg_dists.end(), [](const auto &l, const auto &r) {
        return l.second < r.second;
    });
    float avgsum = 0.0;
    std::cout << "show avg_dists:" << std::endl;
    for (auto i = 0; i < avg_dists.size(); i ++) {
        std::cout << "(" << avg_dists[i].first << ", " << avg_dists[i].second << ") ";
        avgsum += avg_dists[i].second;
    }
    std::cout << std::endl;
    std::cout << "avg_dists.min = " << avg_dists[0].second
              << ", avg_dists.mid = " << avg_dists[avg_dists.size() / 2].second
              << ", avg_dists.avg = " << avgsum / avg_dists.size()
              << ", avg_dists.max = " << avg_dists[avg_dists.size() - 1].second << std::endl;
    float threshold = avg_dists[avg_dists.size() / 2].second * scale_coefficient_;
    int iteration_cnt = 0;
    while (threshold < avg_dists.back().second) {
        t0 = std::chrono::high_resolution_clock::now();
        std::vector<int> re_con_list;
        for (int i = (int)avg_dists.size() - 1; i >= 0; i --) {
            if (avg_dists[i].second > threshold)
                re_con_list.push_back(avg_dists[i].first);
            else
                break;
        }
        std::vector<std::pair<int, int>> hot_cnt(nc_);
        for (auto i = 0; i < nc_; i ++)
            hot_cnt[i].first = i, hot_cnt[i].second = 0;
        std::cout << "re_con_list.size = " << re_con_list.size() << std::endl;
        for (auto &noi : re_con_list) {
            for (auto i = 0; i < invertlist_[noi].size(); i ++) {
                for (auto j = 0; j < top_hot_cls_; j ++) {
                    auto cid = distances[invertlist_[noi][i]][j].first;
                    if (cid == noi)
                        break;
                    hot_cnt[cid].second ++;
                }
            }
        }
        std::sort(hot_cnt.begin(), hot_cnt.end(), [](const auto &l, const auto &r) {
            return l.second > r.second;
        });
//#pragma omp parallel for schedule(dynamic, 1)
        for (auto i = 0; i < max_hot_pts_; i ++) {
            auto re_con_id = hot_cnt[i].first;
            if (invertlist_[re_con_id].size() < 5) {
                std::cout << "error: the cluster " << re_con_id << " has only " << invertlist_[re_con_id].size()
                          << " points, cannot split" << std::endl;
                continue;
            }
            std::vector<std::vector<float>> cents;
            std::vector<std::vector<size_t>> list;
            two_means(re_con_id, cents, list);
//#pragma omp critical
            {
                centroids_[re_con_id].swap(cents[0]);
                centroids_.push_back(cents[1]);
                nc_ ++;
                avg_dists.resize(nc_);
            }
        }
        for (auto i = 0; i < nb_; i ++) {
            distances[i].clear();
            distances[i].reserve(nc_);
        }
        t0 = std::chrono::high_resolution_clock::now();
        calculate_dist2centroids(distances, true);
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Init::the " << iteration_cnt + 1 << "th iteration calculate_dist2centroids totally costs "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " milliseconds." << std::endl;
        t0 = std::chrono::high_resolution_clock::now();
        split_into_limited_cluster(avg_dists, distances);
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Init::the " << iteration_cnt + 1 << "th iteration split_into_limited_cluster totally costs "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " milliseconds." << std::endl;
        std::sort(avg_dists.begin(), avg_dists.end(), [](const auto &l, const auto &r) {
            return l.second < r.second;
        });
        std::cout << "show avg_dists:" << std::endl;
        avgsum = 0.0;
        for (auto i = 0; i < avg_dists.size(); i ++) {
            std::cout << "(" << avg_dists[i].first << ", " << avg_dists[i].second << ") ";
            avgsum += avg_dists[i].second;
        }
        std::cout << std::endl;
        std::cout << "avg_dists.min = " << avg_dists[0].second
                  << ", avg_dists.mid = " << avg_dists[avg_dists.size() / 2].second
                  << ", avg_dists.avg = " << avgsum / avg_dists.size()
                  << ", avg_dists.max = " << avg_dists[avg_dists.size() - 1].second << std::endl;
        threshold = avg_dists[avg_dists.size() / 2].second * scale_coefficient_;
        iteration_cnt ++;
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Init::the " << iteration_cnt << "th iteration totally costs "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " milliseconds." << std::endl;
        if (iteration_cnt > 10) {
            std::cout << "error: too much iteration of resize clusters, better update arguments." << std::endl;
            break;
        }
    }

    /*
     * main ideas:
     * 1. calculate the new centroid of each cluster(dame)
     * 2. calculate the distance to each centroid of all points(dame)
     * 3. calculate sum_dis(use the invertlist_ and distances in step 2)(get from split_into_limited_cluster)
     * 4. count the nearest centroid of the points in the clusters that sum_dis > threshold
     * 5. split the clusters in step 4
     * 6. go to step 2 until there is no sum_dis > threshold in step 4
     */
}

void
Cluster::HealthyCheck() {
    assert(invertlist_.size() == nc_);
    std::vector<bool> flag(nb_, false);
    for (auto i = 0; i < nc_; i ++) {
        if (invertlist_[i].size() > cs_) {
            std::cout << "HealthyCheck error: the " << i << "th cluster's size is " << invertlist_[i].size()
                      << ", which is over limited size: " << cs_ << std::endl;
        }
        for (auto j = 0; j < invertlist_[i].size(); j ++) {
            if (invertlist_[i][j] < 0 || invertlist_[i][j] > nb_) {
                std::cout << "HealthyCheck error: the " << j << "th elem in cluster " << i
                          << " is " << invertlist_[i][j] << ", which is invalid!" << std::endl;
            }
            if (flag[invertlist_[i][j]]) {
                std::cout << "HealthyCheck error: the " << j << "th elem in cluster " << i
                          << " is duplicate, whose id is " << invertlist_[i][j] << std::endl;
            }
            flag[invertlist_[i][j]] = true;
        }
    }
    for (auto i = 0; i < nb_; i ++) {
        if (!flag[i]) {
            std::cout << "HealthyCheck error: element " << i << " has not apperes in inverstlist" << std::endl;
        }
    }
}

void
Cluster::ShowCSHistogram() {
    int mx_sz = 0;
    std::unordered_map<int, int> hist_map;
    for (auto &cls : invertlist_) {
        if (hist_map.find(cls.size()) != hist_map.end())
            hist_map[cls.size()] ++;
        else
            hist_map[cls.size()] = 1;
        mx_sz = std::max(mx_sz, (int)cls.size());
    }
    size_t sssums = 0;
    std::vector<int> histogram(mx_sz + 1, 0);
    int summmm1 = 0;
    for (auto &pr : hist_map) {
        histogram[pr.first] = pr.second;
        summmm1 += pr.first * pr.second;
    }
    for (int kkkkk = (int)histogram.size() - 1; kkkkk >= 0; kkkkk --) {
        if (histogram[kkkkk] > 0) {
            sssums += histogram[kkkkk] * kkkkk;
            std::cout << "there are " << histogram[kkkkk] << " clusters whose size is " << kkkkk << std::endl;
        }
    }
    std::cout << "ShowCSHistogram: sums = " << sssums << ", summm = " << summmm1 << ", nb_ = " << nb_ << std::endl;
}

/*
 *
 */
void
Cluster::Train() {
    /*
     * step1: calculate the new centroid of each cluster
     * step2: calculate the sum_dis of each cluster
     * step3: split the cluster with bit sum_dis if needed
     * step4: re-calculate distances between centroids and points in base
     * step5:
     */
}

/*
 * the points in inverlist_ has already in ascending order, so we can get the init 2-means points of the 1/3th and
 * 2/3th points. than go to iteration.
 */
void
Cluster::two_means(const int cluster_id, std::vector<std::vector<float>>& cts, std::vector<std::vector<size_t>>& list) {
    auto cls = invertlist_[cluster_id].size();
    auto pd = (float*)data_;
    cts.resize(2);
    list.resize(2);
    cts[0].resize(dim_);
    cts[1].resize(dim_);
    memcpy(cts[0].data(), pd + invertlist_[cluster_id][cls / 3] * dim_, dim_ * sizeof(float));
    memcpy(cts[1].data(), pd + invertlist_[cluster_id][cls * 2 / 3] * dim_, dim_ * sizeof(float));
    int iteration_time = 0;
    do {
        std::vector<float> sum_dists(2, 0.0);
        std::vector<float> diss(cls, std::numeric_limits<float>::max());
        list[0].clear();
        list[1].clear();
//    std::vector<int>
        for (auto i = 0; i < cls; i ++) {
            auto dist1 = disf_(cts[0].data(), pd + invertlist_[cluster_id][i] * dim_, &dim_);
            auto dist2 = disf_(cts[1].data(), pd + invertlist_[cluster_id][i] * dim_, &dim_);
            list[(int)(dist1 > dist2)].push_back(invertlist_[cluster_id][i]);
            sum_dists[(int)(dist1 > dist2)] += std::min(dist1, dist2);
        }
        std::cout << "in two means, the " << iteration_time << "th iteration, err = " << sum_dists[0] + sum_dists[1]
                  << ", err0 = " << sum_dists[0] << ", err1 = " << sum_dists[1] << std::endl;
        for (auto i = 0; i < 2; i ++) {
            for (auto j = 0; j < dim_; j ++)
                cts[i][j] = 0.0;
            for (auto j = 0; j < list[i].size(); j ++) {
                for (auto k = 0; k < dim_; k ++)
                    cts[i][k] += *(pd + list[i][j] * dim_ + k);
            }
            for (auto j = 0; j < dim_; j ++)
                cts[i][j] /= list[i].size();
        }
        iteration_time ++;
    } while (iteration_time < 3);
    std::cout << "two-means in cluster " << cluster_id << " done." << std::endl;
}

void
Cluster::calculate_dist2centroids(std::vector<std::vector<std::pair<int, float>>>& distances,
                                  bool full_distances) {

    assert(centroids_.size() == nc_);
    // todo: parallelize
#pragma omp parallel for schedule(static, 4096)
    for (auto i = 0; i < nb_; i ++) {
        for (auto j = 0; j < nc_; j ++) {
            // todo: advance continue
            auto disij = disf_(centroids_[j].data(), (float*)data_ + i * dim_, &dim_);
            if (std::isnan(disij)) {
                std::cout << "error: nan occurs!" << std::endl;
                std::cout << "show centroid[j]:" << std::endl;
                for (auto fff = 0; fff < dim_; fff ++)
                    std::cout << centroids_[j][fff] << " ";
                std::cout << std::endl;
                std::cout << "show the " << i << " vector:" << std::endl;
                auto pfpfpf = (float*)data_ + i * dim_;
                for (auto iff = 0; iff < dim_; iff ++)
                    std::cout << pfpfpf[iff] << " ";
                std::cout << std::endl;
            }
            assert(!std::isnan(disij));
            if (full_distances) {
                distances[i].push_back(std::pair<int, float>(j, disij));
            } else {
                if (!distances[i].empty()) {
                    if (disij < distances[i][0].second) {
                        distances[i][0].second = disij;
                        distances[i][0].first = j;
                    }
                } else {
                    distances[i].push_back(std::pair<int, float>(j, disij));
                }
            }
        }
    }
}

void
Cluster::calculate_centroids(std::vector<std::vector<size_t>>& ivl, bool clean) {
    // todo: parallelize
#pragma omp parallel for schedule(static, 8)
    for (auto i = 0; i < nc_; i ++) {
        for (auto k = 0; k < dim_; k ++)
            centroids_[i][k] = 0.0;
        for (auto j = 0; j < ivl[i].size(); j ++) {
            auto pd = (float*)data_ + ivl[i][j] * dim_;
            for (auto k = 0; k < dim_; k ++) {
                centroids_[i][k] += pd[k];
            }
        }
        for (auto k = 0; k < dim_; k ++)
            centroids_[i][k] /= ivl[i].size();
        if (clean)
            ivl[i].clear();
    }
}

/*
 * preemptive divide
 */
void
Cluster::split_into_limited_cluster(std::vector<std::pair<int, float>>& sum_dists,
                                    std::vector<std::vector<std::pair<int, float>>>& distances) {
    float sum_dis = 0.0;
    assert(sum_dists.size() == nc_);
    auto tstart = std::chrono::high_resolution_clock::now();
    std::vector<int> cluster_size(sum_dists.size(), cs_);
    for (auto i = 0; i < sum_dists.size(); i ++) {
        sum_dists[i].first = i;
        sum_dists[i].second = 0.0;
    }
    claim::Heap<false, std::pair<std::pair<int, float>*, size_t>> hp(nb_);
    auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 5)
    for (size_t i = 0; i < nb_; i ++) {
        sort(distances[i].begin(), distances[i].end(), [](const auto &l, const auto &r) {
            return l.second < r.second;
        });
#pragma omp critical
        {
            hp.Push(std::pair<std::pair<int, float>*, size_t>(distances[i].data(), i));
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "split_into_limited_cluster: sort and push into heap totally costs "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " milliseconds." << std::endl;
    invertlist_.resize(nc_);
    for (auto &lst : invertlist_) {
        lst.reserve(cs_);
        lst.clear();
    }
    t0 = std::chrono::high_resolution_clock::now();
    do {
        auto pr = hp.Pop();
        if (cluster_size[pr.first->first]) {
            invertlist_[pr.first->first].push_back(pr.second);
//            sum_dis += pr.first->second;
            cluster_size[pr.first->first] --;
            sum_dists[pr.first->first].second += pr.first->second;
        } else {
            pr.first ++;
            hp.Push(pr);
        }
    } while (!hp.Empty());
//    std::cout << "show sum_dists:" << std::endl;
    for (auto &sd : sum_dists) {
//        std::cout << "(" << sd.first << ", " << sd.second << ") ";
        sum_dis += sd.second;
        assert(cs_ - cluster_size[sd.first] != 0);
        sd.second /= (cs_ - cluster_size[sd.first]);
    }
//    std::cout << std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "split_into_limited_cluster: split points into cluster totally costs "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " milliseconds." << std::endl;
    std::cout << "split_into_limited_cluster:: sum_dis = " << sum_dis << std::endl;
    auto tend = std::chrono::high_resolution_clock::now();
    std::cout << "split_into_limited_cluster done in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count()
              << " milliseconds." << std::endl;
}

void
Cluster::Split(size_t max_iter_times) {
    int loop_cnt = 0;
    do {
        auto tstart = std::chrono::high_resolution_clock::now();
        auto do_spt = split(max_iter_times);
        auto tend = std::chrono::high_resolution_clock::now();
        if (do_spt == 0)
            break;
        loop_cnt ++;
        std::cout << "the " << loop_cnt << "th loop split " << do_spt << " clusters, costs "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count()
                  << " milliseconds." << std::endl;
    } while (true);
    calculate_err();
}

void
Cluster::Query(const float* pquery,
               size_t nq,
               size_t topk,
               size_t nprobe,
               std::vector<std::vector<size_t>>& ids,
               std::vector<std::vector<float>>& dist) {
    ids.resize(nq);
    dist.resize(nq);
    if (nprobe == 0) {
//        nprobe = (nc_ / 100 + (nc_ % 100 > 50)) * 2;
        nprobe = nc_ * 2 / 100;
    }
    std::cout << "Cluster::Query: nq = " << nq << ", topk = " << topk << ", nprobe = " << nprobe << std::endl;
    std::vector<int> vis_cnt(nq, 0);
#pragma omp for schedule(dynamic, 1)
    for (auto i = 0; i < nq; i ++) {
        vis_cnt[i] = query(pquery + i * dim_, topk, nprobe, ids[i], dist[i]);
    }
    int mx = vis_cnt[0];
    int mn = vis_cnt[0];
    int sm = vis_cnt[0];
    for (auto i = 1; i < nq; i ++) {
        if (mx < vis_cnt[i])
            mx = vis_cnt[i];
        if (mn > vis_cnt[i])
            mn = vis_cnt[i];
        sm += vis_cnt[i];
    }
    std::cout << nq << " queries totally access " << sm << " points."
              << " min = " << mn
              << " max = " << mx
              << " avg = " << (double)sm / nq
              << std::endl;
}

int
Cluster::query(const float* pquery, size_t topk, size_t nprobe, std::vector<size_t>& ids, std::vector<float>& dist) {
    int ret = 0;
    std::vector<std::pair<int, float>> cluster_ids;
    cluster_ids.reserve(nc_);
    for (auto i = 0; i < nc_; i ++) {
        auto dists = disf_(centroids_[i].data(), pquery, &dim_);
        cluster_ids.emplace_back(std::pair<int, float>(i, dists));
    }
    std::sort(cluster_ids.begin(), cluster_ids.end(), [](const auto &l, const auto &r) {
        return l.second < r.second;
    });
    auto pd = (float*) data_;
    std::vector<std::pair<size_t, float>> unsort_ans;
    unsort_ans.reserve(topk * 100);
    for (auto iii = 0; iii < nprobe; iii ++) {
        auto pr = cluster_ids[iii];
        ret += invertlist_[pr.first].size();
        for (auto i = 0; i < invertlist_[pr.first].size(); i ++) {
            auto dist = disf_(pquery, pd + invertlist_[pr.first][i] * dim_, &dim_);
            unsort_ans.emplace_back(invertlist_[pr.first][i], dist);
        }
    }
    std::sort(unsort_ans.begin(), unsort_ans.end(), [] (const auto &l, const auto &r) {
        return l.second < r.second;
    });
    auto len = std::min(topk, unsort_ans.size());
    ids.resize(len);
    dist.resize(len);
    for (auto i = 0; i < len; i ++)
        ids[i] = unsort_ans[i].first, dist[i] = unsort_ans[i].second;
    return ret;
}

void
Cluster::calculate_err() {
    std::vector<float> errs(nc_, 0.0);
    float tot_err = 0.0;
    auto pd = (float*)data_;
#pragma omp parallel for schedule(static, 64)
    for (auto i = 0; i < nc_; i ++) {
        for (auto j = 0; j < invertlist_[i].size(); j ++) {
            errs[i] += disf_(centroids_[i].data(), pd + invertlist_[i][j] * dim_, &dim_);
        }
    }
    for (auto i = 0; i < errs.size(); i ++)
        tot_err += errs[i];
    std::sort(errs.begin(), errs.end());
    std::cout << "calculate_err: "
              << " min = " << errs[0]
              << ", mid = " << errs[errs.size() / 2]
              << ", avg = " << tot_err / nc_
              << ", max = " << errs[nc_ - 1]
              << ", tot_err = " << tot_err
              << std::endl;
}

void
Cluster::clean_empty_cluster(std::vector<std::vector<size_t>>& ivl) {
    assert(ivl.size() == nc_);
    do {
        int i, j;
        for (i = 0; i < nc_; i ++) {
            if (ivl[i].size() == 0) {
                break;
            }
        }
        if (i == nc_) {
            return;
        }
        int mx = 0;
        j = i;
        for (auto ii = 0; ii < nc_; ii ++) {
            if (mx < ivl[ii].size()) {
                mx = ivl[ii].size();
                j = ii;
            }
        }
        std::cout << "the " << i << "th cluster.size = 0, split the " << j << "th cluster." << std::endl;
        std::vector<std::vector<size_t>> ils;
        std::vector<std::vector<float>> cent;
        two_means(j, cent, ils);
        centroids_[i].swap(cent[0]);
        centroids_[j].swap(cent[1]);
        invertlist_[i].swap(ils[0]);
        invertlist_[j].swap(ils[1]);
    } while (true);
}
} // namespace claim
