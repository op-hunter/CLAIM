#pragma once

#include <iostream>
#include <vector>
#include "kmeans.h"
#include "distance.h"


namespace claim {

class Cluster {
 public:
    Cluster(size_t nb, size_t dim, size_t cs, size_t max_iter_time);
    ~Cluster();

    // load points in binary format
    void Load(std::string& source_data_path);
    void SetData(char* pdata);
    void SizeLimited();
    void Train();
    void HealthyCheck();
    void ShowCSHistogram();
    void Kmeans(size_t max_iter_times = 10);
    void Split(size_t max_iter_times = 10); // kmeans first and then split the big cluster into smaller
    void Slice(size_t max_iter_times = 10); // split the dataset first and then do kmeans
    void Query(const float* pquery, size_t nq, size_t topk, size_t nprobe, std::vector<std::vector<size_t>>& ids, std::vector<std::vector<float>>& dist);

 private:
    void split_into_limited_cluster(std::vector<std::pair<int, float>>& sum_dists,
                                    std::vector<std::vector<std::pair<int, float>>>& distances);

    void calculate_centroids(std::vector<std::vector<size_t>>& ivl, bool clean);

    void calculate_dist2centroids(std::vector<std::vector<std::pair<int, float>>>& distances, bool full_distances);

    void two_means(const int cluster_id, std::vector<std::vector<float>>& cts, std::vector<std::vector<size_t>>& list);
    int split(size_t max_iter_times = 10);
    int query(const float* pquery, size_t topk, size_t nprobe, std::vector<size_t>& ids, std::vector<float>& dist);
    void calculate_err();
    void clean_empty_cluster();

 private:
    size_t nb_; // num of points in base
    size_t dim_; // dimension
    size_t cs_; // cluster size
    size_t nc_; // num of clusters;
    size_t iter_; // maximum times of iteration
    float scale_coefficient_; // the coefficient of threshold to decide imbalance clusters
    size_t max_hot_pts_; // max num of re-construct clusters
    size_t top_hot_cls_; // max num of cluster take into statistics
    std::vector<std::vector<float>> centroids_; // centroids of clusters
    std::vector<std::vector<size_t>> invertlist_; // ids of each cluster
    char* data_;
    p_dis disf_;
    bool own_data_;
};

} // namespace claim
