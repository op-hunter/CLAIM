#include <iostream>
#include <string>
#include <cluster.h>
#include <chrono>
#include <fstream>
#include "kmeans.h"
#include "distance.h"

const int SZ = 0;
void LoadQ(char** pquery, size_t &nq) {
    std::string query_file = "/home/zilliz/workspace/data/sift1m_query.bin";
    std::ifstream fin(query_file.c_str(), std::ios::binary);

    unsigned unq, unqd;
    fin.read((char*)&unq, 4);
    fin.read((char*)&unqd, 4);
    nq = unq;
    *pquery = (char*) malloc(nq * unqd * sizeof(float));
    fin.read(*pquery, nq * unqd * sizeof(float));
    fin.close();
    std::cout << "read query done, nq = " << nq << ", dim = " << unqd << std::endl;
}

void Recall(std::vector<std::vector<std::pair<float, size_t>>>& groundtruth,
            std::vector<std::vector<float>>& resultset,
            size_t topk, size_t nq) {

    int tot_cnt = 0;
    std::cout << "recall@" << topk << ":" << std::endl;
    for (unsigned i = 0; i < nq; i ++) {
        int cnt = 0;
        // std::cout << "groundtruth[i][k - 1].first = " << groundtruth[i][k - 1].first << std::endl;
        for (size_t j = 0; j < topk; j ++) {
            if (resultset[i][j] <= groundtruth[i][topk - 1].first)
                cnt ++;
        }
        // std::cout << "cnt = " << cnt << std::endl;
        tot_cnt += cnt;
        std::cout << "query " << i + 1 << " recall@" << topk << " is: " << ((double)(cnt)) / topk * 100 << "%." << std::endl;
    }
    std::cout << "avg recall@100 = " << ((double)tot_cnt) / topk / nq * 100 << "%." << std::endl;
}

int main() {
    char *pquery = nullptr;
    size_t k = 100;
    size_t nq;
    LoadQ(&pquery, nq);
    std::vector<std::vector<std::pair<float, size_t>>> groundtruth;
    groundtruth.resize(nq);
    size_t sz;
    std::ifstream finn("/home/zilliz/workspace/data/sift_ground_truth_100.bin", std::ios::binary);
    for (unsigned i = 0; i < nq; i ++) {
        finn.read((char*)&sz, 8);
//        std::cout << "query " << i + 1 << " has " << sz << " groundtruth ans." << std::endl;
        groundtruth[i].resize(sz);
        finn.read((char*)groundtruth[i].data(), k * 16);
    }
    finn.close();
    std::cout << "show groundtruth:" << std::endl;
    for (auto i = 0; i < nq; i ++) {
        std::cout << "top1: (" << groundtruth[i][0].second << ", " << groundtruth[i][0].first
                  << "), topk: " << groundtruth[i][k - 1].second << ", " << groundtruth[i][k - 1].first
                  << ")" << std::endl;
    }
    // test kmeans
    {
        claim::Cluster kmeans(1000000, 128, SZ, 5);
        std::string datafile = "/home/zilliz/workspace/data/sift1m.bin";
        auto tstart = std::chrono::high_resolution_clock::now();
        auto t0 = std::chrono::high_resolution_clock::now();
        kmeans.Load(datafile);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "kmeans::Load done in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>( t1 - t0 ).count()
                  << " milliseconds." << std::endl;
        std::cout << "Load Done!" << std::endl;
        t0 = std::chrono::high_resolution_clock::now();
        kmeans.Kmeans();
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "kmeans::Kmeans done in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>( t1 - t0 ).count()
                  << " milliseconds." << std::endl;
        std::cout << "Kmeans Done!" << std::endl;
//        t0 = std::chrono::high_resolution_clock::now();
//        kmeans.HealthyCheck();
//        t1 = std::chrono::high_resolution_clock::now();
//        std::cout << "kmeans::HealthyCheck done in "
//                  << std::chrono::duration_cast<std::chrono::milliseconds>( t1 - t0 ).count()
//                  << " milliseconds." << std::endl;
//        std::cout << "HealthyCheck Done!" << std::endl;
        t0 = std::chrono::high_resolution_clock::now();
        kmeans.ShowCSHistogram();
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "kmeans::ShowCSHistogram done in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>( t1 - t0 ).count()
                  << " milliseconds." << std::endl;
        std::cout << "ShowCSHistogram Done!" << std::endl;
        auto tend = std::chrono::high_resolution_clock::now();
        std::cout << "kmeans totally done in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>( tend - tstart ).count()
                  << " milliseconds." << std::endl;

        std::vector<std::vector<size_t>> ids;
        std::vector<std::vector<float>> dis;
        std::cout << "Do Query after kmeans:" << std::endl;
        t0 = std::chrono::high_resolution_clock::now();
        kmeans.Query((float*)pquery, nq, k, 0, ids, dis);
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "kmeans::Query1 done in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>( t1 - t0 ).count()
                  << " milliseconds." << std::endl;
        std::cout << "show ans of kmeans:" << std::endl;
        for (auto i = 0; i < nq; i ++) {
            std::cout << "top1: (" << ids[i][0] << ", " << dis[i][0]
                      << ") topk: (" << ids[i][k - 1] << ", " << dis[i][k - 1] << ")" << std::endl;
        }
        Recall(groundtruth, dis, k, nq);

        /*
        std::cout << "now do split:" << std::endl;
        t0 = std::chrono::high_resolution_clock::now();
        kmeans.Split(4);
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "kmeans::Split done in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>( t1 - t0 ).count()
                  << " milliseconds." << std::endl;
        kmeans.HealthyCheck();
        t0 = std::chrono::high_resolution_clock::now();
        kmeans.ShowCSHistogram();
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "kmeans::ShowCSHistogram done in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>( t1 - t0 ).count()
                  << " milliseconds." << std::endl;
        auto tendd = std::chrono::high_resolution_clock::now();
        std::cout << "kmeans + split totally done in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>( tendd - tstart ).count()
                  << " milliseconds." << std::endl;

        std::cout << "Do Query after re-cluster:" << std::endl;
        t0 = std::chrono::high_resolution_clock::now();
        kmeans.Query((float*)pquery, nq, k, 0, ids, dis);
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "kmeans::Query2 done in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>( t1 - t0 ).count()
                  << " milliseconds." << std::endl;
        std::cout << "show ans of kmeans.Query:" << std::endl;
        for (auto i = 0; i < nq; i ++) {
            std::cout << "top1: (" << ids[i][0] << ", " << dis[i][0]
                      << ") topk: (" << ids[i][k - 1] << ", " << dis[i][k - 1] << ")" << std::endl;
        }
        Recall(groundtruth, dis, k, nq);
        */
    }
    // test claim
    /*
    {
        claim::Cluster cmli(1000000, 128, SZ, 5);
        std::string datafile = "/home/zilliz/workspace/data/sift1m.bin";
        auto tstart = std::chrono::high_resolution_clock::now();
        auto t0 = std::chrono::high_resolution_clock::now();
        cmli.Load(datafile);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "cmli::Load done in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>( t1 - t0 ).count()
                  << " milliseconds." << std::endl;
        std::cout << "Load Done!" << std::endl;
        t0 = std::chrono::high_resolution_clock::now();
        cmli.SizeLimited();
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "cmli::Init done in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>( t1 - t0 ).count()
                  << " milliseconds." << std::endl;
        std::cout << "Init Done!" << std::endl;
        t0 = std::chrono::high_resolution_clock::now();
        cmli.HealthyCheck();
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "cmli::HealthyCheck done in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>( t1 - t0 ).count()
                  << " milliseconds." << std::endl;
        std::cout << "HealthyCheck Done!" << std::endl;
        t0 = std::chrono::high_resolution_clock::now();
        cmli.ShowCSHistogram();
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "cmli::ShowCSHistogram done in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>( t1 - t0 ).count()
                  << " milliseconds." << std::endl;
        std::cout << "ShowCSHistogram Done!" << std::endl;
        auto tend = std::chrono::high_resolution_clock::now();
        std::cout << "cmli totally done in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>( tend - tstart ).count()
                  << " milliseconds." << std::endl;
        std::vector<std::vector<size_t>> ids;
        std::vector<std::vector<float>> dis;
        std::cout << "Do Query after cmli.re-cluster:" << std::endl;
        t0 = std::chrono::high_resolution_clock::now();
        cmli.Query((float*)pquery, nq, k, 0, ids, dis);
        t1 = std::chrono::high_resolution_clock::now();
        std::cout << "cmli::Query done in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>( t1 - t0 ).count()
                  << " milliseconds." << std::endl;
        std::cout << "show ans of cmli.Query:" << std::endl;
        for (auto i = 0; i < nq; i ++) {
            std::cout << "top1: (" << ids[i][0] << ", " << dis[i][0]
                      << ") topk: (" << ids[i][k - 1] << ", " << dis[i][k - 1] << ")" << std::endl;
        }
        Recall(groundtruth, dis, k, nq);
    }
    */

    free(pquery);
    return 0;
}
