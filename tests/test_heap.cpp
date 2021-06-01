#include <iostream>
#include <algorithm>
#include <random>
#include <vector>
#include <cassert>
#include "heap.h"

const int n = 5;
const int m = 10;

int main() {
    std::vector<std::vector<std::pair<int, float>>> lst(n, std::vector<std::pair<int, float>>(m, std::pair<size_t, float>(0, 0.0)));
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    auto         x = rd();
    std::mt19937 generator(x);  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<int> distrii(0, 10);
    std::uniform_real_distribution<float> distrif(0, 100.0);
    for (auto i = 0; i < n; i ++) {
        for (auto j = 0; j < m; j ++) {
            lst[i][j].first = distrii(generator);
            lst[i][j].second = distrif(generator);
        }
    }
    for (auto i = 0; i < n; i ++) {
        sort(lst[i].begin(), lst[i].end(), [](const auto &l, const auto &r) {
            return l.second < r.second;
        });
    }
    std::cout << "show lists: " << std::endl;
    for (auto i = 0; i < n; i ++) {
        for (auto j = 0; j < m; j ++) {
            std::cout << "(" << lst[i][j].first << ", " << lst[i][j].second << ") ";
        }
        std::cout << std::endl;
    }
    claim::Heap<false, std::pair<std::pair<int, float>*, int>> hp(n);
    for (auto i = 0; i < n; i ++) {
        auto pp = (std::pair<int, float>*)lst[i].data();
        hp.Push(std::pair<std::pair<int, float>*, int>(pp, i));
    }
    assert(hp.Size() == n);
    std::cout << "show heap:" << std::endl;
    hp.Show();
    std::vector<int> cnt(n, 0);
    std::cout << "output all data in ascending order:" << std::endl;
    do {
        auto pr = hp.Pop();
//        assert(pr != nullptr);
        std::cout << "bucket id: " << pr.second << ", elem: ("
                  << pr.first->first << ", " << pr.first->second << ")"
                  << std::endl;
        cnt[pr.second] ++;
        if (cnt[pr.second] < m) {
            pr.first ++;
            hp.Push(pr);
        }
    } while (!hp.Empty());
    return 0;
}

