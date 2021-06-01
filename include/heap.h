#pragma once
#include <iostream>
#include <vector>


namespace claim {

template <bool is_max_heap, typename ET>
class Heap {
 public:
    Heap(const size_t sz): sz_(0) {
        hp_.resize(sz + 10);
    }

    ET Top() {
        if (sz_ <= 0) {
            return nullptr;
        }
        return hp_[1];
    }
    ET Pop() {
        if (sz_ <= 0) {
            return ET();
        }
        ET ret = hp_[1];
        hp_[1] = hp_[sz_];
        sz_ --;
        for (auto i = 1; i <= sz_; ) {
            auto target_child = (i << 1);
            if (target_child > sz_) break;
            if ((i << 1 | 1) <= sz_ && cmp(hp_[i << 1 | 1], hp_[target_child]))
                target_child ++;
            if (cmp(hp_[target_child], hp_[i])) {
                std::swap(hp_[target_child], hp_[i]);
                i = target_child;
            } else {
                break;
            }
        }
        return ret;
    }
    void Push(ET et) {
        sz_ ++;
        if (hp_.size() <= sz_ + 1)
            hp_.resize(sz_ + 10);
        hp_[sz_] = et;
        for (auto i = sz_; i > 1; ) {
            if (cmp(hp_[i], hp_[i >> 1]))
                std::swap(hp_[i], hp_[i >> 1]);
            else
                break;
            i >>= 1;
        }
    }
    size_t Size() { return sz_; }
    bool Empty() { return sz_ == 0; }

    void Show() {
        for (auto i = 1; i <= sz_; i ++) {
            std::cout << "i = " << i << ", id = " << hp_[i].second
                      << ", first = " << hp_[i].first->first
                      << ", second = " << hp_[i].first->second
                      << std::endl;
        }
    }

 private:
    bool cmp(ET& a, ET& b) {
        if (is_max_heap)
            return a.first->second > b.first->second;
        else
            return a.first->second < b.first->second;
    }
    size_t sz_;
    std::vector<ET> hp_;
};

} // namespace claim
