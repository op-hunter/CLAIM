#include "distance.h"




namespace claim {

float L2Square(const void* opa, const void* opb, const void* len) {
    float ret = 0.0;
    auto pa = (float*)(opa);
    auto pb = (float*)(opb);
    for (auto i = 0; i < *(size_t*)len; i ++) {
        auto diff = pa[i] - pb[i];
        ret += diff * diff;
    }
    return ret;
}

} // namespace claim
