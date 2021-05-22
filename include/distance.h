#pragma once
#include <iostream>

namespace claim {

using p_dis = float(*)(const void*, const void*, const void*);


float L2Square(const void* opa, const void* opb, const void* len);

} // namespace claim
