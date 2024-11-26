#pragma once

#include <torch/extension.h>

at::Tensor text_sim(
    at::Tensor elements,
    at::Tensor test_elements,
    at::Tensor element_weights,
    at::Tensor test_element_weights,
    int padding_idx);