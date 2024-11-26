import torch
from torch.utils.cpp_extension import load

# Compile and load the CUDA extension with Ninja
text_sim_cuda = load(
    name="text_sim_cuda",
    sources=["cuda/text_sim_cuda.cu"],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-lineinfo'],
    verbose=True
)

# Define test parameters
B, L, N, L2 = 2, 3, 2, 4  # Batch size, Sequence length, Number of test sets, Test set length
padding_idx = 8192

# Input tensors
elements = torch.tensor([ 456,  662, 1248, 1667, 1823, 2446, 2918, 2933, 3935, 4572, 4901, 5021,
        5444, 5491, 6445, 6672, 6810, 7110, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192, 8192, 8192, 8192, 8192], device='cuda:0',
       dtype=torch.int32).view(1,-1)
test_elements = torch.tensor([
        [ 102,  271,  486,  522,  617,  760,  783,  843,  937, 1051, 1139, 1307,
         1328, 1349, 1442, 1510, 1558, 1571, 1619, 1725, 1823, 1878, 1966, 2106,
         2127, 2278, 2304, 2309, 2530, 2533, 2653, 2822, 2933, 3047, 3241, 3442,
         3473, 3513, 3631, 3706, 3770, 3794, 3837, 3906, 4041, 4057, 4082, 4108,
         4185, 4221, 4258, 4280, 4459, 4519, 4546, 4572, 4595, 4762, 4899, 4949,
         4978, 5027, 5157, 5308, 5349, 5394, 5614, 5701, 5938, 5988, 6040, 6051,
         6093, 6101, 6129, 6145, 6172, 6274, 6506, 6516, 6582, 6606, 6748, 6943,
         7178, 7211, 7421, 7437, 7509, 7589, 7835, 7869, 7965, 8024, 8192, 8192,
         8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
         8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
         8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
         8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
         8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
         8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
         8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
         8192, 8192], 
        [141,  456,  662,  801, 1248, 1328, 1667, 1773, 1823, 2360, 2446, 2918,
        2933, 3080, 3134, 3871, 3935, 3948, 4057, 4082, 4290, 4549, 4572, 4595,
        4901, 5021, 5077, 5087, 5444, 5491, 6445, 6564, 6582, 6672, 6810, 7072,
        7110, 7138, 7211, 8179, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
        8192, 8192],
        [ 271,  564,  565,  663,  734,  762,  793,  871, 1063, 1151, 1183, 1328,
         1349, 1434, 1519, 1520, 1654, 1664, 1773, 1878, 1883, 2127, 2426, 2654,
         2711, 2804, 2822, 2923, 3189, 3241, 3267, 3401, 3780, 4057, 4082, 4185,
         4211, 4390, 4468, 4522, 4595, 5077, 5157, 5174, 5253, 5723, 5738, 5781,
         5978, 5982, 6016, 6172, 6324, 6467, 6489, 6547, 6582, 6748, 6804, 6816,
         7043, 7182, 7187, 7211, 7376, 7421, 7753, 7813, 7899, 8025, 8192, 8192,
         8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
         8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
         8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
         8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
         8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
         8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
         8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
         8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
         8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
         8192, 8192]], device='cuda:0', dtype=torch.int32)
# Weights
# element_weights = torch.tensor(
#     [[0.5, 1.0, 0.0], [1.5, 2.0, 0.0]], device="cuda", dtype=torch.float32
# )  # [B, L]
# test_element_weights = torch.tensor(
#     [[0.0, 1.2, 0.8, 0.5], [0.9, 0.7, 1.0, 0.6]], device="cuda", dtype=torch.float32
# )  # [N, L2]

element_weights = torch.ones_like(elements, device="cuda", dtype=torch.float32)
test_element_weights = torch.ones_like(test_elements, device="cuda", dtype=torch.float32)

# Run the kernel
output = text_sim_cuda.text_sim(
    elements, test_elements, element_weights, test_element_weights, padding_idx
)

# Compute the absolutely correct output using a basic Python implementation
def compute_correct_output(elements, test_elements, element_weights, test_element_weights, padding_idx):
    B, L = elements.shape
    N, L2 = test_elements.shape
    correct_output = torch.zeros((B, N), device="cuda", dtype=torch.float32)

    for b in range(B):
        for l in range(L):
            element = elements[b, l]
            if element == padding_idx:
                continue
            for n in range(N):
                for l2 in range(L2):
                    if test_elements[n, l2] == element:
                        correct_output[b, n] += element_weights[b, l] * test_element_weights[n, l2]
    return correct_output

# Compute the correct output
correct_output = compute_correct_output(elements, test_elements, element_weights, test_element_weights, padding_idx)

# Print inputs, kernel output, and correct output
print("Elements:\n", elements)
print("Element Weights:\n", element_weights)
print("Test Elements:\n", test_elements)
print("Test Element Weights:\n", test_element_weights)
print("Kernel Output (B x N):\n", output)
print("Correct Output (B x N):\n", correct_output)

# Compare the results
if torch.allclose(output, correct_output, atol=1e-6):
    print("The kernel output matches the correct output!")
else:
    print("There is a discrepancy between the kernel output and the correct output.")
