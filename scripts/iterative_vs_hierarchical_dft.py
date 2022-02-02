# BSD 3-Clause License
#
# Copyright (c) 2022, ZHENG Leon
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from src import utils
from src import hierarchical_fact as fact
from src.iterative_grad_fact import ButterflyFact_BuiltinComplex

import numpy as np
import scipy
import argparse
import math
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment to compare the hierarchical factorization methods (with balanced or unbalanced tree)"
                    "vs. first-order optimization method. "
                    "Loss is the Froebenius norm between the target matrix and the computed factorization."
                    "We use our own implementation of a first-order optimization method. "
                    "The target matrix is the DFT matrix."
                    "It can be noisy, i.e. we add iid Gaussian white noise on the entries. "
                    "The hierarchical factorization method is performed 8 times, for each of the 8 permutation matrix "
                    "in the BP model. See README.md for more details on these 8 permutations.")
    parser.add_argument("--k", type=int, default=3, help="Number of factors.")
    parser.add_argument("--max_depth", default=-1, type=int, help="Depth of the tree. It corresponds to how deep we go "
                                                                  "in the hierarchical factorization method. "
                                                                  "-1 is full depth.")
    parser.add_argument("--trial", type=int, default=10, help="Number of times we repeat the experiments, by sampling "
                                                              "different noisy version of the target matrix.")
    parser.add_argument("--sigma", type=float, default=0.01, help="Standard deviation on the Gaussian white noise to "
                                                                  "sample the noisy version of the target matrix.")
    parser.add_argument("--results_path", type=str, default="./results.npz")
    return parser.parse_args()


if __name__ == "__main__":
    arg = parse_args()
    acc_result_iterative = []
    time_result_iterative = []
    acc_result_b = []
    time_result_b = []
    acc_result_u = []
    time_result_u = []
    noise_level = []
    size = 2 ** arg.k
    noise = arg.sigma * 0.5
    lr = 0.1

    for trial in range(arg.trial):
        print(f"\nEXP_ID: {trial+1}/{arg.trial}")
        # Method in fast learning butterfly paper
        fact_mat = scipy.linalg.dft(size) + np.random.randn(size, size) * math.sqrt(noise) + 1j * np.random.randn(size, size) * math.sqrt(noise)
        noise_level.append(np.linalg.norm(fact_mat - scipy.linalg.dft(size)))
        bf = ButterflyFact_BuiltinComplex(arg.k, fact_mat, True)
        running_time, accuracy, _, _ = bf.training(lr, optimize='Adam', num_iter=50)
        loss = np.sqrt(2 * accuracy[0].item())
        print(f"Iterative gradient descent method: loss={loss}, time={running_time[0]}")
        acc_result_iterative.append(loss)
        time_result_iterative.append(running_time[0])
        
        # Our method
        best_b = np.inf
        begin_b = time.time()
        for perm in ["000", "001", "010", "011", "100", "101", "110", "111"]:
            p = utils.get_permutation_matrix(arg.k, perm)
            product, factors = fact.project_BP_model_P_fixed(fact_mat, "balanced", p, -1, return_factors=True)
            loss = utils.error_cal(factors + [p], fact_mat, relative = False)
            best_b = min(best_b, loss)
        end_b = time.time()
        print(f"Balanced hierarchical factorization: loss={best_b}, time={end_b - begin_b}")
        acc_result_b.append(best_b)
        time_result_b.append(end_b - begin_b)

        best_u = np.inf
        begin_u = time.time()
        for perm in ["000", "001", "010", "011", "100", "101", "110", "111"]:
            p = utils.get_permutation_matrix(arg.k, perm)
            product, factors = fact.project_BP_model_P_fixed(fact_mat, "comb", p, -1, return_factors=True)
            loss = utils.error_cal(factors + [p], fact_mat, relative = False)
            best_u = min(best_u, loss)
        end_u = time.time()

        print(f"Unbalanced hierarchical factorization: loss={best_u}, time={end_u - begin_u}")
        acc_result_u.append(best_u)
        time_result_u.append(end_u - begin_u)

    print("\nFINAL RESULTS")
    print("Noise level:")
    print(noise_level)
    print("Loss for gradient iterative method:")
    print(acc_result_iterative)
    print("Time for gradient iterative method:")
    print(time_result_iterative)
    print("Loss for hierarchical balanced method:")
    print(acc_result_b)
    print("Time for hierarchical balanced method:")
    print(time_result_b)
    print("Loss for hierarchical unbalanced method:")
    print(acc_result_u)
    print("Time for hierarchical unbalanced method:")
    print(time_result_u)

    np.savez(
        arg.results_path,
        noise=np.array(noise_level),
        acc_iter=np.array(acc_result_iterative), time_iter=np.array(time_result_iterative),
        acc_balanced=np.array(acc_result_b), time_balanced=np.array(time_result_b),
        acc_unbalanced=np.array(acc_result_u), time_unbalanced=np.array(time_result_u)
    )
