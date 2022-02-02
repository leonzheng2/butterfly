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


import matplotlib.pyplot as plt
import scipy
import numpy as np
import time
import argparse

from src.iterative_grad_fact import ButterflyFact_BuiltinComplex
import src.utils as utils
import src.hierarchical_fact as fact


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment to compare running time and approximation error between gradient based method "
                    "hierarchical factorization")
    parser.add_argument("--num_factors", type=int, default=3, help="Number of factors in the matrix factorization.")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--results_path", type=str, default="./results.npz")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    size = 2 ** args.num_factors
    sigma = 0.1
    fact_mat = scipy.linalg.dft(size)
    print(f"Norm of the DFT matrix: {np.linalg.norm(fact_mat)}")

    # Build instance to run iterative gradient-based algorithm
    print("\nITERATIVE GRADIENT-BASED FACTORIZATION")
    bf = ButterflyFact_BuiltinComplex(args.num_factors, fact_mat, True)
    running_time, accuracy, loss_array, running_time_per_iter = bf.training(
        args.learning_rate,
        optimize='Adam',
        num_iter=50,
        time_record=True
    )
    loss_array = np.array([np.sqrt(2 * t.item()) for t in loss_array])
    log_loss_array = np.log10(loss_array)
    running_time_per_iter = np.cumsum(np.array(running_time_per_iter))
    print(f"Approximation error computed as || Z - BP ||: {loss_array[-1]}")
    print(f"Factorization time: {running_time[0]}")

    # Hierarchical factorization methods
    print("\nHIERARCHICAL BALANCED FACTORIZATION METHOD")
    best_balanced = np.inf
    begin_balanced = time.time()
    for perm in ["000", "001", "010", "011", "100", "101", "110", "111"]:
        p = utils.get_permutation_matrix(args.num_factors, perm)
        product, factors = fact.project_BP_model_P_fixed(fact_mat, "balanced", p, -1, return_factors=True)
        loss = utils.error_cal(factors + [p], fact_mat, relative=False)
        best_balanced = min(best_balanced, loss)
    end_balanced = time.time()
    print("Approximation error computed as || Z - BP ||:", best_balanced)
    print("Factorization time:", end_balanced - begin_balanced)

    print("\nHIERARCHICAL UNBALANCED FACTORIZATION METHOD")
    best_comb = np.inf
    begin_comb = time.time()
    for perm in ["000", "001", "010", "011", "100", "101", "110", "111"]:
        p = utils.get_permutation_matrix(args.num_factors, perm)
        product, factors = fact.project_BP_model_P_fixed(fact_mat, "comb", p, -1, return_factors=True)
        loss = utils.error_cal(factors + [p], fact_mat, relative=False)
        best_comb = min(best_comb, loss)
    end_comb = time.time()
    print("Approximation error computed as || Z - BP ||:", best_comb)
    print("Factorization time:", end_comb - begin_comb)

    # Save and plot results
    np.savez(
        args.results_path,
        loss1=loss_array, time1=running_time_per_iter,
        loss2=np.array(best_balanced), time2=np.array(end_balanced - begin_balanced),
        loss3=np.array(best_comb), time3=np.array(end_comb - begin_comb))

    # Log scale
    plt.plot(running_time_per_iter[:50], log_loss_array[:50], color='red', label="ADAM")
    plt.plot(running_time_per_iter[-20:], log_loss_array[-20:], color='blue', label="LBFGS")
    plt.plot([running_time_per_iter[49], running_time_per_iter[50]], [log_loss_array[49], log_loss_array[50]], color="blue")
    plt.scatter(end_balanced - begin_balanced, np.log10(best_balanced), color="black", label="Balanced")
    plt.scatter(end_comb - begin_comb, np.log10(best_comb), color="green", label="Unbalanced")
    plt.legend()
    plt.xlabel("Running time (seconds)")
    plt.ylabel("Logarithm of loss function")
    plt.show()
