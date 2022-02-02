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


import time
import argparse
import src.utils as utils
import src.hierarchical_fact as fact
import scipy.linalg
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment to compare the hierarchical factorization method between two choices of trees:"
                    "comb vs. balanced. For each target matrix generate by the script, we perform a hierarchical "
                    "factorization for each choice of tree, and compare approximation errors and computation times."
                    "The target matrix Z to factorize is either: a random matrix having the butterfly structure, "
                    "the Hadamard matrix or the DFT matrix. The target can be noisy, with Gaussian noise, or noiseless.")
    parser.add_argument("--num_factors", type=int, default=6, help="Number of factors in the matrix factorization.")
    parser.add_argument("--noise_m", type=float, default=0., help="Mean of Gaussian noise on the target matrix.")
    parser.add_argument("--noise_std", type=float, default=0.1,
                        help="Standard deviation of Gaussian noise on the target matrix.")
    parser.add_argument("--matrix", choices=["random", "hadamard", "dft"], default="hadamard",
                        help="Target matrix to factorize. If random, the target matrix is a product of butterfly "
                             "factors whose entries are random Gaussian variables.")
    parser.add_argument("--main_m", type=float, default=0, help="if args.matrix is random, it is the mean value "
                                                                "of the entries sampled from a Gaussian distribution.")
    parser.add_argument("--main_std", type=float, default=1, help="if args.matrix is random, it is the std of the "
                                                                  "entries sampled from a Gaussian distribution.")
    parser.add_argument("--max_depth", default=-1, type=int, help="Depth of the tree. It corresponds to how deep we go "
                                                                  "in the hierarchical factorization method. "
                                                                  "-1 is full depth.")
    parser.add_argument("--repeat", type=int, default=10, help="Repeat the experiment several time, over different "
                                                               "noisy version of the target matrix.")
    parser.add_argument("--perm",
                        choices=["identity", "bit-reversal", "000", "001", "010", "011", "100", "101", "110", "111",
                                 "all_8"],
                        default="identity",
                        help="Choice of the permutation matrix in the BP model."
                             "It can only be one of the 8 permutations matrices, described by 3 binaries. "
                             "See README.md for more details."
                             "identity is the choice where P is the identity matrix, and corresponds to 000."
                             "bit-reversal is the choice where P is the bit-reversal matrix, and corresponds to 001."
                             "When choosing all_8, the factorization is performed 8 times, by fixing each of the eight "
                             "permutations. We only keep the best factorization among the eight at the end."
                        )
    parser.add_argument("--results_path", default="./results.pkl",
                        help="Path to save experiment results. Results are written in a pandas dataframe."
                             "For each row, we save the parameters of the experiment, the quality of the factorization "
                             "mesured as the distance (in Frobenius norm) between the target matrix and the product of "
                             "the computed factors via our factorization method, and finally the computation time.")
    parser.add_argument("--rewrite_results", action="store_true")
    return parser.parse_args()


def main(args):
    results = utils.Stats(
        args.results_path,
        ["num_factors", "noise_m", "noise_std", "matrix", "main_m", "main_std", "tree", "max_depth", "perm",
         "loss_abs", "loss_rel", "time", "matrix_norm", "noise_norm", "noise_over_signal"],
        rewrite=args.rewrite_results
    )
    print(f"========= {args.matrix} matrix with noise_m = {args.noise_m}, noise_std = {args.noise_std}, "
          f"BP projection with {args.num_factors} factors, where perm = {args.perm}, max_depth = {args.max_depth}")

    # Generate the matrix to factorize
    if args.matrix == "random":
        print("Generating random matrix to factorize...")
        supp = utils.support_DFT(args.num_factors)
        matrix = utils.generate_random_matrix(supp, main_m=args.main_m, main_std=args.main_std)
    elif args.matrix == "dft":
        print("Generating DFT matrix to factorize...")
        matrix = scipy.linalg.dft(2 ** args.num_factors)
    else:
        assert args.matrix == "hadamard"
        print("Generating Hadamard matrix to factorize...")
        matrix = scipy.linalg.hadamard(2 ** args.num_factors)

    # Add noise if wanted
    noise_over_signal = None
    target_norm = None
    noise_norm = None

    for exp_id in range(args.repeat):
        print(f"\nEXP_ID: {exp_id+1}/{args.repeat}")
        if args.noise_m or args.noise_std:
            print(f"Adding noise with mean {args.noise_m} and std {args.noise_std}")
            noise = utils.generate_matrix_noise(matrix.shape, args.noise_m, args.noise_std)
            target = matrix + noise
            target_norm = np.linalg.norm(target)
            noise_norm = np.linalg.norm(noise)
            noise_over_signal = noise_norm / target_norm
            print(f"Noise over signal: {100 * noise_over_signal} %")
        else:
            target = matrix
        print("--------------------------------------------------")

        # Permutation
        p = None
        if args.perm != "all_8":
            p = utils.get_permutation_matrix(args.num_factors, args.perm)

        # Hierarchical factorization
        for tree in ["comb", "balanced"]:
            begin = time.time()
            if p is None:
                print(f"BP projection, where P can be any of the 8 permutations, and {tree} tree, max_depth={args.max_depth}")
                product, factors, p = fact.project_BP_model_8_perm_fixed(target, tree, args.max_depth, return_factors=True)
            else:
                print(f"BP projection, with fixed P = {args.perm}, and {tree} tree, max_depth={args.max_depth}")
                product, factors = fact.project_BP_model_P_fixed(target, tree, p, args.max_depth, return_factors=True)
            end = time.time()

            # Loss
            loss_abs = utils.error_cal(factors + [p], target, relative=False)
            loss_rel = utils.error_cal(factors + [p], target, relative=True)
            print("Loss absolute:", loss_abs, f"({100 * loss_rel} %)")
            print("Factorization time:", end - begin)
            print("--------------------------------------------------")

            results.update([args.num_factors, args.noise_m, args.noise_std, args.matrix, args.main_m, args.main_std,
                            tree, args.max_depth, args.perm,
                            loss_abs, loss_rel, end - begin, target_norm, noise_norm, noise_over_signal])

    print("\nRESULTS DATAFRAME")
    print(results.stats)


if __name__ == '__main__':
    args = parse_args()
    main(args)
