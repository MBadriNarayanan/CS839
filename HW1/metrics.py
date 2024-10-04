import argparse

import numpy as np

from collections import Counter


def get_prob_distribution(text):
    char_count = Counter(list(text))
    total_chars = sum(char_count.values())
    prob_dist = {char: count / total_chars for char, count in char_count.items()}
    return prob_dist


def compute_kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def compute_perplexity(prob_dist):
    entropy_sum = sum(p * np.log2(p) for p in prob_dist.values())
    perplexity = 2 ** (-entropy_sum)
    return perplexity


def main():
    print(
        "\n--------------------\nComputing KL Divergence and Perplexity!\n--------------------\n"
    )

    parser = argparse.ArgumentParser(
        description="Calculate the KL Divergence and Perplexity based on character distributions."
    )
    parser.add_argument("--input_file", type=str, help="Path to the input text file.")
    parser.add_argument("--output_file", type=str, help="Path to the output text file.")

    args = parser.parse_args()

    with open(args.input_file, "r") as text_file:
        input_text = text_file.read()
    input_text = input_text.strip()

    with open(args.output_file, "r") as text_file:
        output_text = text_file.read()
    output_text = output_text.strip()

    input_prob_dist = get_prob_distribution(text=input_text)
    output_prob_dist = get_prob_distribution(text=output_text)

    vocab = set(input_prob_dist.keys()).union(set(output_prob_dist.keys()))

    input_prob = np.array([input_prob_dist.get(token, 1e-10) for token in vocab])
    output_prob = np.array([output_prob_dist.get(token, 1e-10) for token in vocab])

    kl_score = compute_kl_divergence(p=input_prob, q=output_prob)
    perplexity = compute_perplexity(prob_dist=input_prob_dist)

    print("KL Divergence: {:.3f}".format(kl_score))
    print("Perplexity: {:.3f}".format(perplexity))

    print(
        "\n--------------------\nComputed KL Divergence and Perplexity!\n--------------------\n"
    )


if __name__ == "__main__":
    main()
