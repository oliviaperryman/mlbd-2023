# 04
import numpy as np


def entropy(p_x):
    """Information content or surprise of event x
    Intuition: low probability = high surprise/information content
    Entropy: Expected surprise/information content
    """
    return -p_x * np.log2(p_x) - (1 - p_x) * np.log2(1 - p_x)


def conditional_entropy_y_given_x_is_true(p_xy, p_xnoty):
    """Conditional entropy of x given y
    Intuition: how much information is needed to describe x after y is known
    """
    p_x = p_xy + p_xnoty
    p_y_given_x = p_xy / p_x
    p_noty_given_x = p_xnoty / p_x

    return -p_y_given_x * np.log2(p_y_given_x) - p_noty_given_x * np.log2(
        p_noty_given_x
    )


def expected_conditional_entropy(p_xy, p_notxy, p_xnoty, p_notxnoty):
    """What is the expected entropy of  Y given X"""
    p_x = p_xy + p_xnoty
    p_notx = p_notxy + p_notxnoty

    return p_x * conditional_entropy_y_given_x_is_true(
        p_xy, p_xnoty
    ) + p_notx * conditional_entropy_y_given_x_is_true(p_notxy, p_notxnoty)


def information_gain(extropy_y, expected_conditional_entropy_y_given_x):
    """How much information is gained about Y when X is known
    How much information about Y do we get by discovering
     whether it is X
    """
    return extropy_y - expected_conditional_entropy_y_given_x


if __name__ == "__main__":
    print(conditional_entropy_y_given_x_is_true(0.24, 0.01))
    print(expected_conditional_entropy(0.24, 0.25, 0.01, 0.5))

    p_xy, p_notxy, p_xnoty, p_notxnoty = (
        0.5,
        0.125,
        0.0001,
        0.375,
    )
    p_y = 0.375

    print(
        information_gain(
            entropy(p_y),
            expected_conditional_entropy(p_xy, p_notxy, p_xnoty, p_notxnoty),
        )
    )
