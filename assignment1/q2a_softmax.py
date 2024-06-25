import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        x_max = np.max(x, axis=1, keepdims=True)
        x = x - x_max
        exp_x = np.exp(x)
        x = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        ### END YOUR CODE
    else:
        # Vector
        ### YOUR CODE HERE
        x_max = np.max(x)
        x = x - x_max
        exp_x = np.exp(x)
        x = exp_x / np.sum(exp_x)
        ### END YOUR CODE

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1, 2]))
    print(test1)
    ans1 = np.array([0.26894142, 0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print(test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print("You should be able to verify these results by hand!\n")


def your_softmax_test():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")
    ### YOUR OPTIONAL CODE HERE
    # Test with a single vector containing negative values
    test4 = softmax(np.array([-1, -2, -3]))
    print(test4)
    ans4 = np.array([0.66524096, 0.24472847, 0.09003057])
    assert np.allclose(test4, ans4, rtol=1e-05, atol=1e-06)

    # Test with a matrix where all rows are the same
    test5 = softmax(np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]))
    print(test5)
    ans5 = np.array([
        [0.09003057, 0.24472847, 0.66524096],
        [0.09003057, 0.24472847, 0.66524096],
        [0.09003057, 0.24472847, 0.66524096]])
    assert np.allclose(test5, ans5, rtol=1e-05, atol=1e-06)

    print("All custom tests passed!\n")
    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    your_softmax_test()
