"""

"""
import matplotlib.pyplot as plt

LAYERS_DIMENSIONS = [12288, 20, 7, 5, 3, 1]  # 5-layer model

def run():
    """

    :return:
    """
    np.random.seed(1)
    print('Start training ...')
    train_x_orig, train_y_orig, test_x_orig, test_y_orig = load_data()

    # standardize sets
    train_x = train_x_orig/255.
    test_x = test_x_orig/255.



if __name__ == '__main__':
        print_pypath()
        run()