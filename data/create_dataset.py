import numpy as np


def draw_sample(type_rel="sin", size=1000, noise=0.1, height=0, period=1):
    # sample from sine or cosine function from given parameters
    iter_sample = int(size / 100)
    x_set = np.array([])
    y_set = np.array([])
    for i in range(iter_sample):
        # sample data
        x = np.arange(0, 4 * np.pi, 0.1)
        if type_rel == "sin":
            y = np.sin(x * period)
        elif type_rel == "cos":
            y = np.cos(x * period)
        noi = np.random.normal(loc=0.0, scale=noise, size=len(y))
        y = y + noi + height
        x_set = np.append(x_set, x)
        y_set = np.append(y_set, y)
    return (x_set, y_set)

