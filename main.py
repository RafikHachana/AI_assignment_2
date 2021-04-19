import warnings

import cv2
import numpy as np
import random
from os import environ
from numba import jit, NumbaDeprecationWarning, NumbaPendingDeprecationWarning

WHOLE_MUTATION_RATE = 0.999
POPULATION_SIZE = 30000  # This is the INITIAL POPULATION SIZE
FITNESS_THRESHOLD = float(input('Enter the initial fitness tolerance (between 1 and 5) :'))

MAX_POPULATION_SIZE = 500000
POPULATION_INCREMENT = 5000
OVERALL_FITNESS_THRESHOLD = 35
MAX_GENERATIONS = int(input("Enter the maximum number of generations : "))

BRUSH_SIZE = 12
BRUSH_OPACITY = 0.5
IMAGE_SCALING_FACTOR = 1

REFERENCE_IMAGE = input(' Enter the image path (or null to use the default image) : ')


# This function is just to suppress certain warnings that are not relevant
def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


# @jit(nopython=True)
def get_image_from_circles(circles: np.arange, shape: tuple):
    image = np.ones(shape=shape) * 200
    for c in circles:
        x = c[0]
        y = c[1]
        color = c[2]
        size = c[3]
        sub_img = image[x - size // 2:x + size // 2, y - size // 2:y + size // 2]
        white_rect = np.uint8(cv2.circle(sub_img.astype('uint8'), (size // 2, size // 2), size // 2, color, -1))
        res = np.uint8(cv2.addWeighted(sub_img.astype('uint8'), 1 - BRUSH_OPACITY, white_rect, BRUSH_OPACITY, 0))

        image[x - size // 2:x + size // 2, y - size // 2:y + size // 2] = res
    return image


# Loading and showing pictures
# Loads a picture to a numpy array
def load_image(path=None):
    res = (cv2.imread(path), cv2.imread('sample.jpg'))[path is None]
    return res


# Displays the image in a popup window
def show_image(image):
    cv2.namedWindow("Making art ...", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Making art ...", image.astype('uint8'))
    cv2.waitKey(10)


# Utility function to downscale the image by a certain integer factor, used in case the image size is too big
# @jit(nopython=True)
def downscale_image(image, factor):
    new_shape = (image.shape[0] // factor, image.shape[1] // factor, 3)
    res = np.zeros(shape=new_shape)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for rgb in range(3):
                res[x // factor, y // factor, rgb] += image[x, y, rgb] // (factor ** 2)
    return res


@jit(nopython=True)
def fitness_function(ref, image):
    res = 0
    image_shape = image.shape
    for x in range(image_shape[0]):
        for y in range(image_shape[1]):
            norm = 0
            for rgb in range(image_shape[2]):
                norm += abs(image[x, y, rgb] - ref[x, y, rgb]) ** 2
            res += np.sqrt(norm)
    return res / (image_shape[0] * image_shape[1])


@jit(nopython=True)
def random_circle(image_shape):
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    size = BRUSH_SIZE
    x = random.randint(size // 2, image_shape[0] - size // 2 - 1)
    y = random.randint(size // 2, image_shape[1] - size // 2 - 1)
    return (x, y, color, size)


# Generates a random population of circles
@jit(nopython=True)
def random_circle_population(sample_size: int, image_shape: tuple):
    # example_tuple = (np.int64,np.int64,(np.int64,np.int64,np.int64),np.int64)
    circles = [(np.uint16(0), np.uint16(0), (np.uint16(0), np.uint16(0), np.uint16(0)), np.uint16(0)) for x in range(0)]
    for j in range(sample_size):
        # x, y, color, size = random_square(image_shape)
        circles.append(random_circle(image_shape))
    return circles


# Performs the crossover (and mutation) of circles using the parents (individuals from the previous generation)
@jit(nopython=True)
def crossover_circles(parents: np.arange, shape):
    choices = len(parents)

    # whole mutation
    mutation = random.uniform(0, 1)
    if mutation <= WHOLE_MUTATION_RATE or choices < 20000:
        return [random_circle(shape), random_circle(shape)]

    # crossover
    p1 = random.randint(0, choices - 10001)
    p2 = p1 + random.randint(0, 10000)

    x = (parents[p1][0] + parents[p2][0]) // 2

    y = (parents[p1][1] + parents[p2][1]) // 2

    b = (parents[p1][2][0] + parents[p2][2][0]) // 2
    g = (parents[p1][2][1] + parents[p2][2][1]) // 2
    r = (parents[p1][2][2] + parents[p2][2][2]) // 2
    color = (b, g, r)
    size = parents[p1][3]
    # return (x, y, color, size)
    res = [(parents[p1][0], parents[p1][1], parents[p2][2], parents[p2][3]), (x, y, color, size)]
    return res


# Function to measure the fitness of a single circle
@jit(nopython=True)
def fitness_function_circle(circle: tuple, ref: np.ndarray):
    shape = ref.shape
    x = circle[0]
    y = circle[1]
    color = circle[2]
    size = circle[3]
    area = np.pi * ((size // 2) ** 2)
    norm = 0
    for a in range(-size // 2, size // 2):
        for b in range(-size // 2, size // 2):
            if (x + a) in range(shape[0]) \
                    and (y + b) in range(shape[1]) \
                    and (a - size / 2) ** 2 + (
                    b - size / 2) ** 2 <= (size // 2) ** 2:
                for rgb in range(3):
                    norm += abs(ref[x + a, y + b, rgb] - color[rgb]) ** 2
    norm = np.sqrt(norm)
    res = norm / area
    return res


# Returns the survivors from a certain generation using the value of the fitness function
def survivors_circles(generation: list, ref: np.ndarray):
    an_iterator = filter(lambda x: fitness_function_circle(x, ref) < FITNESS_THRESHOLD, generation)
    res2 = list(an_iterator)
    print('Number of survivors : ' + str(len(res2)))
    return res2


# Utility function that returns the position (x,y) of a circle
@jit(nopython=True)
def get_position(circle):
    return (circle[0], circle[1])


# Computes the next generation using the survivors of the previous generation
@jit(nopython=True)
def new_generation_circles(sample_size, old, shape):
    res = [(np.uint16(0), np.uint16(0), (np.uint16(0), np.uint16(0), np.uint16(0)), np.uint16(0)) for x in range(0)]
    old.sort(key=get_position)
    while len(res) < sample_size - len(old):
        res += crossover_circles(old, shape)
    if WHOLE_MUTATION_RATE > 0.05:
        res += old
    return res


if __name__ == '__main__':

    suppress_qt_warnings()

    print('Welcome to my artistic genetic algorithm!')
    random.seed()
    np.random.seed()
    img = load_image()
    if REFERENCE_IMAGE != 'null':
        img = load_image(REFERENCE_IMAGE)
    print(img.shape)
    if IMAGE_SCALING_FACTOR > 1:
        img = downscale_image(img, IMAGE_SCALING_FACTOR)
    show_image(img)
    print(img.shape)
    print('Creating a random initial population.')
    generation = random_circle_population(POPULATION_SIZE, img.shape)
    for i in range(MAX_GENERATIONS):
        print('\n\n Generation ' + str(i))
        current = get_image_from_circles(generation, img.shape)
        generation = new_generation_circles(POPULATION_SIZE, survivors_circles(generation, img), img.shape)
        show_image(current)
        overall_fitness = fitness_function(current, img)
        WHOLE_MUTATION_RATE = (min(50, overall_fitness) / 50) ** (i / 20)

        POPULATION_SIZE += POPULATION_INCREMENT
        FITNESS_THRESHOLD *= 0.9975
        FITNESS_THRESHOLD = max(FITNESS_THRESHOLD, 0.5)
        print('Overall fitness : ' + str(overall_fitness))
        print('Current mutation rate: ' + str(WHOLE_MUTATION_RATE))
        print('Current fitness threshold: ' + str(FITNESS_THRESHOLD))
        print('Current population size: ' + str(len(generation)))
        if overall_fitness < OVERALL_FITNESS_THRESHOLD or POPULATION_SIZE > MAX_POPULATION_SIZE:
            generation = survivors_circles(generation, img)
            break

    current = get_image_from_circles(generation, img.shape)
    cv2.imwrite('result_' + str(BRUSH_SIZE) + '_at_' + str(
        BRUSH_OPACITY) + '_with_tolerance_' + str(FITNESS_THRESHOLD) + '_of_' + str(REFERENCE_IMAGE), current.astype('uint8'))
    print("Image saved!")
