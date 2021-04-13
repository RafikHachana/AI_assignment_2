import cv2
import numpy as np
import random

from os import environ
from numba import jit, njit

WHOLE_MUTATION_RATE = 1
SURVIVAL_RATE = 0.1
POPULATION_SIZE = 75000

BRUSH_OPACITY = 0.65
BRUSH_SIZE = 6
IMAGE_SCALING_FACTOR = 1

REFERENCE_IMAGE = 'nature.tiff'


def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"


global img_height
global img_width


# @jitclass
class Individual:
    def __init__(self, circles: list, shape: tuple):
        self.shape = shape
        self.circles = []
        self.image = np.ones(shape=self.shape) * 128
        self.apply_circles(circles)

    # @jit
    def add_circle(self, c: list):
        self.circles.append(c)
        x = c[0]
        y = c[1]
        color = c[2]
        size = c[3]
        # print(color)
        sub_img = self.image[x - size // 2:x + size // 2, y - size // 2:y + size // 2]

        white_rect = cv2.circle(sub_img.astype('uint8'), (size // 2, size // 2), size // 2, color, -1)
        res = cv2.addWeighted(sub_img.astype('uint8'), 1-BRUSH_OPACITY, white_rect, BRUSH_OPACITY, 0)
        # show_image(res)

        # Putting the image back to its position
        self.image[x - size // 2:x + size // 2, y - size // 2:y + size // 2] = res
        '''for a in range(size):
            for b in range(size):
                if (x + a) in range(self.shape[0]) \
                        and (y + b) in  range(self.shape[1]) \
                        and (a - size / 2) ** 2 + (
                        b - size / 2) ** 2 <= size:
                    for rgb in range(3):
                        self.image[x + a, y + b, rgb] = self.image[x + a, y + b, rgb] * 0.75 + color[rgb] * 0.25'''

    # @jit(nopython=True)
    def apply_circles(self, circles: list):
        for c in circles:
            self.add_circle(c)


# Loading and showing pictures
def load_image(path=None):
    res = (cv2.imread(path), cv2.imread(REFERENCE_IMAGE))[path is None]
    return res


@jit
def downscale_image(image, factor):
    new_shape = (image.shape[0] // factor, image.shape[1] // factor, 3)
    res = np.zeros(shape=new_shape)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for rgb in range(3):
                res[x // factor, y // factor, rgb] += image[x, y, rgb] // (factor ** 2)
    return res


@jit
def fitness_function2(image, ref):
    # print(image.shape)
    image_gray = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref.astype('uint8'), cv2.COLOR_BGR2GRAY)

    laplacian = np.array((
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]), dtype="int")

    ref_convolved = cv2.filter2D(ref_gray, -1, laplacian)
    image_convolved = cv2.filter2D(image_gray, -1, laplacian)
    # show_image(image_convolved)
    # print(image_convolved.shape)
    picture1_norm = image_convolved / np.sqrt(np.sum(image_convolved ** 2))
    picture2_norm = ref_convolved / np.sqrt(np.sum(ref_convolved ** 2))
    return fitness_function_gray(image_convolved, ref_convolved) + 100 * (1 - np.sum(picture2_norm * picture1_norm))


def show_image(image):
    cv2.namedWindow("bw", cv2.WND_PROP_FULLSCREEN)
    cv2.imshow("bw", image.astype('uint8'))
    cv2.waitKey(100)


@jit
def fitness_function(ref, img):
    res = 0
    image_shape = img.shape
    for x in range(image_shape[0]):
        for y in range(image_shape[1]):
            norm = 0
            for rgb in range(image_shape[2]):
                norm += abs(img[x, y, rgb] - ref[x, y, rgb]) ** 2
            res += np.sqrt(norm)
    return res / (image_shape[0] * image_shape[1])


@jit
def fitness_function_gray(ref, img):
    res = 0
    image_shape = img.shape
    for x in range(image_shape[0]):
        for y in range(image_shape[1]):
            res += abs(img[x, y] - ref[x, y])
    return res / (image_shape[0] * image_shape[1])


# from STACKOVERFLOW
@jit
def get_image_difference(image_1, image_2):
    gray_1 = cv2.cvtColor(image_1.astype('uint8'), cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(image_2.astype('uint8'), cv2.COLOR_BGR2GRAY)
    first_image_hist = cv2.calcHist([gray_1.astype('uint8')], [0], None, [256], [0, 255])
    second_image_hist = cv2.calcHist([gray_2.astype('uint8')], [0], None, [256], [0, 255])

    img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
    img_template_probability_match = cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
    img_template_diff = 1 - img_template_probability_match

    # taking only 10% of histogram diff, since it's less accurate than template method
    commutative_image_diff = img_template_diff
    return commutative_image_diff


@jit
def crossover(parents):
    choices = len(parents)
    p1 = random.randint(0, choices - 1)
    p2 = random.randint(0, choices - 1)
    circles = []
    # Crossover part
    '''for i in range(min(len(parents[p1].circles),len(parents[p2].circles))):
        source = (parents[p1],parents[p2])[random.randint(0,1)]
        circles.append(source.circles[i])'''
    # print('Crossing 2 parents')
    cut = random.randint(2, min(len(parents[p1].circles), len(parents[p2].circles)))
    circles += parents[p1].circles[:cut]
    circles += parents[p2].circles[cut + 1:]
    res = Individual(circles, parents[0].shape)

    # Mutation part
    # print('Random mutation')
    mutation_rate = 0.8
    mutation = random.uniform(0, 1)
    if mutation <= mutation_rate:
        x, y, color, size = random_square(res.shape)
        res.add_circle([x, y, color, size])
    return res


@jit
def random_square(image_shape):
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    size = BRUSH_SIZE
    x = random.randint(size // 2, image_shape[0] - size // 2 - 1)
    y = random.randint(size // 2, image_shape[1] - size // 2 - 1)
    return x, y, color, size


@jit
def random_sample_circles(sample_size, image_shape):
    res = []
    number_of_circles = 500
    for i in range(sample_size):
        circles = []
        for j in range(number_of_circles):
            # x, y, color, size = random_square(image_shape)
            circles += [random_square(image_shape)]
        res.append(Individual(circles, image_shape))
    return res


# @jit
def survivors(generation, ref):
    survival_rate = 0.2
    number_of_survivors = int(len(generation) * survival_rate)
    res = sorted(generation,
                 key=lambda x: fitness_function(ref, x.image)
                 # '''get_image_difference(ref, x)+0.01*fitness_function(downscale_image(ref,16),downscale_image(x,16))'''
                 )[:number_of_survivors]
    print("Found survivors")
    return res


@jit
def new_generation(size, old):
    res = []
    for i in range(size):
        res.append(crossover(old))
    res.append(old[0])
    print("Made generation")
    return res


# Parth's idea
@jit
def random_circle_population(sample_size: int, image_shape: tuple):
    circles = []
    for j in range(sample_size):
        # x, y, color, size = random_square(image_shape)
        circles += [random_square(image_shape)]
    return circles


@njit
def crossover_circles(parents: list, shape):
    choices = len(parents)
    # crossover
    p1 = random.randint(0, choices - 2)
    p2 = p1 + 1

    #x = parents[p1][0]
    x = (parents[p1][0] + parents[p2][0]) // 2
    #y = parents[p1][0]
    y = (parents[p1][1] + parents[p2][1]) // 2

    #color = parents[p2][2]
    b = (parents[p1][2][0] + parents[p2][2][0]) // 2
    g = (parents[p1][2][1] + parents[p2][2][1]) // 2
    r = (parents[p1][2][2] + parents[p2][2][2]) // 2
    color = (b,g,r)
    size = parents[p1][3]

    # mutation (only color for the moment)
    '''mutation_rate = WHOLE_MUTATION_RATE
    mutation = random.uniform(0, 1)
    if mutation <= mutation_rate:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # position mutation
    mutation_rate = WHOLE_MUTATION_RATE
    mutation = random.uniform(0, 1)
    if mutation <= mutation_rate:
        x *= np.random.normal(loc=0.5, scale=0.5)
        y *= np.random.normal(loc=0.5, scale=0.5)
        x = int(max(min(x, shape[0] - 1), 0))
        y = int(max(min(x, shape[1] - 1), 0))'''

    # whole mutation
    mutation_rate = WHOLE_MUTATION_RATE
    mutation = random.uniform(0, 1)
    if mutation <= mutation_rate:
        return random_square(shape)

    return x, y, color, size


@njit
def fitness_function_circle(circle: tuple, ref: np.ndarray):
    shape = ref.shape
    x = circle[0]
    y = circle[1]
    color = circle[2]
    size = circle[3]
    area = np.pi * ((size // 2) ** 2)
    norm = 0
    for a in range(-size//2, size//2):
        for b in range(-size//2,size//2):
            if (x + a) in range(shape[0]) \
                    and (y + b) in range(shape[1]) \
                    and (a - size / 2) ** 2 + (
                    b - size / 2) ** 2 <= (size // 2) ** 2:
                for rgb in range(3):
                    norm += abs(ref[x + a, y + b, rgb] - color[rgb]) ** 2
    norm = np.sqrt(norm)
    return norm / area


#@njit
def survivors_circles(generation, ref : np.ndarray):
    '''res = []
    for c in generation:
        if fitness_function_circle(c,ref)<100:
            res.append(c)
    return res'''
    survival_rate = SURVIVAL_RATE
    number_of_survivors = int(len(generation) * survival_rate)
    res = sorted(generation, key=lambda x: fitness_function_circle(x, ref))[:number_of_survivors]
    print("Worst survivor : " + str(fitness_function_circle(res[number_of_survivors-1],ref)))
    return res

@njit
def get_position(circle):
    return (circle[0], circle[1])

@njit
def new_generation_circles(sample_size, old, shape):
    res = []
    old = sorted(old, key= get_position)
    for i in range(sample_size - len(old)):
        # res.append(random_square(shape))
        res.append(crossover_circles(old, shape))
    res += old
    #print("Made generation of size " + str(len(res)))
    return res


if __name__ == '__main__':
    sample_size = POPULATION_SIZE
    generations = 50000

    suppress_qt_warnings()

    print('Welcome to my artistic genetic algorithm!')
    random.seed()
    np.random.seed()
    img = load_image()
    print(img.shape)
    # show_image(img)
    img = downscale_image(img, IMAGE_SCALING_FACTOR)
    show_image(img)
    print(img.shape)
    print('Going to create a random sample')
    '''generation = random_sample_circles(sample_size, img.shape)
    show_image(generation[0].image)
    for i in range(generations):
        print('Generation ' + str(i))
        generation = new_generation(sample_size, survivors(generation, img))
        print('Best fitness : ' + str(fitness_function(generation[0].image, img)))
        # print('Image difference : '+str(get_image_difference(generation[0].image, img)))
        print('Fitness function 2 : ' + str(fitness_function2(generation[0].image, img)))

        show_image(generation[0].image)'''
    generation = random_circle_population(sample_size, img.shape)
    for i in range(generations):
        print('Generation ' + str(i))
        current = Individual(generation, img.shape).image
        generation = new_generation_circles(sample_size, survivors_circles(generation, img), img.shape)
        # print(generation)
        show_image(current)
        overall_fitness = fitness_function(current, img)
        WHOLE_MUTATION_RATE = (min(50,overall_fitness)/50)**i
        SURVIVAL_RATE = 0.6 - 0.3*(max(0,50-overall_fitness)/50)**1
        print('Overall fitness : ' + str(overall_fitness))
        print('Current mutation rate: ' + str(WHOLE_MUTATION_RATE))
        print('Current survival rate: ' + str(SURVIVAL_RATE))
        # print('Image difference : '+str(get_image_difference(generation[0].image, img)))
        #print('Fitness function 2 : ' + str(fitness_function2(current, img)))

    # show_image(generation[0].image)
