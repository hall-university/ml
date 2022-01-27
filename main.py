import logging.config
import math
from statistics import mean


logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '%(asctime)s [%(processName)s] [%(threadName)s] | %(module)s | %(filename)s | %(funcName)s | [%(name)s] [%(levelname)s]  %(message)s'
        },
    },
    'handlers': {
        'console': {
            'formatter': 'simple',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'level': 'DEBUG'
        },
    },
    'loggers': {
        '': {'handlers': ['console'], 'level': logging.DEBUG, 'propagate': True},
    }
})


logger = logging.getLogger(__name__)


population = [2908457, 2090836, 2151836, 1020767, 2508464, 3364176, 5324519, 1002575, 2128483, 1193348, 2298811, 4593358, 1265415, 1445478, 3469464, 1717970]
marriages = [13599, 10294, 10911, 4875, 11405, 17361, 24924, 4822, 11287, 6135, 11461, 22765, 6051, 6978, 17437, 8183]


dependent_x_collection = population
independent_y_collection = marriages


class ModelResult:
    def __init__(self, slope, intercept, determinant):
        self._slope = slope
        self._intercept = intercept
        self._determinant = determinant


def predict(a, b, x):
    return (a * x) + b


def main():
    assert len(dependent_x_collection) == len(independent_y_collection)
    collection_len = len(dependent_x_collection)

    for i in range(1, collection_len):
        break_point = i + 1

        dependent_x_collection_dynamic = dependent_x_collection[:break_point]
        independent_y_collection_dynamic = independent_y_collection[:break_point]

        dependent_mean = mean(dependent_x_collection_dynamic)
        independent_mean = mean(independent_y_collection_dynamic)

        dependent_mean_difference = [
            dependent - dependent_mean for dependent in dependent_x_collection_dynamic
        ]
        independent_mean_difference = [
            independent - independent_mean for independent in independent_y_collection_dynamic
        ]

        diff_product = map(
            lambda mean_difference_tuple: mean_difference_tuple[0] * mean_difference_tuple[1],
            zip(dependent_mean_difference, independent_mean_difference)
        )
        dependent_mean_difference_power = map(
            lambda dependent_mean_difference_value: pow(dependent_mean_difference_value, 2),
            dependent_mean_difference
        )
        independent_mean_difference_power = map(
            lambda independent_mean_difference_value: pow(independent_mean_difference_value, 2),
            independent_mean_difference
        )

        diff_product_sum = sum(diff_product)
        dependent_mean_difference_power_sum = sum(dependent_mean_difference_power)
        independent_mean_difference_power_sum = sum(independent_mean_difference_power)

        slope = diff_product_sum / dependent_mean_difference_power_sum
        intercept = independent_mean - (dependent_mean * slope)

        predicates = [predict(slope, intercept, x) for x in dependent_x_collection_dynamic]
        predicate_mean_difference = [predicate - independent_mean for predicate in predicates]
        predicate_mean_difference_pow = map(
            lambda predicate_mean_difference_value: pow(predicate_mean_difference_value, 2),
            predicate_mean_difference
        )
        predicate_mean_difference_pow_sum = sum(predicate_mean_difference_pow)
        determinant = predicate_mean_difference_pow_sum / independent_mean_difference_power_sum

        logger.info(f'iteration: {i}')
        logger.info(f'diff_product_sum: {diff_product_sum}')
        logger.info(f'independent_mean_difference_power_sum: {independent_mean_difference_power_sum}')
        logger.info(f'slope: {slope}')
        logger.info(f'intercept: {intercept}')
        logger.info(f'R^2: {determinant}')

        predict_for = 2900000
        predicate = predict(slope, intercept, predict_for)
        logger.info(f'predicate: {predicate}')
        print()


def count_slope(dependent_collection, independent_collection, dependent_mean, independent_mean):
    merged_collection = list(zip(dependent_collection, independent_collection))
    return (
        sum([(dependent - dependent_mean) * (independent - independent_mean) for dependent, independent in merged_collection]) /  # noqa
        sum([pow((dependent - dependent_mean), 2) for dependent, _ in merged_collection])
    )


def count_intercept(slope, dependent_mean, independent_mean):
    return independent_mean - (slope * dependent_mean)


def count_determinant(slope, intercept, independent_mean, dependent_collection, independent_collection):
    return (
        sum([pow(predict(slope, intercept, x) - independent_mean, 2) for x in dependent_collection]) /
        sum([pow(y - independent_mean, 2) for y in independent_collection])
    )


def get_dynamic_collection(data, break_point):
    return data[:break_point]


def main_v2():
    assert len(dependent_x_collection) == len(independent_y_collection)
    collection_len = len(dependent_x_collection)

    # Start from the second element, to prevent incorrect calculations
    for i in range(1, collection_len):
        break_point = i + 1

        dependent_x_collection_dynamic = get_dynamic_collection(
            dependent_x_collection,
            break_point
        )
        independent_y_collection_dynamic = get_dynamic_collection(
            independent_y_collection,
            break_point
        )

        dependent_mean = mean(
            dependent_x_collection_dynamic
        )
        independent_mean = mean(
            independent_y_collection_dynamic
        )

        slope = count_slope(
            dependent_x_collection_dynamic,
            independent_y_collection_dynamic,
            dependent_mean,
            independent_mean
        )
        intercept = count_intercept(
            slope,
            dependent_mean,
            independent_mean
        )
        determinant = count_determinant(
            slope,
            intercept,
            independent_mean,
            dependent_x_collection_dynamic,
            independent_y_collection_dynamic
        )

        logger.info(f'iteration: {i}')
        logger.info(f'slope: {slope}')
        logger.info(f'intercept: {intercept}')
        logger.info(f'R^2: {determinant}')

        predict_for = 2900000
        predicate = math.floor(predict(slope, intercept, predict_for))
        logger.info(f'predicate: {predicate}')
        print()


def cli():
    print('1. Predict value\n2. Train model\n3. Predict during training\n-- Anything else to close')
    try:
        option = int(input('Select option: '))
        match option:
            case 1:
                print('predict')
            case 2:
                print('train')
            case 3:
                print('predict and train')
            case _:
                raise Exception('Exit')
    except:
        exit(1)


def runner():
    while True:
        cli()


if __name__ == '__main__':
    runner()
    # main()
    # main_v2()
