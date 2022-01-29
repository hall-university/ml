import logging.config
import math
from statistics import mean
from typing import Optional


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
        self.slope = slope
        self.intercept = intercept
        self.determinant = determinant


model_result: Optional[ModelResult] = None


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
        sum([pow(((slope * x) + intercept) - independent_mean, 2) for x in dependent_collection]) /
        sum([pow(y - independent_mean, 2) for y in independent_collection])
    )


def get_dynamic_collection(data, break_point):
    return data[:break_point]


def train(dependent_x_collection, independent_y_collection, break_point, iteration):
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

    logger.info(f'iteration: {iteration}')
    logger.info(f'slope: {slope}')
    logger.info(f'intercept: {intercept}')
    logger.info(f'R^2: {determinant}')

    global model_result
    model_result = ModelResult(slope, intercept, determinant)

    return iteration, slope, intercept, determinant


def predict(a, b, x):
    result = math.floor((a * x) + b)
    logger.info(f'predicate: {result}')
    print()
    return result


def main_runner(option: int, predict_for: Optional[int] = None):
    assert len(dependent_x_collection) == len(independent_y_collection)
    collection_len = len(dependent_x_collection)

    # Start from the second element, to prevent incorrect calculations
    for i in range(1, collection_len):
        break_point = i + 1
        iteration, slope, intercept, determinant = train(
            dependent_x_collection, independent_y_collection, break_point, i
        )

        if option == 3:
            predict(slope, intercept, predict_for)


def cli():
    global model_result

    print('1. Predict value\n2. Train model\n3. Predict during training\n-- Anything else to close')
    try:
        option = int(input('Select option: '))
        match option:
            case 1:
                if model_result:
                    predict_for = int(input('Predict for: '))
                    predict(model_result.slope, model_result.intercept, predict_for)
                else:
                    print('\tTrain model first! Option no. 2')
            case 2:
                main_runner(option)
            case 3:
                predict_for = int(input('Predict for: '))
                main_runner(option, predict_for)
            case _:
                raise Exception('Exit')
    except:
        exit(1)


def runner():
    while True:
        cli()


if __name__ == '__main__':
    runner()
