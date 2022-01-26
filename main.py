import logging.config
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

        diff_product_sum = sum(diff_product)
        independent_mean_difference_power_sum = sum(dependent_mean_difference_power)

        slope = diff_product_sum / independent_mean_difference_power_sum
        intercept = independent_mean - (dependent_mean * slope)

        logger.debug(f'iteration: {i}')
        logger.debug(f'diff_product_sum: {diff_product_sum}')
        logger.debug(f'independent_mean_difference_power_sum: {independent_mean_difference_power_sum}')
        logger.debug(f'slope: {slope}')
        logger.debug(f'intercept: {intercept}')
        print()


if __name__ == '__main__':
    main()
# https://www.statystyczny.pl/regresja-liniowa/
