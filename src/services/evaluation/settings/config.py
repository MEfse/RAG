"""Application settings with .env file support."""
import time

import logging
logger = logging.getLogger(__name__)


def time_decorator(func):
    """
    Декоратор для измерения времени выполнения функции.

    Args:
        func (Callable): Функция, время выполнения которой необходимо замерить.

    Returns:
        Callable: Обертка функции, которая замеряет время выполнения.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"Ошибка при выполнении функции '{func.__name__}' за {execution_time:.4f} секунд. Ошибка: {e}", exc_info=True)
            raise
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Время выполнения функции '{func.__name__}': {execution_time:.4f} секунд")
        return result
    return wrapper
