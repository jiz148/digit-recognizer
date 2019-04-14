"""
# test_utils_logger

@author: Jason Zhu
@email: jason_zhuyx@hotmail.com

"""
import logging
import unittest

from dr.utils.logger_formatter import factory
from dr.utils.logger import get_logger
from dr.utils.logger import print_info
from dr.utils.logger import raise_ni

LOGGER = get_logger(__name__)


class LoggerTests(unittest.TestCase):
    """
    LoggerTests includes all unit tests for dr.urils.logger module
    """
    @classmethod
    def teardown_class(cls):
        logging.shutdown()

    def setUp(self):
        """setup for test"""
        pass

    def tearDown(self):
        """tearing down at the end of the test"""
        pass

    def test_formatter_factory(self):
        """
        test dr.utils.logger_formatter.factory
        """
        err_logger = logging.getLogger(__name__)
        err_logger.addHandler(logging.StreamHandler())
        err_handler = err_logger.handlers[0]
        err_handler.setFormatter(fmt=factory(use_color=False))
        err_logger.info('Logging information (no color) ...')
        err_handler.setFormatter(fmt=factory())
        err_logger.info('Logging information ...')
        err_logger.fatal('Logging fail ...')
        err_logger.error('Logging error ...')
        err_logger.warning('Logging warning ...')
        err_logger.debug('Logging debug ...')

    def test_get_logger(self):
        """
        test dr.utils.logger.get_logger
        """
        print_info()
        # import re
        # removing prefix /^tests\./ for `python -m unitest`
        # logger_name = re.sub(r'^tests\.', '', LOGGER.name)
        # self.assertEqual(logger_name, 'test_utils_logger')
        self.assertEqual(LOGGER.name, 'dr.tests.test_utils_logger')
        LOGGER.debug('LOGGER name: %s', LOGGER.name)
        LOGGER.warning('LOGGER warning ...')

    def test_raise_ni(self):
        """
        test dr.utils.logger.raise_ni
        """
        method_name = 'some_method_name'
        msg = 'must implement {} in derived class'.format(method_name)
        with self.assertRaises(NotImplementedError) as context:
            result = raise_ni(method_name)
            self.assertIsNone(result)
        self.assertEqual(msg, str(context.exception))
