# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-


def test_suite():
    """Returns unittest.TestSuite of desispec tests"""
    import unittest
    from os.path import dirname
    module_dir = dirname(dirname(__file__))
    print(module_dir)
    return unittest.defaultTestLoader.discover(module_dir,
                                               top_level_dir=dirname(module_dir))
