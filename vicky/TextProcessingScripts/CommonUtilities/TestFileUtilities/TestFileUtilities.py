# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import os
import sys
import unittest

# <codecell>

scripts_path = os.path.abspath('..')
sys.path.append(scripts_path)

# <codecell>

from FileUtilities import *

# <codecell>

class TestFileUtilities(unittest.TestCase):
    
    def test_return_file_content(self):
        test_file_content = return_file_content('test_text_file.txt')
        self.assertEqual(test_file_content, 'This is a textfile used to test the Darts Processing modules.')
        
    def test_save_and_load_pickle_file(self):
        test_var = 'This is a test string'
        save_pickle_file(test_var, "test_variable.p")
        check_test_var = load_pickle_file("test_variable.p")
        self.assertEqual(test_var, check_test_var)


# <codecell>

if __name__ == '__main__':
    unittest.main()
