# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

"""
This module consists of functions useful for reading and saving files.

"""

# <codecell>

import pickle

# <codecell>

def return_file_content(file_path_name):
    ''' 
    Reads the file from file_path_name and returns the content. All files compatible to be read using file.read()
    can be read using this function.
    
    Parameters
    ----------
    file_path_name : string
        This variable includes the name of the file to be read with the path of its location. If the file is in the  current working directory then the path is not necessary.
            
    Returns
    -------
    file_content : string
        This variable consists of content of the file
    '''
    f = open(file_path_name,'r')
    file_content = f.read()
    f.close()
    return file_content

def save_pickle_file(var, file_path_name):
    '''
    Saves the variable `var` as pickle file. This function doesn't return anything.
    
    Parameters
    ----------
    var : Python variable
        This variable can be of any valid type including string, dictionary, dataframe and etc.
    
    file_path_name : string
        Name of the pickle file with its path
    
    '''
    with open(file_path_name, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def load_pickle_file(file_path_name):
    '''
    Return a pickle file as a python variable. 
    
    Parameters
    ----------
    file_path_name : string
        Name of the pickle file with its path
        
    Returns
    -------
    var : Python variable
        Python variable with the pickle file content.
    '''
    with open(file_path_name, 'rb') as handle:
        var = pickle.load(handle)
        return var
