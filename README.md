# Asynchronous Evolutionary Algorithm

This is a python implementation of asynchronous evolutionary algorithm. 
The main algorithm is implemented in [asynch_ea.py](https://github.com/LeniLeGoff/py_async_ea/blob/master/asynch_ea.py)
It is heavily based on DEAP and scoop.

To test it there are two scripts: [test_asynch_ea.py](https://github.com/LeniLeGoff/py_async_ea/blob/master/test_asynch_ea.py) and [test_nested_asynch_ea.py](https://github.com/LeniLeGoff/py_async_ea/blob/master/test_nested_async_ea.py)

At the moment one experiment is implemented using the algorithm: [modular_2d_walker.py](https://github.com/LeniLeGoff/py_async_ea/blob/master/modular_2d_walker.py)
This experiment evolve 2d creatures to solve a walking task. The algorithm is a nested optimisation process with the asynch_ea to evolve the shape of the creatures and a simple genetic algorithm to optimise their controllers. 

To run it you have to clone [gym_rem2D](https://github.com/LeniLeGoff/gym_rem2D) repository. And copy or create a symbolic link of the folder ModularER_2D in the task folder.
To create a symbolic link:

    cd tasks
    ln -s path/to/gym_rem2D/ModularER_2D

