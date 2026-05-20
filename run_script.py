from pipt.loop.assimilation import Assimilate
from pipt import pipt_init
from input_output import read_config
from subsurface.multphaseflow.jutul_darcy import JutulDarcy
import numpy as np


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(29_11_1997)

    # Read configuration file
    kwda, kwsim, kwen = read_config.read('config.yaml')

    # Rmove adjoint info
    kwsim.pop('adjoints', None)

    # Add options
    #kwda['analysis'] = 'approx'
    kwda['localization'] = {'field': [50, 50], 'autoadaloc': 0.3}

    sim = JutulDarcy(kwsim)
   
    # Initialize
    analysis = pipt_init.init_da(kwda, kwen, sim)
    # Run Data Assimilation
    assimilator = Assimilate(analysis)
    assimilator.run()
