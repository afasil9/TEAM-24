import sys

import numpy as np


def par_print(comm, string):
    if comm.rank == 0:
        print(string)
        sys.stdout.flush()


def interpolate_by_tags(function, value_dict, domain_tags):
    for tag, value in value_dict.items():
        function.interpolate(lambda x: np.full_like(x[0], value), domain_tags.find(tag))
    function.x.scatter_forward()
