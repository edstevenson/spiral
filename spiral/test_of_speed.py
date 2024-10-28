import numpy as np
from utils import polar_to_cart_log, polar_to_cart_log_CV
import cProfile

Nphi, Nr = 1000, 1000
polar_data = np.random.rand(Nphi, Nr)
r = np.logspace(0, 1, Nr)
x = np.linspace(-10, 10, 2000)
y = np.linspace(-10, 10, 2000)


profiler = cProfile.Profile()
profiler.enable()

# Call the function with sample data
result = polar_to_cart_log_CV(polar_data, r, x, y, linear=True)
# result = polar_to_cart_log(polar_data, r, x, y, order=3)

profiler.disable()
profiler.print_stats(sort='cumulative')
