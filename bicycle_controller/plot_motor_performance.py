import numpy as np
import matplotlib.pyplot as plt
import pint

# Setup unit registry
u = pint.UnitRegistry(system='mks')

# Motor parameters
name = 'Dayton 4Z143'
gearing_ratio = 500
max_torque = 5.63 * u.inch * u.lbf
max_speed = 1750 * u.rpm

# Convert to base units
max_torque = max_torque.to_base_units().magnitude
max_speed = max_speed.magnitude

# Gear down the motor
max_torque *= gearing_ratio
max_speed /= gearing_ratio


# Plots
plt.figure()
plt.plot([0, max_speed], [max_torque, 0])
plt.grid()
plt.xlabel('Motor Speed [RPM]')
plt.ylabel('Motor Torque [N m]')
plt.title('Motor Graph')
plt.show()


