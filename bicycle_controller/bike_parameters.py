# General parameters for my Fuji bike.  Most of these paramters are measured,
# and some are taken from the bike datasheet.

import math


# general parameters for my Fuji bike
# w = 1.0 # [m]
# c = 0.067 # [m]
# lamda = math.radians(18) # [rad]
# g = 9.81 # [m/s/s]

# # Rear wheel R
# r_R = 0.35 # [m]
# m_R = 2 # [kg]
# I_Rxx = 0.1405/2 # [kg*m^2]
# I_Ryy = 0.28/2 # [kg*m^2]

# # Rear Body and frame assembly B
# x_B = 0.4 # [m]
# z_B = -1.2 # [m]
# m_B = 75 # [kg]
# I_Bxx = 3.1 # [kg*m^2]
# I_Bxz = 0.8 # [kg*m^2]
# I_Byy = 3.67 # [kg*m^2]
# I_Bzx = I_Bxz # [kg*m^2]
# I_Bzz = 0.93 # [kg*m^2]

# # front Handlebar and fork assembly H
# x_H = 0.9 # [m]
# z_H = -0.7 # [m]
# m_H = 4 # [kg]
# I_Hxx = 0.05892 # [kg*m^2]
# I_Hxz = -0.00756 # [kg*m^2]
# I_Hyy = 0.06 # [kg*m^2]
# I_Hzx = I_Hxz # [kg*m^2]
# I_Hzz = 0.00708 # [kg*m^2]

# # Front wheel F
# r_F = 0.35 # [m]
# m_F = 2 # [kg]
# I_Fxx = 0.1405 # [kg*m^2]
# I_Fyy = 0.28 # [kg*m^2]

# --------Input Parameters ---------------#
# general parameters for my Fuji bike
w = 1.32 # [m]
c = 0.082 # [m]
lamda = math.radians(27) # [rad]
g = 9.81 # [m/s/s]

# Rear wheel R
r_R = 0.25 # [m]
m_R = 1 # [kg]
I_Rxx = 0.1405/2 # [kg*m^2]
I_Ryy = 0.28/2 # [kg*m^2]

# Rear Body and frame assembly B
x_B = 0.3 # [m]
z_B = -0.58 # [m]
m_B = 78-5.7 # [kg]
I_Bxx = 9.2 # [kg*m^2]
I_Bxz = 2.4 # [kg*m^2]
I_Byy = 11 # [kg*m^2]
I_Bzx = I_Bxz # [kg*m^2]
I_Bzz = 2.8 # [kg*m^2]

# front Handlebar and fork assembly H
x_H = 1.2 # [m]
z_H = -0.6 # [m]
m_H = 5.7 # [kg]
I_Hxx = 0.05892 # [kg*m^2]
I_Hxz = -0.00756 # [kg*m^2]
I_Hyy = 0.06 # [kg*m^2]
I_Hzx = I_Hxz # [kg*m^2]
I_Hzz = 0.00708 # [kg*m^2]

# Front wheel F
r_F = 0.25 # [m]
m_F = 1 # [kg]
I_Fxx = 0.1405/2 # [kg*m^2]
I_Fyy = 0.28/2 # [kg*m^2]
#-----------------------------------------#

# Calculate total mass and center of mass location
m_T = m_R + m_B + m_H + m_F
x_T = (x_B*m_B + x_H*m_H + w*m_F)/m_T
z_T = (-r_R*m_R + z_B*m_B + z_H*m_H - r_F*m_F)/m_T

# Calculate total mass moment of inertia
I_Txx = I_Rxx + I_Bxx + I_Hxx + I_Fxx + m_R*(r_R**2) + m_B*(z_B**2) + m_H*(z_H**2) + m_F*(r_F**2)
I_Txz = I_Bxz + I_Hxz - m_B*x_B*z_B - m_H*x_H*z_H + m_F*w*r_F

# Calculate symmetric mass moment of inertias
I_Rzz = I_Rxx
I_Fzz = I_Fxx

# Calculate total z-axis mass moment of inertia
I_Tzz = I_Rzz + I_Bzz + I_Hzz + I_Fzz + m_B*(x_B**2) + m_H*(x_H**2) + m_F*(w**2)

# Calculate mass and center of mass location for the front assembly
m_A = m_H + m_F
x_A = (x_H*m_H + w*m_F)/m_A
z_A = (z_H*m_H - r_F*m_F)/m_A

# Calculate the mass moment of inertias for the fron assembly
I_Axx = I_Hxx + I_Fxx + m_H*((z_H-z_A)**2) + m_F*((r_F+z_A)**2)
I_Axz = I_Hxz - m_H*(x_H-x_A)*(z_H-z_A) + m_F*(w-x_A)*(r_F+z_A)
I_Azz = I_Hzz + I_Fzz + m_H*((x_H-x_A)**2) + m_F*((w-x_A)**2)

# Perpendicular distance of the center of mass of the fron assembly and the steering axis
u_A = (x_A - w - c)*math.cos(lamda) - z_A*math.sin(lamda)

# Mass moment of inertia about the steering axis of the front assembly
I_Alamdalamda = m_A*(u_A**2) + I_Axx*(math.sin(lamda)**2) + 2*I_Axz*math.sin(lamda)*math.cos(lamda) + I_Azz*(math.cos(lamda)**2)
I_Alamdax = -m_A*u_A*z_A + I_Axx*math.sin(lamda) + I_Axz*math.cos(lamda)
I_Alamdaz = m_A*u_A*x_A + I_Axz*math.sin(lamda) + I_Azz*math.cos(lamda)

# Trail ratio
mu = c/w*math.cos(lamda)

# gyrostatic coefficients
S_R = I_Ryy/r_R
S_F = I_Fyy/r_F
S_T = S_R + S_F
S_A = m_A*u_A + mu*m_T*x_T

# Mass matrix components
M_phiphi = I_Txx
M_phidelta = I_Alamdax + mu*I_Txz
M_deltaphi = M_phidelta
M_deltadelta = I_Alamdalamda + 2*mu*I_Alamdaz + (mu**2)*I_Tzz

# Mass matrix
M = [[M_phiphi,M_phidelta], [M_deltaphi,M_deltadelta]]

# Non-velocity dependent stiffness components
K_0phiphi = m_T*z_T
K_0phidelta = -S_A
K_0deltaphi = K_0phidelta
K_0deltadelta = -S_A*math.sin(lamda)

# Non-velocity dependent stiffnes matrix
K_0 = [[K_0phiphi,K_0phidelta], [K_0deltaphi,K_0deltadelta]]

# velocity dependent stiffness components
K_2phiphi = 0
K_2phidelta = (S_T-m_T*z_T)/w*math.cos(lamda)
K_2deltaphi = 0
K_2deltadelta = (S_A+S_F*math.sin(lamda))/w*math.cos(lamda)

# velocity dependent stiffness matrix
K_2 = [[K_2phiphi,K_2phidelta], [K_2deltaphi,K_2deltadelta]]

# Damping constant components
C_1phiphi = 0
C_1phidelta = mu*S_T + S_F*math.cos(lamda) + I_Txz/w*math.cos(lamda) - mu*m_T*z_T
C_1deltaphi = -(mu*S_T + S_F*math.cos(lamda))
C_1deltadelta = I_Alamdaz*math.cos(lamda)/w + mu*(S_A+I_Tzz/w*math.cos(lamda))

# Damping constant matrix
C_1 = [[C_1phiphi,C_1phidelta], [C_1deltaphi,C_1deltadelta]]

# print(C_1)