import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math


def Rot2D(X, Y, Alpha):  # rotates point (X,Y) on angle alpha with respect to Origin
    RX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RX, RY

#defining t as a symbol (it will be the independent variable)
t = sp.Symbol('t')
#here r, phi, x, y, Vx, Vy, Wx, Wy, xC are functions of 't'
r = 5 - 0.5*t
phi = 2*t

x = r*sp.cos(phi)
y = r*sp.sin(phi)

Vx = sp.diff(x, t)
print("Vx =", Vx)

Vy = sp.diff(y, t)
print("Vy =", Vy)

Vmod = sp.sqrt(Vx*Vx+Vy*Vy)
Wx = sp.diff(Vx, t)
print("Wx =",Wx)
Wy = sp.diff(Vy, t)
print("Wy =",Wy)
Wmod = sp.sqrt(Wx*Wx+Wy*Wy)
# and here really we could escape integrating, just don't forget that it's absolute value of V here we should differentiate
Wtau = sp.diff(Vmod, t)
Wn = sp.sqrt(Wmod*Wmod - Wtau*Wtau)
# this is the value of rho but in the picture you should draw the radius, don' t forget!
Xr = - Vy*(Vx*Vx+Vy*Vy)/(Vx*Wy-Wx*Vy)
Yr = + Vx*(Vx*Vx+Vy*Vy)/(Vx*Wy-Wx*Vy)
rho = sp.sqrt(Xr*Xr + Yr*Yr)

#constructing and filling arrays with corresponding values
a = np.arange(0, 10, 0.01)
f = sp.lambdify(t, x, "numpy")
X = f(a)
f = sp.lambdify(t, y, "numpy")
Y = f(a)
f = sp.lambdify(t, Vx, "numpy")
VX = f(a)
f = sp.lambdify(t, Vy, "numpy")
VY = f(a)
f = sp.lambdify(t, Vmod, "numpy")
VMOD = f(a)
f = sp.lambdify(t, Xr, "numpy")
XR = f(a)
f = sp.lambdify(t, Yr, "numpy")
YR = f(a)
f = sp.lambdify(t, Wtau, "numpy")
WTAU = f(a)
f = sp.lambdify(t, Wn, "numpy")
WN = f(a)
f = sp.lambdify(t, rho, "numpy")
RHO = f(a)

# here we start to plot
fig = plt.figure()
fig.set(facecolor = "#A9A9A9")

ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
R = 2
ax1.set(xlim=[-2*R, 2*R], ylim=[-3*R, 3*R])
# plotting a trajectory
ax1.plot(X, Y)

P, = ax1.plot(X[0], Y[0], marker='o')
# of the velocity vector of this point (line)
#velocity vector
VLine, = ax1.plot([X[0], X[0]+VX[0]], [Y[0], Y[0]+VY[0]], color="r")
# Vector of radius of curvature
RLine, = ax1.plot([X[0], X[0]+XR[0]], [Y[0], Y[0]+YR[0]], color="c")
#vector of tangential acceleration
WTLine, = ax1.plot([X[0], X[0]+VX[0]/VMOD[0]*WTAU[0]], [Y[0], Y[0]+VY[0]/VMOD[0]*WTAU[0]], color="b")
#vector of normal acceleration
WNLine, = ax1.plot([X[0], X[0]+XR[0]/RHO[0]*WN[0]], [Y[0], Y[0]+YR[0]/RHO[0]*WN[0]], color="m")
# Arrows of the vectors
ArrowX = np.array([-0.2*R, 0, -0.2*R])
ArrowY = np.array([0.1*R, 0, -0.1*R])
RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
VArrow, = ax1.plot(RArrowX + X[0]+VX[0], RArrowY + Y[0]+VY[0], color="r")

ArrowXR = np.array([-0.2*R, 0, -0.2*R])
ArrowYR = np.array([0.1*R, 0, -0.1*R])
RArrowXR, RArrowYR = Rot2D(ArrowXR, ArrowYR, math.atan2(YR[0], XR[0]))
VArrowR, = ax1.plot(RArrowX + X[0] + XR[0], RArrowY + Y[0] + YR[0], color="c")

ArrowXWT = np.array([-0.2*R, 0, -0.2*R])
ArrowYWT = np.array([0.1*R, 0, -0.1*R])
RArrowXWT, RArrowYWT = Rot2D( ArrowXWT, ArrowYWT, math.atan2(VY[0]/VMOD[0]*WTAU[0], VX[0]/VMOD[0]*WTAU[0]))
WTArrow, = ax1.plot(RArrowXWT + X[0]+VX[0]/VMOD[0]*WTAU[0], RArrowYWT + Y[0]+VY[0]/VMOD[0]*WTAU[0], color="b")

ArrowXWN = np.array([-0.2*R, 0, -0.2*R])
ArrowYWN = np.array([0.1*R, 0, -0.1*R])
RArrowXWN, RArrowYWN = Rot2D(
    ArrowXWN, ArrowYWN, math.atan2(YR[0]/RHO[0]*WN[0], XR[0]/RHO[0]*WN[0]))
WNArrow, = ax1.plot(RArrowXWN + X[0]+XR[0]/RHO[0]*WN[0], RArrowYWN + Y[0]+YR[0]/RHO[0]*WN[0], color="m")

# function for recounting the positions
def anima(i):
    P.set_data(X[i], Y[i])
    VLine.set_data([X[i], X[i]+VX[i]], [Y[i], Y[i]+VY[i]])
    RLine.set_data([X[i], X[i]+XR[i]], [Y[i], Y[i]+YR[i]])
    WTLine.set_data([X[i], X[i]+VX[i]/VMOD[i]*WTAU[i]], [Y[i], Y[i]+VY[i]/VMOD[i]*WTAU[i]])
    WNLine.set_data([X[i], X[i]+XR[i]/RHO[i]*WN[i]], [Y[i], Y[i]+YR[i]/RHO[i]*WN[i]])

    RArrowXR, RArrowYR = Rot2D(ArrowXR, ArrowYR, math.atan2(YR[i], XR[i]))
    VArrowR.set_data(RArrowXR + X[i] + XR[i], RArrowYR + Y[i] + YR[i])

    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(RArrowX + X[i]+VX[i], RArrowY + Y[i]+VY[i])

    RArrowXWT, RArrowYWT = Rot2D(ArrowXWT, ArrowYWT, math.atan2(
        VY[i]/VMOD[i]*WTAU[i], VX[i]/VMOD[i]*WTAU[i]))
    WTArrow.set_data(RArrowXWT + X[i]+VX[i]/VMOD[i]
                     * WTAU[i], RArrowYWT + Y[i]+VY[i]/VMOD[i]*WTAU[i])

    RArrowXWN, RArrowYWN = Rot2D(ArrowXWN, ArrowYWN, math.atan2(YR[i]/RHO[i]*WN[i], XR[i]/RHO[i]*WN[i]))
    WNArrow.set_data(RArrowXWN + X[i]+XR[i]/RHO[i]
                     * WN[i], RArrowYWN + Y[i]+YR[i]/RHO[i]*WN[i])
    return P, WTLine, VLine, WNLine, RLine, VArrowR, VArrow, WTArrow, WNArrow,


# animation function
anim = FuncAnimation(fig, anima,
                     frames=1000, interval=80, blit=True)

plt.show()
