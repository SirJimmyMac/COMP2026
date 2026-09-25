import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad


# ---------------------------------------
# 1. Define symbolic variable
# ---------------------------------------

z = sp.symbols('z')


# ---------------------------------------
# 2. User inputs
# ---------------------------------------

expr_string = input("Enter f(z): ")
R = float(input("Enter contour radius: "))

# Convert typed string into a symbolic SymPy expression
expr = sp.sympify(expr_string)

# Convert symbolic expression into a numerical function
f = sp.lambdify(z, expr, "numpy")


# ---------------------------------------
# 3. Define the contour
#    Counterclockwise circle centered at 0
# ---------------------------------------

def contour(t):
    return R * np.exp(1j * t)


def contour_derivative(t):
    return 1j * R * np.exp(1j * t)


# ---------------------------------------
# 4. Numerical contour integral
#    LHS = integral_C f(z) dz
# ---------------------------------------

def contour_integral(f, contour, contour_derivative):

    def integrand(t):
        return f(contour(t)) * contour_derivative(t)

    real_part = quad(
        lambda t: np.real(integrand(t)),
        0,
        2 * np.pi
    )[0]

    imag_part = quad(
        lambda t: np.imag(integrand(t)),
        0,
        2 * np.pi
    )[0]

    return real_part + 1j * imag_part


lhs = contour_integral(
    f,
    contour,
    contour_derivative
)


# ---------------------------------------
# 5. Numerical winding number
#
# n(C,p) = 1/(2*pi*i) integral_C dz/(z-p)
# ---------------------------------------

def winding_number(contour, contour_derivative, pole):

    def integrand(t):
        return contour_derivative(t) / (
            contour(t) - complex(pole)
        )

    real_part = quad(
        lambda t: np.real(integrand(t)),
        0,
        2 * np.pi
    )[0]

    imag_part = quad(
        lambda t: np.imag(integrand(t)),
        0,
        2 * np.pi
    )[0]

    integral = real_part + 1j * imag_part

    n = integral / (2 * np.pi * 1j)

    # Winding number should be an integer
    return int(round(np.real(n)))


# ---------------------------------------
# 6. Residue-theorem RHS
# ---------------------------------------

# Get denominator of f(z)
denominator = sp.denom(expr)

# Find poles automatically by solving denominator = 0
poles = sp.solve(denominator, z)

print()
print("Poles found automatically:", poles)
print()

rhs_sum = 0


for pole in poles:

    # Find residue symbolically
    residue = sp.residue(expr, z, pole)

    # Find winding number numerically
    winding = winding_number(
        contour,
        contour_derivative,
        pole
    )

    print("Pole =", pole)
    print("Residue =", residue)
    print("Winding number =", winding)
    print()

    rhs_sum += winding * complex(residue)


rhs = 2 * np.pi * 1j * rhs_sum


# ---------------------------------------
# 7. Compare LHS and RHS
# ---------------------------------------

error = abs(lhs - rhs)

print("--------------------------------")
print("Numerical contour integral")
print("LHS =", lhs)

print()

print("Residue theorem")
print("RHS =", rhs)

print()

print("Absolute error =", error)


# ---------------------------------------
# 8. Plot contour and poles
# ---------------------------------------

t_values = np.linspace(
    0,
    2 * np.pi,
    500
)

z_values = contour(t_values)

plt.figure(figsize=(6, 6))


# Plot contour
plt.plot(
    np.real(z_values),
    np.imag(z_values),
    label="Contour C"
)


# Plot poles
for pole in poles:

    pole_complex = complex(pole)

    plt.plot(
        np.real(pole_complex),
        np.imag(pole_complex),
        "x",
        markersize=10
    )

    plt.text(
        np.real(pole_complex) + 0.1,
        np.imag(pole_complex) + 0.1,
        f"pole = {pole}"
    )


# Draw real and imaginary axes
plt.axhline(0)
plt.axvline(0)


plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.title("Contour and Poles")

plt.axis("equal")
plt.grid(True)
plt.legend()

plt.show()
#%%
#%%
# ---------------------------------------
# Automated stress tests
# ---------------------------------------

import numpy as np
import sympy as sp
from scipy.integrate import quad

z = sp.symbols('z')


def contour_integral_test(expr, R):

    f = sp.lambdify(z, expr, "numpy")

    def contour(t):
        return R * np.exp(1j * t)

    def contour_derivative(t):
        return 1j * R * np.exp(1j * t)

    def integrand(t):
        return f(contour(t)) * contour_derivative(t)

    real_part = quad(
        lambda t: np.real(integrand(t)),
        0,
        2 * np.pi
    )[0]

    imag_part = quad(
        lambda t: np.imag(integrand(t)),
        0,
        2 * np.pi
    )[0]

    return real_part + 1j * imag_part


def winding_number_test(R, pole):

    def contour(t):
        return R * np.exp(1j * t)

    def contour_derivative(t):
        return 1j * R * np.exp(1j * t)

    def integrand(t):
        return contour_derivative(t) / (
            contour(t) - complex(pole)
        )

    real_part = quad(
        lambda t: np.real(integrand(t)),
        0,
        2 * np.pi
    )[0]

    imag_part = quad(
        lambda t: np.imag(integrand(t)),
        0,
        2 * np.pi
    )[0]

    integral = real_part + 1j * imag_part

    n = integral / (2 * np.pi * 1j)

    return int(round(np.real(n)))


test_cases = [

    {
        "name": "One pole inside",
        "expr": 1 / ((z - 1) * (z + 2)),
        "R": 1.5
    },

    {
        "name": "Two poles inside",
        "expr": 1 / ((z - 1) * (z + 1)),
        "R": 2.0
    },

    {
        "name": "No poles inside",
        "expr": 1 / ((z - 3) * (z + 4)),
        "R": 1.5
    }

]


for test in test_cases:

    expr = test["expr"]
    R = test["R"]

    lhs = contour_integral_test(expr, R)

    denominator = sp.denom(expr)
    poles = sp.solve(denominator, z)

    rhs_sum = 0

    for pole in poles:

        residue = sp.residue(expr, z, pole)

        winding = winding_number_test(
            R,
            pole
        )

        rhs_sum += winding * complex(residue)

    rhs = 2 * np.pi * 1j * rhs_sum

    error = abs(lhs - rhs)

    print("--------------------------------")
    print("Test:", test["name"])
    print("Function:", expr)
    print("Radius:", R)
    print("Poles:", poles)
    print("LHS =", lhs)
    print("RHS =", rhs)
    print("Error =", error)
    print()
# %%
