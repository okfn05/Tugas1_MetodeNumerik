import pandas as pd
import numpy as np
from math import sqrt

# -----------------------------
# Definisi sistem persamaan
# -----------------------------
def f1(x, y): return x**2 + x*y - 10
def f2(x, y): return y + 3*x*y**2 - 57
def F(vec): return np.array([f1(vec[0], vec[1]), f2(vec[0], vec[1])])

def g1A(x, y): return (-y + sqrt(y**2 + 40)) / 2
def g2A(x, y): return (-1 + sqrt(1 + 684*x)) / (6*x)

# -----------------------------
# Metode Iterasi Jacobi
# -----------------------------
def iterasi_jacobi(x0, y0, tol=1e-6, maxit=100):
    data = []
    x, y = x0, y0
    data.append((0, x, y, 0, 0))
    for r in range(1, maxit+1):
        x_new = g1A(x, y)
        y_new = g2A(x, y)
        dx, dy = abs(x_new - x), abs(y_new - y)
        data.append((r, x_new, y_new, dx, dy))
        if max(dx, dy) < tol:
            break
        x, y = x_new, y_new
    df = pd.DataFrame(data, columns=["Iterasi","x","y","deltaX","deltaY"]).round(9)
    return df["x"].iloc[-1], df["y"].iloc[-1], int(df["Iterasi"].iloc[-1]), df

# -----------------------------
# Metode Iterasi Seidel
# -----------------------------
def iterasi_seidel(x0, y0, tol=1e-6, maxit=100):
    data = []
    x, y = x0, y0
    data.append((0, x, y, 0, 0))
    for r in range(1, maxit+1):
        x_new = g1A(x, y)
        y_new = g2A(x_new, y)
        dx, dy = abs(x_new - x), abs(y_new - y)
        data.append((r, x_new, y_new, dx, dy))
        if max(dx, dy) < tol:
            break
        x, y = x_new, y_new
    df = pd.DataFrame(data, columns=["Iterasi","x","y","deltaX","deltaY"]).round(9)
    return df["x"].iloc[-1], df["y"].iloc[-1], int(df["Iterasi"].iloc[-1]), df

# -----------------------------
# Newton-Raphson
# -----------------------------
def jacobian(x, y): return np.array([[2*x + y, x],[3*y**2, 1 + 6*x*y]])
def newton(x0, y0, tol=1e-6, maxit=50):
    data = []
    x, y = x0, y0
    data.append((0, x, y, 0, 0))
    for r in range(1, maxit+1):
        J = jacobian(x, y)
        delta = np.linalg.solve(J, -F([x, y]))
        x_new, y_new = x + delta[0], y + delta[1]
        dx, dy = abs(x_new - x), abs(y_new - y)
        data.append((r, x_new, y_new, dx, dy))
        if max(dx, dy) < tol:
            break
        x, y = x_new, y_new
    df = pd.DataFrame(data, columns=["Iterasi","x","y","deltaX","deltaY"]).round(9)
    return df["x"].iloc[-1], df["y"].iloc[-1], int(df["Iterasi"].iloc[-1]), df

# -----------------------------
# Secant (Broyden)
# -----------------------------
def broyden(x0, y0, tol=1e-6, maxit=50):
    data = []
    x = np.array([x0, y0], dtype=float)
    data.append((0, x[0], x[1], 0, 0))
    B = jacobian(x[0], x[1])
    for r in range(1, maxit+1):
        s = np.linalg.solve(B, -F(x))
        x_new = x + s
        dx, dy = abs(x_new[0] - x[0]), abs(x_new[1] - x[1])
        data.append((r, x_new[0], x_new[1], dx, dy))
        if max(dx, dy) < tol:
            break
        y = F(x_new) - F(x)
        B = B + np.outer((y - B @ s), s) / (s @ s)
        x = x_new
    df = pd.DataFrame(data, columns=["Iterasi","x","y","deltaX","deltaY"]).round(9)
    return df["x"].iloc[-1], df["y"].iloc[-1], int(df["Iterasi"].iloc[-1]), df

# -----------------------------
# Main Program
# -----------------------------
if __name__ == "__main__":
    x0, y0 = 1.5, 3.5

    print("Tebakan awal:", x0, y0)

    xj, yj, itj, df_jacobi = iterasi_jacobi(x0, y0)
    print(f"IT Jacobi: x={xj:.6f}, y={yj:.6f}, iter={itj}")

    xs, ys, its, df_seidel = iterasi_seidel(x0, y0)
    print(f"IT Seidel: x={xs:.6f}, y={ys:.6f}, iter={its}")

    xn, yn, itn, df_newton = newton(x0, y0)
    print(f"Newton-Raphson: x={xn:.6f}, y={yn:.6f}, iter={itn}")

    xb, yb, itb, df_broyden = broyden(x0, y0)
    print(f"Secant (Broyden): x={xb:.6f}, y={yb:.6f}, iter={itb}")

    # Simpan ke Excel
    with pd.ExcelWriter("hasil_iterasi.xlsx", engine="openpyxl") as writer:
        df_jacobi.to_excel(writer, sheet_name="Jacobi", index=False)
        df_seidel.to_excel(writer, sheet_name="Seidel", index=False)
        df_newton.to_excel(writer, sheet_name="Newton", index=False)
        df_broyden.to_excel(writer, sheet_name="Secant_Broyden", index=False)

    print("File Excel 'hasil_iterasi.xlsx' berhasil dibuat.")
