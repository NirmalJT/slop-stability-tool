import matplotlib.pyplot as plt
import math


def calculate_fos(c, phi, gamma, H, slope_angle):

    phi_rad = math.radians(phi)
    beta = math.radians(slope_angle)

    resisting_force = c + (gamma * H * math.cos(beta) * math.tan(phi_rad))
    driving_force = gamma * H * math.sin(beta)

    fos = resisting_force / driving_force

    return round(fos, 3)


def plot_slope(H, slope_angle):

    import numpy as np

    length = H / math.tan(math.radians(slope_angle))

    x = [0, length, length + 5]
    y = [H, 0, 0]

    plt.figure()

    # plot slope
    plt.plot(x, y, marker="o", label="Slope")

    center_x = length / 2
    center_y = -H

    min_radius = H * 1.2
    max_radius = H * 2.5

    radii = np.linspace(min_radius, max_radius, 20)

    critical_radius = None
    min_fos = 999

    for r in radii:

        theta = np.linspace(30,150,100)

        arc_x = center_x + r * np.cos(np.radians(theta))
        arc_y = center_y + r * np.sin(np.radians(theta))

        fos = r / (H + 1)

        if fos < min_fos:
            min_fos = fos
            critical_radius = r

        plt.plot(arc_x, arc_y, color="gray", alpha=0.4)

    # plot critical surface
    theta = np.linspace(30,150,100)

    arc_x = center_x + critical_radius * np.cos(np.radians(theta))
    arc_y = center_y + critical_radius * np.sin(np.radians(theta))

    plt.plot(arc_x, arc_y, color="red", linewidth=3, label="Critical Slip Surface")

    plt.title("Slope Stability Analysis")
    plt.xlabel("Distance")
    plt.ylabel("Height")

    plt.legend()
    plt.grid()

    plt.savefig("static/slope.png")

    plt.close()


def plot_fos_vs_slope(c, phi, gamma, H):

    import numpy as np

    slopes = [20, 30, 40, 50, 60]

    fos_values = []

    for s in slopes:
        fos = calculate_fos(c, phi, gamma, H, s)
        fos_values.append(fos)

    plt.figure()

    plt.plot(slopes, fos_values, marker="o")

    plt.title("FOS vs Slope Angle")
    plt.xlabel("Slope Angle (degrees)")
    plt.ylabel("Factor of Safety")

    plt.grid()

    plt.savefig("static/fos_graph.png")

    plt.close()