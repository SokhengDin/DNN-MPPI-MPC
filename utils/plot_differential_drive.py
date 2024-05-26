import numpy as np
import matplotlib.pyplot as plt

from models.differentialSim import DifferentialSimulation

simulation = DifferentialSimulation()
plt.figure(figsize=(12, 7))

class Simulation:

    @staticmethod
    def plot_animation(x: float, y: float, yaw: float):
        
        plt.clf()
        plt.gcf().canvas.mpl_connect('key_release_event',
                                    lambda event: [exit(0) if event.key == 'escape' else None] 
                                    )
        simulation.generate_each_wheel_and_draw(x, y, yaw)
        plt.axis("equal")
        plt.title("Differential Drive Robot Simulation")
        plt.grid(True)
        plt.pause(0.001)