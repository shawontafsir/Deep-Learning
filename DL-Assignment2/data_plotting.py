import matplotlib.pyplot as plt


class PlotGeneration:

    @classmethod
    def line_plotting(cls, array_2d, labels):
        for i in range(len(array_2d)):
            plt.plot(array_2d[i], label=labels[i])
            plt.xlabel("Number of Iterations")
            plt.ylabel("Loss")

        plt.legend()
        plt.show()
