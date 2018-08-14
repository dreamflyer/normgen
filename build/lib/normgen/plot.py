import matplotlib.pyplot as plt
import image_util

from mpl_toolkits.axes_grid1 import Grid

import numpy

import keras

plt.ion()

class SimplePlot():
    def __init__(self, name):
        self.figure, self.ax = plt.subplots()

        self.figure.canvas.set_window_title(name + " loss")
        self.lines, = self.ax.plot([],[])
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(0, 1)
        self.ax.grid()

    def update(self, ydata):
        xdata = list(range(len(ydata)))

        self.lines, = self.ax.plot([],[])
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(0, len(xdata))
        self.ax.grid()
        self.on_running(xdata, ydata)

    def on_running(self, xdata, ydata):
        #Update data (with the new _and_ the old points)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

class ImagePlot():
    def __init__(self):
        self.figure = plt.figure(figsize=(6 * 0.9, 10 * 0.9))

        self.ax = Grid(self.figure, rect=111, nrows_ncols=(5, 1), axes_pad = 0.0)

    def compose(self, input, output, fake):
        return numpy.concatenate((input, output, fake), 3)

    def update(self, input, output, fake):
        images = self.compose(input, output, fake)

        for i in range(len(images)):
            img = images[i]
            self.ax[i].imshow(image_util.convert_to_rgb(img, False))
            self.ax[i].axis('off')

        plt.tight_layout()

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()