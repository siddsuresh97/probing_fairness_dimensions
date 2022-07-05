import os
import numpy as np
import psychopy.visual
import psychopy.event



def draw_and_save_circle(save_dir, color_list, radius_list):
    win = psychopy.visual.Window(
        size=[400, 400],
        units="pix",
        fullscr=False,
        color=[1, 1, 1]
    )
    for color in color_list:
        for radius in radius_list:
            circle = psychopy.visual.Circle(
                win=win,
                units="norm",
                radius=radius,
                fillColor=color,
                lineColor=[-1, -1, -1], 
                edges=128
            )
            circle.draw()
            win.getMovieFrame(buffer="back")
            win.saveMovieFrames(os.path.join(save_dir, 'color_{}_radius_{}.png'.format(color, radius))) 
            win.flip()
    win.close()
    return
