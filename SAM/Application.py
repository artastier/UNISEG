# @author Damien GAUTIER
# This python file corresponds to an application allowing to :
# - Load the SAM predictor
# - Choose the image to segment and the ground truth mask associated
# - Select the points belonging to the mask (left click) and the points belonging to the background (right click)
# - Predict the masks with those points prompt
# - Save the figure with the masks predicted and the points chosen

# The button to choose the image first took the image and then the ground truth mask.
# It also starts the embedding which can take some time (around 20 or 25 seconds) and can
# freeze your computer.
# Once an image has been selected, you can predict masks as much as you like. But, the prediction is always made with
# the last points selected with the window when you click on the Button : "Select points to prompt".
# For selecting the points belonging to the mask use the left click and the right click for the points belonging
# to the background.
# Once you have selected all the points you wanted just close the window opened when clicking on "Select points to
# prompt".

# Color legend plotted :
# POINTS :
# - mask (left click) : green
# - background (right click) : red
# MASKS :
# - ground truth mask : green
# - predicted mask by SAM : red
# - intersection of ground truth mask and predicted mask : yellow

# imports
# UI
from tkinter import *
from tkinter.filedialog import askopenfilename
# SAM model
from segment_anything import sam_model_registry, SamPredictor
# Analysing
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import skimage.io as io
# for test
import time
# Compare ground_truth and prediction
from Compare import color_enhanced
import os
import datetime


class App:

    def __init__(self):
        # Variables for plotting and computing
        self.fig = None
        self.img_points_selected = None
        self.img_gray = None
        self.gt_mask = None
        self.mask_path = None
        self.image_path = None
        self.img = None
        self.img_rgb = None
        # Variables for storing the prompts
        self.input_coords = []
        self.input_labels = []

        # Creating the window for the app
        self.window = Tk()
        self.window.title("Application for SAM")
        height = 600
        width = 600
        self.window.minsize(width=width, height=height)

        # Adding all the components
        self.frame = Frame(self.window)
        # The button for choosing the image
        img_choice_button = Button(self.frame, font=("Courrier", 25), bg='white',
                               fg='#41B77F', text="Load \n >image \n >ground truth mask", command=self.img_choice)
        img_choice_button.pack(pady=25, fill=X)
        # The button for prompting points
        points_button = Button(self.frame, text="Select points to prompt", font=("Courrier", 25), bg='white',
                               fg='#41B77F',
                               command=self.points_selection)
        points_button.pack(pady=25, fill=X)
        # The button for predicting
        predict_button = Button(self.frame, font=("Courrier", 25), bg='white',
                               fg='#41B77F', text="Predict", command=self.predict)
        predict_button.pack(pady=25, fill=X)
        # The button for saving the result
        img_save_button = Button(self.frame, font=("Courrier", 25), bg='white',
                               fg='#41B77F', text="Save the prediction", command=self.img_save)
        img_save_button.pack(pady=25, fill=X)
        self.frame.pack(expand=True)

        # Loading the SAM model
        self.sam_checkpoint = "sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        tic = time.perf_counter()
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.predictor = SamPredictor(self.sam)
        toc = time.perf_counter()
        print(f'Charging the predictor took : {toc - tic}s')

    def predict(self):
        masks, _, _ = self.predictor.predict(
            point_coords=np.array(self.input_coords),
            point_labels=np.array(self.input_labels),
            box=None,
            multimask_output=False,
        )

        # Showing the masks generated
        self.fig, ax = plt.subplots(1, 2)
        ax[0].imshow(color_enhanced(self.img_gray, self.gt_mask, masks[0]), cmap='gray')
        ax[0].set_title(
            'Green: Truth Red: Predic Yellow: Inter')
        # Showing the points selected
        ax[1].imshow(cv2.cvtColor(self.img_points_selected, cv2.COLOR_BGR2RGB))
        self.show_points(coords=np.array(self.input_coords), labels=np.array(self.input_labels), ax=ax[1], marker_size=(72./self.fig.dpi)**2)
        ax[1].set_title('Green: In Red: Background')
        plt.suptitle(f'Prediction for the : {os.path.basename(self.image_path)}')
        plt.axis('on')
        plt.show()

    def points_selection(self):
        # Set parameters
        self.input_coords = []  # Clear the storage
        self.input_labels = []  # of the previous prompt
        window_name = 'Window'
        key_for_exit_window = 'q'
        self.img_points_selected = copy.deepcopy(self.img)

        cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)

        # displaying the image
        cv2.imshow(window_name, self.img_points_selected)

        # setting mouse handler for the image and calling the click_event() function
        cv2.setMouseCallback(window_name, self.click_event,
                             param={'window_name': window_name, 'image': self.img_points_selected})

        # wait for a specific key to be pressed or window closing to exit
        wait_time = 1000
        while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            keyCode = cv2.waitKey(wait_time)
            if (keyCode & 0xFF) == ord(key_for_exit_window):
                # close the window
                cv2.destroyAllWindows()
                break

    def img_choice(self):
        # Clear the storage of the previous prompt because the image will change
        self.input_coords = []
        self.input_labels = []

        # Loading the following image
        self.image_path = askopenfilename()
        self.img = cv2.imread(self.image_path, 1)
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img_gray = io.imread(self.image_path, as_gray=True).astype(float)

        # Loading the mask
        self.mask_path = askopenfilename()
        self.gt_mask = io.imread(self.mask_path, as_gray=True).astype(float)

        # Setting it to the predictor
        toc = time.perf_counter()
        self.predictor.set_image(self.img_rgb)
        tac = time.perf_counter()
        print(f"embedding the image took : {tac - toc}s")

    # function to display the coordinates of the points clicked on the image
    def click_event(self, event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates with the point on the image window
            params['image'][y, x] = [0, 255, 0]
            cv2.imshow(params['window_name'], params['image'])

            # storing the point selected
            self.input_coords.append([x, y])
            self.input_labels.append(1)

        # checking for right mouse clicks
        if event == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates with the point on the image window
            params['image'][y, x] = [0, 0, 255]
            cv2.imshow(params['window_name'], params['image'])

            # storing the point selected
            self.input_coords.append([x, y])
            self.input_labels.append(0)

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=1):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',marker='o', s=marker_size)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',marker='o', s=marker_size)

    def img_save(self):
        ts = datetime.datetime.now().strftime("%Y%m%d-%H.%M.%S")
        nb_in = self.input_labels.count(1)
        nb_out = self.input_labels.count(0)
        self.fig.savefig(f"../results/{ts}--{nb_in}i-{nb_out}o--{os.path.basename(self.image_path)}")
        print(f"Saved image to ../results/{ts}--{nb_in}i-{nb_out}o--{os.path.basename(self.image_path)}")

if __name__ == "__main__":
    app = App()
    app.window.mainloop()
