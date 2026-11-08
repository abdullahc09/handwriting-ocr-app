import cv2
import numpy as np
def split_page(page_image):
    numpy_image = np.array(page_image.convert("RGB"))
    grayscale_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
    chosen_threshold, bandw_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_ink_counts = np.count_nonzero(bandw_image, axis=1)
    line_ranges = []
    inside_line = False
    start_row = None
    ink_threshold = 100
    minimum_line_height = 20
    for row_index, ink_count in enumerate(row_ink_counts):
        if ink_count >= ink_threshold and inside_line == False:
            inside_line = True
            start_row = row_index
        elif ink_count <= ink_threshold and inside_line == True:
            inside_line = False
            line_range = start_row, row_index
            if row_index - start_row >= minimum_line_height:
                
                line_ranges.append(line_range)
    return bandw_image, row_ink_counts, line_ranges