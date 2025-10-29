"""

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import random

import numpy as np
from PIL import Image, ImageFilter
from ultralytics import YOLO


class Scrambler:
    def __init__(self):
        # This way the model should be downloaded automatically
        self.model = YOLO('yolo11x-seg.pt')

    def process_img(self, img_path, block_size=5, blur_radius=0, out_file=None, b_verbose=False, b_show=True):
        """

        :param img_path: path to the image to be processed
        :param block_size: size of the "blocks" (or "patches" if you prefer) to be randomly swapped around
        :param blur_radius: size of the blur to be added to the scrambled background; default = 0 = no blur
        :param out_file: path to the output file to be generated of the processed image; default = None = output is not saved
        :param b_verbose: if True, print some additional info; False by default
        :param b_show: if True, will show the original and processed images
        :return: processed image as PIL image
        """
        # Open image
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        # Process image with YOLOv11
        res = self.model(img_path)[0]

        # Extract people
        mask_people = self._extract_people(res)
        if mask_people is None:
            print("No people detected in image.")
            return
        mask_people, resize_h, resize_w = mask_people

        # Make image with only people
        resized_img = img.resize((resize_w, resize_h))
        if b_show:
            resized_img.show()

        np_image = np.array(resized_img)

        filter_people = np_image*mask_people[:,:,None]
        # Debugging:
        # img_people = Image.fromarray(filter_people, mode='RGB')
        # img_people.show()

        # Make image without people
        mask_no_people = mask_people.astype(np.int8)
        mask_no_people = np.abs((mask_no_people - 1)).astype(np.uint8)
        filter_no_people = np_image*mask_no_people[:,:,None]
        # Debugging:
        # img_no_people = Image.fromarray(filter_no_people, mode='RGB')
        # img_no_people.show()

        # Scramble image without people
        np_scrambled_img = self._scramble(filter_no_people, block_size=block_size, b_verbose=b_verbose)

        # Blur scrambled image?
        if blur_radius > 0:
            img_scrambled = Image.fromarray(np_scrambled_img, mode='RGB')
            img_scrambled = img_scrambled.filter(ImageFilter.BoxBlur(radius=blur_radius))
            # Beware of blurry edges that bleed into the people, as these generate artifacts when
            # putting the blurred image and the people back together again
            # That why we multiply by 'mask_no_people[:,:,None]' again
            np_scrambled_img = np.array(img_scrambled)*mask_no_people[:,:,None]

        # Debugging:
        # Image.fromarray(np_scrambled_img, mode='RGB').show()
        # Image.fromarray(filter_people, mode='RGB').show()

        # Put people back in scrambled image
        np_res_image = np_scrambled_img + filter_people
        res_image = Image.fromarray(np_res_image, mode='RGB')
        if b_show:
            res_image.show()
        if out_file is not None:
            res_image.save(fp=out_file)

        return res_image

    def _extract_people(self, res):
        idxs_people = set()
        # Gather indices of results corresponding to people
        for idx_res, idx_class in enumerate(res.boxes.cls):
            if idx_class == 0:
                idxs_people.add(idx_res)

        if not idxs_people:
            return None


        # Extract masks corresponding to people
        _, h, w = res.masks[0].shape

        mask_people = np.zeros((h, w))
        for idx_mask, mask in enumerate(res.masks):
            if idx_mask not in idxs_people:
                continue

            mask_np = mask.data.squeeze(0).cpu().numpy()
            mask_people += mask_np
        mask_people = np.clip(mask_people, 0., 1.).astype(np.uint8)

        return mask_people, h, w

    def _scramble(self, np_img, block_size, b_verbose=False):
        # Areas of block_size x block_size that are full black are ignored, since we assume these correspond to
        # the people that have been cut out of the image
        blocks = []
        at_block = -1
        y_range = np_img.shape[0]//block_size
        x_range = np_img.shape[1]//block_size

        if b_verbose:
            print(f"np_img.shape = {np_img.shape}")
            print(f"block_size = {block_size}")
            print(f"x_range = {x_range} -- y_range = {y_range}")
        for i in range(0, np_img.shape[0], block_size):
            if i + block_size > np_img.shape[0]:
                continue
            for j in range(0, np_img.shape[1], block_size):
                if j + block_size > np_img.shape[1]:
                    continue
                at_block += 1
                block = np_img[i:i+block_size,j:j+block_size,:]
                # Not only black pixels?
                # print(np.count_nonzero(block==0))
                # if np.sum(block):
                # Alternative: skip if more than 1/3rd of values is 0; 'block' = RGB, so 3 channels
                if np.count_nonzero(block == 0) < (block_size**2):  # How many black pixels?
                    blocks.append(at_block)
                # else:
                #     Image.fromarray(block, mode='RGB').show()
                #     input()


        # Shuffle block IDs; this operation is INPLACE
        blocks_orig = blocks.copy()
        self.rng = random.SystemRandom()
        for _ in range(5):
            self.rng.shuffle(blocks)

        # Recreated image with shuffled blocks
        np_img_shuffled = np_img.copy()
        for idx_block, block in enumerate(blocks):
            # Get i index
            x_block = block % x_range
            if x_block > x_range:
                print(f"Exceeding x range: {x_block}")
                exit()
            # Get j index
            y_block = block // x_range
            if y_block > y_range:
                print(f"Exceeding y range: {y_block}")
                exit()
            # Extract block from original image
            px_block = np_img[y_block*block_size:(y_block+1)*block_size,x_block*block_size:(x_block+1)*block_size,:]
            # Put block in correct place
            j_put = (blocks_orig[idx_block] % x_range)*block_size
            i_put = (blocks_orig[idx_block] // x_range)*block_size
            orig_block = np_img_shuffled[i_put:i_put+block_size,j_put:j_put+block_size,:]
            if np.sum(orig_block) == 0:
                print("Wazzefuk? Empty block in original image?!")
            if orig_block.shape != px_block.shape:
                print(f"Not the same shape: orig = {orig_block.shape} vs. blur = {px_block.shape}")
                continue
            np_img_shuffled[i_put:i_put+block_size,j_put:j_put+block_size,:] = px_block

        # img_shuffled = Image.fromarray(np_img_shuffled, mode='RGB')

        return np_img_shuffled


if __name__ == '__main__':
    # Create Scrambler instance
    proc = Scrambler()
    # Replace the line below to point at the image you want to parse
    _img = 'test_image_unsplash.jpg'
    # Parse image
    # block_size: size of the blocks (or patches if you prefer) that will be randomly swapped around
    # blur_radius: size of the blur to be applied to the scrambled part
    res = proc.process_img(_img, block_size=3, blur_radius=0, b_show=True)
