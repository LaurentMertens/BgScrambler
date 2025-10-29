# Background Scrambler
This package contains a simple tool that allows to scramble the background of images. By background, we mean anything but the people in the image. This is practical for, e.g., psychology or neuroscience experiments dealing with human vision and/or social cognition.

As an example, it turns this image:

<div align="center">
<img src="https://github.com/LaurentMertens/BgScrambler/blob/main/bg_scrambler/test_image_unsplash.jpg" width="640px">
</div>

into this image:
<div align="center">
<img src="https://github.com/LaurentMertens/BgScrambler/raw/main/bg_scrambler/res_image.jpg" width="640px">
</div>

## Installation
To install this application, either clone this repository, or simply install using
```pip install bg-scrambler```.

## Usage
The application will keep the original image size, if its largest side is a multiple of 32 (a requisite of YOLOv11),
up till sizes of 2016px. If the largest side exceeds 2016, the image will be resized so that this side becomes 2016px.

Note that the scrambler returns the processed image. By default, the output is NOT saved to disk.\
To save the output (i.e., the processed image) to disk, specify ```out_file='path/to/output.jpg'``` (or ```png```) when calling ```process_img()```.

### When cloning the repo
See the example in ```scrambler.py```, reprised here:
```python
if __name__ == '__main__':
    # Create Scrambler instance
    proc = Scrambler()
    # Replace the line below to point at the image you want to parse
    _img = 'test_image_unsplash.jpg'
    # Parse image
    # block_size: size of the blocks (or patches if you prefer) that will be randomly swapped around
    # blur_radius: size of the blur to be applied to the scrambled part
    res = proc.process_img(_img, block_size=3, blur_radius=0, out_file=None, b_show=True)
```

### When installing the package through pip:
A short demo:
```python
from bg_scrambler import scrambler
import os


if __name__ == '__main__':
    proc = scrambler.Scrambler()
    _img = os.path.join(os.path.dirname(scrambler.__file__), 'test_image_unsplash.jpg')

    # Parse image
    # block_size: size of the blocks (or patches if you prefer) that will be randomly swapped around
    # blur_radius: size of the blur to be applied to the scrambled part
    res = proc.process_img(_img, block_size=3, blur_radius=0, out_file=None, b_show=True)
```

## Licensing
This repository is made available under an MIT license (see [LICENSE.md](./LICENSE.md)).
This is in agreement with the original repository, which also uses an MIT license.

The test image is taken from Unsplash:\
[https://unsplash.com/photos/a-person-sitting-on-a-couch-with-a-laptop-X1GZqv-F7Tw](https://unsplash.com/photos/a-person-sitting-on-a-couch-with-a-laptop-X1GZqv-F7Tw)\
The image is distributed under the [Unsplash license](https://unsplash.com/license).

Author: Laurent Mertens\
Mail: [contact@laurentmertens.com](contact@laurentmertens.com)
