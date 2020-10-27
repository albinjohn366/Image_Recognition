from PIL import Image, ImageFilter

# Open image
image = Image.open('spiderman.png').convert('RGB')

# Using edge detection kernel
filter = image.filter(ImageFilter.Kernel(size=(3, 3), kernel=[-1, -1, -1, -1, 8,
                                                              -1, -1, -1, -1],
                                         scale=1))
filter.show()
