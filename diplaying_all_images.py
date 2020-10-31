import pygame
import sys
import tensorflow
from recognize import recognize

# initializing pygame
pygame.init()

# Display screen
size = width, height = (400, 600)
window = pygame.display.set_mode(size)

# Setting required variables
model = tensorflow.keras.models.load_model(sys.argv[1])
numbers = ['one.jpg', 'two.jpg', 'three.jpg', 'four.jpg', 'five.jpg',
           'six.jpg', 'seven.jpg', 'eight.jpg', 'nine.jpg']
offset = 10
cell_size_h = int((width - 20) / 4)
cell_size_v = int((height - 20) / 9)
large_font = pygame.font.Font(pygame.font.get_default_font(), 50)

while True:
    window.fill((0, 0, 0))
    # To exit from the window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    # Displaying each image in all formats
    for num, number in enumerate(numbers):
        prediction = recognize(sys.argv[1], number, (num + 1))
        column = ['image_{}.png'.format((num + 1)),
                  'greyscale_image_{}.png'.format(
                      (num + 1)), 'reshaped_image_{}.png'.format((num + 1))]
        for i in range(3):
            image = pygame.transform.scale(pygame.image.load(column[i]),
                                           (cell_size_h, cell_size_v))
            window.blit(image, (offset + (i * cell_size_h),
                                offset + (num * cell_size_v)))
        prediction_text = large_font.render(str(prediction), True, (255, 255,
                                                                    255))
        prediction_text_rect = prediction_text.get_rect()
        prediction_text_rect.center = (offset + (3 * cell_size_h) + int(
            cell_size_h / 2), offset + (num * cell_size_v) + int(cell_size_v
                                                                 / 2))
        window.blit(prediction_text, prediction_text_rect)

    # Updating the display
    pygame.display.update()
