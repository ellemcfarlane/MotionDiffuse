import os
import re

from PIL import Image, ImageDraw, ImageFont, ImageSequence


def combine_gifs_with_timestep(gif_paths, output_path):
    """
    Combines multiple GIF files into a single GIF, adding a timestep extracted from each file's name.

    :param gif_paths: List of paths to the GIF files.
    :param output_path: Path where the combined GIF will be saved.
    """
    frames = []
    for gif_path in gif_paths:
        # Extract timestep from filename using regular expressions
        timestep = re.search(r'sample_tensor\(\[(\d+)\]\)', gif_path)
        if timestep:
            timestep = timestep.group(1)
        else:
            print(f"error: Timestep not found in filename {gif_path}. Skipping this file.")
            exit(1)

        start_i = 0
        n_frames_keep = 30
        tot_frames = 30
        min_frames = 20
        # Open the GIF
        with Image.open(gif_path) as img:
            # Loop over each frame in the GIF
            # so lower the timestep, the more frames we keep
            n_frames_keep = (999-int(timestep))/999 * tot_frames + min_frames
            for i, frame in enumerate(ImageSequence.Iterator(img)):
                if i >= start_i:
                    # if int(timestep) >= 50 and i >= (n_frames_keep + start_i):
                    #     break  # Stop after 10 frames
                    # elif int(timestep) < 50 and int(timestep) >= 20 and i >= n_frames_keep + start_i:
                    #     break
                    # elif int(timestep) < 20 and int(timestep) > 0 and i >= n_frames_keep + start_i:
                    #     break
                    if int(timestep) > 0 and i >= n_frames_keep + start_i:
                        break
                    # elif int(timestep) == 999 and i >= n_frames_keep + start_i:
                    #     break
                    # Convert the frame to RGB mode and draw the timestep on it
                    frame = frame.convert("RGBA")
                    d = ImageDraw.Draw(frame)

                    # Load a font - you can adjust the size and font type as needed
                    font_path = '/work3/s222376/MotionDiffuse2/text2motion/datasets/arial.ttf'
                    font_large = ImageFont.truetype(font_path, 20)  # Large font for the timestep number
                    font_small = ImageFont.truetype(font_path, 20)  # Small font for the word 'timestep'

                    # Calculate text size and position
                    number_size = d.textsize(timestep, font=font_large)
                    # label_size = d.textsize("timestep", font=font_small)
                    # total_width = number_size[0] + label_size[0]
                    # x_number = (frame.width - total_width) // 2
                    # y = 10  # 10 pixels from the top
                    y_offset = 100
                    x_offset = 20
                    center = (frame.width // 2 - x_offset, frame.height // 2 - y_offset)
                    # Draw the timestep number and label
                    color = (0,0,0)
                    d.text((10,10), f"t={timestep}", font=font_large, fill=color)
                    # d.text((x_number + number_size[0], y + (number_size[1] - label_size[1]) // 2), "timestep", font=font_small, fill=color)

                    frames.append(frame)

    # Save the frames as a new GIF
    frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0)

# Example usage
dir = "/work3/s222376/MotionDiffuse2/text2motion/gifs/md_fulem_2g_excl_196_seed42/"
# list all files in dir
# gif_paths = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
# # filter but contians 'sample_tensor' and 'happiness'
# gif_paths = [f for f in gif_paths if 'sample_tensor' in f and 'happiness' in f]
# # reverse sort by timestep
# gif_paths.sort(key=lambda f: int(re.search(r'sample_tensor\(\[(\d+)\]\)', f).group(1)), reverse=True)
# print(gif_paths)
times = [999, 80, 10, 0]
gif_paths = [f'sample_tensor([{t}])_happiness.gif' for t in times]
full_gif_paths = [os.path.join(dir, gif_path) for gif_path in gif_paths]
output_path = f'{dir}combined_gif.gif'  # Replace with your desired output file path
combine_gifs_with_timestep(full_gif_paths, output_path)
print(f"Combined GIF saved to {output_path}")