from PIL import Image
import itertools

def merge_images(config, uJy, a_or_i_list, Mth_list, dustnum_list, hand_list, separationx=10, separationy=10, right_margin=60, top_margin=0, single_disk=None, tag=''):
    # Generate filenames
    filenames = []
    for a_or_i, hand, dustnum, Mth in itertools.product(a_or_i_list, hand_list, dustnum_list, Mth_list):
        if single_disk:
            diskname = f'{a_or_i}-{Mth}Mth-dust{dustnum}-{hand}'
            filename = f'/home/es833/Proj42_results/{single_disk}/chi2_plots/{diskname}_chi2_polar.png'
        else:
            # diskname = f'{config}_{uJy}uJy_{a_or_i}-{Mth}Mth-dust{dustnum}-{hand}'
            # filename = f'/home/es833/Proj42_results/{config}_{uJy}uJy_simulated_observations/{diskname}chi2_polar{tag}.png'
            diskname = f'{config}_{uJy}uJy_{a_or_i}-{Mth}Mth-dust{dustnum}'
            filename = f'/home/es833/Proj42_results/gofish_profiles/profile_{diskname}.png'

        filenames.append(filename)

    # Assuming all images have the same size
    sample_image = Image.open(filenames[0])
    img_width, img_height = sample_image.size

    # Calculate total width and height of the output image
    total_width = img_width * len(Mth_list) + separationx * (len(Mth_list) - 1) + right_margin
    if len(a_or_i_list) > 1:
        total_height = img_height * len(a_or_i_list) + separationy * (len(a_or_i_list) - 1)
    else:
        total_height = img_height * len(dustnum_list) + separationy * (len(dustnum_list) - 1) + top_margin

    # Create a blank white image with the calculated width and height
    combined_img = Image.new('RGB', (total_width, total_height), 'white')

    y_offset = top_margin
    for dustnum in reversed(dustnum_list):
    # for a_or_i in a_or_i_list:
        x_offset = 0
        for Mth in Mth_list:
            # Find the filename for the current combination
            if single_disk:
                diskname = f'{a_or_i}-{Mth}Mth-dust{dustnum_list[0]}-{hand_list[0]}'
                filename = f'/home/es833/Proj42_results/{single_disk}/chi2_plots/{diskname}_chi2_polar.png'
            else:
                # diskname = f'{config}_{uJy}uJy_{a_or_i}-{Mth}Mth-dust{dustnum}-{hand}'
                # filename = f'/home/es833/Proj42_results/{config}_{uJy}uJy_simulated_observations/{diskname}chi2_polar{tag}.png'
                # # print(filename)
                diskname = f'{config}_{uJy}uJy_{a_or_i}-{Mth}Mth-dust{dustnum}'
                filename = f'/home/es833/Proj42_results/gofish_profiles/profile_{diskname}.png'

            img = Image.open(filename)
            combined_img.paste(img, (x_offset, y_offset))
            x_offset += img_width + separationx
            # print(img_width)
        y_offset += img_height + separationy

    if len(a_or_i_list) == 1:
        eos = 'adi' if a_or_i_list[0] == 'a' else 'iso' if a_or_i_list[0] == 'i' else a_or_i_list[0]
    else:
        eos = 'special'
        
    if single_disk:
        combined_img.save(f"/home/es833/Proj42_results/mismatching_dust{dustnum_list[0]}.png", dpi=(330,330))
    else:
        combined_img.save(f"/home/es833/Proj42_results/{config}_{uJy}uJy_{eos}s_{hand_list[0]}_combined{tag}.png", dpi=(300,300))

# Example usage:
config = "C4C7"
uJy = 35
single_disk = None
a_or_i_list = ['both']
Mth_list = ['03','1','3']
dustnum_list = ['0','1','2','3']
hand_list = ['R']
tag = '_rh-axis' 

# merge_images(config, uJy, a_or_i_list, Mth_list, dustnum_list, hand_list, separationx=100, separationy=30, right_margin=180, top_margin=20, single_disk=single_disk, tag=tag)

merge_images(config, uJy, a_or_i_list, Mth_list, dustnum_list, hand_list, separationx=-20, separationy=-60, right_margin=40)

# def merge_4_images(filenames, separationx=10, separationy=10, output_filename="merged_image.png"):
#     """
#     Merge 4 images in a 2x2 grid.

#     Parameters:
#     - filenames: List of 4 image filenames.
#     - separationx: Horizontal separation between images.
#     - separationy: Vertical separation between images.
#     - output_filename: Name of the output merged image file.
#     """
    
#     # Assuming all images have the same size
#     sample_image = Image.open(filenames[0])
#     img_width, img_height = sample_image.size

#     # Calculate total width and height of the output image
#     total_width = img_width * 2 + separationx + 100
#     total_height = img_height * 2 + separationy + 80

#     # Create a blank white image with the calculated width and height
#     combined_img = Image.new('RGB', (total_width, total_height), 'white')

#     # Place images on the blank canvas
#     for i in range(2):
#         for j in range(2):
#             img = Image.open(filenames[i*2 + j])
#             x_offset = j * (img_width + separationx)
#             y_offset = i * (img_height + separationy)
#             combined_img.paste(img, (x_offset, y_offset))

#     # Save the merged image
#     combined_img.save(output_filename, dpi=(330,330))

# # Example usage:
# prefix = '/home/es833/Proj42_results/'
# filenames = [f"{prefix}DR_Tau/chi2_plots/a-01Mth-dust2-R_chi2_polar.png", f"{prefix}a-03Mth-dust2-R_chi2_polar.png", f"{prefix}a-1Mth-dust2-R_chi2_polar.png", f"{prefix}a-3Mth-dust2-R_chi2_polar.png"]
# merge_4_images(filenames, output_filename="/home/es833/Proj42_results/FT_Tau/FT_Tau_4circles.png")




################################################################################

 
