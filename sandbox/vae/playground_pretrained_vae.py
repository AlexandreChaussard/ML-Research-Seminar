from src.model.vae.vae import generate_data

import src.utils.utils as utils
import src.utils.viz as viz

model = utils.load_model("vae_generator_200_cifar10")

# Generate new samples
generated_imgs = generate_data(model, n_data=5)
# Display the results
viz.display_images(generated_imgs)
