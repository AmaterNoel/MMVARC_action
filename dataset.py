import os
import torch
import random
import numpy as np
import pandas as pd
import torchvision.transforms as T
from torchvision.transforms import Compose
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageOps, ImageFilter
from transformers import CLIPTokenizer, CLIPTextModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_prompt(class_list):
    temp_prompt = []
    for c in class_list:
        temp_prompt.append(c)
    return temp_prompt

def get_text_features(class_list):
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    act_prompt = get_prompt(class_list)
    texts = tokenizer(act_prompt, padding=True, return_tensors="pt")
    text_class = text_model(**texts).pooler_output.detach()
    return text_class

def get_dict():
    action_dict = {'Abseiling': 0, 'Attacking': 1, 'Attending': 2, 'Barking': 3, 'Being carried': 4, 'Being carried in mouth': 5, 'Being dragged': 6, 'Being eaten': 7, 'Biting': 8, 'Building nest': 9, 'Calling': 10, 'Camouflaging': 11, 'Carrying': 12, 'Carrying in mouth': 13, 'Chasing': 14, 'Chirping': 15, 'Climbing': 16, 'Coiling': 17, 'Competing for dominance': 18, 'Dancing': 19, 'Dancing on water': 20, 'Dead': 21, 'Defecating': 22, 'Defensive rearing': 23, 'Detaching as a parasite': 24, 'Digging': 25, 'Displaying defensive pose': 26, 'Disturbing another animal': 27, 'Diving': 28, 'Doing a back kick': 29, 'Doing a backward tilt': 30, 'Doing a chin dip': 31, 'Doing a face dip': 32, 'Doing a neck raise': 33, 'Doing a side tilt': 34, 'Doing push up': 35, 'Doing somersault': 36, 'Drifting': 37, 'Drinking': 38, 'Dying': 39, 'Eating': 40, 'Entering its nest': 41, 'Escaping': 42, 'Exiting cocoon': 43, 'Exiting nest': 44, 'Exploring': 45, 'Falling': 46, 'Fighting': 47, 'Flapping': 48, 'Flapping tail': 49, 'Flapping its ears': 50, 'Fleeing': 51, 'Flying': 52, 'Gasping for air': 53, 'Getting bullied': 54, 'Giving birth': 55, 'Giving off light': 56, 'Gliding': 57, 'Grooming': 58, 'Hanging': 59, 'Hatching': 60, 'Having a flehmen response': 61, 'Hissing': 62, 'Holding hands': 63, 'Hopping': 64, 'Hugging': 65, 'Immobilized': 66, 'Jumping': 67, 'Keeping still': 68, 'Landing': 69, 'Lying down': 70, 'Laying eggs': 71, 'Leaning': 72, 'Licking': 73, 'Lying on its side': 74, 'Lying on top': 75, 'Manipulating object': 76, 'Molting': 77, 'Moving': 78, 'Panting': 79, 'Pecking': 80, 'Performing sexual display': 81, 'Performing allo-grooming': 82, 'Performing allo-preening': 83, 'Performing copulatory mounting': 84, 'Performing sexual exploration': 85, 'Performing sexual pursuit': 86, 'Playing': 87, 'Playing dead': 88, 'Pounding': 89, 'Preening': 90, 'Preying': 91, 'Puffing its throat': 92, 'Pulling': 93, 'Rattling': 94, 'Resting': 95, 'Retaliating': 96, 'Retreating': 97, 'Rolling': 98, 'Rubbing its head': 99, 'Running': 100, 'Running on water': 101, 'Sensing': 102, 'Shaking': 103, 'Shaking head': 104, 'Sharing food': 105, 'Showing affection': 106, 'Sinking': 107, 'Sitting': 108, 'Sleeping': 109, 'Sleeping in its nest': 110, 'Spitting': 111, 'Spitting venom': 112, 'Spreading': 113, 'Spreading wings': 114, 'Squatting': 115, 'Standing': 116, 'Standing in alert': 117, 'Startled': 118, 'Stinging': 119, 'Struggling': 120, 'Surfacing': 121, 'Swaying': 122, 'Swimming': 123, 'Swimming in circles': 124, 'Swinging': 125, 'Tail swishing': 126, 'Trapped': 127, 'Turning around': 128, 'Undergoing chrysalis': 129, 'Unmounting': 130, 'Unrolling': 131, 'Urinating': 132, 'Walking': 133, 'Walking on water': 134, 'Washing': 135, 'Waving': 136, 'Wrapping itself around prey': 137, 'Wrapping prey': 138, 'Yawning': 139}
    animal_dict = {'Abalone': 0, 'Acyrthosiphon Pisum Aphid': 1, 'Aedes Aegypti Mosquito': 2, 'Aedes Aegypti Mosquito Larva': 3, 'Aesculapian Snake': 4, 'African Bullfrog': 5, 'African Civet': 6, 'African Clawed Toad': 7, 'African Finfoot': 8, 'African Golden Cat': 9, 'African Oryx': 10, 'African Painted Dog': 11, 'African Penguin': 12, 'African Wild Boar': 13, 'Agouti': 14, 'Agraulis Vanilla Caterpillar': 15, 'Alauda Arvensis Bird': 16, 'Albatross': 17, 'Aldabrachelys Gigantea Tortoise': 18, 'Alfalfa Leafcutting Bee': 19, 'Alligator': 20, 'Alpine Newt': 21, 'Altantic Stingray': 22, 'Amazon Milk Frog': 23, 'Anartia Jatrophae': 24, 'Anas Crecca Bird': 25, 'Anas Platyrhynchos Bird': 26, 'Anisolabis Maritima': 27, 'Annulated Tree Boa': 28, 'Anole Lizard': 29, 'Anopheles Gambiae Mosquito': 30, 'Ant': 31, 'Anteater': 32, 'Antelope': 33, 'Anthus Pratensis Bird': 34, 'Antipaluria Urichi Webspinner': 35, 'Aphid': 36, 'Aphidius Ervi Parasitoid Wasp': 37, 'Aphis Fabae Alates': 38, 'Aphis Fabae Aphid': 39, 'Apis Mellifera Honey Bee': 40, 'Araneus Diadematus Spider': 41, 'Archer Fish': 42, 'Archispirostreptus Gigas Giant African Millipede': 43, 'Ardea Alba Egret': 44, 'Ardeotis Kori Bird': 45, 'Argentine Ant': 46, 'Arizona Bark Scorpion': 47, 'Asian Glossy Starling Bird': 48, 'Asian Lion': 49, 'Asian Small-Clawed Otter': 50, 'Asian Tapir': 51, 'Asiatic Golden Cat': 52, 'Atheris Hispida Viper': 53, 'Atheris Nitschei Viper': 54, 'Atheris Squamigera': 55, 'Atlantic Blue Tang Fish': 56, 'Australian Bowerbird': 57, 'Avicularia Spider': 58, 'Azure Tit Bird': 59, 'Babirusa': 60, 'Backswimmer': 61, 'Badger': 62, 'Bald Eagle': 63, 'Banana Slug': 64, 'Banded Rubber Frog': 65, 'Banded Woodpecker': 66, 'Bare-Faced Curassow': 67, 'Barking Deer': 68, 'Barracuda Fish': 69, 'Barramundi Fish': 70, 'Barred Warbler Bird': 71, 'Basilisk Lizard': 72, 'Bat': 73, 'Battus Philenor Hirsuta': 74, 'Battus Philenor Hirsuta Caterpillar': 75, 'Bearded Pig': 76, 'Beaver': 77, 'Bee': 78, 'Beetle': 79, 'Bengal Tiger': 80, 'Big Headed Ant': 81, 'Bighorn Sheep': 82, 'Binturong': 83, 'Bird': 84, 'Bison': 85, 'Black Bean Aphid': 86, 'Black Bearded Draco': 87, 'Black Click Beetle': 88, 'Black Duiker': 89, 'Black Durgon Fish': 90, 'Black Grouse': 91, 'Black Headed Python': 92, 'Black Mamba': 93, 'Black Necked Spitting Cobra': 94, 'Black Stork': 95, 'Black Swan': 96, 'Black-Winged Stilt': 97, 'Blackbird': 98, 'Blue Dasher Dragonfly': 99, 'Blue Orchard Bee': 100, 'Blue Orchard Bee Larva': 101, 'Blue Poison Dart Frog': 102, 'Blue Whale': 103, 'Blue-Ringed Octopus': 104, 'Bluehead Wrasse Fish': 105, 'Bluethroat': 106, 'Boa': 107, 'Boar': 108, 'Bogong Moth': 109, 'Bombardier Beetle': 110, 'Bongo': 111, 'Bordered Mantis': 112, 'Boreal Owl': 113, 'Bornean Orangutan': 114, 'Botaurus Stellaris Bird': 115, 'Bothriechis': 116, 'Bothrops Asper': 117, 'Bottlenose Dolphin': 118, 'Box Jellyfish': 119, 'Boxer Crab': 120, 'Brittle Star': 121, 'Brown Garden Snail': 122, 'Brown-Banded Cockroach': 123, 'Buffalo': 124, 'Bull Shark': 125, 'Bullfinch': 126, 'Bullfrog': 127, 'Bullfrog Tadpole': 128, 'Bumblebee': 129, 'Bushmaster Snake': 130, 'Butterfly': 131, 'Butterfly Fish': 132, 'Cabbage White Caterpillar': 133, 'Caddisfly': 134, 'Caddisfly Larva': 135, 'Calamari': 136, 'Calidris Apina Bird': 137, 'Calidris Minuta Bird': 138, 'California Newt': 139, 'California Newt Young': 140, 'California Oak Moth': 141, 'California Oak Moth Larva': 142, 'California Oak Moth Pupa': 143, 'California Rock Crab': 144, 'California Sea Lion': 145, 'Camberwell Beauty Butterfly': 146, 'Camel': 147, 'Capuchin Monkey': 148, 'Carcharhinus Galapagensis': 149, 'Carcharhinus Limbatus': 150, 'Carcharias Taurus': 151, 'Carcharodon Carcharias': 152, 'Cardinal Fish': 153, 'Caretta Caretta Turtle': 154, 'Caribou': 155, 'Carolina Duck': 156, 'Carribean Rock Mantis Shrimp': 157, 'Caspian Tern': 158, 'Cat': 159, 'Caterpillar': 160, 'Catfish': 161, 'Cattle': 162, 'Chaffinch Bird': 163, 'Chameleon': 164, 'Chamois Goat Antelope': 165, 'Charadrius Dubius Bird': 166, 'Cheetah': 167, 'Chelonia Mydas Turtle': 168, 'Chicken': 169, 'Chimpanzee': 170, 'Chrysoperla Rufilabris Green Lacewing Larva': 171, 'Circus Aeruginosus Bird': 172, 'Citrine Wagtail': 173, 'Civet Cat': 174, 'Clitarchus Hookeri Common Stick Insect': 175, 'Clouded Monitor Lizard': 176, 'Clownfish': 177, 'Coati': 178, 'Cobra': 179, 'Cockroach': 180, 'Colugo': 181, 'Common Basilisk Lizard': 182, 'Common Bleak': 183, 'Common Buzzard': 184, 'Common Chiffchaff Bird': 185, 'Common Crane': 186, 'Common Crowned Pigeon': 187, 'Common Cuckoo Bird': 188, 'Common Eastern Bumblebee': 189, 'Common Eider': 190, 'Common Goldeneye': 191, 'Common Greenshank': 192, 'Common Quail': 193, 'Common Redshank': 194, 'Common Rosefinch Bird': 195, 'Common Snipe': 196, 'Common Whitethroat Bird': 197, 'Common Wood Pigeon': 198, 'Cone Snail': 199, 'Convergent Ladybug': 200, 'Coommon Eider': 201, 'Copper Shark': 202, 'Coral Mimic Snake': 203, 'Coral Snake': 204, 'Corncrake': 205, 'Coronate Medusa Jellyfish': 206, 'Coronella Austriaca Snake': 207, 'Corroboree Frog': 208, 'Cotesia Glomerata Wasp': 209, 'Cougar': 210, 'Cow': 211, 'Coyote': 212, 'Crab': 213, 'Crane': 214, 'Crane Fly': 215, 'Crested Grebe Bird': 216, 'Cricket': 217, 'Crocodile': 218, 'Crocodylus Acutus Crocodile': 219, 'Crocodylus Palustris Crocodile': 220, 'Crotalus Willardi Ridge Nosed Rattlesnake': 221, 'Crow': 222, 'Crowned Eagle': 223, 'Cryptic Kelp Crab': 224, 'Cryptic Mantis': 225, 'Cuckoo Bird': 226, 'Culex Pipiens Mosquito': 227, 'Cuttlefish': 228, 'Cyclosa Conica': 229, 'Daddy Longlegs Spider': 230, 'Dampwood Termite': 231, 'Damsel Fish': 232, 'Damselfly': 233, 'Damselfly Nymph': 234, 'Danaus Plexippus Monarch Butterfly': 235, 'Danaus Plexippus Monarch Butterfly Caterpillar': 236, 'Danaus plexippus Monarch Butterfly': 237, 'Danube Bleak Fish': 238, 'Danube Salmon': 239, 'Darner Damselfly': 240, 'Darner Damselfly Larva': 241, 'Decorator Crab': 242, 'Deer': 243, 'Dendroaspis Polylepis Black Mamba': 244, 'Desert Fox': 245, 'Desert Rain Frog': 246, 'Dholes': 247, 'Diana Monkey': 248, 'Dice Snake': 249, 'Dingo Dog': 250, 'Dispholidus Typus Snake': 251, 'Diving Bell Water Spider': 252, 'Dog': 253, 'Dog Faced Water Snake': 254, 'Dolphin': 255, 'Draco Lizard': 256, 'Dragonfly': 257, 'Drosophila Melanogaster Fruit Fly': 258, 'Duck': 259, 'Dumbo Octopus': 260, 'Eagle': 261, 'Earthworm': 262, 'Eastern Montpellier Snake': 263, 'Echina': 264, 'Eel': 265, 'Eelgrass Isopod': 266, 'Eelgrass Sea Hare': 267, 'Egret': 268, 'Elegant Bronzeback Snake': 269, 'Elephant': 270, 'Emerita Analoga': 271, 'Ensatina': 272, 'Estuarine Crocodile': 273, 'Eurasian Skylark Bird': 274, 'Eurasian Wren Bird': 275, 'Eurasian Wryneck Bird': 276, 'European Robin Bird': 277, 'European Serin Bird': 278, 'European Turtle Dove': 279, 'Eyelash Pit Viper': 280, 'Feather Star': 281, 'Feathery Tailed Flatted Bug': 282, 'Fer-De-Lance Snake': 283, 'Fiddler Crab': 284, 'Firebrat Insect': 285, 'Firecrest Bird': 286, 'Firefly': 287, 'Fish': 288, 'Fishing Cat': 289, 'Flame Skimmer Dragonfly': 290, 'Flamingo': 291, 'Flamingo Young': 292, 'Fly': 293, 'Flying Ant': 294, 'Flying Snake': 295, 'Foam Nest Frog': 296, 'Forest Buffalo': 297, 'Forest Cobra': 298, 'Forficula Auricularia': 299, 'Formica Accreta': 300, 'Fox': 301, 'Frilled Neck Lizard': 302, 'Frog': 303, 'Frog Tadpole': 304, 'Fruit Bat': 305, 'Fruit Fly': 306, 'Gaboon Viper': 307, 'Galapagos Tortoise': 308, 'Galeocerdo Cuvier': 309, 'Gallinago Gallinago Bird': 310, 'Garganey': 311, 'Gaur': 312, 'Gavialis Gangeticus Crocodile': 313, 'Gazelle': 314, 'Gecarcinus Lateralis Crab': 315, 'Gecko': 316, 'Genet': 317, 'Giant Forest Hog': 318, 'Giant Galapagos Tortoise': 319, 'Giant Ground Gecko': 320, 'Giant Ground Pangolin': 321, 'Giant Hornet': 322, 'Giant Salamander': 323, 'Giant Tortoise': 324, 'Giant Trevally': 325, 'Gibbon': 326, 'Gila Monster': 327, 'Giraffe': 328, 'Glass Frog': 329, 'Glossina Morsitans Morsitans': 330, 'Glossina Morsitans Morsitans Larva': 331, 'Goat': 332, 'Goby Fish': 333, 'Goldcrest Bird': 334, 'Golden Eagle': 335, 'Golden Orb Spider': 336, 'Golden Oriole': 337, 'Golden Poison Frog': 338, 'Goldfinch': 339, 'Goose': 340, 'Gorilla': 341, 'Graphocephala Atropunctata Sharpshooter': 342, 'Grass Snake': 343, 'Grass Spider': 344, 'Grass Warbler Bird': 345, 'Grasshopper': 346, 'Grasshopper Warbler': 347, 'Gray Angelfish': 348, 'Gray Whale': 349, 'Grayling Fish': 350, 'Great Argus': 351, 'Great Curassow': 352, 'Great Diving Beetle': 353, 'Great Egret': 354, 'Great Grey Shrike': 355, 'Great Grey Shrike Young': 356, 'Great Ramshorn Snail': 357, 'Great Reed Warbler Bird': 358, 'Great Snipe': 359, 'Greater Flamingo': 360, 'Greater Mouse Deer': 361, 'Greater Racket Tail Drongo': 362, 'Grebe Bird': 363, 'Green Bottle Fly': 364, 'Green Crested Lizard': 365, 'Green Iguana': 366, 'Green Mamba': 367, 'Green Pea Aphid': 368, 'Green Woodpecker': 369, 'Greta Oto Glasswing Butterfly': 370, 'Greta Oto Glasswing Butterfly Caterpillar': 371, 'Grey Bird': 372, 'Grey Heron': 373, 'Grey Langur': 374, 'Greylag Goose': 375, 'Grizzly Bear': 376, 'Gryllus Lineaticeps': 377, 'Gull': 378, 'Guttural Toad': 379, 'Gyalopion Canum Snake': 380, 'Habronattus Clypeatus': 381, 'Hairworm': 382, 'Hairworm Egg': 383, 'Hairworm Larva': 384, 'Hairy Caterpillar': 385, 'Halyomorpha Halys': 386, 'Hamadryas Feronia Variable Cracker Butterfly': 387, 'Hammerhead Shark': 388, 'Hartebeest': 389, 'Harvester Ant': 390, 'Hawfinch Bird': 391, 'Hawk': 392, 'Hawksbill Turtle': 393, 'Hazel Grouse Bird': 394, 'Hedgehog': 395, 'Helmeted Guineafowl': 396, 'Hermit Crab': 397, 'Heron': 398, 'Hippodamia Convergens Ladybug': 399, 'Hippopotamus': 400, 'Hognosed Pit Viper': 401, 'Homalodisca Vitripennis Sharpshooter': 402, 'Honey Bee': 403, 'Hoopoe': 404, 'Hornbill': 405, 'Horned Adder': 406, 'Horse': 407, 'Horse Young': 408, 'Horseshoe Crab': 409, 'House Centipede': 410, 'House Cricket': 411, 'Housefly': 412, 'Hoverfly': 413, 'Hoverfly Young': 414, 'Hummingbird': 415, 'Hydroid': 416, 'Hyena': 417, 'Ibex': 418, 'Ibis Hagedash': 419, 'Icterine Warbler Bird': 420, 'Iguana': 421, 'Indian Chameleon': 422, 'Indian Walking Stick': 423, 'Insect': 424, 'Ixos Mcclellandii': 425, 'Jack Snipe Bird': 426, 'Jackal': 427, 'Jaculus Jaculus': 428, 'Jaguarundi Cat': 429, 'Javan Spitting Cobra': 430, 'Jellyfish': 431, 'Jerusalam Cricket': 432, 'Jeweled Cockroach Wasp': 433, 'Jumping Spider': 434, 'Kangaroo': 435, 'Kangaroo Rat': 436, 'Kangaroo Young': 437, 'Keeltail Needlefish': 438, 'King Cobra': 439, 'King Colobus': 440, 'Kingfisher': 441, 'Ladybug': 442, 'Lampropeltis Getula Snake': 443, 'Lampropeltis Pyromelana Snake': 444, 'Lampropeltis Splendida Snake': 445, 'Lampropeltis Zonata Snake': 446, 'Lanius Excubitor': 447, 'Larus Canus Bird': 448, 'Larus Ridibundus Bird': 449, 'Larva': 450, 'Laticauda Saintgironsi Sea Krait': 451, 'Latrodectus Hersperus Western Widow Spider': 452, 'Leaf Insect': 453, 'Leaf-Tailed Gecko': 454, 'Leafcutter Ant': 455, 'Leatherback Sea Turtle': 456, 'Leech': 457, 'Leopard': 458, 'Leopard Gecko': 459, 'Leopard Seal': 460, 'Lesser Mousedeer': 461, 'Lesser Spot Nosed Monkey': 462, 'Lesser Sunda Pit Viper': 463, 'Lichanura Trivirgata Snake': 464, 'Lightfoot Crab': 465, 'Lion': 466, 'Lion Young': 467, 'Lionfish': 468, 'Little Crake Bird': 469, 'Little Egret': 470, 'Lizard': 471, 'Lobster': 472, 'Long Horned Beetle': 473, 'Long Tail Scorpion': 474, 'Long Tailed Macaque': 475, 'Long-Toed Water Beetle': 476, 'Luscinia Luscinia Nightingale Bird': 477, 'Macaque': 478, 'Madagascar Hissing Cockroach': 479, 'Madagascar Hissing Cockroach Mite': 480, 'Malabar Pit Viper': 481, 'Malayan Flying Fox': 482, 'Malayan Porcupine': 483, 'Malayan Sun Bear': 484, 'Malayan Tapir': 485, 'Malayan Water Monitor Lizard': 486, 'Mallard Duck': 487, 'Mandarin Duck': 488, 'Mandrill': 489, 'Manta Ray': 490, 'Mantis': 491, 'Mantis Shrimp': 492, 'Many Horned Adder': 493, 'Marbled Cat': 494, 'Marbled Rubber Frog': 495, 'Marine Iguana': 496, 'Markhor Goat': 497, 'Maroon Macaque': 498, 'Marsh Frog': 499, 'Marsh Harrier Bird': 500, 'Marshbuck': 501, 'Mayfly': 502, 'Mayfly Larva': 503, 'Meerkat': 504, 'Megalorchestia Californiana Beach Hopper': 505, 'Melanosuchus Niger Crocodile': 506, 'Metlapilcoatlus Mexicanus Jumping Pit Viper': 507, 'Metrius Bombardier Beetle': 508, 'Micruroides Euryxanthus Snake': 509, 'Millipede': 510, 'Mimic Blenny Fish': 511, 'Mimic Poison Frog': 512, 'Minke Whale': 513, 'Mistle Thrush': 514, 'Mitred Leaf Monkey': 515, 'Mojave Rattlesnake': 516, 'Mojave Rattlesnake Young': 517, 'Mongoose': 518, 'Monitor Lizard': 519, 'Monkey': 520, 'Monkey Young': 521, 'Monster Frog': 522, 'Montpellier Snake': 523, 'Moose': 524, 'Moray Eel': 525, 'Morning Gecko': 526, 'Morpho Butterfly': 527, 'Mosquito': 528, 'Mosquito Larva': 529, 'Moss Crab': 530, 'Motacilla Alba Bird': 531, 'Motacilla Flava': 532, 'Moth': 533, 'Mountain Gorilla': 534, 'Mountain Pygmy Possum': 535, 'Mountain Yellow-Legged Frog': 536, 'Mouse': 537, 'Mouse Deer': 538, 'Mozambique Spitting Cobra': 539, 'Mudskipper': 540, 'Naja Nivea Snake': 541, 'Namaqua Dwarf Adder': 542, 'Namaqua Dwarf Chameleon': 543, 'Nasutitermes Nigriceps Termite': 544, 'Natrix Natrix Snake': 545, 'Natrix Tessellata Snake': 546, 'Nautilus': 547, 'Nerillid Worm': 548, 'Newt': 549, 'Newt Young': 550, 'Nightingale Bird': 551, 'Nilgai': 552, 'Nine Banded Armadillo': 553, 'Northern Pacific Rattlesnake': 554, 'Nose-Horned Viper': 555, 'Nudibranch': 556, 'Numenius Arquata Bird': 557, 'Nursery Web Spider': 558, 'Nuthatch Bird': 559, 'Nymphidium Sp. Caterpillar': 560, 'Ocelot': 561, 'Octopus': 562, 'Ocyropsis Ctenophore': 563, 'Olive Colobus': 564, 'Opossum': 565, 'Orange Clownfish': 566, 'Orangutan': 567, 'Orb Spider': 568, 'Orb-Weaver Spider': 569, 'Orca': 570, 'Oriental Pied Hornbill': 571, 'Oriental Whip Snake': 572, 'Ostrich': 573, 'Otter': 574, 'Owl': 575, 'Pacman Frog': 576, 'Panda': 577, 'Pangolin': 578, 'Paradise Tree Snake': 579, 'Parrot': 580, 'Peacock': 581, 'Peacock Mantis Shrimp': 582, 'Pelican': 583, 'Penguin': 584, 'Perch Fish': 585, 'Phalacrocorax Carbo Bird': 586, 'Philomachus Pugnax Ruff Bird': 587, 'Phyllium Giganteum': 588, 'Pig-Tailed Macaque': 589, 'Pika': 590, 'Pike Perch Fish': 591, 'Pill Bug': 592, 'Pink Skunk Clownfish': 593, 'Pipevine Swallowtail Caterpillar': 594, 'Pistol Shrimp': 595, 'Pit Viper': 596, 'Pituophis Catenifer Snake': 597, 'Planarian Schmidtea Mediterranea Flatworm': 598, 'Plantain Squirrel': 599, 'Platypus': 600, 'Polar Bear': 601, 'Polyergus Mexicanus': 602, 'Porcupine': 603, 'Portia Jumping Spider': 604, 'Postman Butterfly': 605, 'Postman Butterfly Caterpillar': 606, 'Prawn': 607, 'Praying Mantis': 608, 'Predaceous Diving Beetle': 609, 'Proboscis Monkey': 610, 'Promecognathus Crassus Beetle': 611, 'Puff Adder': 612, 'Pufferfish': 613, 'Puffin': 614, 'Pygmy Hippopotamus': 615, 'Pygmy Owl': 616, 'Pygmy Seahorse': 617, 'Python': 618, 'Quail': 619, 'Rabbit': 620, 'Racoon': 621, 'Raffles Banded Langur': 622, 'Rain Frog': 623, 'Rat': 624, 'Rat Snake': 625, 'Rattlesnake': 626, 'Raven': 627, 'Red Ant': 628, 'Red Bellied Black Snake': 629, 'Red Crossbill': 630, 'Red River Hog': 631, 'Red Ruffed Lemur': 632, 'Red Spitting Cobra': 633, 'Red-Backed Shrike Bird': 634, 'Red-Backed Shrike Bird Young': 635, 'Red-Blacked Squirrel Monkey': 636, 'Red-Eyed Tree Frog': 637, 'Red-Ruffed Lemur': 638, 'Red-Throated Pipit': 639, 'Redback Spider': 640, 'Reed Bunting Bird': 641, 'Reef Octopus': 642, 'Remiz Pendulinus Bird': 643, 'Reticulated Python': 644, 'Rhacodactylus Trachyrhynchus Gecko': 645, 'Rhamnophis Aethiopissa Snake': 646, 'Rhesus Macaque': 647, 'Rhincodon Typus': 648, 'Rhinoceros': 649, 'Rhinoceros Viper': 650, 'Rhinoceros Young': 651, 'Ring-Tailed Lemur': 652, 'Rinkhals Snake': 653, 'Robin Bird': 654, 'Roe Deer': 655, 'Rooster': 656, 'Round Face Bat Fish': 657, 'Royal Grammas Fish': 658, 'Saddleback Galapagos Tortoise': 659, 'Sagittarius Serpentarius Bird': 660, 'Salamander': 661, 'Salmon': 662, 'Salticidae Jumping Spider': 663, 'Saltwater Crocodile': 664, 'Sambar Deer': 665, 'Sand Bubbler Crab': 666, 'Sand Frog': 667, 'Sand Hopper': 668, 'Sardine': 669, 'Scallop': 670, 'Scorpion': 671, 'Scutigera Coleoptera House Centipede': 672, 'Sea Bird': 673, 'Sea Cockroach': 674, 'Sea Cucumber': 675, 'Sea Goldies': 676, 'Sea Lion': 677, 'Sea Otter': 678, 'Sea Slug': 679, 'Sea Snail': 680, 'Sea Snake': 681, 'Sea Spider': 682, 'Sea Toad Fish': 683, 'Sea Turtle': 684, 'Sea Urchin': 685, 'Seagull': 686, 'Seahorse': 687, 'Seal': 688, 'Sedge Warbler Bird': 689, 'Sergeant Major Fish': 690, 'Serow': 691, 'Seven Banded Civet': 692, 'Shark': 693, 'Shearwater Bird': 694, 'Shield Bug': 695, 'Shoebill Bird': 696, 'Shorebird': 697, 'Short-Clawed Otter': 698, 'Shrimp': 699, 'Side Blotched Lizard': 700, 'Sidewinder Rattlesnake': 701, 'Singing Nightingale': 702, 'Skate': 703, 'Skink': 704, 'Skylark': 705, 'Slender Hognosed Pit Viper': 706, 'Slender-Snouted Crocodile': 707, 'Sloth': 708, 'Sloth Bear': 709, 'Slow Loris': 710, 'Smew': 711, 'Smooth-Coated Otter': 712, 'Snail': 713, 'Snake': 714, 'Snouted Cobra': 715, 'Snow Leopard': 716, 'Socotra Cormorant': 717, 'Song Thrush Bird': 718, 'Sooty Mangabey': 719, 'Southern Grasshopper Mouse': 720, 'Sparrowhawk': 721, 'Spectacled Cobra': 722, 'Spider': 723, 'Spider Crab': 724, 'Spiny Flower Mantis': 725, 'Spotted Deer': 726, 'Spotted Whistling Duck': 727, 'Spotted Wood Owl': 728, 'Squid': 729, 'Squirrel': 730, 'Sri Lankan Leopard': 731, 'Stag Beetle': 732, 'Stagmomantis Limbata Bordered Mantis': 733, 'Starfish': 734, 'Starling Bird': 735, 'Sterlet Fish': 736, 'Stick Insect': 737, 'Stingray': 738, 'Stoat': 739, 'Stock Dove': 740, 'Stonefish': 741, 'Stork': 742, 'Stork-Billed Kingfisher': 743, 'Strange-Horned Chameleon': 744, 'Strawberry Poison-Dart Frog': 745, 'Stump-Tailed Macaque': 746, 'Sumatran Hog Badger': 747, 'Sumatran Orangutan': 748, 'Sumatran Serow': 749, 'Sumatran Tiger': 750, 'Sunbear': 751, 'Surgeonfish': 752, 'Swan': 753, 'Syllid Worm': 754, 'Tachybaptus Ruficollis Bird': 755, 'Tadpole': 756, 'Tamandua': 757, 'Tarantula': 758, 'Tarantula Hawk Wasp': 759, 'Tawny Owl': 760, 'Tayra': 761, 'Teddy Bear Crab': 762, 'Tench Fish': 763, 'Termite': 764, 'Tern': 765, 'Terrapin': 766, 'Tetragnatha Versicolor': 767, 'Texas Brown Tarantula': 768, 'Thamin Deer': 769, 'Thamnophis Cyrtopsis Snake': 770, 'Thelotornis Snake': 771, 'Three-Toed Woodpecker': 772, 'Thrush Nightingale Bird': 773, 'Tiger': 774, 'Tiger Salamander': 775, 'Tiger Shark': 776, 'Tinamou': 777, 'Tit Bird': 778, 'Toad': 779, 'Toadfish': 780, 'Tomato Frog': 781, 'Tomistoma Schlegelii Gharial': 782, 'Tortoise': 783, 'Trap-Door Spider': 784, 'Tree Climbing Crab': 785, 'Tree Hopper': 786, 'Tree Snake': 787, 'Tringa Erythropus Bird': 788, 'Tringa Glareola Bird': 789, 'Tringa Nebularia Bird': 790, 'Tringa Ochropus Bird': 791, 'Trissolcus Japonicus Wasp': 792, 'Tropical Reed Frog': 793, 'Trout': 794, 'Trout Young': 795, 'Turdus Merula Blackbird': 796, 'Turret Spider': 797, 'Turtle': 798, 'Turtle Dove': 799, 'Two-Toed Sloth': 800, 'Vampire Squid': 801, 'Vanellus Vanellus Bird': 802, 'Vanessa Cardui Caterpillar': 803, 'Vimba Fish': 804, 'Viper': 805, 'Vipera Berus Snake': 806, 'Vulture': 807, 'Walking Stick': 808, 'Walrus': 809, 'Wandering Alabatross': 810, 'Wasp': 811, 'Water Dipper Bird': 812, 'Water Insect': 813, 'Water Lily Frog': 814, 'Water Rail Bird': 815, 'Water Scorpion': 816, 'Water Strider': 817, 'Water Threader': 818, 'Weaver Ant': 819, 'Wedge Tailed Eagle': 820, 'Western Chimpanzee': 821, 'Western Chimpanzee Young': 822, 'Western Diamondback': 823, 'Western Diamondback Young': 824, 'Western Dwarf Chameleon': 825, 'Western Pine Beetle': 826, 'Western Pine Beetle Larva': 827, 'Western Pond Turtle': 828, 'Western Widow Spider': 829, 'Western-Crest Guineafowl': 830, 'Whale': 831, 'Whinchat Bird': 832, 'Whiskered Tern Bird': 833, 'Whistling Duck': 834, 'White And Gray Wagtail Bird': 835, 'White Cockatoo Bird': 836, 'White Nosed Coati': 837, 'White Rhinoceros': 838, 'White Speckled Rattlesnake': 839, 'White Throated Dipper Bird': 840, 'White Tiger': 841, 'White Wig Marine Iguana': 842, 'White-Backed Woodpecker': 843, 'White-Bellied Fish Eagle': 844, 'White-Breasted Guineafowl': 845, 'White-Breasted Waterhen': 846, 'White-Faced Saki Monkey': 847, 'Whooper Swan': 848, 'Wild Boar': 849, 'Wild Dog': 850, 'Wild Red-Tailed Boa': 851, 'Wildebeest': 852, 'Winter Ant': 853, 'Wolf': 854, 'Wombat': 855, 'Wood Cricket': 856, 'Wood Warbler': 857, 'Woodlark Bird': 858, 'Woodpecker': 859, 'Worm': 860, 'Wormlion': 861, 'Wormlion Larva': 862, 'Wren': 863, 'Xystocheir Dissecta Millipede': 864, 'Yellow Cuttlefish': 865, 'Yellow Striped Tree Skink': 866, 'Yellow Watchman Goby Fish': 867, 'Yellow Wrasse Fish': 868, 'Yellow-Backed Duiker': 869, 'Yellow-Eyed Ensatina': 870, 'Yellow-Throated Martens': 871, 'Yellowhammer': 872, 'Yellowhammer Young': 873, 'Yellowtail Fusilier Fish': 874, 'Zamenis Longissiumus Snake': 875, 'Zebra': 876, 'Zebra-Duiker': 877, 'Zooplankton': 878, 'Zygiella X-Notata': 879}
    return action_dict, animal_dict

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# action
class AnimalKingdom_action(Dataset):
    def __init__(self, root, type='train', total_length=8, action_class_list=[], random_shift=False):
        self.root = os.path.expanduser(root)
        self.type = type
        self.total_length = total_length
        self.anno_path = os.path.join(self.root, 'action_recognition', 'annotation', 'AR_metadata_animal.xlsx')
        self.action_num_class = len(action_class_list)
        self.random_shift = False

        self.transform = self.get_train_transforms()

        # read the data file
        try:
            self.label_list, self.file_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

        self.data_len = len(self.label_list)

    def _parse_annotations(self):
        label_list = []
        file_list = []
        df = pd.read_excel(self.anno_path, sheet_name='Sheet1')
        df = df[df['type'] == self.type]
        len_df = len(df)
        video_ids = df['video_id'].tolist()
        action_lists = df['action_labels'].astype(str).tolist()
        for i, video_id in enumerate(video_ids):
            path = os.path.join(self.root, 'action_recognition', 'dataset', 'image', video_id)
            files = sorted(os.listdir(path))
            file_list += [files]
            count = len(files)
            action_labels = [int(l) for l in action_lists[i].split(',')]
            label_list.append([path, count, action_labels])
        print("数据集读取完成")
        return label_list, file_list

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        images_names = self.file_list[index]
        label_lists = self.label_list[index]
        indices = self._sample_indices(label_lists[1])
        return self._get(label_lists, images_names, indices)

    def _sample_indices(self, num_frames):
        if num_frames <= self.total_length:
            indices = np.linspace(0, num_frames - 1, self.total_length, dtype=int)
        else:
            ticks = np.linspace(0, num_frames, self.total_length + 1, dtype=int)
            if self.random_shift:
                indices = ticks[:-1] + np.random.randint(ticks[1:] - ticks[:-1])
            else:
                indices = ticks[:-1] + (ticks[1:] - ticks[:-1]) // 2
        return indices

    def _get(self, label_lists, image_names, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(label_lists[0], image_names[idx])
            except:
                print('ERROR: Could not read image "{}"'.format(os.path.join(label_lists[0], image_names[idx])))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        action_label = np.zeros(self.action_num_class)
        action_label[label_lists[2]] = 1.0
        return process_data, action_label

    def _load_image(self, directory, image_name):
        return [Image.open(os.path.join(directory, image_name)).convert('RGB')]

    def get_train_transforms(self,):
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        input_size = 224
        unique = Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                          GroupRandomHorizontalFlip(True),
                          GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                          GroupRandomGrayscale(p=0.2),
                          GroupGaussianBlur(p=0.0),
                          GroupSolarization(p=0.0)])
        common = Compose([Stack(roll=False),
                          ToTorchFormatTensor(div=True),
                          GroupNormalize(input_mean, input_std)])
        transforms = Compose([unique, common])
        return transforms

# action_eval
class AnimalKingdom_action_eval(Dataset):
    def __init__(self, root, type='head_type', total_length=8, action_class_list=[], random_shift=False):
        self.root = os.path.expanduser(root)
        self.type = type
        self.total_length = total_length
        self.anno_path = os.path.join(self.root, 'action_recognition', 'annotation', 'AR_metadata_eval.xlsx')
        self.action_num_class = len(action_class_list)
        self.random_shift = False

        self.transform = self.get_train_transforms()

        # read the data file
        try:
            self.label_list, self.file_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

        self.data_len = len(self.label_list)

    def _parse_annotations(self):
        label_list = []
        file_list = []
        df = pd.read_excel(self.anno_path, sheet_name='Sheet1')
        if self.type != 'overall':
            df = df[df[self.type] == True]
        len_df = len(df)
        video_ids = df['video_id'].tolist()
        action_lists = df['action_labels'].astype(str).tolist()
        for i, video_id in enumerate(video_ids):
            path = os.path.join(self.root, 'action_recognition', 'dataset', 'image', video_id)
            files = sorted(os.listdir(path))
            file_list += [files]
            count = len(files)
            action_labels = [int(l) for l in action_lists[i].split(',')]
            label_list.append([path, count, action_labels])
        print("数据集读取完成 ", len_df)
        return label_list, file_list

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        images_names = self.file_list[index]
        label_lists = self.label_list[index]
        indices = self._sample_indices(label_lists[1])
        return self._get(label_lists, images_names, indices)

    def _sample_indices(self, num_frames):
        if num_frames <= self.total_length:
            indices = np.linspace(0, num_frames - 1, self.total_length, dtype=int)
        else:
            ticks = np.linspace(0, num_frames, self.total_length + 1, dtype=int)
            if self.random_shift:
                indices = ticks[:-1] + np.random.randint(ticks[1:] - ticks[:-1])
            else:
                indices = ticks[:-1] + (ticks[1:] - ticks[:-1]) // 2
        return indices

    def _get(self, label_lists, image_names, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(label_lists[0], image_names[idx])
            except:
                print('ERROR: Could not read image "{}"'.format(os.path.join(label_lists[0], image_names[idx])))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        action_label = np.zeros(self.action_num_class)
        action_label[label_lists[2]] = 1.0
        return process_data, action_label

    def _load_image(self, directory, image_name):
        return [Image.open(os.path.join(directory, image_name)).convert('RGB')]

    def get_train_transforms(self,):
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        input_size = 224
        unique = Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                          GroupRandomHorizontalFlip(True),
                          GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                          GroupRandomGrayscale(p=0.2),
                          GroupGaussianBlur(p=0.0),
                          GroupSolarization(p=0.0)])
        common = Compose([Stack(roll=False),
                          ToTorchFormatTensor(div=True),
                          GroupNormalize(input_mean, input_std)])
        transforms = Compose([unique, common])
        return transforms


# ----------------------------------------transforms--------------------------------------------------
class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

    @staticmethod
    def fill_fc_fix_offset(image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 2
        h_step = (image_h - crop_h) // 2

        ret = list()
        ret.append((0, 0))  # left
        ret.append((1 * w_step, 1 * h_step))  # center
        ret.append((2 * w_step, 2 * h_step))  # right

        return ret


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_sth=False):
        self.is_sth = is_sth

    def __call__(self, img_group, is_sth=False):
        v = random.random()
        if not self.is_sth and v < 0.5:

            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            return ret
        else:
            return img_group


class GroupRandomColorJitter(object):
    """Randomly ColorJitter the given PIL.Image with a probability
    """

    def __init__(self, p=0.8, brightness=0.4, contrast=0.4,
                 saturation=0.2, hue=0.1):
        self.p = p
        self.worker = T.ColorJitter(brightness=brightness, contrast=contrast,
                                    saturation=saturation, hue=hue)

    def __call__(self, img_group):

        v = random.random()
        if v < self.p:
            ret = [self.worker(img) for img in img_group]

            return ret
        else:
            return img_group


class GroupRandomGrayscale(object):
    """Randomly Grayscale flips the given PIL.Image with a probability
    """

    def __init__(self, p=0.2):
        self.p = p
        self.worker = T.Grayscale(num_output_channels=3)

    def __call__(self, img_group):

        v = random.random()
        if v < self.p:
            ret = [self.worker(img) for img in img_group]

            return ret
        else:
            return img_group

class GroupGaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img_group):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return [img.filter(ImageFilter.GaussianBlur(sigma))  for img in img_group]
        else:
            return img_group

class GroupSolarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img_group):
        if random.random() < self.p:
            return [ImageOps.solarize(img)  for img in img_group]
        else:
            return img_group

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                rst = np.concatenate(img_group, axis=2)
                # plt.imshow(rst[:,:,3:6])
                # plt.show()
                return rst

class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        mean = self.mean * (tensor.size()[0]//len(self.mean))
        std = self.std * (tensor.size()[0]//len(self.std))
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)

        if len(tensor.size()) == 3:
            # for 3-D tensor (T*C, H, W)
            tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        elif len(tensor.size()) == 4:
            # for 4-D tensor (C, T, H, W)
            tensor.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        return tensor