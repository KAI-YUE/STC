from .dataset import UserDataset, assign_user_data
from .networks import *

nn_registry = {
"naiveMLP": NaiveMLP,
"naiveCNN": NaiveCNN,

"bn": FullPrecision_BN,
"lenet": LeNet_5,
"vgg": VGG_7
}
