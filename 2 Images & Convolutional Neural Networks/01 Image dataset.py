
# Compose transformations
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128,128)),
])

# Create Dataset using ImageFolder
dataset_train = ImageFolder(
    "clouds_train",
    transform=train_transforms,
)
