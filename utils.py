import torch
import torchvision
from dataset import RGBDdataset
from torch.utils.data import DataLoader
from torch.nn.functional import normalize


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir,
    train_maskdir,
    train_depth_dir,
    val_dir,
    val_maskdir,
    val_depth_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = RGBDdataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        depth_dir=train_depth_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = RGBDdataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        depth_dir=val_depth_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # y = y.to(device).unsqueeze(1)
            y = y.to(device)
            y = y.permute(0, 3, 1, 2)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            # print(preds.size())
            # print(y.size())

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    # print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            # preds = torch.nn.ReLU(model(x))
            preds = (preds > 0.5).float()
            preds = preds.float()
            y = y.float()
            preds = normalize(preds, p=1.0, dim=0)
            y = normalize(y, p=1.0, dim=0)
            y = y.permute(0, 3, 1, 2)
            # preds = preds.permute(0, 2, 3, 1)
            print(preds.size())
            print(y.size())

        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y, f"{folder}{idx}.png")

    model.train()
