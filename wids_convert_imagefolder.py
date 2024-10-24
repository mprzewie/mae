import os
import sys
from torchvision.datasets import ImageNet, ImageFolder
import webdataset as wds
import typer
from pathlib import Path

app = typer.Typer()


@app.command()
def convert(root: str, odir: str = "./shards", maxcount: int = 1000):
    Path(odir).mkdir(exist_ok=True)
    
    assert os.path.isdir(root)
    assert os.path.isdir(os.path.join(root, "train"))
    assert os.path.isdir(os.path.join(root, "val"))
    assert os.path.isdir(odir)

    assert not os.path.exists(os.path.join(odir, "train-000000.tar"))
    assert not os.path.exists(os.path.join(odir, "val-000000.tar"))

    # dataset = ImageNet(root=root, split="val")

    for fold in ["val", "train"]:
        dataset = ImageFolder(f"{root}/{fold}")
        opat = os.path.join(odir, f"imagenet-{fold}-%06d.tar")
        output = wds.ShardWriter(opat, maxcount=maxcount)

        
        for i in range(len(dataset)):
            if i % maxcount == 0:
                print(i, file=sys.stderr)
            img, label = dataset[i]
            output.write({"__key__": "%08d" % i, "jpg": img, "cls": label})
        
        output.close()



if __name__ == "__main__":
    app()
