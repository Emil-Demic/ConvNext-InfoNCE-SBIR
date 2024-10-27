# ConvNext-InfoNCE-SBIR

Implementation of a simple model for sketch based image retrieval using a siamese network with ConvNext Small as the encoder
and trained with InfoNCE loss.

## FSCOCO dataset
You can download the dataset from the official [FSCOCO website](https://fscoco.github.io).
Place the extracted dataset inside the folder.
## Pretrained models
You can download the pretrained model for both train/test splits on FSCOC:
- [Normal split](https://drive.google.com/file/d/14RxTcdbueG7j5RnGtYMko7dp4vP7qoOy/view?usp=sharing)
- [Unseen split](https://drive.google.com/file/d/1mKj0B8a4hvxQga6dQkONbdvI84Jo6Qy-/view?usp=sharing) 

Place the downloaded pth file in models folder.

### Example folder structure:
```
    ConvNext-InfoNCE-SBIR
    ├── models
    │   ├── model_normal.pth
    │   ├── model_unseen.pth
    │   └── ...
    ├── fscoco
    │   ├── raster_sketches
    │   ├── images
    │   └── ...
    ├── train.py
    ├── eval.py
    ...
```

## Installation

To install required libraries run:

```
pip install -r requirements.txt
```
This will install everything required to evaluate the model and run the streamlit application. \
It will install cpu version of torch. If you want to use cuda install appropriate torch version separately.


## Running the code

You can train the model using *train.py* and evaluate the pretrained models using *eval.py*. 

### Available command line arguments
| Argument       | Description                             | Default  |
|----------------|-----------------------------------------|----------|
| `--no_cuda`    | Disables CUDA training.                 | `False`  |
| `--save`       | Save trained model state dict.          | `False`  |
| `--val_unseen` | Use unseen user train/val split         | `False`  |
| `--epochs`     | Number of epochs to train.              | `10`     |
| `--lr`         | Initial learning rate.                  | `0.0001` |
| `--temp`       | Temperature parameter for InfoNCE loss. | `0.05`   |
| `--batch_size` | Number of samples in each batch.        | `60`     |
| `--model_path` | path to saved model.                    | `''`     |
| `--seed`       | Seed for reproducibility.               | `42`     |

On completion you will find a *result.html* file inside the folder which contains the visualized results. \
When opened in a browser it will display the 10 most similar images to the sketch according to model predictions. \
The correct image will have a red outline. If no image in a row has a red outline the correct image was not among \
the first 10 most similar results.


## Streamlit app
To run the streamlit example app run the command inside the *streamlit-app* folder:
```
streamlit run app.py
```
In the sidebar you can select witch dataset to use: 
- FSCOCO unseen (test set of FSCOCO dataset)
- My Photos (a toy dataset of random personal photos replicating a somewhat realistic use case scenario on a tiny scale)


The app uses the pretrained model trained on the unseen FSCOCO data split which need to be
placed in the models folder for the app to work.\
When you finish drawing the sketch on the canvas press the *search* button the app will show the 10 most 
similar photos according to the model.

