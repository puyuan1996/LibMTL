## Office-31 and Office-Home

```eval_rst
The Office-31 dataset :cite:`saenko2010adapting` consists of three classification tasks on three domains: Amazon, DSLR, and Webcam, where each task has 31 object categories. It can be download `here <https://www.cc.gatech.edu/~judy/domainadapt/#datasets_code>`_. This dataset contains 4,110 labeled images and we randomly split these samples, with 60\% for training, 20\% for validation, and the rest 20\% for testing. 

The Office-Home dataset :cite:`venkateswara2017deep` has four classification tasks on four domains: Artistic images (abbreviated as Art), Clip art, Product images, and Real-world images. It can be download `here <https://www.hemanthdv.org/officeHomeDataset.html>`_. This dataset has 15,500 labeled images in total and each domain contains 65 classes. We divide the entire data into the same proportion as the Office-31 dataset. 

Both datasets belong to the multi-input setting in MTL. Thus, the ``multi_input`` must be ``True`` for both of the two office datasets.

The training codes are available in ``examples/office``. We use the ResNet-18 network pretrained on the ImageNet dataset followed by a fully connected layer as a shared encoder among tasks and a fully connected layer is applied as a task-specific output layer for each task. All the input images are resized to 3x224x224.
```

### Run a Model

The script ``train_office.py`` is the main file for training and evaluating a MTL model on the Office-31 or Office-Home dataset. A set of command-line arguments is provided to allow users to adjust the training parameter configuration. 

Some important  arguments are described as follows.

```eval_rst
- ``weighting``: The weighting strategy. Refer to `here <../_autoapi/LibMTL/weighting/index.html>`_.
- ``arch``: The MTL architecture. Refer to `here <../_autoapi/LibMTL/architecture/index.html>`_.
- ``gpu_id``: The id of gpu. The default value is '0'.
- ``seed``: The random seed for reproducibility. The default value is 0.
- ``optim``: The type of the optimizer. We recommend to use 'adam' here.
- ``dataset``: Training on Office-31 or Office-Home. Options: 'office-31', 'office-home'.
- ``dataset_path``: The path of the Office-31 or Office-Home dataset.
- ``bs``: The batch size of training, validation, and test data. The default value is 64.
```

The complete command-line arguments and their descriptions can be found by running the following command.

```shell
python main.py -h
```

If you understand those command-line arguments, you can train a MTL model by running a command like this. 

```shell
python main.py --weighting WEIGHTING --arch ARCH --dataset_path PATH --gpu_id GPU_ID --multi_input
```

### References

```eval_rst
.. bibliography::
   :style: unsrt
   :filter: docname in docnames
```
