from typing_extensions import Literal, List
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)
import numpy as np

class TrainArgs(Tap):
    """:class:`TrainArgs` includes :class:`CommonArgs` along with additional arguments used for training a Chemprop model."""

    # General arguments
    dataset_type: Literal['regression', 'classification', 'multiclass', 'spectra'] = None
    """Type of dataset."""
    data_path: str
    """Path to data CSV file."""
    target_columns: List[str] = None
    """
    Name of the columns containing target values.
    By default, uses all columns except the SMILES column and the :code:`ignore_columns`.
    """
    loss_function: Literal['mse', 'bounded_mse', 'binary_cross_entropy', 'cross_entropy', 'mcc', 'sid', 'wasserstein', 'mve', 'evidential', 'dirichlet'] = None
    """Choice of loss function. Loss functions are limited to compatible dataset types."""
    separate_val_path: str = None
    """Path to separate val set, optional."""
    separate_test_path: str = None
    """Path to separate test set, optional."""
    split_type: Literal['random', 'scaffold_balanced', 'predetermined', 'crossval', 'cv', 'cv-no-test', 'index_predetermined', 'random_with_repeated_smiles'] = 'random'
    """Method of splitting the data into train/val/test."""
    split_sizes: List[float] = None
    """Split proportions for train/validation/test sets."""
    split_key_molecule: int = 0
    """The index of the key molecule used for splitting when multiple molecules are present and constrained split_type is used, like scaffold_balanced or random_with_repeated_smiles.
       Note that this index begins with zero for the first molecule."""
    num_folds: int = 1
    """Number of folds when performing cross validation."""
    seed: int = 0
    """
    Random seed to use when splitting data into train/val/test sets.
    When :code`num_folds > 1`, the first fold uses this seed and all subsequent folds add 1 to the seed.
    """
    metric = None
    """
    Metric to use during evaluation. It is also used with the validation set for early stopping.
    Defaults to "auc" for classification, "rmse" for regression, and "sid" for spectra.
    """
    save_dir: str = None
    """Directory where model checkpoints will be saved."""
    save_smiles_splits: bool = False
    """Save smiles for each train/val/test splits for prediction convenience later."""
    test: bool = False
    """Whether to skip training and only test the model."""
    save_preds: bool = False
    """Whether to save test split predictions during training."""
    quiet: bool = False
    """Whether to suppress all non-essential output to the console."""

    # Model arguments
    timesteps: int = 1000
    """Number of diffusion steps."""
    comp_embed_model = 'molformer'
    """Model to use for computing compound embeddings. default='molformer'."""
    train_data_size: int = 0
    """Size of the training data set."""


    # Training arguments
    cuda: bool = True
    """Whether to use GPU."""
    epochs: int = 200
    """Number of epochs to run."""
    warmup_epochs: int = 2
    """
    Number of epochs during which learning rate increases linearly from :code:`init_lr` to :code:`max_lr`.
    Afterwards, learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr`.
    """
    batch_size: int = 128
    """Batch size to use during training."""
    init_lr: float = 1e-4
    """Initial learning rate."""
    max_lr: float = 1e-3
    """Maximum learning rate."""
    final_lr: float = 1e-4
    """Final learning rate."""
    num_lrs: int = 1
    """Number of learning rate cycles to perform during training."""
    grad_clip: float = None
    """Maximum magnitude of gradient during training."""
    parallel: bool = False
    """Whether to use multiple GPUs for training."""
    early_stop_criterion: int = 30
    """Number of epochs with no improvement on validation metric before early stopping."""

    def __init__(self, *args, **kwargs) -> None:
        super(TrainArgs, self).__init__(*args, **kwargs)
        self._task_names = None
        self._crossval_index_sets = None
        self._task_names = None
        self._num_tasks = None
        self._features_size = None
        self._train_data_size = None


    @property
    def minimize_score(self) -> bool:
        """Whether the model should try to minimize the score metric or maximize it."""
        return self.metric in {'rmse', 'mae', 'mse', 'cross_entropy', 'binary_cross_entropy', 'sid', 'wasserstein', 'bounded_mse', 'bounded_mae', 'bounded_rmse'}

    @property
    def task_names(self) -> List[str]:
        """A list of names of the tasks being trained on."""
        return self._task_names

    @task_names.setter
    def task_names(self, task_names: List[str]) -> None:
        self._task_names = task_names

    @property
    def num_tasks(self) -> int:
        """The number of tasks being trained on."""
        return len(self.task_names) if self.task_names is not None else 0

    def process_args(self) -> None:
        super(TrainArgs, self).process_args()


        # Process SMILES columns
        self.smiles_columns = 0

        
        # Validate split size entry and set default values
        if self.split_sizes is None:
            if self.separate_val_path is None and self.separate_test_path is None: # separate data paths are not provided
                self.split_sizes = [0.8, 0.1, 0.1]
            elif self.separate_val_path is not None and self.separate_test_path is None: # separate val path only
                self.split_sizes = [0.8, 0., 0.2]
            elif self.separate_val_path is None and self.separate_test_path is not None: # separate test path only
                self.split_sizes = [0.8, 0.2, 0.]
            else: # both separate data paths are provided
                self.split_sizes = [1., 0., 0.]

        else:
            if not np.isclose(sum(self.split_sizes), 1):
                raise ValueError(f'Provided split sizes of {self.split_sizes} do not sum to 1.')
            if any([size < 0 for size in self.split_sizes]):
                raise ValueError(f'Split sizes must be non-negative. Received split sizes: {self.split_sizes}')


            if len(self.split_sizes) not in [2,3]:
                raise ValueError(f'Three values should be provided for train/val/test split sizes. Instead received {len(self.split_sizes)} value(s).')

            if self.separate_val_path is None and self.separate_test_path is None: # separate data paths are not provided
                if len(self.split_sizes) != 3:
                    raise ValueError(f'Three values should be provided for train/val/test split sizes. Instead received {len(self.split_sizes)} value(s).')
                if self.split_sizes[0] == 0.:
                    raise ValueError(f'Provided split size for train split must be nonzero. Received split size {self.split_sizes[0]}')
                if self.split_sizes[1] == 0.:
                    raise ValueError(f'Provided split size for validation split must be nonzero. Received split size {self.split_sizes[1]}')

            elif self.separate_val_path is not None and self.separate_test_path is None: # separate val path only
                if len(self.split_sizes) == 2: # allow input of just 2 values
                    self.split_sizes = [self.split_sizes[0], 0., self.split_sizes[1]]
                if self.split_sizes[0] == 0.:
                    raise ValueError('Provided split size for train split must be nonzero.')
                if self.split_sizes[1] != 0.:
                    raise ValueError(f'Provided split size for validation split must be 0 because validation set is provided separately. Received split size {self.split_sizes[1]}')

            elif self.separate_val_path is None and self.separate_test_path is not None: # separate test path only
                if len(self.split_sizes) == 2: # allow input of just 2 values
                    self.split_sizes = [self.split_sizes[0], self.split_sizes[1], 0.]
                if self.split_sizes[0] == 0.:
                    raise ValueError('Provided split size for train split must be nonzero.')
                if self.split_sizes[1] == 0.:
                    raise ValueError('Provided split size for validation split must be nonzero.')
                if self.split_sizes[2] != 0.:
                    raise ValueError(f'Provided split size for test split must be 0 because test set is provided separately. Received split size {self.split_sizes[2]}')


            else: # both separate data paths are provided
                if self.split_sizes != [1., 0., 0.]:
                    raise ValueError(f'Separate data paths were provided for val and test splits. Split sizes should not also be provided. Received split sizes: {self.split_sizes}')

        # Test settings
        if self.test:
            self.epochs = 0
