import pandas as pd
from comet_ml import Experiment


class CometMLWriter:
    """
    Class for experiment tracking via CometML.

    See https://www.comet.com/docs/v2/.
    """

    def __init__(
        self,
        logger,
        project_config,
        project_name="triplexnet",
        workspace=None,
        api_key=None,
        experiment_name=None,
        run_name=None,
        mode="online",
        **kwargs,
    ):
        """
        Initialize CometML experiment.

        Args:
            logger: Python logger instance.
            project_config (dict): full experiment configuration.
            project_name (str): CometML project name.
            workspace (str | None): CometML workspace name.
            api_key (str | None): CometML API key (or use env COMET_API_KEY).
            experiment_name (str | None): experiment name for CometML.
            run_name (str | None): run name.
            mode (str): 'online' or 'offline'.
            **kwargs: additional arguments.
        """
        self.logger = logger
        self.mode = mode
        self.run_name = run_name or experiment_name or "experiment"

        self.experiment = Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
            auto_metric_logging=False,
            auto_param_logging=False,
            auto_histogram_weight_logging=False,
            auto_histogram_gradient_logging=False,
            auto_histogram_activation_logging=False,
            display_summary_level=0,
        )

        if experiment_name:
            self.experiment.set_name(experiment_name)

        self.experiment.log_parameters(project_config)

        self.step = 0
        self.mode_step = "train"

        logger.info(f"CometML experiment initialized: {self.experiment.url}")
        logger.info(f"Experiment key: {self.experiment.get_key()}")

    def set_step(self, step, mode="train"):
        """
        Set current step and mode.

        Args:
            step (int): current step/iteration number.
            mode (str): 'train' or 'val'/'test'.
        """
        self.step = step
        self.mode_step = mode

    def _object_name(self, object_name):
        """
        Add mode prefix to object name.

        Args:
            object_name (str): base object name.

        Returns:
            str: prefixed object name.
        """
        return f"{self.mode_step}_{object_name}"

    def add_scalar(self, scalar_name, scalar):
        """
        Log scalar value.

        Args:
            scalar_name (str): name of the scalar.
            scalar (float): scalar value.
        """
        self.experiment.log_metric(
            self._object_name(scalar_name),
            scalar,
            step=self.step,
        )

    def add_scalars(self, scalars):
        """
        Log multiple scalars.

        Args:
            scalars (dict): dictionary of scalar names and values.
        """
        for name, value in scalars.items():
            self.add_scalar(name, value)

    def add_image(self, image_name, image):
        """
        Log image.

        Args:
            image_name (str): name of the image.
            image (np.ndarray or PIL.Image): image to log.
        """
        self.experiment.log_image(
            image,
            name=self._object_name(image_name),
            step=self.step,
        )

    def add_audio(self, audio_name, audio, sample_rate=None):
        """
        Log audio.

        Args:
            audio_name (str): name of the audio.
            audio (np.ndarray): audio waveform.
            sample_rate (int | None): sample rate.
        """
        self.experiment.log_audio(
            audio,
            sample_rate=sample_rate,
            name=self._object_name(audio_name),
            step=self.step,
        )

    def add_text(self, text_name, text):
        """
        Log text.

        Args:
            text_name (str): name of the text.
            text (str): text content.
        """
        self.experiment.log_text(
            text,
            metadata={"name": self._object_name(text_name)},
            step=self.step,
        )

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        """
        Log histogram.

        Args:
            hist_name (str): name of the histogram.
            values_for_hist (np.ndarray): values for histogram.
            bins (int | None): number of bins.
        """
        self.experiment.log_histogram_3d(
            values_for_hist,
            name=self._object_name(hist_name),
            step=self.step,
        )

    def add_table(self, table_name, table: pd.DataFrame):
        """
        Log table/dataframe.

        Args:
            table_name (str): name of the table.
            table (pd.DataFrame): pandas DataFrame.
        """
        self.experiment.log_table(
            f"{self._object_name(table_name)}.csv",
            tabular_data=table,
            step=self.step,
        )

    def add_figure(self, figure_name, figure):
        """
        Log matplotlib figure.

        Args:
            figure_name (str): name of the figure.
            figure: matplotlib figure object.
        """
        self.experiment.log_figure(
            figure_name=self._object_name(figure_name),
            figure=figure,
            step=self.step,
        )

    def finish(self):
        """End the experiment."""
        self.experiment.end()
        self.logger.info("CometML experiment ended")
