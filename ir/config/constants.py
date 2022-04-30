class slack:
    # TODO: Insert your credentials here.
    TOKEN: str = 'xxxx-0000000000000-0000000000000-xxxxxxxxxxxxxxxxxxxxxxxx'
    CHANNEL: str = 'deep-learning-research-lab-bot'
    WEBHOOK_URL: str = 'https://hooks.slack.com/services/XXXXXXXXXXX/XXXXXXXXXXX/XXXXXXXXXXXXXXXXXXXXXXXX'
    USER_ID_TO_MENTION: str = 'your.name'


class net:
    LAYER_NAME_PREFIX: str = 'W'


class vis:
    AXIS_LIM_MARGIN: float = .01


class frame:
    MODEL_AXES_LIM: float = 1.5
    MODEL_ALPHA: float = .5


class env:
    AGG_ARTIFACTS_DIRNAME: str = 'agg'


class gif:
    total_duration: int = 12


class agg:
    IDX_OF_LAST_RVALUED_FRAME_COLUMN_NAME: str = 'idx_of_last_rvalued_frame'
    LAST_RVALUED_LOSS_COLUMN_NAME: str = 'last_rvalued_loss'
