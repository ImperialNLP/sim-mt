# First the basic types
from .alignments import AlignmentsDataset
from .entities import EntitiesDataset
from .numpy import NumpyDataset
from .text import TextDataset
from .objdet import ObjectDetectionsDataset


# Second the selector function
def get_dataset(type_):
    return {
        'numpy': NumpyDataset,
        'text': TextDataset,
        'objectdetections': ObjectDetectionsDataset,
        'entities': EntitiesDataset,
        'alignments': AlignmentsDataset,
    }[type_.lower()]


# Should always be at the end
from .multimodal import MultimodalDataset
