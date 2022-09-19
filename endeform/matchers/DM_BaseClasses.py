# Base classes for descriptor matchers
from abc import abstractmethod, ABCMeta


class BaseMatcher(metaclass=ABCMeta):
    """
    BaseMatcher

    Abstract base class for matchers
    """

    @abstractmethod
    def match(self, descriptors1, descriptors2):
        """Match keypoints between two sets of keypoints

        Parameters
        ----------
        descriptors1 : np.array
            First set of input keypoints' descriptors
        descriptors2 : np.array
            Second set of input keypoints' descriptors

        Returns
        -------
        match
            The found matches as a list of cv2.DMatch objects
        """
        return NotImplemented
