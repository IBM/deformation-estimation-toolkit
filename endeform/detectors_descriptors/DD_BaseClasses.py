from abc import abstractmethod, ABCMeta


class DDBase:
    """This is the basest of all descriptor AND detector classes.
    Currently its only purpose is to let every class inherit Pol's `__repr__` method
    """

    def __repr__(self):
        """Dump entire internal state into a string.
        Includes all instance variables
        """
        _description = "Class Name: {}".format(self.__class__.__name__)
        for key in self.__dict__.keys():
            _description += f"\n\t{key} = {self.__dict__[key]}"
        return _description


class BaseDetector(DDBase, metaclass=ABCMeta):
    """The base class for a keypoint detector"""

    @abstractmethod
    def detect(self, img, mask=None, **kwargs):
        """Detects keypoints in `img`

        Parameters
        ----------
        img : np.array of np.uint8
            h-by-w-by-c array. c=1: greyscale image, c=3: color image.
            image to detect keypoints in
        mask : h-by-w array of bool, optional
            Only find keypoints not overlapping pixels with mask==False. By default, mask==True everywhere

        Returns
        -------
        keypoints:
            array of cv2.KeyPoint objects
        """
        return NotImplemented


class BaseDescriptor(DDBase, metaclass=ABCMeta):
    _binary = False
    """ double underscore would make this "private" but also mangle it during inheritance, 
                     so we'll go with a single _ """

    def isbinarydescriptor(self):
        return self._binary

    @abstractmethod
    def compute(self, img, keypoints, **kwargs):
        """Describes keypoints in `img` taking the values in `keypoints`

        Parameters
        ----------
        img : np.array of np.uint8
            h-by-w-by-c array. c=1: greyscale image, c=3: color image.
            image to detect keypoints in
        keypoints : An iterable (list, numpy.array, ..) of OpenCv keypoints
            For example returned by the self.detect() method


        Returns
        -------
        keypoints:
            array of cv2.KeyPoint objects: (potentially subset of input) keypoints for which descriptors were computed
        descriptors:
            np.array with one row per output keypoint.
        """
        return NotImplemented


class BaseBinaryDescriptor(BaseDescriptor, metaclass=ABCMeta):
    _binary = True


class BaseDetectorDescriptor(BaseDetector, BaseDescriptor):
    @abstractmethod
    def detect_and_compute(self, img, mask=None, **kwargs):
        """Detect and describe keypoints in `img` ignoring locations given by 'mask'
        This is a chain of self.detect() followed by self.compute()

        Parameters
        ----------
        img : np.array of np.uint8
            h-by-w-by-c array. c=1: greyscale image, c=3: color image.
            image to detect keypoints in
        mask : h-by-w array of bool, optional
            Only find keypoints not overlapping pixels with mask==False. By default, mask==True everywhere

        Returns
        -------
        keypoints:
            array of cv2.KeyPoint objects: keypoints which were detected and for which descriptors were computed
        descriptors:
            np.array with one row per output keypoint.
        """
        return NotImplemented

    @abstractmethod
    def detectAndCompute(self, *args, **kwargs):
        """
        This is a facade for 'self.detect_and_compute()' to provide a compatible interface for OpenCV.
        It HAS to be an alias for `detect_and_compute`
        """
        raise NotImplementedError()


class BaseDetectorBinaryDescriptor(BaseDetectorDescriptor, metaclass=ABCMeta):
    _binary = True
