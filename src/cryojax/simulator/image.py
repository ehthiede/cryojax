"""
Image formation models.
"""

from __future__ import annotations

__all__ = ["ImagePipeline"]

from typing import Union, Optional
from functools import cached_property

from .filter import Filter
from .mask import Mask
from .specimen import Specimen
from .helix import Helix
from .scattering import ScatteringConfig
from .instrument import Instrument
from .ice import Ice, NullIce
from ..utils import fftn, irfftn
from ..core import field, Module
from ..types import RealImage, ComplexImage, Image, Real_


class ImagePipeline(Module):
    """
    Base class for an imaging model.

    Call an ``Image`` or its ``render``, ``sample``,
    or ``log_probability`` routines to evaluate the model.

    Attributes
    ----------
    specimen :
        The specimen from which to render images.
    scattering :
        The image and scattering model configuration.
    instrument :
        The abstraction of the electron microscope.
    solvent :
        The solvent around the specimen.
    filters :
        A list of filters to apply to the image.
    masks :
        A list of masks to apply to the image.
    observed :
        The observed data in real space. This must be the same
        shape as ``scattering.shape``. Note that the user
        should preprocess the observed data before passing it
        to the image, such as applying the ``filters`` and
        ``masks``.
    """

    specimen: Union[Specimen, Helix] = field()
    scattering: ScatteringConfig = field()
    instrument: Instrument = field(default_factory=Instrument)
    solvent: Ice = field(default_factory=NullIce)

    filters: list[Filter] = field(default_factory=list)
    masks: list[Mask] = field(default_factory=list)
    observed: Optional[RealImage] = field(default=None)

    def render(self, view: bool = True) -> RealImage:
        """
        Render an image given a parameter set.

        Parameters
        ----------
        view : `bool`
            If ``True``, view the cropped,
            masked, and rescaled image in real
            space. If ``False``, return the image
            at this place in the pipeline.
        """
        # Compute image in detector plane
        optics_image = self.specimen.scatter(
            self.scattering,
            exposure=self.instrument.exposure,
            optics=self.instrument.optics,
        )
        # Compute image at detector pixel size
        pixelized_image = self.instrument.detector.pixelize(
            irfftn(optics_image), resolution=self.specimen.resolution
        )
        if view:
            pixelized_image = self.view(pixelized_image, real=True)

        return pixelized_image

    def sample(self, view: bool = True) -> RealImage:
        """
        Sample the an image from a realization of the noise.

        Parameters
        ----------
        view : `bool`, optional
            If ``True``, view the protein signal overlayed
            onto the noise. If ``False``, just return
            the noise given at this place in the pipeline.
        """
        # Determine pixel size
        if self.instrument.detector.pixel_size is not None:
            pixel_size = self.instrument.detector.pixel_size
        else:
            pixel_size = self.specimen.resolution
        # Frequencies
        freqs = self.scattering.padded_freqs / pixel_size
        # The specimen image at the detector pixel size
        pixelized_image = self.render(view=False)
        # The ice image at the detector pixel size
        ice_image = self.solvent.scatter(
            self.scattering,
            resolution=pixel_size,
            optics=self.instrument.optics,
        )
        ice_image = irfftn(ice_image)
        # Measure the detector readout
        image = pixelized_image + ice_image
        noise = self.instrument.detector.sample(freqs, image=image)
        detector_readout = image + noise
        if view:
            detector_readout = self.view(detector_readout, real=True)

        return detector_readout

    def log_probability(self) -> Real_:
        """Evaluate the log-probability of the data given a parameter set."""
        raise NotImplementedError

    def __call__(self, view: bool = True) -> Union[Image, Real_]:
        """
        Evaluate the model at a parameter set.

        If ``Image.observed = None``, sample an image from
        a noise model. Otherwise, compute the log likelihood.
        """
        if self.observed is None:
            return self.sample(view=view)
        else:
            return self.log_probability()

    def view(self, image: Image, real: bool = False) -> RealImage:
        """
        View the image. This function applies
        filters, crops the image, then applies masks.
        """
        # Apply filters
        if real:
            if len(self.filters) > 0:
                image = irfftn(self.filter(fftn(image)))
        else:
            image = irfftn(self.filter(image))
        # View
        image = self.mask(self.scattering.crop(image))
        return image

    def filter(self, image: ComplexImage) -> ComplexImage:
        """Apply filters to image."""
        for filter in self.filters:
            image = filter(image)
        return image

    def mask(self, image: RealImage) -> RealImage:
        """Apply masks to image."""
        for mask in self.masks:
            image = mask(image)
        return image

    @cached_property
    def residuals(self) -> RealImage:
        """Return the residuals between the model and observed data."""
        simulated = self.render()
        residuals = self.observed - simulated
        return residuals
