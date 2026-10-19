"""Compatibility imports for the baseline cure estimators."""

from .mixture_cure import MixtureCureModel
from .promotion_time import PromotionTimeModel

__all__ = ["MixtureCureModel", "PromotionTimeModel"]
