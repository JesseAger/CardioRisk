from django.urls import path
from . import HeartPredictView

urlPatterns = [
    path("predict-heart/", HeartPredictView.as_view(), name="predict-heart"),
]