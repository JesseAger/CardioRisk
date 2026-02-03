from django.urls import path
from predictor.views import HeartPredictView

urlpatterns = [
    path("predict-heart/", HeartPredictView.as_view(), name="predict-heart"),
]