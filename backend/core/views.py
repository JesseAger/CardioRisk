from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework import status

from predictor.ml import HeartModelService

from .models import Patient, PatientVitals, Doctor, HeartPrediction
from .serializers import (
    PatientSerializer,
    PatientVitalsSerializer,
    DoctorSerializer,
    HeartPredictionSerializer,
)


class PatientViewSet(viewsets.ModelViewSet):
    queryset = Patient.objects.all().order_by("-created_at")
    serializer_class = PatientSerializer

    @action(detail=True, methods=["post"], url_path="predict-heart")
    def predict_heart(self, request, pk=None):
        patient = self.get_object()  # ✅ this is a Patient

        # Optional: link to a vitals row for this patient
        vitals_id = request.data.get("vitals_id")
        vitals_obj = None
        if vitals_id is not None:
            try:
                vitals_obj = PatientVitals.objects.get(id=vitals_id, patient=patient)
            except PatientVitals.DoesNotExist:
                return Response(
                    {"error": "vitals_id not found for this patient"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        # Run prediction (expects the 13 ML fields in request.data)
        try:
            result = HeartModelService.predict(request.data)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        pred = HeartPrediction.objects.create(
            patient=patient,
            vitals=vitals_obj,
            risk_probability=result["risk_probability"],
            threshold=result["threshold"],
            prediction=bool(result["prediction"]),
            input_payload={k: request.data.get(k) for k in result["features_used"]},
        )

        return Response(HeartPredictionSerializer(pred).data, status=status.HTTP_201_CREATED)

class PatientVitalsViewSet(viewsets.ModelViewSet):
    queryset = PatientVitals.objects.select_related("patient").all().order_by("-recorded_at")
    serializer_class = PatientVitalsSerializer

class DoctorViewSet(viewsets.ModelViewSet):
    queryset = Doctor.objects.all().order_by("-created_at")
    serializer_class = DoctorSerializer
