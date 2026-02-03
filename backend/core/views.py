from rest_framework import viewsets
from .models import Patient, PatientVitals, Doctor
from .serializers import PatientSerializer, PatientVitalsSerializer, DoctorSerializer


class PatientViewSet(viewsets.ModelViewSet):
    queryset = Patient.objects.all().order_by("-created_at")
    serializer_class = PatientSerializer


class PatientVitalsViewSet(viewsets.ModelViewSet):
    queryset = PatientVitals.objects.select_related("patient").all().order_by("-recorded_at")
    serializer_class = PatientVitalsSerializer


class DoctorViewSet(viewsets.ModelViewSet):
    queryset = Doctor.objects.all().order_by("-created_at")
    serializer_class = DoctorSerializer
