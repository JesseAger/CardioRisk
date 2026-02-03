# from rest_framework import serializers
# from .models import Patient, PatientVitals, Doctor


# class PatientSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Patient
#         fields = [
#             "id",
#             "first_name",
#             "last_name",
#             "email",
#             "phone",
#             "address",
#             "created_at",
#         ]
#         read_only_fields = ["id", "created_at"]


# class PatientVitalsSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = PatientVitals
#         fields = [
#             "id",
#             "patient",
#             "pulse",
#             "temperature",
#             "respiratory_rate",
#             "spo2",
#             "recorded_at",
#         ]
#         read_only_fields = ["id", "recorded_at"]


# class DoctorSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Doctor
#         fields = [
#             "id",
#             "first_name",
#             "last_name",
#             "email",
#             "phone",
#             "is_available",
#             "created_at",
#         ]
#         read_only_fields = ["id", "created_at"]

from rest_framework import serializers
from .models import Patient, PatientVitals, Doctor, HeartPrediction

class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = ["id", "first_name", "last_name", "email", "phone", "address", "created_at"]
        read_only_fields = ["id", "created_at"]

class PatientVitalsSerializer(serializers.ModelSerializer):
    class Meta:
        model = PatientVitals
        fields = ["id", "patient", "pulse", "temperature", "respiratory_rate", "spo2", "recorded_at"]
        read_only_fields = ["id", "recorded_at"]

class DoctorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Doctor
        fields = ["id", "first_name", "last_name", "email", "phone", "is_available", "created_at"]
        read_only_fields = ["id", "created_at"]

class HeartPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = HeartPrediction
        fields = [
            "id",
            "patient",
            "vitals",
            "risk_probability",
            "threshold",
            "prediction",
            "input_payload",
            "created_at",
        ]
        read_only_fields = ["id", "risk_probability", "threshold", "prediction", "created_at"]