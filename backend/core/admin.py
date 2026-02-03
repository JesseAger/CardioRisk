from django.contrib import admin
from .models import Patient, Doctor, PatientVitals


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ("first_name", "last_name", "email", "phone", "created_at")
    search_fields = ("first_name", "last_name", "email", "phone")


@admin.register(Doctor)
class DoctorAdmin(admin.ModelAdmin):
    list_display = ("first_name", "last_name", "email", "phone", "is_available", "created_at")
    list_filter = ("is_available",)
    search_fields = ("first_name", "last_name", "email", "phone")


@admin.register(PatientVitals)
class PatientVitalsAdmin(admin.ModelAdmin):
    list_display = ("patient", "pulse", "temperature", "spo2", "recorded_at")
    list_filter = ("recorded_at",)
    search_fields = ("patient__first_name", "patient__last_name", "patient__email")
