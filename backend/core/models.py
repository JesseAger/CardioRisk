from django.db import models
from django.utils import timezone


class Patient(models.Model):
    first_name = models.CharField(max_length=80)
    last_name = models.CharField(max_length=80)

    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=30, unique=True)

    address = models.TextField(blank=True)

    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.first_name} {self.last_name}"


class Doctor(models.Model):
    first_name = models.CharField(max_length=80)
    last_name = models.CharField(max_length=80)

    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=30, unique=True)

    is_available = models.BooleanField(default=True)

    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Dr. {self.first_name} {self.last_name}"


class PatientVitals(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name="vitals")

    # You can expand these later (SpO2, glucose, BP, etc.)
    pulse = models.PositiveIntegerField(help_text="beats per minute")
    temperature = models.DecimalField(max_digits=4, decimal_places=1, help_text="Celsius")

    # optional fields (safe to add now)
    respiratory_rate = models.PositiveIntegerField(null=True, blank=True, help_text="breaths per minute")
    spo2 = models.PositiveIntegerField(null=True, blank=True, help_text="oxygen saturation %")

    recorded_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Vitals for {self.patient} @ {self.recorded_at:%Y-%m-%d %H:%M}"
