from rest_framework.routers import DefaultRouter
from .views import PatientViewSet, PatientVitalsViewSet, DoctorViewSet

router = DefaultRouter()
router.register(r"patients", PatientViewSet, basename="patients")
router.register(r"vitals", PatientVitalsViewSet, basename="vitals")
router.register(r"doctors", DoctorViewSet, basename="doctors")

urlpatterns = router.urls
