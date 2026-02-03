from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import HeartPredictSerializer
from .ml import HeartModelService


class HeartPredictView(APIView):
    def post(self, request):
        serializer = HeartPredictSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            result = HeartModelService.predict(serializer.validated_data)
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response(result, status=status.HTTP_200_OK)
