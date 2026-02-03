from rest_framework import serializers


class HeartPredictSerializer(serializers.Serializer):
    # Keep everything explicit (good for documentation & future wearable endpoint)
    age = serializers.IntegerField(min_value=0, max_value=120)
    sex = serializers.IntegerField(min_value=0, max_value=1)  # 0=female, 1=male (dataset convention)
    cp = serializers.IntegerField()
    trestbps = serializers.IntegerField()
    chol = serializers.IntegerField()
    fbs = serializers.IntegerField(min_value=0, max_value=1)
    restecg = serializers.IntegerField()
    thalch = serializers.IntegerField()
    exang = serializers.IntegerField(min_value=0, max_value=1)
    oldpeak = serializers.FloatField()
    slope = serializers.IntegerField()
    ca = serializers.IntegerField()
    thal = serializers.IntegerField()
    # Additional fields to be added for wearable data in the future