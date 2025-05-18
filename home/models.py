from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Feedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    rating = models.IntegerField()
    comment = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Feedback from {self.user.username} - Rating: {self.rating}"
