from django.db import models

class conversations(models.Model):
    conversation_id = models.IntegerField()
    user_prompt = models.TextField()
    response = models.TextField()
    date = models.DateField(auto_now=True)
    user_score = models.IntegerField()
    actions_taken = models.TextField()
    response_id = models.TextField()