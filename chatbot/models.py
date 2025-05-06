import uuid
from django.conf import settings
from django.db import models


class LlmModel(models.Model):
    """Which LLM model was called"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, unique=True)
    model_id = models.CharField(max_length=100, unique=True, help_text="Identifier used by the provider (e.g., gpt-4o)")
    provider_id = models.CharField(max_length=100, null=True, blank=True, help_text="Identifier of the provider (e.g., openai)")
    label = models.CharField(max_length=100, help_text="Human-friendly name (e.g., 'OpenAI GPT-4')")
    disabled = models.BooleanField(default=False, help_text="Mark model as disabled")

    def __str__(self): return self.label


class SystemPrompt(models.Model):
    """Which system prompt was used along with a query"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, unique=True)
    name = models.CharField(max_length=100, help_text="Human-friendly identifier for the prompt")
    version = models.FloatField(default=1)
    prompt = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('name', 'version')  # Ensure a specific prompt has unique versions
        ordering = ['-created_at']

    def __str__(self): return f"Prompt: {self.name} (v{self.version})"


class Conversation(models.Model):
    """An entry representing one conversation between the user and LLM"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, unique=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name='conversations')
    label = models.CharField(max_length=200, blank=True, null=True, help_text="Human-friendly label")
    started_at = models.DateTimeField(auto_now_add=True)

    class Meta: ordering = ['-started_at']

    def __str__(self): return f"Conversation {self.id} for {self.user.username if self.user else 'Anonymous'}"


class Query(models.Model):
    """One prompt-response between the user and LLM within a conversation"""

    class Rating(models.IntegerChoices):
        """Thumbs up or down rating for a response"""
        THUMBS_DOWN = -1, 'Thumbs Down'
        NO_RATING = 0, 'No Rating'
        THUMBS_UP = 1, 'Thumbs Up'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, unique=True)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='queries')
    query_num = models.PositiveIntegerField(help_text="Order of the query within the conversation")
    llm_model = models.ForeignKey(LlmModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='queries')
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, help_text="Datetime of the final response")
    raw_query = models.TextField(help_text="The user's raw input")
    response = models.TextField(null=True, blank=True, help_text="The LLM's final response")
    rating = models.IntegerField(choices=Rating.choices, default=Rating.NO_RATING, null=True, blank=True)

    class Meta:
        ordering = ['started_at']  # Order queries within a conversation chronologically
        verbose_name_plural = "Queries"  # Correct pluralization

    def __str__(self):
        return f"Query {self.query_num} of Conversation {self.conversation.label if self.conversation.label else self.conversation.id}"


class Step(models.Model):
    """Represents one step in the chain used to generate a response for a query"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, unique=True)
    query = models.ForeignKey(Query, on_delete=models.CASCADE, related_name='steps')
    step_num = models.PositiveIntegerField(help_text="Step number to generate the query")

    llm_model = models.ForeignKey(LlmModel, on_delete=models.SET_NULL, null=True, blank=True, related_name='calls', help_text="LLM used in this specific step")
    system_prompt = models.ForeignKey(SystemPrompt, on_delete=models.SET_NULL, null=True, blank=True, related_name='calls')
    call_id = models.CharField(max_length=100, help_text="Used to identify the step in the graph")

    step_input = models.TextField(null=True, blank=True, help_text="LLM input being passed to this step")
    step_output = models.TextField(null=True, blank=True, help_text="LLM output coming from this step")

    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, help_text="Datetime of the final response")

    class Meta:
        unique_together = ('query', 'step_num')  # Ensure step numbers are unique per query
        ordering = ['-query', 'step_num']  # Order steps within a query chronologically

    def __str__(self): return f"Step {self.step_num} of Query {self.query.id}"
