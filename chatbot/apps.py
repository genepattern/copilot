# chatbot/apps.py
from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)


class ChatbotConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chatbot'

    def ready(self):
        logger.info("ChatbotConfig is ready.")