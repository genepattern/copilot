from django.apps import AppConfig
import asyncio
import logging

logger = logging.getLogger(__name__)


class ChatbotConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chatbot'

    # This will hold the ServiceHelper instance
    service_helper_instance = None
    # This will hold the asyncio.Lock, initialized per app lifecycle
    _instance_lock = None

    def ready(self):
        """
        This method is called when Django starts.
        It's a good place for setup that needs to happen once per process.
        """
        from .services import ServiceHelper # Import here to avoid circular imports

        # Initialize the lock here to ensure it's bound to the current event loop
        # and is part of the AppConfig's lifecycle.
        if not ChatbotConfig._instance_lock:
            ChatbotConfig._instance_lock = asyncio.Lock()

        # Using a background task or a separate thread for initial async setup
        # is often necessary if you want to kick it off immediately.
        # However, for a singleton accessed via `instance()`,
        # it will be initialized on first access.
        logger.info("ChatbotConfig is ready. ServiceHelper will be initialized on first access.")

        # Optional: If you need to force initialization immediately on startup,
        # you could run an async function in a new thread or a sync wrapper.
        # However, for a lazy singleton, `instance()` handles this.
        # Example (if you absolutely need to preload):
        # import threading
        # def _initial_load():
        #     loop = asyncio.new_event_loop()
        #     asyncio.set_event_loop(loop)
        #     loop.run_until_complete(ServiceHelper().create())
        #     loop.close()
        # threading.Thread(target=_initial_load).start()
