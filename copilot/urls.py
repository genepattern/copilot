from django.contrib import admin
from django.urls import path, include
from chatbot.views import ChatInterfaceView

urlpatterns = [
    # GenePattern Copilot webapp
    path('', ChatInterfaceView.as_view(), name='chat-interface'),

    # Copilot admin interface
    path('admin/', admin.site.urls),

    # Copilot API endpoints
    path('api/', include('chatbot.urls', namespace='chatbot_api')),

    # Browsable API login / logout pages
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
]