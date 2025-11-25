from django.contrib import admin
from django.urls import path, include
from chatbot.views import ChatInterfaceView, TestInterfaceView, TokenSummaryView, ConversationListTemplateView, ConversationDetailTemplateView, MatrixTestView, TestPromptsAPIView

urlpatterns = [
    # GenePattern Copilot webapp
    path('', ChatInterfaceView.as_view(), name='chat-interface'),

    # Method test view
    path('test/', TestInterfaceView.as_view(), name='test-interface'),

    # Matrix testing view and API
    path('matrix/', MatrixTestView.as_view(), name='matrix-test'),
    path('matrix/test-prompts/', TestPromptsAPIView.as_view(), name='matrix-test-prompts'),

    # Token usage summary (admin only)
    path('tokens/', TokenSummaryView.as_view(), name='token-summary'),

    # Conversation admin views (admin only)
    path('conversations/', ConversationListTemplateView.as_view(), name='conversation-list-admin'),
    path('conversations/<uuid:conversation_id>/', ConversationDetailTemplateView.as_view(), name='conversation-detail-admin'),

    # Validation results page
    path('validation/', include('validation.urls', namespace='validation')),

    # Copilot admin interface
    path('admin/', admin.site.urls),

    # Copilot API endpoints
    path('api/', include('chatbot.urls', namespace='chatbot_api')),

    # Browsable API login / logout pages
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
]