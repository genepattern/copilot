from django.urls import path, include
from .views import (ChatAPIView, ConversationDetailView, ResponseRatingView, LlmModelViewSet, ModelsAPIView,
                    LoginAPIView, LogoutAPIView, RefreshTokenAPIView, ConversationListView, TokenSummaryAPIView)
from rest_framework import routers


app_name = 'chatbot'


router = routers.DefaultRouter()
router.register(r'llm-models', LlmModelViewSet, basename='llm-models')

urlpatterns = [
    # API endpoint to handle user login and logout
    path('login/', LoginAPIView.as_view(), name='login'),
    path('logout/', LogoutAPIView.as_view(), name='logout'),
    path('refresh/', RefreshTokenAPIView.as_view(), name='refresh'),

    # API endpoint to retrieve the list of available models
    path('models/', ModelsAPIView.as_view(), name='models'),

    # API endpoint to post a query (handles new/existing conversations)
    path('chat/', ChatAPIView.as_view(), name='chat-message'),

    # API endpoint to rate a specific response
    path('rate/<query_id>/', ResponseRatingView.as_view(), name='query-rate'),

    # API endpoint to list and retrieve conversations
    path('conversations/', ConversationListView.as_view(), name='conversation-list'),
    path('conversations/<id>/', ConversationDetailView.as_view(), name='conversation-detail'),

    # API endpoint for token usage statistics (admin only)
    path('token-summary/', TokenSummaryAPIView.as_view(), name='token-summary-api'),

    # Django REST Framework's browsable API
    path('', include(router.urls)),
]