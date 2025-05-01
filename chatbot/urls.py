from django.urls import path, include
from .views import ChatAPIView, ConversationDetailView, ResponseRatingView, LlmModelViewSet, ModelsAPIView
from rest_framework import routers


app_name = 'chatbot'


router = routers.DefaultRouter()
router.register(r'llm-models', LlmModelViewSet, basename='llm-models')

urlpatterns = [
    # API endpoint to retrieve the list of available models
    path('models/', ModelsAPIView.as_view(), name='models'),

    # API endpoint to post a query (handles new/existing conversations)
    path('chat/', ChatAPIView.as_view(), name='chat-message'),

    # API endpoint to rate a specific response
    path('rate/<query_id>/', ResponseRatingView.as_view(), name='query-rate'),

    # API endpoint to retrieve a specific conversation's details and history
    path('conversations/<id>/', ConversationDetailView.as_view(), name='conversation-detail'),

    # Django REST Framework's browsable API
    path('', include(router.urls)),
]