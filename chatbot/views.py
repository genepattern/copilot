from django.views.generic import TemplateView
from django.shortcuts import get_object_or_404
from rest_framework import generics, status, views, viewsets
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from .models import Conversation, Query, LlmModel
from .serializers import (
    ConversationSerializer,
    QuerySerializer,
    ChatInputSerializer,
    QueryRatingSerializer, LlmModelSerializer
)
from .services import handle_chat_message


class ChatAPIView(views.APIView):
    """
    API endpoint for handling chat interactions.
    POST: Send a new message (creates a conversation if needed).
    """
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = ChatInputSerializer(data=request.data)
        if serializer.is_valid():
            user = request.user
            conversation_id = serializer.validated_data.get('conversation_id')
            user_query = serializer.validated_data['query']
            model_id = serializer.validated_data.get('model_id')
            html = serializer.validated_data.get('html')

            # Call the service layer to handle the logic
            query_instance, error_message = handle_chat_message(user=user, conversation_id=conversation_id,
                                                                user_query=user_query, model_id=model_id)

            # Check if there was an error in processing
            if error_message: return Response({ "error": error_message }, status=status.HTTP_400_BAD_REQUEST)

            # Return the newly created query object
            if query_instance:
                output_serializer = QuerySerializer(query_instance, context={ 'request': request, 'html': html })
                return Response(output_serializer.data, status=status.HTTP_201_CREATED)

            # Should not happen if error_message is handled, but as a fallback
            else: return Response({"error": "Failed to process chat message."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ConversationDetailView(generics.RetrieveAPIView):
    """
    API endpoint to retrieve details and history of a specific conversation.
    GET: Retrieve conversation by ID.
    """
    queryset = Conversation.objects.prefetch_related('queries__steps', 'queries__llm_model').all()  # Optimize queries
    serializer_class = ConversationSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'id'  # Use the UUID field

    def get_queryset(self):
        # Ensure users can only access their own conversations
        return super().get_queryset().filter(user=self.request.user)


class ResponseRatingView(views.APIView):
    """
    API endpoint for rating a specific query response.
    PATCH: Update the rating of a query.
    """
    permission_classes = [AllowAny]

    def patch(self, request, query_id, *args, **kwargs):
        # Ensure the query exists and belongs to the user's conversation
        query = get_object_or_404(Query, id=query_id, conversation__user=request.user)

        serializer = QueryRatingSerializer(data=request.data)
        if serializer.is_valid():
            rating_value = serializer.validated_data['rating']
            query.rating = rating_value
            query.save(update_fields=['rating'])

            # Return the updated query or just a success message
            return Response({ 'response': 'Thanks for the feedback' }, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ChatInterfaceView(TemplateView):
    """
    Serves the main HTML chat interface. Requires user to be logged in.
    """
    template_name = 'index.html'


class LlmModelViewSet(viewsets.ModelViewSet):
    queryset = LlmModel.objects.all()
    serializer_class = LlmModelSerializer
