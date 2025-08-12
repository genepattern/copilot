from adrf.views import APIView as AsyncAPIView
from asgiref.sync import sync_to_async
from django.contrib.auth import authenticate, login, logout
from django.views.generic import TemplateView
from django.shortcuts import get_object_or_404
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie
from rest_framework import generics, status, views, viewsets
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Conversation, Query, LlmModel, UserProfile
from .serializers import (
    ConversationSerializer,
    QuerySerializer,
    ChatInputSerializer,
    QueryRatingSerializer, LlmModelSerializer
)
from .services import handle_chat_message


class ModelsAPIView(views.APIView):
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        """
        API endpoint to retrieve the list of available models.
        GET: List all available LLM models.
        """
        models = LlmModel.objects.all()
        serializer = LlmModelSerializer(models, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class ChatAPIView(AsyncAPIView):
    """
    API endpoint for handling chat interactions.
    POST: Send a new message (creates a conversation if needed).
    """
    permission_classes = [AllowAny]

    def validate_and_get_data(self, serializer):
        if serializer.is_valid(): return serializer.validated_data
        else: return None

    @sync_to_async
    def serialize_output(self, serializer): return serializer.data

    @sync_to_async
    def serialize_errors(self, serializer): return serializer.errors

    async def post(self, request, *args, **kwargs):
        serializer = ChatInputSerializer(data=request.data)
        data = await sync_to_async(self.validate_and_get_data)(serializer)
        if data:
            user = request.user
            conversation_id = serializer.validated_data.get('conversation_id')
            user_query = serializer.validated_data['query']
            model_id = serializer.validated_data.get('model_id')
            method_id = serializer.validated_data.get('method_id')
            api_key = serializer.validated_data.get('api_key')
            html = serializer.validated_data.get('html')

            # If the call doesn't specify an API key, check the user profile
            if not api_key and user.is_authenticated:
                try:
                    profile = await sync_to_async(UserProfile.objects.get)(user=user)
                    api_key = profile.gp_api_key
                except UserProfile.DoesNotExist: pass  # Profile doesn't exist, no API key available

            # Call the service layer to handle the logic
            query_instance, error_message = await handle_chat_message(user=user, conversation_id=conversation_id,
                                                                      user_query=user_query, model_id=model_id,
                                                                      method_id=method_id, api_key=api_key)

            # Check if there was an error in processing
            if error_message: return Response({ "error": error_message }, status=status.HTTP_400_BAD_REQUEST)

            # Return the newly created query object
            if query_instance:
                output_serializer = QuerySerializer(query_instance, context={ 'request': request, 'html': html })
                data = await self.serialize_output(output_serializer)
                return Response(data, status=status.HTTP_201_CREATED)

            # Should not happen if error_message is handled, but as a fallback
            else: return Response({"error": "Failed to process chat message."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        errors = await self.serialize_errors(serializer)
        return Response(errors, status=status.HTTP_400_BAD_REQUEST)


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


@method_decorator(ensure_csrf_cookie, name='dispatch')
class ChatInterfaceView(TemplateView):
    """
    Serves the main chat interface.
    """
    template_name = 'index.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['user'] = self.request.user
        return context


@method_decorator(ensure_csrf_cookie, name='dispatch')
class TestInterfaceView(TemplateView):
    """
    Serves the test chat interface.
    """
    template_name = 'test.html'


class LlmModelViewSet(viewsets.ModelViewSet):
    queryset = LlmModel.objects.all()
    serializer_class = LlmModelSerializer


class LoginAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')

        if username and password:
            user = authenticate(username=username, password=password)
            if user:
                login(request, user)

                # Store API key in user profile
                profile, created = UserProfile.objects.get_or_create(user=user)

                return Response({
                    'success': True,
                    'username': user.username,
                    'is_staff': user.is_staff,
                    'api_key': profile.gp_api_key if hasattr(profile, 'gp_api_key') else None
                })

        return Response({'success': False, 'error': 'Invalid credentials'},
                        status=status.HTTP_401_UNAUTHORIZED)


class LogoutAPIView(APIView):
    permission_classes = [AllowAny]
    def post(self, request):
        # Clear API key from user profile if user is authenticated
        if request.user.is_authenticated:
            try:
                profile = UserProfile.objects.get(user=request.user)
                profile.gp_api_key = None
                profile.save(update_fields=['gp_api_key'])
            except UserProfile.DoesNotExist: pass  # Profile doesn't exist, nothing to clear

        logout(request)
        return Response({'success': True})