from adrf.views import APIView as AsyncAPIView
from asgiref.sync import sync_to_async
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.mixins import UserPassesTestMixin
from django.views.generic import TemplateView
from django.shortcuts import get_object_or_404
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie
from django.db.models import Sum, Count, Q
from django.utils import timezone
from datetime import timedelta
from rest_framework import generics, status, views, viewsets
from rest_framework.permissions import IsAuthenticated, AllowAny, IsAdminUser
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Conversation, Query, LlmModel, UserProfile, TokenCount
from .serializers import (
    ConversationSerializer,
    QuerySerializer,
    ChatInputSerializer,
    QueryRatingSerializer, LlmModelSerializer,
    ConversationListSerializer
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


class ConversationListView(generics.ListAPIView):
    """
    API endpoint to list the current user's conversations (sidebar list).
    """
    serializer_class = ConversationListSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Conversation.objects.filter(user=self.request.user).prefetch_related('queries')


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
            files = serializer.validated_data.get('files', [])

            # If the call doesn't specify an API key, check the user profile
            if not api_key and user.is_authenticated:
                try:
                    profile = await sync_to_async(UserProfile.objects.get)(user=user)
                    # Only use the profile API key if it's not None or empty
                    if profile.gp_api_key:
                        api_key = profile.gp_api_key
                except UserProfile.DoesNotExist: pass  # Profile doesn't exist, no API key available

            # Call the service layer to handle the logic
            query_instance, error_message = await handle_chat_message(
                user=user,
                conversation_id=conversation_id,
                user_query=user_query,
                model_id=model_id,
                method_id=method_id,
                api_key=api_key,
                files=files
            )

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

    def get_serializer_context(self):
        context = super().get_serializer_context()
        # Ensure conversation responses are returned as HTML for rendering in chat history
        context['html'] = True
        return context


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


@method_decorator(ensure_csrf_cookie, name='dispatch')
class TokenSummaryView(UserPassesTestMixin, TemplateView):
    """
    Serves the token summary page - admin only.
    """
    template_name = 'token_summary.html'

    def test_func(self):
        return self.request.user.is_staff

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['user'] = self.request.user
        return context


class TokenSummaryAPIView(APIView):
    """
    API endpoint for token usage statistics - admin only.
    GET: Retrieve token usage summary by user for a given timeframe.
    """
    permission_classes = [IsAdminUser]

    def get(self, request, *args, **kwargs):
        # Get timeframe parameter (default to 'today')
        timeframe = request.query_params.get('timeframe', 'today')

        # Calculate date range based on timeframe
        now = timezone.now()
        if timeframe == 'today':
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif timeframe == '7days':
            start_date = now - timedelta(days=7)
        elif timeframe == '30days':
            start_date = now - timedelta(days=30)
        else:
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Query token counts for the timeframe
        token_data = TokenCount.objects.filter(
            timestamp__gte=start_date,
            token_type=TokenCount.TokenType.TOTAL
        ).values('user__username', 'user__id').annotate(
            total_tokens=Sum('token_count'),
            request_count=Count('id'),
            estimated_count=Count('id', filter=Q(estimated=True))
        ).order_by('-total_tokens')

        # Format the data
        user_stats = []
        total_all_tokens = 0
        total_all_requests = 0

        for item in token_data:
            username = item['user__username'] if item['user__username'] else 'Anonymous'
            tokens = item['total_tokens'] or 0
            requests = item['request_count'] or 0
            estimated = item['estimated_count'] or 0

            user_stats.append({
                'username': username,
                'user_id': item['user__id'],
                'total_tokens': tokens,
                'request_count': requests,
                'estimated_count': estimated,
                'is_anonymous': item['user__username'] is None
            })

            total_all_tokens += tokens
            total_all_requests += requests

        # Get model breakdown
        model_stats = TokenCount.objects.filter(
            timestamp__gte=start_date,
            token_type=TokenCount.TokenType.TOTAL
        ).values('llm_model__model_id', 'llm_model__label').annotate(
            total_tokens=Sum('token_count'),
            request_count=Count('id')
        ).order_by('-total_tokens')

        model_breakdown = []
        for item in model_stats:
            if item['llm_model__model_id']:
                model_breakdown.append({
                    'model_id': item['llm_model__model_id'],
                    'model_label': item['llm_model__label'],
                    'total_tokens': item['total_tokens'] or 0,
                    'request_count': item['request_count'] or 0
                })

        return Response({
            'timeframe': timeframe,
            'start_date': start_date.isoformat(),
            'end_date': now.isoformat(),
            'total_tokens': total_all_tokens,
            'total_requests': total_all_requests,
            'user_stats': user_stats,
            'model_stats': model_breakdown
        }, status=status.HTTP_200_OK)