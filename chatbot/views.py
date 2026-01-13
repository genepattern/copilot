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
from .models import Conversation, Query, LlmModel, UserProfile, TokenCount, Step
from .serializers import (
    ConversationSerializer,
    QuerySerializer,
    ChatInputSerializer,
    QueryRatingSerializer, LlmModelSerializer,
    ConversationListSerializer
)
from .services import handle_chat_message
import yaml
from pathlib import Path
import logging
import base64
from urllib.parse import unquote

# Configure logging
logger = logging.getLogger(__name__)


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


class ChatAPIView(APIView):
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
            method_id = serializer.validated_data.get('method_id')
            api_key = serializer.validated_data.get('api_key')
            html = serializer.validated_data.get('html')
            files = serializer.validated_data.get('files', [])

            # Log initial state
            logger.info(f"ChatAPIView.post - User: {user.username if user.is_authenticated else 'anonymous'}")
            logger.info(f"  API key from request: {api_key is not None}")
            if api_key:
                logger.info(f"  Request API key type: {type(api_key)}, length: {len(api_key)}")

            # If the call doesn't specify an API key, check the user profile
            if not api_key and user.is_authenticated:
                try:
                    profile = UserProfile.objects.get(user=user)
                    logger.info(f"  Found UserProfile for {user.username}")
                    logger.info(f"  Profile gp_api_key exists: {profile.gp_api_key is not None}")
                    if profile.gp_api_key:
                        logger.info(f"  Profile gp_api_key type: {type(profile.gp_api_key)}, length: {len(profile.gp_api_key)}")
                        logger.info(f"  Profile gp_api_key value: '{profile.gp_api_key[:8]}...'")
                    # Only use the profile API key if it's not None or empty
                    if profile.gp_api_key:
                        api_key = profile.gp_api_key
                        logger.info(f"  ✓ Using API key from user profile")
                    else:
                        logger.info(f"  ✗ Profile API key is None or empty")
                except UserProfile.DoesNotExist:
                    logger.info(f"  ✗ No UserProfile found for {user.username}")
                    pass  # Profile doesn't exist, no API key available

            # Log final API key state before calling service
            logger.info(f"  Final api_key being passed to service: {api_key is not None}")
            if api_key:
                logger.info(f"  Final API key type: {type(api_key)}, length: {len(api_key)}, value: '{api_key[:8]}...'")

            # Call the service layer to handle the logic (now synchronous)
            query_instance, error_message = handle_chat_message(
                user=user,
                conversation_id=conversation_id,
                user_query=user_query,
                model_id=model_id,
                method_id=method_id,
                api_key=api_key,
                files=files
            )

            # Check if there was an error in processing
            if error_message:
                return Response({"error": error_message}, status=status.HTTP_400_BAD_REQUEST)

            # Return the newly created query object
            if query_instance:
                output_serializer = QuerySerializer(query_instance, context={'request': request, 'html': html})
                return Response(output_serializer.data, status=status.HTTP_201_CREATED)

            # Should not happen if error_message is handled, but as a fallback
            else:
                return Response({"error": "Failed to process chat message."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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
            return Response({'response': 'Thanks for the feedback'}, status=status.HTTP_200_OK)

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
            except UserProfile.DoesNotExist:
                pass  # Profile doesn't exist, nothing to clear

        logout(request)
        return Response({'success': True})


class RefreshTokenAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            # Get the GenePattern cookie
            gp_cookie = request.COOKIES.get('GenePattern')

            if not gp_cookie:
                return Response({'error': 'Could not refresh GenePattern token'},
                              status=status.HTTP_400_BAD_REQUEST)

            # Parse the cookie
            parts = gp_cookie.split('|')
            if len(parts) != 2:
                return Response({'error': 'Could not refresh GenePattern token'},
                              status=status.HTTP_400_BAD_REQUEST)

            username = parts[0]
            encoded_password = parts[1]
            decoded_password = base64.b64decode(unquote(encoded_password)).decode('utf-8')

            # Import the connect_to_genepattern function
            from copilot.auth import connect_to_genepattern

            # Call GenePattern to refresh the token
            user, token = connect_to_genepattern(username, decoded_password)

            # Store API key in user profile
            profile, created = UserProfile.objects.get_or_create(user=user)
            profile.gp_api_key = token
            profile.save()

            return Response({
                'success': True,
                'message': 'Token refreshed successfully'
            })

        except Exception as e:
            logger.error(f'Error refreshing GenePattern token: {str(e)}')
            return Response({'error': 'Could not refresh GenePattern token'},
                          status=status.HTTP_400_BAD_REQUEST)


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
        total_estimated_cost = 0.0

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

        # Calculate total estimated cost from all token counts in the timeframe
        all_token_counts = TokenCount.objects.filter(timestamp__gte=start_date)
        for tc in all_token_counts:
            total_estimated_cost += tc.get_estimated_cost()

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
            'estimated_cost': round(total_estimated_cost, 4),
            'user_stats': user_stats,
            'model_stats': model_breakdown
        })


class ConversationListAdminView(generics.ListAPIView):
    """
    Admin API endpoint to list all conversations.
    """
    serializer_class = ConversationListSerializer
    permission_classes = [IsAdminUser]

    def get_queryset(self):
        return Conversation.objects.all().prefetch_related('queries')


class ConversationDetailAdminView(generics.RetrieveAPIView):
    """
    Admin API endpoint to retrieve details of a specific conversation.
    """
    queryset = Conversation.objects.prefetch_related('queries__steps', 'queries__llm_model').all()
    serializer_class = ConversationSerializer
    permission_classes = [IsAdminUser]
    lookup_field = 'id'

    def get_queryset(self):
        return super().get_queryset()

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['html'] = True
        return context


@method_decorator(ensure_csrf_cookie, name='dispatch')
class ConversationListTemplateView(UserPassesTestMixin, TemplateView):
    """
    Serves the conversation list page - admin only.
    """
    template_name = 'conversation_list.html'

    def test_func(self):
        return self.request.user.is_staff

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['user'] = self.request.user

        # Get user filter from query parameter
        user_filter = self.request.GET.get('user', None)

        # Get page number from query parameter (default to 1)
        try:
            page = int(self.request.GET.get('page', 1))
            if page < 1:
                page = 1
        except (ValueError, TypeError):
            page = 1

        page_size = 100
        offset = (page - 1) * page_size

        # Get all conversations with their first query for fallback labeling
        conversations = Conversation.objects.all().prefetch_related('queries', 'user')

        # Apply user filter if provided
        if user_filter == 'anonymous':
            conversations = conversations.filter(user__isnull=True)
        elif user_filter:
            conversations = conversations.filter(user__username=user_filter)

        # Order by most recent
        conversations = conversations.order_by('-started_at')

        # Get total count for pagination
        total_count = conversations.count()
        total_pages = (total_count + page_size - 1) // page_size  # Ceiling division

        # Apply pagination
        conversations = conversations[offset:offset + page_size]

        conversation_data = []
        for conv in conversations:
            first_query = conv.queries.order_by('query_num').first()
            fallback_label = ''
            if first_query and not conv.label:
                fallback_label = first_query.raw_query[:50] + ('...' if len(first_query.raw_query) > 50 else '')

            conversation_data.append({
                'id': conv.id,
                'label': conv.label if conv.label else fallback_label,
                'user': conv.user,
                'started_at': conv.started_at,
            })

        context['conversations'] = conversation_data
        context['filtered_user'] = user_filter
        context['current_page'] = page
        context['total_pages'] = total_pages
        context['total_count'] = total_count
        context['has_previous'] = page > 1
        context['has_next'] = page < total_pages
        context['previous_page'] = page - 1
        context['next_page'] = page + 1

        # Generate page range for pagination display (show max 10 pages)
        page_range_start = max(1, page - 5)
        page_range_end = min(total_pages, page + 5)
        context['page_range'] = range(page_range_start, page_range_end + 1)

        return context


@method_decorator(ensure_csrf_cookie, name='dispatch')
class ConversationDetailTemplateView(UserPassesTestMixin, TemplateView):
    """
    Serves the conversation detail page - admin only.
    """
    template_name = 'conversation_detail.html'

    def test_func(self):
        return self.request.user.is_staff

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['user'] = self.request.user

        conversation_id = kwargs.get('conversation_id')
        conversation = get_object_or_404(
            Conversation.objects.prefetch_related(
                'queries__steps__llm_model',
                'queries__llm_model'
            ),
            id=conversation_id
        )

        context['conversation'] = conversation
        context['queries'] = conversation.queries.all().order_by('query_num')

        return context


@method_decorator(ensure_csrf_cookie, name='dispatch')
class MatrixTestView(TemplateView):
    """
    Serves the matrix testing interface.
    """
    template_name = 'matrix.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['user'] = self.request.user

        # Get user's API key if authenticated
        user_api_key = ''
        if self.request.user.is_authenticated:
            try:
                profile = UserProfile.objects.get(user=self.request.user)
                user_api_key = profile.gp_api_key if profile.gp_api_key else ''
            except UserProfile.DoesNotExist:
                pass

        context['user_api_key'] = user_api_key
        return context


class TestPromptsAPIView(APIView):
    """
    API endpoint to retrieve pre-defined test prompts from the test_prompts directory.
    GET: List all available test prompts.
    """
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        from django.conf import settings

        # Get the test_prompts directory
        test_prompts_dir = Path(settings.BASE_DIR) / 'test_prompts'

        if not test_prompts_dir.exists():
            return Response([], status=status.HTTP_200_OK)

        prompts = []

        # Load all YAML files from the directory
        for file_path in sorted(test_prompts_dir.glob('*.yaml')):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)

                    if data and isinstance(data, dict):
                        # Extract prompt data
                        prompt_data = {
                            'id': file_path.stem,  # Use filename without extension as ID
                            'name': data.get('name', file_path.name),
                            'description': data.get('description', ''),
                            'prompt': data.get('prompt', ''),
                            'combinations': data.get('combinations', [])
                        }
                        prompts.append(prompt_data)
            except Exception as e:
                # Skip files that can't be parsed
                print(f"Error loading test prompt {file_path}: {e}")
                continue

        return Response(prompts, status=status.HTTP_200_OK)
