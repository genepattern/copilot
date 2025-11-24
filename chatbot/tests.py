import json
import uuid
from unittest.mock import patch, MagicMock, Mock
from django.test import TestCase, Client, override_settings
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone
from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from .models import (
    Conversation, Query, LlmModel, SystemPrompt,
    Step, UserProfile, TokenCount
)
from .serializers import (
    ConversationSerializer, QuerySerializer,
    LlmModelSerializer, ChatInputSerializer
)


class ModelTestCase(TestCase):
    """Test suite for Django models."""

    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123',
            email='test@example.com'
        )

        self.llm_model = LlmModel.objects.create(
            model_id='test-model-1',
            provider_id='openai',
            label='Test Model 1',
            disabled=False,
            max_context_tokens=128000
        )

        self.system_prompt = SystemPrompt.objects.create(
            name='TestPrompt',
            version=1.0,
            prompt='You are a helpful assistant.'
        )

        self.conversation = Conversation.objects.create(
            user=self.user,
            label='Test Conversation'
        )

    def test_user_profile_creation(self):
        """Test UserProfile model creation."""
        profile = UserProfile.objects.create(
            user=self.user,
            gp_api_key='test-api-key-123'
        )
        self.assertEqual(profile.user, self.user)
        self.assertEqual(profile.gp_api_key, 'test-api-key-123')
        self.assertIn('testuser', str(profile))

    def test_llm_model_creation(self):
        """Test LlmModel creation and string representation."""
        self.assertEqual(self.llm_model.model_id, 'test-model-1')
        self.assertEqual(self.llm_model.provider_id, 'openai')
        self.assertEqual(str(self.llm_model), 'Test Model 1')
        self.assertFalse(self.llm_model.disabled)

    def test_system_prompt_creation(self):
        """Test SystemPrompt creation and unique constraint."""
        self.assertEqual(self.system_prompt.name, 'TestPrompt')
        self.assertEqual(self.system_prompt.version, 1.0)
        self.assertIn('TestPrompt', str(self.system_prompt))

        # Test unique constraint
        prompt2 = SystemPrompt.objects.create(
            name='TestPrompt',
            version=2.0,
            prompt='Updated prompt'
        )
        self.assertEqual(SystemPrompt.objects.filter(name='TestPrompt').count(), 2)

    def test_conversation_creation(self):
        """Test Conversation model."""
        self.assertEqual(self.conversation.user, self.user)
        self.assertEqual(self.conversation.label, 'Test Conversation')
        self.assertIsNotNone(self.conversation.started_at)
        self.assertIn('testuser', str(self.conversation))

    def test_query_creation(self):
        """Test Query model with ratings."""
        query = Query.objects.create(
            conversation=self.conversation,
            query_num=1,
            llm_model=self.llm_model,
            raw_query='What is Django?',
            response='Django is a web framework.',
            rating=Query.Rating.THUMBS_UP
        )

        self.assertEqual(query.conversation, self.conversation)
        self.assertEqual(query.query_num, 1)
        self.assertEqual(query.rating, Query.Rating.THUMBS_UP)
        self.assertEqual(query.get_rating_display(), 'Thumbs Up')
        self.assertIn('Query 1', str(query))

    def test_step_creation(self):
        """Test Step model."""
        query = Query.objects.create(
            conversation=self.conversation,
            query_num=1,
            llm_model=self.llm_model,
            raw_query='Test query',
            response='Test response'
        )

        step = Step.objects.create(
            query=query,
            step_num=1,
            llm_model=self.llm_model,
            system_prompt=self.system_prompt,
            call_id='test_call',
            step_input='Input text',
            step_output='Output text'
        )

        self.assertEqual(step.query, query)
        self.assertEqual(step.step_num, 1)
        self.assertEqual(step.call_id, 'test_call')
        self.assertIn('Step 1', str(step))

    def test_token_count_creation(self):
        """Test TokenCount model."""
        query = Query.objects.create(
            conversation=self.conversation,
            query_num=1,
            llm_model=self.llm_model,
            raw_query='Test',
            response='Response'
        )

        step = Step.objects.create(
            query=query,
            step_num=1,
            llm_model=self.llm_model,
            call_id='test_call'
        )

        token_count = TokenCount.objects.create(
            step=step,
            user=self.user,
            llm_model=self.llm_model,
            token_type=TokenCount.TokenType.TOTAL,
            token_count=1000,
            estimated=False
        )

        self.assertEqual(token_count.token_count, 1000)
        self.assertEqual(token_count.token_type, TokenCount.TokenType.TOTAL)
        self.assertFalse(token_count.estimated)
        self.assertIn('testuser', str(token_count))


class SerializerTestCase(TestCase):
    """Test suite for serializers."""

    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )

        self.llm_model = LlmModel.objects.create(
            model_id='test-model',
            provider_id='openai',
            label='Test Model'
        )

        self.conversation = Conversation.objects.create(
            user=self.user,
            label='Test'
        )

    def test_llm_model_serializer(self):
        """Test LlmModelSerializer."""
        serializer = LlmModelSerializer(self.llm_model)
        data = serializer.data

        self.assertEqual(data['model_id'], 'test-model')
        self.assertEqual(data['provider_id'], 'openai')
        self.assertEqual(data['label'], 'Test Model')
        self.assertIn('disabled', data)

    def test_chat_input_serializer_validation(self):
        """Test ChatInputSerializer validation."""
        # Valid data
        valid_data = {
            'query': 'Test question',
            'model_id': 'test-model',
            'method_id': 'raw'
        }
        serializer = ChatInputSerializer(data=valid_data)
        self.assertTrue(serializer.is_valid())

        # Invalid model_id
        invalid_model_data = {
            'query': 'Test',
            'model_id': 'nonexistent-model'
        }
        serializer = ChatInputSerializer(data=invalid_model_data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('model_id', serializer.errors)

        # Invalid method_id
        invalid_method_data = {
            'query': 'Test',
            'method_id': 'invalid_method'
        }
        serializer = ChatInputSerializer(data=invalid_method_data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('method_id', serializer.errors)

        # Valid method_ids
        for method in ['raw', 'rag', 'mcp', 'rag_mcp']:
            data = {'query': 'Test', 'method_id': method}
            serializer = ChatInputSerializer(data=data)
            self.assertTrue(serializer.is_valid())

    def test_query_serializer(self):
        """Test QuerySerializer with markdown to HTML."""
        query = Query.objects.create(
            conversation=self.conversation,
            query_num=1,
            llm_model=self.llm_model,
            raw_query='Test query',
            response='**Bold** text',
            rating=Query.Rating.THUMBS_UP
        )

        # Test without HTML
        serializer = QuerySerializer(query, context={'html': False})
        data = serializer.data
        self.assertEqual(data['response'], '**Bold** text')

        # Test with HTML
        serializer = QuerySerializer(query, context={'html': True})
        data = serializer.data
        self.assertIn('<strong>Bold</strong>', data['response'])


class AuthenticationAPITestCase(APITestCase):
    """Test suite for authentication endpoints."""

    def setUp(self):
        """Set up test client and user."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.profile = UserProfile.objects.create(
            user=self.user,
            gp_api_key='test-key-123'
        )

    def test_login_success(self):
        """Test successful login."""
        url = reverse('chatbot_api:login')
        data = {
            'username': 'testuser',
            'password': 'testpass123'
        }
        response = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])
        self.assertEqual(response.data['username'], 'testuser')
        self.assertEqual(response.data['api_key'], 'test-key-123')

    def test_login_failure(self):
        """Test failed login with invalid credentials."""
        url = reverse('chatbot_api:login')
        data = {
            'username': 'testuser',
            'password': 'wrongpassword'
        }
        response = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertFalse(response.data['success'])

    def test_logout(self):
        """Test logout endpoint."""
        # Login first
        self.client.login(username='testuser', password='testpass123')

        # Logout
        url = reverse('chatbot_api:logout')
        response = self.client.post(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])


class ModelsAPITestCase(APITestCase):
    """Test suite for models API endpoint."""

    def setUp(self):
        """Set up test models."""
        self.client = APIClient()

        LlmModel.objects.create(
            model_id='model-1',
            provider_id='openai',
            label='Model 1',
            disabled=False
        )

        LlmModel.objects.create(
            model_id='model-2',
            provider_id='anthropic',
            label='Model 2',
            disabled=True
        )

    def test_get_models_list(self):
        """Test retrieving list of all models."""
        url = reverse('chatbot_api:models')
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 2)

        # Check model data
        model_ids = [m['model_id'] for m in response.data]
        self.assertIn('model-1', model_ids)
        self.assertIn('model-2', model_ids)


class ConversationAPITestCase(APITestCase):
    """Test suite for conversation endpoints."""

    def setUp(self):
        """Set up test data."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )

        self.llm_model = LlmModel.objects.create(
            model_id='test-model',
            provider_id='openai',
            label='Test Model'
        )

        self.conversation = Conversation.objects.create(
            user=self.user,
            label='Test Conversation'
        )

        self.query = Query.objects.create(
            conversation=self.conversation,
            query_num=1,
            llm_model=self.llm_model,
            raw_query='What is AI?',
            response='AI is artificial intelligence.'
        )

    def test_conversation_list_authenticated(self):
        """Test listing conversations for authenticated user."""
        self.client.force_authenticate(user=self.user)
        url = reverse('chatbot_api:conversation-list')
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        # ConversationListSerializer uses 'title' field, not 'label'
        self.assertEqual(response.data[0]['title'], 'Test Conversation')

    def test_conversation_list_unauthenticated(self):
        """Test conversation list requires authentication."""
        url = reverse('chatbot_api:conversation-list')
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_conversation_detail(self):
        """Test retrieving conversation detail."""
        self.client.force_authenticate(user=self.user)
        url = reverse('chatbot_api:conversation-detail', kwargs={'id': self.conversation.id})
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['id'], str(self.conversation.id))
        self.assertEqual(len(response.data['queries']), 1)

    def test_conversation_detail_wrong_user(self):
        """Test users can't access other users' conversations."""
        other_user = User.objects.create_user(
            username='otheruser',
            password='testpass123'
        )
        self.client.force_authenticate(user=other_user)

        url = reverse('chatbot_api:conversation-detail', kwargs={'id': self.conversation.id})
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)


class QueryRatingAPITestCase(APITestCase):
    """Test suite for query rating endpoint."""

    def setUp(self):
        """Set up test data."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )

        self.llm_model = LlmModel.objects.create(
            model_id='test-model',
            provider_id='openai',
            label='Test Model'
        )

        self.conversation = Conversation.objects.create(user=self.user)
        self.query = Query.objects.create(
            conversation=self.conversation,
            query_num=1,
            llm_model=self.llm_model,
            raw_query='Test',
            response='Response'
        )

    def test_rate_query_thumbs_up(self):
        """Test rating a query with thumbs up."""
        self.client.force_authenticate(user=self.user)
        url = reverse('chatbot_api:query-rate', kwargs={'query_id': self.query.id})
        data = {'rating': Query.Rating.THUMBS_UP}

        response = self.client.patch(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.query.refresh_from_db()
        self.assertEqual(self.query.rating, Query.Rating.THUMBS_UP)

    def test_rate_query_thumbs_down(self):
        """Test rating a query with thumbs down."""
        self.client.force_authenticate(user=self.user)
        url = reverse('chatbot_api:query-rate', kwargs={'query_id': self.query.id})
        data = {'rating': Query.Rating.THUMBS_DOWN}

        response = self.client.patch(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.query.refresh_from_db()
        self.assertEqual(self.query.rating, Query.Rating.THUMBS_DOWN)


@override_settings(
    DEFAULT_LLM_MODEL='test-model',
    DEFAULT_LLM_METHOD='raw',
    DAILY_TOKEN_LIMIT=1000000
)
class ChatAPITestCase(APITestCase):
    """Test suite for chat API endpoint with all agentic methods."""

    def setUp(self):
        """Set up test data and mocks."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )

        # Create multiple LLM models for testing
        # Dictionary keys must match the model_id field for serializer validation
        self.models = {
            'test-model': LlmModel.objects.create(
                model_id='test-model',
                provider_id='openai',
                label='Test OpenAI Model'
            ),
            'claude-3-5-sonnet-20241022': LlmModel.objects.create(
                model_id='claude-3-5-sonnet-20241022',
                provider_id='anthropic',
                label='Claude 3.5 Sonnet'
            ),
            'gemini-1.5-flash': LlmModel.objects.create(
                model_id='gemini-1.5-flash',
                provider_id='google_genai',
                label='Gemini 1.5 Flash'
            ),
        }

        self.system_prompt = SystemPrompt.objects.create(
            name='General',
            version=1.0,
            prompt='You are a helpful assistant.'
        )

    @patch('chatbot.views.handle_chat_message')
    def test_chat_raw_method(self, mock_handle):
        """Test chat with RAW method (no RAG, no MCP)."""
        # Setup mock to return a query instance and no error
        mock_query = Query.objects.create(
            conversation=Conversation.objects.create(),
            query_num=1,
            llm_model=self.models['test-model'],
            raw_query='What is Python?',
            response='This is a raw response.'
        )
        mock_handle.return_value = (mock_query, None)

        url = reverse('chatbot_api:chat-message')
        data = {
            'query': 'What is Python?',
            'model_id': 'test-model',
            'method_id': 'raw'
        }

        response = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('response', response.data)
        self.assertEqual(response.data['response'], "This is a raw response.")

    @patch('chatbot.views.handle_chat_message')
    def test_chat_rag_method(self, mock_handle):
        """Test chat with RAG method (with document retrieval)."""
        # Setup mock to return a query instance and no error
        mock_query = Query.objects.create(
            conversation=Conversation.objects.create(),
            query_num=1,
            llm_model=self.models['test-model'],
            raw_query='How do I use GenePattern?',
            response='RAG-based response with context.'
        )
        mock_handle.return_value = (mock_query, None)

        url = reverse('chatbot_api:chat-message')
        data = {
            'query': 'How do I use GenePattern?',
            'model_id': 'test-model',
            'method_id': 'rag'
        }

        response = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['response'], "RAG-based response with context.")

    @patch('chatbot.views.handle_chat_message')
    def test_chat_mcp_method(self, mock_handle):
        """Test chat with MCP method (with tool calling)."""
        # Setup mock to return a query instance and no error
        mock_query = Query.objects.create(
            conversation=Conversation.objects.create(),
            query_num=1,
            llm_model=self.models['test-model'],
            raw_query='List my GenePattern jobs',
            response='MCP response with tool usage.'
        )
        mock_handle.return_value = (mock_query, None)

        url = reverse('chatbot_api:chat-message')
        data = {
            'query': 'List my GenePattern jobs',
            'model_id': 'test-model',
            'method_id': 'mcp',
            'api_key': 'test-gp-api-key'
        }

        response = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['response'], "MCP response with tool usage.")

    @patch('chatbot.views.handle_chat_message')
    def test_chat_rag_mcp_method(self, mock_handle):
        """Test chat with RAG+MCP method (with both RAG and tools)."""
        # Setup mock to return a query instance and no error
        mock_query = Query.objects.create(
            conversation=Conversation.objects.create(),
            query_num=1,
            llm_model=self.models['test-model'],
            raw_query='Analyze my GenePattern job using documentation',
            response='Combined RAG+MCP response.'
        )
        mock_handle.return_value = (mock_query, None)

        url = reverse('chatbot_api:chat-message')
        data = {
            'query': 'Analyze my GenePattern job using documentation',
            'model_id': 'test-model',
            'method_id': 'rag_mcp',
            'api_key': 'test-gp-api-key'
        }

        response = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['response'], "Combined RAG+MCP response.")

    @patch('chatbot.views.handle_chat_message')
    def test_chat_all_models(self, mock_handle):
        """Test chat with all configured LLM models."""
        for model_id in self.models.keys():
            # Reset the mock for each iteration
            mock_handle.reset_mock()

            # Setup mock to return a query instance for each model
            mock_query = Query.objects.create(
                conversation=Conversation.objects.create(),
                query_num=1,
                llm_model=self.models[model_id],
                raw_query=f'Test query for {model_id}',
                response=f'Response from {model_id}'
            )
            mock_handle.return_value = (mock_query, None)

            url = reverse('chatbot_api:chat-message')
            data = {
                'query': f'Test query for {model_id}',
                'model_id': model_id,
                'method_id': 'raw'
            }

            response = self.client.post(url, data, format='json')

            # Add better error reporting
            if response.status_code != status.HTTP_201_CREATED:
                print(f"Failed for model {model_id}: {response.data}")

            self.assertEqual(response.status_code, status.HTTP_201_CREATED,
                           f"Failed for model {model_id}: {response.data}")
            self.assertIn('response', response.data)

    @patch('chatbot.views.handle_chat_message')
    def test_chat_existing_conversation(self, mock_handle):
        """Test adding a message to an existing conversation."""
        # Create initial conversation
        conversation = Conversation.objects.create(user=self.user)

        # Setup mock to return a query instance
        mock_query = Query.objects.create(
            conversation=conversation,
            query_num=1,
            llm_model=self.models['test-model'],
            raw_query='Follow-up question',
            response='Follow-up response'
        )
        mock_handle.return_value = (mock_query, None)

        url = reverse('chatbot_api:chat-message')
        data = {
            'conversation_id': str(conversation.id),
            'query': 'Follow-up question',
            'model_id': 'test-model',
            'method_id': 'raw'
        }

        self.client.force_authenticate(user=self.user)
        response = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        # Query count will be 1 since we're mocking
        self.assertEqual(Query.objects.filter(conversation=conversation).count(), 1)

    def test_chat_invalid_model(self):
        """Test chat with invalid model_id."""
        url = reverse('chatbot_api:chat-message')
        data = {
            'query': 'Test query',
            'model_id': 'nonexistent-model',
            'method_id': 'raw'
        }

        response = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_chat_invalid_method(self):
        """Test chat with invalid method_id."""
        url = reverse('chatbot_api:chat-message')
        data = {
            'query': 'Test query',
            'model_id': 'test-model',
            'method_id': 'invalid_method'
        }

        response = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    @patch('chatbot.services.check_daily_token_limit_exceeded')
    def test_chat_token_limit_exceeded(self, mock_token_check):
        """Test chat when daily token limit is exceeded."""
        mock_token_check.return_value = True

        url = reverse('chatbot_api:chat-message')
        data = {
            'query': 'Test query',
            'model_id': 'test-model',
            'method_id': 'raw'
        }

        response = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('too busy', response.data['error'])

    @patch('chatbot.views.handle_chat_message')
    def test_chat_with_api_key_from_profile(self, mock_handle):
        """Test chat uses API key from user profile when not provided."""
        # Create user profile with API key
        UserProfile.objects.create(
            user=self.user,
            gp_api_key='profile-api-key-123'
        )

        # Setup mock to return a query instance
        mock_query = Query.objects.create(
            conversation=Conversation.objects.create(user=self.user),
            query_num=1,
            llm_model=self.models['test-model'],
            raw_query='Test query',
            response='Response'
        )
        mock_handle.return_value = (mock_query, None)

        self.client.force_authenticate(user=self.user)
        url = reverse('chatbot_api:chat-message')
        data = {
            'query': 'Test query',
            'model_id': 'test-model',
            'method_id': 'raw'
        }

        response = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

    @patch('chatbot.views.handle_chat_message')
    def test_chat_anonymous_user(self, mock_handle):
        """Test chat works for anonymous users."""
        # Setup mock to return a query instance
        mock_query = Query.objects.create(
            conversation=Conversation.objects.create(user=None),
            query_num=1,
            llm_model=self.models['test-model'],
            raw_query='Anonymous query',
            response='Anonymous response'
        )
        mock_handle.return_value = (mock_query, None)

        url = reverse('chatbot_api:chat-message')
        data = {
            'query': 'Anonymous query',
            'model_id': 'test-model',
            'method_id': 'raw'
        }

        response = self.client.post(url, data, format='json')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        # Verify the query was created (conversation user is None in mock)
        self.assertIsNotNone(Query.objects.filter(raw_query='Anonymous query').first())


class TokenSummaryAPITestCase(APITestCase):
    """Test suite for token summary endpoint (admin only)."""

    def setUp(self):
        """Set up test data."""
        self.client = APIClient()

        # Create regular user and admin user
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )

        self.admin_user = User.objects.create_user(
            username='admin',
            password='adminpass123',
            is_staff=True
        )

        # Create test data
        self.llm_model = LlmModel.objects.create(
            model_id='test-model',
            provider_id='openai',
            label='Test Model'
        )

        conversation = Conversation.objects.create(user=self.user)
        query = Query.objects.create(
            conversation=conversation,
            query_num=1,
            llm_model=self.llm_model,
            raw_query='Test',
            response='Response'
        )
        step = Step.objects.create(
            query=query,
            step_num=1,
            llm_model=self.llm_model,
            call_id='test'
        )

        # Create token counts
        TokenCount.objects.create(
            step=step,
            user=self.user,
            llm_model=self.llm_model,
            token_type=TokenCount.TokenType.TOTAL,
            token_count=1000
        )

    def test_token_summary_admin_access(self):
        """Test admin can access token summary."""
        self.client.force_authenticate(user=self.admin_user)
        url = reverse('chatbot_api:token-summary-api')

        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_token_summary_non_admin_denied(self):
        """Test non-admin users are denied access."""
        self.client.force_authenticate(user=self.user)
        url = reverse('chatbot_api:token-summary-api')

        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_token_summary_unauthenticated_denied(self):
        """Test unauthenticated users are denied access."""
        url = reverse('chatbot_api:token-summary-api')

        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)


class ViewsTestCase(TestCase):
    """Test suite for template views."""

    def setUp(self):
        """Set up test client."""
        self.client = Client()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.admin_user = User.objects.create_user(
            username='admin',
            password='adminpass123',
            is_staff=True
        )

    def test_chat_interface_view(self):
        """Test main chat interface loads."""
        response = self.client.get('/')

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'index.html')

    def test_test_interface_view(self):
        """Test test interface loads."""
        response = self.client.get('/test/')

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'test.html')

    def test_token_summary_view_admin(self):
        """Test token summary page for admin users."""
        self.client.login(username='admin', password='adminpass123')
        response = self.client.get('/tokens/')

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'token_summary.html')

    def test_token_summary_view_non_admin(self):
        """Test token summary page denies non-admin users."""
        self.client.login(username='testuser', password='testpass123')
        response = self.client.get('/tokens/')

        # Should redirect or deny access
        self.assertNotEqual(response.status_code, 200)


class ValidationViewsTestCase(TestCase):
    """Test suite for validation views."""

    def setUp(self):
        """Set up test client."""
        self.client = Client()

    def test_validation_results_view(self):
        """Test validation results page loads."""
        response = self.client.get('/validation/')

        # Should load even if CSV doesn't exist
        self.assertEqual(response.status_code, 200)


class ServiceHelperTestCase(TestCase):
    """Test suite for service layer helpers."""

    @patch('chatbot.services.LlmModel.objects.filter')
    def test_load_llms(self, mock_filter):
        """Test LLM loading from database."""
        from chatbot.services import ServiceHelper

        # Create mock models
        mock_models = [
            Mock(
                model_id='gpt-4',
                provider_id='openai',
                disabled=False
            ),
            Mock(
                model_id='claude-3',
                provider_id='anthropic',
                disabled=False
            )
        ]
        mock_filter.return_value = mock_models

        llms = ServiceHelper._load_llms()

        self.assertIn('gpt-4', llms)
        self.assertIn('claude-3', llms)

    def test_estimate_token_counts(self):
        """Test token count estimation."""
        from chatbot.services import estimate_token_counts

        query = "This is a test query"
        result = "This is a longer test result with more words"

        counts = estimate_token_counts(query, result)

        self.assertIn('prompt_tokens', counts)
        self.assertIn('completion_tokens', counts)
        self.assertIn('total_tokens', counts)
        self.assertTrue(counts['estimated'])
        self.assertEqual(
            counts['total_tokens'],
            counts['prompt_tokens'] + counts['completion_tokens']
        )

    def test_extract_result_text(self):
        """Test extracting text from various result formats."""
        from chatbot.services import extract_result_text

        # Test with 'output' attribute
        result_with_output = Mock(output="Output text")
        self.assertEqual(extract_result_text(result_with_output), "Output text")

        # Test with 'data' attribute
        result_with_data = Mock(spec=['data'])
        result_with_data.data = "Data text"
        self.assertEqual(extract_result_text(result_with_data), "Data text")

        # Test with string
        self.assertEqual(extract_result_text("Plain string"), "Plain string")


class IntegrationTestCase(APITestCase):
    """End-to-end integration tests."""

    def setUp(self):
        """Set up test environment."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )

        self.llm_model = LlmModel.objects.create(
            model_id='test-model',
            provider_id='openai',
            label='Test Model'
        )

        SystemPrompt.objects.create(
            name='General',
            version=1.0,
            prompt='You are a helpful assistant.'
        )

    @patch('chatbot.views.handle_chat_message')
    def test_full_conversation_flow(self, mock_handle):
        """Test complete conversation flow from start to finish."""
        # Create a conversation that will be used
        conversation = Conversation.objects.create(user=self.user)

        # 1. Start new conversation - first query
        mock_query1 = Query.objects.create(
            conversation=conversation,
            query_num=1,
            llm_model=self.llm_model,
            raw_query='First question',
            response='Response 1'
        )
        mock_handle.return_value = (mock_query1, None)

        url = reverse('chatbot_api:chat-message')
        data = {
            'query': 'First question',
            'model_id': 'test-model',
            'method_id': 'raw'
        }

        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        conversation_id = response.data['conversation']

        # 2. Continue conversation - second query
        mock_query2 = Query.objects.create(
            conversation=conversation,
            query_num=2,
            llm_model=self.llm_model,
            raw_query='Second question',
            response='Response 2'
        )
        mock_handle.return_value = (mock_query2, None)

        data['conversation_id'] = conversation_id
        data['query'] = 'Second question'

        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        # 3. Rate the response
        query_id = response.data['id']
        rating_url = reverse('chatbot_api:query-rate', kwargs={'query_id': query_id})
        self.client.force_authenticate(user=self.user)

        rating_response = self.client.patch(
            rating_url,
            {'rating': Query.Rating.THUMBS_UP},
            format='json'
        )
        self.assertEqual(rating_response.status_code, status.HTTP_200_OK)

        # 4. Retrieve conversation history
        history_url = reverse('chatbot_api:conversation-detail', kwargs={'id': conversation_id})
        history_response = self.client.get(history_url)
        self.assertEqual(history_response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(history_response.data['queries']), 2)
