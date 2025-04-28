from django.contrib import admin
from .models import LlmModel, SystemPrompt, Conversation, Query, Step


@admin.register(LlmModel)
class LlmModelAdmin(admin.ModelAdmin):
    list_display = ('label', 'provider_id', 'model_id', 'disabled')
    search_fields = ('label', 'provider_id', 'model_id')


@admin.register(SystemPrompt)
class SystemPromptAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'version', 'created_at', 'prompt_preview')
    list_filter = ('name', 'version', 'created_at')
    search_fields = ('prompt',)

    def prompt_preview(self, obj):
        return obj.prompt[:100] + '...' if len(obj.prompt) > 100 else obj.prompt

    prompt_preview.short_description = 'Prompt Preview'


class QueryInline(admin.TabularInline):  # Or StackedInline
    model = Query
    extra = 0  # Don't show extra empty forms
    fields = ('query_num', 'llm_model', 'started_at', 'ended_at', 'raw_query', 'response', 'rating')
    readonly_fields = ('query_num', 'started_at', 'ended_at')  # Don't allow editing timestamp
    show_change_link = True


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'started_at', 'label', 'query_count')
    list_filter = ('user', 'started_at')
    search_fields = ('id', 'user', 'label')
    inlines = [QueryInline]
    readonly_fields = ('id', 'started_at')

    def query_count(self, obj): return obj.queries.count()
    query_count.short_description = 'Queries'


class StepInline(admin.TabularInline):
    model = Step
    extra = 0
    fields = ('step_num', 'llm_model', 'system_prompt', 'started_at', 'ended_at', 'step_input_preview', 'step_output_preview')
    readonly_fields = ('started_at', 'ended_at', 'step_input_preview', 'step_output_preview')
    ordering = ('step_num',)

    def step_input_preview(self, obj): return obj.step_input[:50] + '...' if obj.step_input and len(obj.step_input) > 50 else obj.step_input
    step_input_preview.short_description = 'Input Preview'

    def step_output_preview(self, obj): return obj.step_output[:50] + '...' if obj.step_output and len(obj.step_output) > 50 else obj.step_output
    step_output_preview.short_description = 'Output Preview'


@admin.register(Query)
class QueryAdmin(admin.ModelAdmin):
    list_display = ('id', 'conversation', 'query_num', 'llm_model', 'started_at', 'ended_at', 'query_preview', 'response_preview', 'rating')
    list_filter = ('conversation', 'llm_model', 'rating')
    search_fields = ('id', 'raw_query', 'response', 'llm_model')
    inlines = [StepInline]
    readonly_fields = ('id', 'query_num', 'started_at', 'ended_at')

    def query_preview(self, obj): return obj.raw_query[:100] + '...' if len(obj.raw_query) > 100 else obj.raw_query
    query_preview.short_description = 'Query Preview'

    def response_preview(self, obj): return obj.response[:100] + '...' if obj.response and len(obj.response) > 100 else obj.response
    response_preview.short_description = 'Response Preview'


@admin.register(Step)
class StepAdmin(admin.ModelAdmin):
    list_display = ('id', 'query', 'step_num', 'llm_model', 'system_prompt', 'call_id', 'started_at', 'ended_at')
    list_filter = ('query', 'step_num', 'llm_model', 'system_prompt')
    search_fields = ('id', 'llm_model', 'step_input', 'step_output')
    readonly_fields = ('id', 'started_at', 'ended_at')
