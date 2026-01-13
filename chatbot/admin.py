from django.contrib import admin
from django.db.models import Sum
from .models import LlmModel, SystemPrompt, Conversation, Query, Step, TokenCount


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
    list_display = ('id', 'user', 'started_at', 'label', 'query_count', 'estimated_cost')
    list_filter = ('user', 'started_at')
    search_fields = ('id', 'user', 'label')
    inlines = [QueryInline]
    readonly_fields = ('id', 'started_at', 'estimated_cost_display')

    def query_count(self, obj): return obj.queries.count()
    query_count.short_description = 'Queries'

    def estimated_cost(self, obj):
        """Calculate total estimated cost for all token counts in this conversation"""
        total_cost = 0.0
        for query in obj.queries.all():
            for step in query.steps.all():
                for token_count in step.token_counts.all():
                    total_cost += token_count.get_estimated_cost()
        return f"${total_cost:.4f}"
    estimated_cost.short_description = 'Estimated Cost'

    def estimated_cost_display(self, obj):
        """Display estimated cost for the change view"""
        return self.estimated_cost(obj)
    estimated_cost_display.short_description = 'Estimated Cost'


class TokenCountInline(admin.TabularInline):
    model = TokenCount
    extra = 0
    fields = ('token_type', 'token_count', 'estimated', 'timestamp')
    readonly_fields = ('timestamp',)
    ordering = ('timestamp',)


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
    list_display = ('id', 'conversation', 'query_num', 'llm_model', 'started_at', 'ended_at', 'rating')
    list_filter = ('llm_model', 'started_at', 'rating')
    search_fields = ('raw_query', 'response')
    inlines = [StepInline]
    readonly_fields = ('id', 'started_at', 'ended_at')


@admin.register(Step)
class StepAdmin(admin.ModelAdmin):
    list_display = ('id', 'query', 'step_num', 'call_id', 'llm_model', 'started_at', 'ended_at', 'token_summary')
    list_filter = ('llm_model', 'call_id', 'started_at')
    search_fields = ('step_input', 'step_output', 'call_id')
    inlines = [TokenCountInline]
    readonly_fields = ('id', 'started_at', 'ended_at')

    def token_summary(self, obj):
        """Display token count summary for this step"""
        token_counts = obj.token_counts.all()
        if not token_counts:
            return "No token data"

        summary = []
        for tc in token_counts:
            estimated_str = " (est)" if tc.estimated else ""
            summary.append(f"{tc.get_token_type_display()}: {tc.token_count}{estimated_str}")
        return " | ".join(summary)

    token_summary.short_description = 'Token Usage'


@admin.register(TokenCount)
class TokenCountAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'llm_model', 'token_type', 'token_count', 'estimated', 'call_id', 'timestamp')
    list_filter = ('token_type', 'estimated', 'llm_model', 'user', 'timestamp')
    search_fields = ('call_id',)
    readonly_fields = ('id', 'timestamp')
    date_hierarchy = 'timestamp'
