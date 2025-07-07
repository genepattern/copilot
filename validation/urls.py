from django.urls import path
from validation.views import validation_results, get_grouped_data

app_name = 'validation'


urlpatterns = [
    # Validation data endpoint
    path("get-grouped-data/", get_grouped_data, name="get_grouped_data"),

    # Validation results page
    path('', validation_results, name='validation_results'),
]