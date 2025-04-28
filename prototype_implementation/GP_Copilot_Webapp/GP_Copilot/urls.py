"""GP_Copilot URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from copilot.views import * 
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    
    ############ Renders pages ##############
    path('', index), ## gpt-4o page
    path('experiment/', prompt_and_vecstore_experimenter_API, 
         name = 'experiment'), ## experiment page
    path('llama', llama_page), ## llama page
    path("validation-results/", validation_results, name="validation_results"), ## validation page
    path('haiku/', haiku_page, name='haiku'),
    ##########################################
    
    ############ API Endpoints ###############
    path('feedback/', response_API, name = 'feedback'),
    path('langgraph_view/', langgraph_API, name='langgraph_chatbot'),
    path('experiment_llm/', experiment_view_API, name = 'experiment inference'),
    path('upload_files/', upload_files_API, name='upload_files'),
    path('list_uploaded_files/', list_uploaded_files_API, name='list_uploaded_files'),
    path('delete_uploaded_file/', delete_uploaded_file_API, name='delete_uploaded_file'),
    path('get_document_count/', get_documents_count_API, name='get_document_count'),
    path('response_view/', response_API, name = 'response_view'),
    path("api/get-grouped-data/", get_grouped_data_API, name="get_grouped_data"),
    ##########################################


] + static(settings.STATIC_URL, document_root=settings.STATIC_URL)