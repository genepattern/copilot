import json
import os
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from markdown_it import MarkdownIt
from bs4 import BeautifulSoup

CSV_FILE_PATH = os.path.join("validation", "validation_results_full.csv")


def validation_results(request):
    """
    Display the initial validation results page with default filters and groupings.
    """
    if not os.path.exists(CSV_FILE_PATH):
        return render(request, "validation_results.html", {"error": "CSV file not found"})

    df = pd.read_csv(CSV_FILE_PATH)

    system_prompt_filters = {str(k): v for k, v in enumerate(df.SystemPrompt.unique())}

    doc_filter = 'empty'
    document_store_versions = list(df.DocumentStoreVersion.unique())
    system_prompt_options = list(system_prompt_filters.keys())

    return render(request, "validation_results.html", {
        "document_store_versions": document_store_versions,
        "system_prompt_options": system_prompt_options,
    })


def markdown_to_html(markdown_text):
    parser = MarkdownIt()
    html = parser.render(markdown_text)
    soup = BeautifulSoup(html, 'html.parser')
    for a in soup.find_all('a', href=True): a['target'] = '_blank'
    return str(soup)


@csrf_exempt
def get_grouped_data(request):
    """
    API endpoint to return grouped data based on selected DocumentStore Version and System Prompt.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            doc_filter = data.get("document_store_version", "empty")

            system_prompt_key = data.get("system_prompt", "0")

            if not os.path.exists(CSV_FILE_PATH):
                return JsonResponse({"error": "CSV file not found"}, status=404)

            df = pd.read_csv(CSV_FILE_PATH)

            # Ensure valid key exists in dictionary
            system_prompt_filters = {str(k): v for k, v in enumerate(df.SystemPrompt.unique())}
            systemprompt_filter = system_prompt_filters.get(system_prompt_key, system_prompt_filters["0"])

            df_filtered = df[(df["DocumentStoreVersion"] == doc_filter) & (df["SystemPrompt"] == systemprompt_filter)]

            grouped = df_filtered.groupby(["UserPrompt", 'real_answer'])

            grouped_data = []
            for (user_prompt, real_answer), group in grouped:
                grouped_data.append({
                    "UserPrompt": markdown_to_html(user_prompt),
                    "RealAnswer": markdown_to_html(real_answer),
                    "responses": [
                        {
                            "ID": response["ID"],
                            "Response": markdown_to_html(response["Response"]),
                            "ModelType": markdown_to_html(response["ModelType"])
                        }
                        for response in group[['ID', "Response", 'ModelType']].to_dict(orient="records")
                    ]
                })

            print(len(grouped_data))

            return JsonResponse({"grouped_data": grouped_data})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)
