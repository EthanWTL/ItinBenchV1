# Dataset Link: 

Here is the link to the huggingface dataset: [Link](https://huggingface.co/datasets/EthanWTL81/ItinBench)

# Generation
### 1. Sole Planning
``` 
python sole_planning.py
```
Avaliable Parameters (Default in **bold** for lower cost):

```--model```: **gpt4o**, mistral, gemini, llama3.1

```--task```: allDataNoRoute, allDataRouteOP, **filteredDataRouteOP**

```--numPlan```: **1**, 2, 3, ..., n

```--OPENAI_API_KEY```, ```--MISTRAL_API_KEY```, ```--GEMINI_API_KEY```: The api key for openai, mistral, or gemini. Must match the model choosed. 

**Example Usage**
```
python sole_planning.py --model gpt4o --task filteredDataRouteOP --numPlan 1 --OPENAI_API_KEY your_api_key_here
```

**Note**: Experiementing with GPT4o is recommended for cost and easier to request for API. Llama uses Ollama as the server thus need to configure locally. 

---

### 2. Tool Use
```
python tool_use.py
```
Avaliable Parameters (Default in **bold** for lower cost):

```--model```: **gpt4o**, mistral, gemini, llama3.1

```--numPlan```: **1**, 2, 3, ..., n

```--apikey```: The api key for openai, mistral, or gemini. Must match the model choosed. 

**Example Usage**
```
python tool_use.py --model gpt4o --numPlan 1 --apikey your_api_key_here
```
---
# Extraction

Extraction use GPT4o to make sure the key information extraction is correct. so apikey need to be provided.

**Example usage**
```
python extraction.py --model your_model --task your_task --numPlan number_of_plan --apikey openai_api_key
```
---
# Evaluation
All evaluation code is provided. Here we provide the example for the main evaluation.

```--model```: **gpt4o**, mistral, gemini, llama3.1
```--task```: allDataNoRoute, allDataRouteOP, **filteredDataRouteOP**, toolUsePlans
```--numPlan```: number of plans you want to evaluate
```--threshold```: the preference match rate for each plan. Default is 75%, which means every 3 out of 4 preference entries need to be satisfied. 

**Example usage**
```
python extraction.py --model gpt4o --task filteredDataRouteOP --numPlan 10 --threshold 0.75
```

**Note:** the evaluation process involved the calculation of TSP algorithm thus requires a certain computation time when the number of plans goes up and depends on the computing resource.
