# ItinBenchV1

# Dataset Link: 

[Link](https://huggingface.co/datasets/EthanWTL81/ItinBench)

---

# Generation

### Sole Planning
``` 
python sole_planning.py
```
Avaliable Parameters (Default in **bold** for lower cost):

```--model```: **gpt4o**, mistral, gemini, llama3.1

```--task```: allDataNoRoute, allDataRouteOP, **filteredDataRouteOP**

```--numPlan```: **1**, 2, 3, ..., n

```--OPENAI_API_KEY```, ```--MISTRAL_API_KEY```, ```--GEMINI_API_KEY```: The api key for openai, mistral, or gemini. Must match the model choosed. 

**Note**: Experiementing with GPT4o is recommended for cost and easier to request for API. Llama uses Ollama as the server thus need to configure locally. 
