# ClassificationPrompt
This project explores the design and optimization of prompts to improve classification accuracy using large language models (LLMs). Contains code, examples, and best practices for various prompt strategies to tackle text classification, topic identification, and more
# Introduction

Welcome! In this MLS, we are going to look into the E-commerce aggregators business. Aggregator platforms face multiple challenges due to their unorganised nature. One such problem is the task of categorising products. Since the sellers on the platform aren't organised, they tend to mislabel some of their products. This leads to confusion in the customers leading to dissatisfaction and ultimately revenue loss. The management had tried to educate the sellers on labelling through hand-outs however, that hasn't been successfull. So they are looking to automate labeling using AI.

**Task 1: Auto-Labelling System**

Initially, our focus is on categories with the highest incidence of mislabeling. Currently, we are facing a 27% mislabeling in the system. One of the reasons this is happening is because there are a lot of common words between the skin care and hair care categories. Both of them have words like oil, powder, wash, etc. It is also confusing as sometimes products for body hair can be categorised as skin care and other times as hair care. Similarly, products for scalp can be basketed into skin care or hair care.Our job is to reduce this as much as possible.

1. Skin Care
2. Hair Care

# Task 2: Customer Intent Analysis**

Management also has received concerns about the usability of the website. They aim to enhance the user experience on our platform by gaining insights into the problems customers are facing. One good source of understanding where customers are having trouble is by looking at the concerns raised by customers with call support. Understanding customer intent in chat support messages will inform the management where improvements are needed in the platform's UI/UX, making navigation and usage more intuitive.

The data science team is entrusted with classifying customer intent into the relevant categories.

Due to resource constraints, we must utilize a small dataset for model training.

# TOOlS

[VScoce]
[Github]
[AzureOpenAI]

# Import all Python packages required to access the Azure Open AI API.

# Import additional packages required to access datasets and create examples.

All the required packages can installed and imported from requirements.txt file, here Requirements.txt file in only specified to read the conetents but not to load it.

# Authentication

create a [config.jcon]file to acess AzureOpenAI API.

# Utilities

While developing the solution, we need to be mindful of the costs it will incurr for the business. Even a good solution that comes at a high cost is not useful for the business. For LLMs, costs are associated with the number of tokens consumed. Let's create a function using tiktoken to understand the number of tokens we are using in each of out prompts. This information will be cruicial while deciding the final technique we are going to use to solve the problem.

# Task 1: Auto-Label Classificaation

Let's have a look at the data and get a feel of it, Prepare Data.

count
Category	
Hair Care	100
Skin Care	100

Note how the dataset is evenly balanced with equal number of reviews assembled for each of the category. This makes our life easy.

Since this is a classification exercise with a balanced dataset, we can use accuracy as our metric. We need to also be mindful of the tokens consumed for each prompt as this is going to be a perpetual task for the business as new products are added everyday.

# Test and Train Split

Let us split the data into two segments - one segment that gives us a pool to draw few-shot examples from and another segment that gives us a pool of gold examples which will be used for testing.

In summary, we extract a dataset from a corpus by processing required fields. Each example should contain the text input and an annotated label. Once we create examples and gold examples from this dataset, this curated dataset is stored in a format appropriate for reuse (e.g., JSON).

To select gold examples for this session, we sample randomly from the test data using a `random_state=42`. This ensures that the examples from multiple runs of the sampling are the same (i.e., they are randomly selected but do not change between different runs of the notebook). Note that we are doing this only to keep execution times low for illustration. In practise, large number of gold examples facilitate robust estimates of model accuracy.

With everything setup, let's start working on our prompts.

# Step 3: Derive Prompt

Create Prompts

Let's create a zero-shot prompt for this scenario. We need to make sure that LLM outputs only the category label and not explanation. So, let's add explicit instructions for that.

zero_shot_system_message = """
Classify the following product desciption presented in the input into one of the following categories.
Categories - ['Hair Care', 'Skin Care']
Product description will be delimited by triple backticks in the input.
Answer only 'Hair Care' or 'Skin Care'. Nothing Else. Do not explain your answer.
"""
# Now lets create Zero Shot prompt.

We can check the no.of tokens consumed by function defined during Utilities

Let's also cap the max_token parameter to 4 so that the model doesn't output explanations. We are capping it at 4 instead of 2 because we want to leave a little lee-way for punctuation marks and sub-words token that the model might output in the middle of the text. It is better to use regex later than to prematurely over-constrain the LLM output.

response = client.chat.completions.create(
    model=deployment_name,
    messages=zero_shot_prompt+user_input,
    temperature=0, # <- Note the low temperature
    max_tokens=4 # <- Note how we restrict the output to not more than 2 tokens
)
print(response.choices[0].message.content)

Let's create a generic evaluation function that can be used with all the prompting techniques that we are going to use.

"""
    Return the accuracy score for predictions on gold examples.
    For each example, we make a prediction using the prompt. Gold labels and
    model predictions are aggregated into lists and compared to compute the
    accuracy.

    Args:
        prompt (List): list of messages in the Open AI prompt format
        gold_examples (str): JSON string with list of gold examples
        user_message_template (str): string with a placeholder for product description
        samples_to_output (int): number of sample predictions and ground truths to print

    Output:
        accuracy (float): Accuracy computed by comparing model predictions.

# Now lets go with Few shot where the system meseage is similar but we need to provide examples in the case of few shot.

 To assemble few-shot examples, we will need to sample the required number of reviews from the training data. One approach would be to  first subset the different categories and then select samples from these subsets.       

 To avoid the biases, it is important to have a balanced set of examples that are arranged in random order. Let us create a Python function that generates bias-free examples (our function implements the workflow presented below):  

 Let's create a function to create few show prompt from our examples.

 That is 3x more token usage than zero-shot. Unless it gives significatnly better results, zero-shot will be the preferred one.

# Prompt 3: Chain-of-Thought

For the CoT prompt, we add detailed step-by-step instructions to the few shot system message instructing the model to carefully ponder before assigning the label. Apart from this addition, there are no further changes from the few-shot prompt.

The examples remain the same while the system message changes.

We can see that token consumption per example is highest in cot_fewshot followed by fewshot and the least by zero-shot. As the business has to process a lot of products, we need to make sure the token consumption is low as openAI charges the business per token basis. Even small improvements in the token consumption while keeping the accuracies can have a huge impact.

# Task 2: Intent Detection

Let's proceed to the second task, which involves determining customer intent from support queries. We have received a raw dataset of these queries and are required to conduct a preliminary analysis. Our goal is to identify various categories of inquiries and determine the most frequent ones. Based on this analysis, management will establish a labeling team to manually classify the queries into the top three categories, with a fourth category, 'Others,' for queries that do not fit into the top three.

Following this step, we will undertake a classification task using OpenAI's tools. If the results are satisfactory (i.e., accuracy greater than 85%), we will apply the model to the entire dataset to determine the actual frequency of each category. The category with the highest occurrence will be prioritized for action.

During this session, our focus will be on the initial exploration, construction, and evaluation of the classification task using a large language model (LLM).

# Step 2: Assemble and Explore Data

We are given an unlabeled dataset. First, let's try to understand what sort of problems are present in the data and which of them are most frequent using an LLM.

Let's concatenate all the responses into one string and then pass it to the LLM and ask it to provide us this preliminary analysis.

# Let us now craft a prompt.

The response from the LLM will vary every time we re-generate it. Since we are doing an unsupervised task and since the LLM has no concrete idea about the problem beforehand, it is expected that there will be some variance. We can iterate over this multiple times and check the outputs each times to figure out the most consistent responses. By doing that, we have a higher chance of actually finding out which of the labels is most frequent. Ideally this should be done on the whole data corpus. Each time sampling a different subset (so that we can fit the context length of the LLM). But in this case-study, we do not have the whole datset (thousands of customer queries). Hence, we don't need to sample the dataset. Instead, we can run the LLM through the data multiple times and find out the most frequent query category (most probably).

You are encouraged to re-run the above cell multiple times (4-5) to check which are the most consistent labels. Find 2-3 labels that are most consistent in those runs. From our runs, we found the following labels consistent - Modifying order(change order), track order, payment issues, account related issues. Of these three labels, modifying orders, payment issues and lastly track order are the most important ones to fix as they are directly related to revenue. Account related issues is a second priority. Let's look at change order, track oder and payment issues can be further investigated at large scale to understand which of these three is most problematic.

The whole classification task would have been a bigger project than actually correcting the three issues if it weren't for LLMs. The NLP classification itself would have taken multiple weeks. However, with LLMs, the whole task can be done within a week.

Using this preliminary exploration, a labeling team has been setup to categorize a sample of the data into the following categories - 'track order', 'change order', 'payment issues', and 'others'.

Let's import the labeled data.

Let's check the different categories of customer inquiries.

# Derive Prompts for Zero Shot, few shot by repeating the same steps used for labelling tasks.

After Evaluating Propmts, Let's have a feel of the data before we move forward.

If you run this with multiple samples, you see that the mislabelling is sometimes due to mislabelling during the manual labeling process and the LLM is actually getting it right and the problem is with the manual labeling itself. In cases where it's not due to manual mislabeling, it is an edge case where the label could have gone either way. Overall our analysis shows that the LLM model is a good proxy and can be used to find the most troubling UX experience.

We can go ahead and use the model to segregate user queries. Post this step, a dashboard can be prepared to show the most frequent labels we have found. This will reveal the most frequent problem encountered by the users.