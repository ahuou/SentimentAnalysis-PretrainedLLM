# Emotional Text Analysis Repository

This repository contains a collection of tools for analyzing and modeling emotional content in text data. It includes several Python scripts, datasets, and resources designed for sentiment analysis.

## Repository Structure

### Python Scripts
- **data_evaluator.py**: Evaluates model performance using various metrics and provides classification reports.
- **data_modeliser.py**: Sets up and runs machine learning models for text-based emotion detection.
- **data_player.py**: Processes text data for visualization and statistical analysis.
- **data_retriever.py**: Manages the retrieval of text data from databases or file systems.
- **data_tester.py**: Tests the functions for accuracy and efficiency in data processing.
- **merger.py**: Merges multiple datasets or outputs into a single format.

### Directories
- **baselines/**: Stores baseline models for comparison.
- **dataset_emoDetectInText/**: Contains datasets formatted for emotion detection.
- **dataset_textEmotion/**: Stores datasets annotated for text-based emotion analysis.
- **prompts/**: Includes prompts for data collection or testing.
- **results/**: Holds output from scripts and models like analysis results and metrics.

## Key Functionalities

- **Emotion Analysis**: Advanced NLP techniques to detect emotions in text.
- **Data Cleaning and Preprocessing**: Automated tools for preparing text data for analysis.
- **Performance Evaluation**: Utilities to measure the effectiveness of sentiment analysis models.
- **Data Integration**: Scripts to consolidate datasets from various sources.

## Setup and Requirements

This repository uses llama_cpp_python repository and benefits from GPU acceleration, but to do so one needs to 
setup the python wheels and cuda drivers. Here is a useful issue that helped me fix that problem with my Windows install:
- https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html (overall )
- https://github.com/abetlen/llama-cpp-python/issues/721 (fixing GPU not being used)

In order to use the repository, one should first run the data_retriever script which will create 2 dataframes
corresponding to cleaned, tokenized versions of each dataset

Then, the script data_player allows to get basic statistics about both datasets, and the tester script
helps debugging the data_retriever module.

## Running the model

Finally the data_modeliser module can be run with the following command serving as example:
`python .\data_modeliser.py --prompts_file "prompts/DS2_3_Shot_prompt.txt" --model_path "mistral-7b-instruct-v0.2.Q4_K_M.gguf" --batch_size 0 --marker 'DS2' --start 14270 --random_few_shot --classified`

Let us discuss the arguments to this function quickly: 

- [`--prompts_file`](--prompts_file): This should be a text file, with a line giving the context, separated by a
return, followed by another line which is the actual prompt. As we will see, this file structure will depend on further arguments

- [`--model_path`](--model_path): This should not be changed, the default value (mistral) is the only one that
returns satisfying results

- [`--batch_size`](--batch_size): For llama, batch prompting isn't supported, a possible extension would be to check
how feasible it would be with Mistral

- [`--marker`](--marker): Used to help with saving and differentiating the results

- [`--start`](--start): In the case of getting some error mid-run, this parameter will run the modeliser module
from the line "start" in the database

- [`--random_few_shot`](--random_few_shot): This enables the model to get 3 random examples of text + prediction 
during prompting (this is known as few-shot prompting, not fine-tuning)

- [`--classified`](--classified): This decides whether during few-shot prompting, we should provide 1 example 
of each label. It is recommended to keep default values as it biases the model due to the label's non-uniform repartition

## Evaluating the model

In order to evaluate the model, one should run the data_evaluator script, one can understand with the example call:
`evaluator("results/DS1_0_Shot_res.txt", "simple", normalized=True, mode="discrete", lowBound=0.5, highBound=0.5, majMap='Negative')`

- [`--res_to_eval`](--res_to_eval): Path to the pickled result file obtained at the end of the data_modeliser module run

- [`--type`](--type): This should be set to "simple" if using results for dataset1, and "tweet" if using results for dataset2

- [`--normalized`](--normalized): This decides whether the heatmaps for the confusion matrices should sum to 1 or to the size of the dataset

- [`--mode`](--mode): This should be set to "discrete" if we expect the model to predict a sentiment, and to continuous if the model returns a score from 0 to 1

- [`--lowBound`](--lowBound): In the case of getting a score, this decides manually the mapping score->sentiment (< lowBound -> Negative)

- [`--highBound`](--highBound): In the case of getting a score, this decides manually the mapping score->sentiment (> highBound -> Positive)

- [`--majMap`](--majMap): In the case the model predicts an answer that can't be approximated to one of the sentiments, transforms its output to the selected sentiment majMap

- [`--testPercent`](--testPercent): Allows us to measure performance on the same subset of the DS as the baselines


## Comparing the model to the baselines

In order to know how good our evaluation of the model's performance with respect to our prompt is we compare it to different types of baselines.
These baselines can be statistical, regression-based (using simpler ML methods than LLMs). There is also an extra baseline which runs a very specific pretrained LLM (Roberta)
