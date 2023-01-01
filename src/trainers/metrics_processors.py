
import math
import time
import os
import sys
import scipy
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from tqdm import tqdm

import wandb
import logging
logger = logging.getLogger(__name__)

from utils.vqaEval import VQAEval
from utils.text_cleaner import TextCleaner

class MetricsProcessor():
    '''
    Metrics processor, general class definitions
    This is to save all metrics that we want to compute
    Data required for metrics computation should be passed in
    And each metrics module will compute metrics and append an entry in metrics_dict
    '''
    def __init__(self) -> None:
        pass

    def compute_metrics(self, data_dict):
        '''
        Compute metrics
        '''
        log_dict = EasyDict({
            "metrics": {},
            "artifacts": {},
        })
        for metrics in self.config.metrics:
            compute_func = getattr(self, metrics.name)
            logger.info(f"Running metrics {str(metrics)}...")
            log_dict = compute_func(metrics, data_dict, log_dict)
            # print(f"Metrics columns {log_dict.metrics.keys()} ")

        return log_dict
        
    def compute_accuracy(self, module, data_dict, log_dict):
        batch_predictions = data_dict['batch_predictions']
        acc_array = []

        for prediction in batch_predictions:
            question_id = prediction['question_id']
            annotation = self.data_loader.data.vqa_data.lookup.get(question_id, None)
            if annotation is None:
                logger.error(f'Annotation not found for question_id: {question_id}')
                raise ValueError(f'Annotation not found for question_id: {question_id}; the dataset might not be correct!')
            if prediction['answer'] in annotation['answers']:
                acc_array.append(1)
            else:
                acc_array.append(0)
        
        log_dict.metrics['accuracy'] = np.mean(np.array(acc_array))
        return log_dict


    def compute_exact_match(self, module, data_dict, log_dict):
        '''
        Compute exact match
        '''
        ##############################
        ##    Compute Exact Match   ##
        ##############################
        # Beam search in this context is to rank all answer proposals by their scores
        # And the results are answers with top scores

        batch_answers = data_dict['batch_answers']
        batch_generation_outputs_for_docs = data_dict['batch_generation_outputs_for_docs']
        batch_loss_with_doc_scores = data_dict['batch_loss_with_doc_scores']

        n_beams = 5
        cleaner = TextCleaner()
        exact_match_results = {
            'exact_match_at_{}'.format(beam+1): [] for beam in range(n_beams)
            }

        for answer_list, answer_proposals, answer_loss in zip(batch_answers, batch_generation_outputs_for_docs, batch_loss_with_doc_scores):
            answer_list = cleaner.clean_texts(answer_list)
            answer_proposals = cleaner.clean_texts(answer_proposals)
            
            indices = np.argsort(answer_loss)
            ranked_answer_proposals = []
            # the lower the loss, the higher the prob
            for index in indices:
                if answer_proposals[index] not in ranked_answer_proposals:
                    ranked_answer_proposals.append(answer_proposals[index])

            beam_count = 0
            for i in range(n_beams):
                if i < len(ranked_answer_proposals):
                    if ranked_answer_proposals[i] in answer_list:
                        # find exact match
                        beam_count = 1
                exact_match_results['exact_match_at_{}'.format(i+1)].append(beam_count)
        
        # Take average
        for metric in exact_match_results.keys():
            exact_match_results[metric] = np.mean(np.array(exact_match_results[metric]))

        log_dict.metrics.update(exact_match_results)

        return log_dict

    def compute_retrieval_metrics(self, module, data_dict, log_dict) -> dict:
        """
        Evaluate the retrieval performance of models
        recall, precision, gold_precision, gold_recall etc.
        successful_hit, successful_no_hit, etc.
        Args:
            batch_answers: gold answers
            batch_retrieved_docs: retrieved docs
            batch_generation_outputs_for_docs: the generation output for each retrieved doc
            batch_loss_with_doc_scores: list of loss with doc scores
        Returns:
            retrieval_metrics: dict of retrieval metrics
        """
        
        batch_answers = data_dict['batch_answers']
        batch_retrieved_docs = data_dict['batch_retrieved_docs']
        batch_generation_outputs_for_docs = data_dict['batch_generation_outputs_for_docs']
        batch_loss_with_doc_scores = data_dict['batch_loss_with_doc_scores']

        def most_frequent(List):
            return max(set(List), key = List.count)

        retrieved_docs = batch_retrieved_docs
        log_result = {
            'recall': [],
            'precision': [],
            'gold_precision': [],
            'gold_recall': [],
        }
        labels = []
        for answer_list, docs in zip(batch_answers, retrieved_docs):
            
            filtered_answer_list = [ans for ans in answer_list if ans != '']
            gold_answer = most_frequent(filtered_answer_list)

            unique_answers = list(set(answer_list))
            
            doc_texts = [doc['content'] for doc in docs]
            if 'add_null_document' in self.config.model_config.modules:
                doc_texts = doc_texts[1:] # ignore the null document!
            found_answers = []
            found_gold_answers = []
            K = len(doc_texts)
            this_batch_labels = [0] * len(doc_texts)

            for index, passage_data in enumerate(doc_texts):
                for answer in unique_answers:
                    if answer.lower() in passage_data.lower():
                        found_answers.append(answer)
                        this_batch_labels[index] = 1
                        break
                if gold_answer.lower() in passage_data.lower():
                    found_gold_answers.append(answer)
                    this_batch_labels[index] = 1

            labels.append(this_batch_labels)
                    
            if len(found_answers) > 0:
                # At least one answer is retireved
                log_result['recall'].append(1)
            else:
                log_result['recall'].append(0)
            # The proportion of retrieved knowledge has an answer
            log_result['precision'].append(len(found_answers) / K)

            if len(found_gold_answers) > 0:
                # if gold answer is found
                log_result['gold_recall'].append(1)
            else:
                log_result['gold_recall'].append(0)
            # The proportion of retrieved knowledge has the gold answer
            log_result['gold_precision'].append(len(found_gold_answers) / K)


        ##############################
        ##    Compute Retriever hit ##
        ##############################
        cleaner = TextCleaner()
        retriever_hit_results = {
            'successful_hit': [],
            'successful_no_hit': [],
            'failed_hit': [],
            'failed_no_hit': [],
            'selected_successful_hit': [],
            'selected_successful_no_hit': [],
            'selected_failed_hit': [],
            'selected_failed_no_hit': [],
        }
        for answer_list, docs, answer_proposals, answer_loss in zip(batch_answers, retrieved_docs, batch_generation_outputs_for_docs, batch_loss_with_doc_scores):
            doc_texts = [doc['content'] for doc in docs]
            indices = np.argsort(answer_loss)
            answer_list = cleaner.clean_texts(answer_list)
            answer_proposals = cleaner.clean_texts(answer_proposals)

            picked_answer = answer_proposals[indices[0]]
            picked_doc = doc_texts[indices[0]]

            for index, doc_text in enumerate(doc_texts):
                prediction_with_this_doc = answer_proposals[index]
                
                # whether the prediction has an exact match
                exact_match = prediction_with_this_doc in answer_list
                # whether the prediction in the doc
                contain_answer = prediction_with_this_doc in doc_text

                if exact_match and contain_answer:
                    retriever_hit_results['successful_hit'].append(1)
                else:
                    retriever_hit_results['successful_hit'].append(0)

                if exact_match and not contain_answer:
                    retriever_hit_results['successful_no_hit'].append(1)
                else:
                    retriever_hit_results['successful_no_hit'].append(0)
                
                if not exact_match and contain_answer:
                    retriever_hit_results['failed_hit'].append(1)
                else:
                    retriever_hit_results['failed_hit'].append(0)
                
                if not exact_match and not contain_answer:
                    retriever_hit_results['failed_no_hit'].append(1)
                else:
                    retriever_hit_results['failed_no_hit'].append(0)
        
            
            # whether the prediction has an exact match
            exact_match = picked_answer in answer_list
            # whether the prediction in the doc
            contain_answer = picked_answer in picked_doc

            if exact_match and contain_answer:
                retriever_hit_results['selected_successful_hit'].append(1)
            else:
                retriever_hit_results['selected_successful_hit'].append(0)

            if exact_match and not contain_answer:
                retriever_hit_results['selected_successful_no_hit'].append(1)
            else:
                retriever_hit_results['selected_successful_no_hit'].append(0)
            
            if not exact_match and contain_answer:
                retriever_hit_results['selected_failed_hit'].append(1)
            else:
                retriever_hit_results['selected_failed_hit'].append(0)
            
            if not exact_match and not contain_answer:
                retriever_hit_results['selected_failed_no_hit'].append(1)
            else:
                retriever_hit_results['selected_failed_no_hit'].append(0)

        log_result.update(retriever_hit_results)

        for metric in log_result.keys():
            log_result[metric] = np.mean(np.array(log_result[metric]))
        
        log_result['n_retrieved_docs'] = K


        log_dict.metrics.update(log_result)
        return log_dict



    def compute_okvqa_scores(self, module, data_dict, log_dict) -> dict:
        """
        Compute OkVQA scores
        """
        try:
            metrics_to_log = {}
            ##############################
            ##    Compute VQA Scores    ##
            ##############################
            mode = data_dict['mode']
            answers = data_dict['batch_predictions']

            # create vqa object and vqaRes object
            vqa_helper = self.data_loader.data.okvqa_data.vqa_helpers[mode]
            vqaRes = vqa_helper.loadResFromDict(answers)

            # create vqaEval object by taking vqa and vqaRes
            vqaEval = VQAEval(vqa_helper, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

            # evaluate results
            """
            If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
            By default it uses all the question ids in annotation file
            """
            vqaEval.evaluate()

            # print accuracies
            print ("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
            print ("Per Question Type Accuracy is the following:")
            for quesType in vqaEval.accuracy['perQuestionType']:
                print ("%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType]))
            print ("\n")
            print ("Per Answer Type Accuracy is the following:")
            for ansType in vqaEval.accuracy['perAnswerType']:
                print ("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            print ("\n")

            metrics_to_log['accuracy_overall'] = vqaEval.accuracy['overall']
            for quesType in vqaEval.accuracy['perQuestionType']:
                metrics_to_log[f'accuracy_QuestionType_{quesType}'] = vqaEval.accuracy['perQuestionType'][quesType]
            for ansType in vqaEval.accuracy['perAnswerType']:
                metrics_to_log[f'accuracy_AnswerType_{ansType}'] = vqaEval.accuracy['perAnswerType'][ansType]
            
            log_dict.metrics.update(metrics_to_log)
            return log_dict
        except Exception as e:
            if self.trainer.state.stage in ['sanity_check']:
                logger.info( f'Failed to compute OKVQA scores: {e}'\
                            'This could be due to the fact that OKVQA parser requires all questions to evaluate'\
                            'the accuracy. Ignore this error if this is the sanity check.')
            else:
                logger.error(f'Failed to compute OKVQA scores: {e}')
            return log_dict

    def compute_DPR_scores(self, module, data_dict, log_dict) -> dict:
        """
        Compute DPR scores
        """
        batch_result = data_dict['batch_result']
        Ks = data_dict['Ks']

        # Total number of questions
        count = len(batch_result)
        result = {
            'precision': np.zeros(len(Ks)),
            'recall': np.zeros(len(Ks)),
            'gold_precision': np.zeros(len(Ks)),
            'gold_recall': np.zeros(len(Ks)),
        }

        for re in tqdm(batch_result):
            result['precision'] += re['precision']/count
            result['recall'] += re['recall']/count
            result['gold_precision'] += re['gold_precision']/count
            result['gold_recall'] += re['gold_recall']/count
        
        log_result = EasyDict()
        for metrics_name, np_array in result.items():
            for index, K in enumerate(Ks):
                log_result[f'{metrics_name}_at_{K}'] = float(np_array[index])

        log_dict.metrics.update(log_result)
        return log_dict
        