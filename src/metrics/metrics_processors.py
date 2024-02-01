
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
import pickle
import torch

import evaluate
from utils.dirs import *

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
        if module.get('evaluate_alternative_answers', None):
            alter_acc_array = []
        for prediction in batch_predictions:
            question_id = prediction['question_id']
            annotation = self.data_loader.data.vqa_data.lookup.get(question_id, None)
            if annotation is None:
                logger.error(f'Annotation not found for question_id: {question_id}')
                raise ValueError(f'Annotation not found for question_id: {question_id}; the dataset might not be correct!')
            if prediction['answer'].lower() in annotation['answers']:
                acc_array.append(1)
            else:
                acc_array.append(0)

            if module.get('evaluate_alternative_answers', None):
                if prediction['answer'].lower() in annotation['answers'] + annotation.get('alternative_answers', []):
                    alter_acc_array.append(1)
                else:
                    alter_acc_array.append(0)
        
        log_dict.metrics['accuracy'] = np.mean(np.array(acc_array))
        if module.get('evaluate_alternative_answers', None):
            log_dict.metrics['accuracy_with_alternative_answers'] = np.mean(np.array(alter_acc_array))
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

    def compute_exact_match_with_numeric_values(self, module, data_dict, log_dict):
        '''
        Compute exact match
        '''
        ##############################
        ##    Compute Exact Match   ##
        ##############################
        # Beam search in this context is to rank all answer proposals by their scores
        # And the results are answers with top scores

        batch_answers = data_dict['batch_answers']
        batch_predictions = data_dict['batch_predictions']
        batch_numeric_ranges = data_dict['batch_numeric_ranges']
        
        cleaner = TextCleaner()
        exact_match_results = {
            'accuracy': []
        }
        
        for answer_list, answer_proposal, numeric_ranges in zip(batch_answers, batch_predictions, batch_numeric_ranges):
            correct = False
            # lower case
            answer_list = [ans.lower() for ans in answer_list]
            # answer_proposals = [ans.lower() for ans in answer_proposals]
            answer_proposal = answer_proposal['answer']
            answer_proposal = answer_proposal.lower()
            answer_proposal = cleaner.clean_texts([answer_proposal])[0]
            answer_list = cleaner.clean_texts(answer_list)
            # answer_proposals = cleaner.clean_texts(answer_proposals)
            
            proposal = answer_proposal
            # print(answer_list, answer_proposal, numeric_ranges)
            if proposal in answer_list:
                correct = True
            if numeric_ranges is not None:
                # check if the proposal is within the numeric range
                try:
                    proposal = float(proposal)
                    print(proposal, numeric_ranges)
                    if proposal >= numeric_ranges[0] and proposal <= numeric_ranges[1]:
                        correct = True
                        break
                except:
                    pass
            if correct:
                exact_match_results['accuracy'].append(1)
            else:
                exact_match_results['accuracy'].append(0)

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
            batch_question_ids: list of question ids
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
        batch_question_ids = data_dict['batch_question_ids']

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
        for question_id, answer_list, docs in zip(batch_question_ids, batch_answers, retrieved_docs):
            
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
            
            def check_contain_entity(ans, doc_to_check):
                doc_id = doc_to_check['title']
                triplet = self.data_loader.data.fvqa_data.triplets.get(doc_id, None)
                if triplet is None:
                    logger.error(f'triplet id {doc_id} not found in the data!')
                    return False
                else:
                    triplet_entities = [triplet.e1_label.lower(), triplet.e2_label.lower()]
                    if ans in triplet_entities:
                        return True
                    else:
                        return False
            

            if 'use_triplet_in_retrieval_metrics' in self.config.model_config.modules:
                item = self.data_loader.data.vqa_data.lookup.get(question_id, None)
                ref_triplet_ids = []
                for i in item.facts.values():
                    ref_triplet_ids.extend(i)
                
                for index, passage_data in enumerate(docs):
                    
                    if passage_data['title'] in ref_triplet_ids:
                        this_batch_labels[index] = 1
                        found_answers.append(passage_data['title'])
                        found_gold_answers.append(passage_data['title'])
                    # for answer in unique_answers:
                    #     if check_contain_entity(answer.lower(), passage_data):
                    #         found_answers.append(answer)
                    #         this_batch_labels[index] = 1
                    #         break
                    # if check_contain_entity(gold_answer.lower(), passage_data):
                    #     found_gold_answers.append(answer)
                    #     this_batch_labels[index] = 1

            else:
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
        from utils.vqa_tools import VQA
        from utils.vqaEval import VQAEval

        try:
            metrics_to_log = {}
            ##############################
            ##    Compute VQA Scores    ##
            ##############################
            mode = data_dict['mode']
            answers = data_dict['batch_predictions']

            torch.distributed.barrier()
            num_processes = torch.distributed.get_world_size()

            if not os.path.exists(self.config.ckpt_dir):
                create_dirs([self.config.ckpt_dir])

            # save tmp files for each process
            tmp_dir = os.path.join(self.config.ckpt_dir, f"tmp_{self.global_rank}.pkl")
            with open(tmp_dir, 'wb') as f:
                pickle.dump(answers, f)
            logger.info(f"Save tmp file {tmp_dir} for process {self.global_rank}.")
            
            torch.distributed.barrier()
            # load tmp files for each process
            all_answers = []
            for i in range(num_processes):
                tmp_dir = os.path.join(self.config.ckpt_dir, f"tmp_{i}.pkl")
                with open(tmp_dir, 'rb') as f:
                    all_answers.extend(pickle.load(f))
                logger.info(f"Load tmp file {tmp_dir} for process {i}.")
            
            torch.distributed.barrier()
            logger.info(f"extended answers from {len(answers)} to {len(all_answers)}")
            answers = all_answers

            # Convert all question_id into int before passing to VQA helper
            for ans in answers:
                ans['question_id'] = int(ans['question_id'])
                

            # create vqa object and vqaRes object
            vqa_helpers = EasyDict({
                'train': VQA(module.vqa_data_path.annotation_files.train, 
                                module.vqa_data_path.question_files.train),
                'test': VQA(module.vqa_data_path.annotation_files.test, 
                                module.vqa_data_path.question_files.test),
            })
            vqa_helper = vqa_helpers[mode]
            
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
        Compute DPR scores (Pseudo Relevance)
        This metrics is based on string matching. 
        """
        batch_result = data_dict['batch_retrieval_result']
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
            if 'answers' not in re.keys():
                # This metric can not be evaluated
                return log_dict
            
            top_ranking_passages = re['top_ranking_passages']
            answers = re['answers']
            gold_answer = re['gold_answer']

            for indexK, K in enumerate(Ks):
                found_answers = []
                found_gold_answers = []
                for passage_data in top_ranking_passages[:K]:
                    
                    for answer in answers:
                        if answer.lower() in passage_data['content'].lower():
                            found_answers.append(answer)
                            break
                    if gold_answer.lower() in passage_data['content'].lower():
                        found_gold_answers.append(answer)
                    
                if len(found_answers) > 0:
                    # At least one answer is retireved
                    result['recall'][indexK] += 1
                # The proportion of retrieved knowledge has an answer
                result['precision'][indexK] += len(found_answers) / K

                if len(found_gold_answers) > 0:
                    # if gold answer is found
                    result['gold_recall'][indexK] += 1
                # The proportion of retrieved knowledge has the gold answer
                result['gold_precision'][indexK] += len(found_gold_answers) / K

        result['precision'] = result['precision']/count
        result['recall'] = result['recall']/count
        result['gold_precision'] = result['gold_precision']/count
        result['gold_recall'] = result['gold_recall']/count
        
        log_result = EasyDict()
        for metrics_name, np_array in result.items():
            for index, K in enumerate(Ks):
                log_result[f'{metrics_name}_at_{K}'] = float(np_array[index])

        log_dict.metrics.update(log_result)
        return log_dict




    def compute_DPR_scores_with_pos_ids(self, module, data_dict, log_dict) -> dict:
        """
        Compute DPR scores (ground truth)
        This metrics is based on ground truth data. 
        """
        batch_result = data_dict['batch_retrieval_result']
        Ks = data_dict['Ks']
        max_K = max(Ks)

        field = module.get('field', 'pos_item_ids')

        # Total number of questions
        count = len(batch_result)
        result = {
            'precision': np.zeros(len(Ks)),
            'recall': np.zeros(len(Ks)),
        }


        for re in tqdm(batch_result):
            top_ranking_passages = re['top_ranking_passages']
            pos_item_ids = re[field]

            hit = []
            
            for passage_data in top_ranking_passages[:max_K]:
                if passage_data['passage_id'] in pos_item_ids:
                    hit.append(1)
                else:
                    hit.append(0)
            
            # print("---------------------")
            # print('pos_item_ids:', pos_item_ids)
            # print('top_ranking_passages', [passage_data['passage_id'] for passage_data in top_ranking_passages[:max_K]])
            # print("hit:", hit)
            # print("---------------------")
            # input()
            
            for indexK, K in enumerate(Ks):
                if sum(hit[:K]) > 0:
                    # At least one answer is retireved
                    result['recall'][indexK] += 1
                # The proportion of retrieved knowledge has an answer
                result['precision'][indexK] += sum(hit[:K]) / K

        result['precision'] = result['precision']/count
        result['recall'] = result['recall']/count
        
        log_result = EasyDict()
        for metrics_name, np_array in result.items():
            for index, K in enumerate(Ks):
                log_result[f'{field}_{metrics_name}_at_{K}'] = float(np_array[index])

        log_dict.metrics.update(log_result)
        return log_dict



    def compute_BLEU_scores(self, module, data_dict, log_dict) -> dict:
        """
        Compute BLEU scores for description retrieval
        This metrics is based on string matching. 
        """

        batch_result = data_dict['batch_retrieval_result']
        Ks = data_dict['Ks']
        max_K = max(Ks)

        # Total number of questions
        count = len(batch_result)
        result = {
            'bleu': np.zeros(len(Ks)),
            'precisions[0]': np.zeros(len(Ks)),
            'precisions[1]': np.zeros(len(Ks)),
            'precisions[2]': np.zeros(len(Ks)),
            'precisions[3]': np.zeros(len(Ks)),
            'brevity_penalty': np.zeros(len(Ks)),
            'length_ratio': np.zeros(len(Ks)),
            'translation_length': np.zeros(len(Ks)),
            'reference_length': np.zeros(len(Ks)),
        }

        """
        >>> predictions = ["hello there general kenobi", "foo bar foobar"]
        >>> references = [
        ...     ["hello there general kenobi", "hello there !"],
        ...     ["foo bar foobar"]
        ... ]
        >>> bleu = evaluate.load("bleu")
        >>> results = bleu.compute(predictions=predictions, references=references)
        >>> print(results)
        {'bleu': 1.0, 'precisions': [1.0, 1.0, 1.0, 1.0], 'brevity_penalty': 1.0, 'length_ratio': 1.1666666666666667, 'translation_length': 7, 'reference_length': 6}
        """

        bleu = evaluate.load("bleu")

        for re in tqdm(batch_result):
            top_ranking_passages = re['top_ranking_passages']
            pos_item_ids = re['pos_item_ids']
            pos_item_contents = re['pos_item_contents']
            pos_item_contents = list(set(pos_item_contents))

            references = [pos_item_contents]*max_K
            predictions = []

            for passage_data in top_ranking_passages:
                predictions.append(passage_data['content'])

            print(predictions[:3])
            print(references[:3])
            for index, K in enumerate(Ks):
                output_scores = bleu.compute(predictions=predictions[:K], references=references[:K])
                result['bleu'][index] += output_scores['bleu']
                result['precisions[0]'][index] += output_scores['precisions'][0]
                result['precisions[1]'][index] += output_scores['precisions'][1]
                result['precisions[2]'][index] += output_scores['precisions'][2]
                result['precisions[3]'][index] += output_scores['precisions'][3]
                result['brevity_penalty'][index] += output_scores['brevity_penalty']
                result['length_ratio'][index] += output_scores['length_ratio']
                result['translation_length'][index] += output_scores['translation_length']
                result['reference_length'][index] += output_scores['reference_length']

        
        log_result = EasyDict()
        for metrics_name, np_array in result.items():
            for index, K in enumerate(Ks):
                log_result[f'{metrics_name}_at_{K}'] = float(np_array[index] / count)

        log_dict.metrics.update(log_result)
        
        return log_dict