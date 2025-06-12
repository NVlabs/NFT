from qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer
from qwen_math_eval_toolkit.grader import math_equal as qwen_math_equal
    
from multiprocessing import Process, Queue
def qwen_math_equal_subprocess(prediction, reference,  timeout_seconds=10):
    def worker(q, prediction, reference):
        result = qwen_math_equal(prediction=prediction, reference=reference, timeout=False)
        q.put(result)

    q = Queue()
    p = Process(target=worker, args=(q, prediction, reference))
    p.start()
    
    # 添加超时处理
    p.join(timeout=timeout_seconds)  # 等待进程完成，最多等待 timeout_seconds 秒
    
    # 如果进程还在运行，则终止它并返回 False
    if p.is_alive():
        p.terminate()
        p.join()  # 确保进程被完全清理
        return False
        
    # 如果进程正常完成，获取结果
    try:
        return q.get_nowait()
    except:
        return False   

import re 
def preprocess_box_response_for_qwen_prompt(sequence, answer):
    # breakpoint()
    model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', sequence, flags=re.DOTALL,count = 1)
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    extract_answer = qwen_extract_answer(model_output, data_name="math") #TODO: check the data_name, hard code here for now
    
    if qwen_math_equal_subprocess(prediction=extract_answer, reference=answer):
        return 1.0
    else:
        return 0.0
