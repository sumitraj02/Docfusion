import json


class LLMPrompt:
    def __init__(self):
        self.user_first_4_results = []

    def prompt_for_user_based_search(self, search_result):
        for key, value in search_result["user_based_search"].items():
            for entry in value:
                if len(self.user_first_4_results) < 4:
                    self.user_first_4_results.append(entry)
                else:
                    break

        user_query_llm_input = json.dumps(self.user_first_4_results, indent=4)

        prompt = f'''    
            I am providing a JSON containing text excerpts that match a user's query with a similarity score of 0.85 or higher. Your task is to generate a concise, well-structured, and contextually relevant summary based on this content.

            Instructions:
            - The summary should be clear, concise, and directly relevant to the user’s query.
            - Retain the most important insights while eliminating redundancy.
            - Ensure the summary is logically structured and coherent.
            - If multiple texts provide different perspectives, synthesize them into a unified and meaningful response.
            - Avoid unnecessary details, focusing only on highly relevant information.
            Input:
                {user_query_llm_input}

            Note:  Do not include text like I understand or here is your summary and Do not mension heading at start.
            Output Format:
            Provide a concise, well-structured, and contextually relevant summary based on the retrieved texts.
        '''

        with open('./extracted/userbased.txt','w',encoding='utf-8') as data:
            data.write(str(prompt))

        return prompt
    

    def prompt_for_intro(self, search_result):
        intros =  search_result.get("default_results", {}).get("Introduction", {})
        if not intros:  
            intros = search_result["user_based_search"].items()

        intro_formatted_data = {"Introduction": intros}

        # Convert to JSON string
        prompt = f'''
            I am providing a JSON containing introductions from multiple academic papers, along with their similarity scores. Your task is to generate a concise, well-structured, and academically written summarized introduction that effectively presents the background, motivation, and objectives of the given papers.

            Instructions:
            - The summary should be formal, academic, and engaging.
            - Clearly introduce the topic, research problem, and significance of the study.
            - Retain key background information while avoiding redundancy.
            - Ensure logical coherence and a smooth transition of ideas.
            - Highlight common themes, research gaps, and objectives from the provided texts.
            
            Input: {intro_formatted_data}

            Note:  Do not include text like I understand or here is your summary and Do not mension heading at start.

            Output Format:
            Provide a well-structured and academically written introduction that encapsulates the key elements of the provided introductions.
        '''
        with open('./extracted/introduction.txt','w',encoding='utf-8') as data:
            data.write(str(prompt))

        return prompt
    
    def prompt_for_abstract(self, search_result):
        abstracts =  search_result.get("default_results", {}).get("Abstract", {})

        if not abstracts:  
            abstracts = search_result["user_based_search"].items()

        abstract_formatted_data = {"Abstract": abstracts}

        # Convert to JSON string
        abstract_llm_input = json.dumps(abstract_formatted_data, indent=4)

        prompt = f'''
            I am providing a JSON containing abstracts from multiple academic papers, along with their similarity scores. Your task is to generate a concise, well-structured, and academically written summarized abstract that captures the core ideas, key findings, and main contributions of the given abstracts.

            Instructions:
            - The summary should be formal and academic in tone.
            - Retain the most significant insights from the abstracts while eliminating redundancy.
            - Ensure the summary is coherent and logically structured, maintaining a clear flow of ideas.
            - Where applicable, highlight any common themes, key methodologies, or conclusions.
            - Avoid unnecessary details while ensuring completeness and clarity.
            
            Input: {abstract_llm_input}

            Note:  Do not include text like I understand or here is your summary and Do not mension heading at start.

            Output Format:
            Provide a single summarized abstract in clear, academic language.
        '''
        
        with open('./extracted/abstract.txt','w',encoding='utf-8') as data:
            data.write(str(abstract_llm_input))

        return prompt

    def prompt_for_conclusion(self, search_result):
        # breakpoint()
        conclusions =  search_result.get("default_results", {}).get("Conclusion", {})

        if not conclusions:  
            conclusions = search_result["user_based_search"].items()

        conclusion_formatted_data = {"Conclusion": conclusions}

        # Convert to JSON string
        conclusion_llm_input = json.dumps(conclusion_formatted_data, indent=4)

        prompt = f'''
        I am providing a JSON containing conclusions from multiple academic papers, along with their similarity scores. Your task is to generate a concise, well-structured, and academically written summarized conclusion that effectively synthesizes the key findings, implications, and future directions of the given papers.

        Instructions:
        - The summary should be formal and academic in tone.
        - Clearly state the main findings and their significance.
        - Highlight common conclusions while avoiding redundancy.
        - Discuss practical implications, limitations, and possible future research directions.
        - Ensure coherence, logical flow, and clarity in presenting the summary.
        
        Input: {conclusion_llm_input}

        Note:  Do not include text like I understand or here is your summary and Do not mension heading at start.

        Output Format:
        Provide a well-structured and academically written conclusion that encapsulates the key takeaways and potential future directions from the provided conclusions.        

        '''
        
        with open('./extracted/conclusion.txt','w',encoding='utf-8') as data:
            data.write(str(conclusion_llm_input))

        return prompt
    
    def prompt_for_reference(self, search_result):
        references =  search_result.get("default_results", {}).get("References", {})

        if not references:  
            references = search_result["user_based_search"].items()

        reference_formatted_data = {"References": references}

        # Convert to JSON string
        reference_llm_input = json.dumps(reference_formatted_data, indent=4)

        prompt = f'''
        I am providing extracted text containing references from multiple academic papers. Your task is to provide extract, organize, deduplicate, and format these references into a properly structured academic reference section output, You can also remove some irrilated papers from it.

        Example Output:
        [1]   Zeiler, M. D. and Fergus, “Visualizing and understanding convolutional networks”. European Conference on Computer Vision, vol 8689. Springer, Cham, pp. 818-833, 2014. 
        [2]   Yann LeCun, Yoshua Bengio, Geoffery  Hinton,  “Deep  Learning”, Nature, Volume 521, pp. 436-444, Macmillan Publishers, May 2015.

        Input:
        {reference_llm_input}

        Note:  Do not include text like I understand or here is your summary and Do not mension heading at start, Name of the Paper should be in double quotes "Visualizing and understanding convolutional networks".

        Output:
        Provide a output which is well-structured, serialized with [N] where N is a number, deduplicated, and properly formatted reference list in a consistent academic citation style. Do not include any additional text or explanations—only the final formatted references.
        '''
        
        with open('./extracted/reference.txt','w',encoding='utf-8') as data:
            data.write(str(reference_llm_input))

        return prompt
    
    def prompt_for_methodology(self, search_result):
        methodologys =  search_result.get("default_results", {}).get("Methodology", {})
        if not methodologys:  
            methodologys = search_result["user_based_search"].items()

        methodology_formatted_data = {"Methodology": methodologys}

        try:
            methodology_llm_input = json.dumps(methodology_formatted_data, indent=4)
        except TypeError as e:
            print("JSON Serialization Error:", e)
            print("Data that caused the issue:", methodology_formatted_data)
            return None

        prompt = f'''

        I am providing a JSON containing methodologies from multiple academic papers, along with their similarity scores. Your task is to generate a concise, well-structured, and academically written summarized methodology that accurately captures the research approach, experimental setup, and techniques used in the given papers.

        Instructions:
        - The summary should be formal and academic in tone.
        - Clearly describe the research design, data sources, techniques, and procedures used.
        - Retain key methodological details while eliminating redundancy.
        - Ensure the summary is coherent, logically structured, and technically precise.
        - If multiple methodologies are provided, highlight common approaches and differences, if relevant.
        
        Input:{methodology_llm_input}

        Note:  Do not include text like I understand or here is your summary and Do not mension heading at start.
        
        Output Format:
        Provide a well-structured and academically written methodology summary that effectively synthesizes the approaches used in the given papers.

        '''
        
        with open('./extracted/methodology.txt','w',encoding='utf-8') as data:
            data.write(str(methodology_llm_input))

        return prompt
    
    def prompt_for_result(self, search_result):
        results = search_result.get("default_results", {}).get("Results") 

        if not results:  
            results = search_result["user_based_search"].items()

        result_formatted_data = {"Results": results}

        # Convert to JSON string
        result_llm_input = json.dumps(result_formatted_data, indent=4)

        prompt = f'''
        I am providing a JSON containing results from multiple academic papers, along with their similarity scores. Your task is to generate a concise, well-structured, and academically written summarized results section that effectively presents the key findings, trends, and insights from the given papers.

        Instructions:
        - The summary should be formal and academic in tone.
        - Clearly highlight the main findings, patterns, and statistical outcomes from the provided texts.
        - Retain key quantitative and qualitative insights while avoiding redundancy.
        - Ensure the summary is coherent, logically structured, and concise.
        - If applicable, mention comparisons, significant improvements, or deviations observed in the results.

        Input: {result_llm_input}

        Note:  Do not include text like I understand or here is your summary and Do not mension heading at start.


        Output Format:
        Provide a well-structured and academically written results summary that effectively synthesizes the key outcomes from the given papers.
        '''
        
        with open('./extracted/results.txt','w',encoding='utf-8') as data:
            data.write(str(result_llm_input))

        return prompt
    
    def prompt_for_lit_review(self, references):

        prompt=f'''

        You are an AI model designed to generate a well-structured Literature Review section in IEEE format based on given references. Your task is to synthesize the key findings, methodologies, and contributions of the provided papers while maintaining an academic writing style.

        Instructions:
        - Summarize Key Findings - Extract and summarize relevant insights from each reference, ensuring that similar studies are grouped logically.
        - Cite Properly - Use IEEE citation format, e.g., "Handwriting digit recognition has been extensively studied using neural networks [1]."
        - Maintain Logical Flow - Organize the literature review into a coherent structure, categorizing related studies.
        - Use Formal Language - Ensure the text aligns with academic writing standards and maintains objectivity.
        - Avoid Direct Copying - Rewrite and paraphrase information in a scholarly manner.
        Example Input:
        [1] Abu Ghosh, M.M., & Maghari, A.Y. (2017). A Comparative Study on Handwriting Digit Recognition Using Neural Networks. *IEEE*.  
        [2] Alizadeh, S., & Fazel, A. (2017). Convolutional Neural Networks for Facial Expression Recognition. *Computer Vision and Pattern Recognition*. Cornell University Library.
        Expected Output:
        
        Handwriting digit recognition has been widely explored using neural networks. Abu Ghosh and Maghari [1] conducted a comparative analysis of different neural network architectures, demonstrating that convolutional neural networks (CNNs) outperform traditional multilayer perceptron models in terms of accuracy and robustness. Their study highlights the importance of feature extraction and layer depth in achieving high classification performance.

        Similarly, CNNs have also been applied to facial expression recognition. Alizadeh and Fazel [2] proposed a deep learning approach that utilizes convolutional layers to automatically extract features from facial images, achieving state-of-the-art accuracy. Their work underscores the effectiveness of deep networks in recognizing complex patterns in visual data.

        By leveraging CNNs, both studies demonstrate the adaptability of deep learning in computer vision applications, reinforcing the need for optimized architectures tailored to specific recognition tasks.

        Note:  Do not include text like I understand or here is your summary and Do not mension heading at start, Do not mention more then one reference together ie [3,4] is not allowed.

        INPUT:
        {references}
        '''
        
        with open('./extracted/lit_review.txt','w',encoding='utf-8') as data:
            data.write(str(prompt))

        return prompt
    
    def prompt_for_caption(self, caption): 
        prompt = f'''
            You are an expert in academic writing. Your task is to generate a clear, informative, and concise figure caption for a research paper. 

            **Context:** The following text describes a figure from the paper. Extract the key information and create a caption that highlights the most relevant aspects.

            **Figure Description:** 
            "{caption}"

            **Instructions:**
            - Summarize the key idea conveyed by the figure.
            - Ensure the caption is clear, precise, and relevant to the topic.
            - Use formal academic language.
            - Keep it concise (one or two sentences).

            **Output Format:** 
            A standalone caption that accurately represents the figure.

            **Example Output:** 
            "Figure X: Visualization of [main concept], demonstrating [key insight] as observed in [data or context]."
        '''
        with open('./extracted/caption_image_prompt.txt', 'w', encoding='utf-8') as data:
            data.write(str(prompt))

        return prompt
