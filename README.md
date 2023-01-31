# Knowledge-Selection

Task-oriented Dialog system이 사용자의 질문에 대해 알맞은 답변 및 서비스를 제공하기 위해서는, 대화의 주제와 관련된 지식을 활용할 수 있어야 한다. 따라서 시스템은 knowledge base에서 가장 적절한 knowledge snippet을 선택할 수 있어야 하며, 이때 데이터베이스나 api에 의해 다뤄지는 내부 지식으로 해결할 수 없는 사용자의 요청에 대해 응답하기 위해서 external unstructured knowledge를 활용한다. 이에 본 논문은 knowledge-seeking-turn detection, knowledge selection, and knowledge-grounded generation, 이렇게 3단계로 나누어 knowledge-grounded Task-oriented conversational system을 구현한다. 특히 knowledge selection 단계를 세분화하여 domain-classification, entity-extraction, and snippet-ranking 태스크로 이루어진 계층 구조를 제안한다. 각 태스크는 pre-trained language model을 advanced techniques과 함께 사용함으로써 응답을 생성하는데 사용할 knowledge snippet을 최종적으로 결정한다. 또한 이전 task를 통해 얻은 domain, entity 정보를 knowledge로 활용하여 후보의 범위를 줄임으로써 knowledge 선택의 성능과 효율성을 향상시키고, 실험을 통해 이를 입증한다. 

## Tasks
<img width="800" alt="KakaoTalk_Image_2023-01-31-17-17-22" src="https://user-images.githubusercontent.com/57340671/215704606-9af05872-33e3-4a0b-b8b0-63fb280767a3.png">
사용자의 발화를 보다 체계적으로 이해하여 의사결정을 내리기 위해, 대화 history를 통해 domain을 결정하고, 이어서 각 도메인에 따른 entity를 결정하는 과정을 진행한다. 이후 entity별로 나열된 여러 지식 snippet들 중 응답 생성에 사용할 적절한 snippet을 결정하기 위해 순위를 매겨 candidate를 filtering하는 과정으로 나누어서 나타낸다.

### 1. Domain Classification: domainClassifiaction.py

<img width="300" alt="sensors-23-00685-g003" src="https://user-images.githubusercontent.com/57340671/215705816-509bf60f-c3d0-4ffa-97b5-640e331612fd.png">

대화 생성에 알맞은 knowledge candidate를 고르기 위해서는 먼저 대화 history를 기반으로 대화의 맥락을 제대로 이해해야 하므로, 지식을 필요로 하는 대화 turn에 대해 진행되고 있는 대화의 알맞은 domain을 파악하는 task
데이터셋에 구성된 도메인은 hotel, restaurant, taxi, bus, attraction 이렇게 5개로 존재하므로 5개 중 하나로 분류


### 2. Entity Extraction: entityExtraction.py
<img width="400" alt="sensors-23-00685-g005" src="https://user-images.githubusercontent.com/57340671/215705867-9a74d9c7-598f-4421-b0f5-c575e86e081f.png">|<img width="600" alt="sensors-23-00685-g004" src="https://user-images.githubusercontent.com/57340671/215705872-a4b9a30b-3e91-4986-9ff6-1c84a4a4c0aa.png">

대화 history에서 정확히 어떤 entity에 대해 이야기하고 있는지를 확인하기 위해 진행되는 단계로, 일반적으로 사람 사이에 이루어지는 대화에는 사용자가 원하는 정보를 직접 발화에 포함해 질문을 하기 마련이므로 해당 task에서 우리가 얻기 원하는 entity는 대화에 포함되어 있을 확률이 높다. 따라서 이 문제를 사용자와의 진행된 대화 history로부터 어디에 entity가 위치하고 있는지를 추출해내는 Named Entity Recognition(NER)으로 설계하여 token classification으로 구현

### 3. Snippet Ranking: snippetRanking.py

<img width="600" alt="sensors-23-00685-g006" src="https://user-images.githubusercontent.com/57340671/215705915-132cf009-9bb1-4f4b-af54-68687db96e52.png">
많은 snippet들 중에서 어떤 것을 knowledge로 결정해 응답 생성에 활용할지 결정하기 위해서는, 대화 history와 각 knowledge snippet과의 관련성을 계산하여 순위를 매김으로써 알맞은 snippet을 후보로 분류하는 과정이 최종적으로 필요하므로 negative sampling을 적용


* All : 모든 document를 candidate로 사용해서 훈련을 진행
* Positive Sample : conversational turn에 대해 domain, entity가 일치하는 경우에 해당하는 snippet들만을 candidate로 사용
* Random : positive sample과 해당 개수만큼 random으로 negative sample를 선택하여 positive sample과 함께 candidate를 구성
* InDomain : 구성된 conversational turn에 대해 Domain은 일치하지만 다른 entity에 대해 positive sample의 개수만큼 random하게 negative를 선택하여 positive sample과 함께 candidate로 사용한다. 이때 도메인 중 entity가 존재하지 않는 taxi와 train의 경우에는, random과 같은 방법으로 구성


## Data
[DSTC9 Track 1 - Beyond Domain APIs: Task-oriented Conversational Modeling with Unstructured Knowledge Access](https://github.com/alexa/alexa-with-dstc9-track1-dataset)

* an augmented version of [MultiWoz 2.1](https://github.com/budzianowski/multiwoz)

## Citation
실험 결과 및 자세한 세부사항은 아래 참고
```
@article{lee2023knowledge,
  title={A Knowledge-Grounded Task-Oriented Dialogue System with Hierarchical Structure for Enhancing Knowledge Selection},
  author={Lee, Hayoung and Jeong, Okran},
  journal={Sensors},
  volume={23},
  number={2},
  pages={685},
  year={2023},
  publisher={Multidisciplinary Digital Publishing Institute}
} 
```
