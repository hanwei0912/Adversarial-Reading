### [Data Adaptive Traceback for Vision-Language Foundation Models in Image Classification](https://arxiv.org/html/2407.08787v1)
The existing adaptation methods do not consider the missing knowledge, which may lead to crucial task-related knowledge for the downstream tasks being ignored. To address this issue, we propose a new adaptation framework called Data Adaptive Traceback (DAT). Specifically, we utilize a zero-shot-based method to extract the most downstream task-related subset of the pre-training data to enable the downstream tasks. Furthermore, we adopt a pseudo-label-based semi-supervised technique to reuse the pre-training images and a vision-language contrastive learning method to address the confirmation bias issue in semi-supervised learning. 

### [From Past to Future: Rethinking Eligibility Traces](https://arxiv.org/abs/2312.12972)
In this paper, we introduce a fresh perspective on the challenges of credit assignment and policy evaluation. First, we
delve into the nuances of eligibility traces and explore instances where their updates may result in unexpected credit assignment to preceding states. From this investigation emerges
the concept of a novel value function, which we refer to as
the bidirectional value function. Unlike traditional state value
functions, bidirectional value functions account for both future
expected returns (rewards anticipated from the current state
onward) and past expected returns (cumulative rewards from
the episode’s start to the present). We derive principled update
equations to learn this value function and, through experimentation, demonstrate its efficacy in enhancing the process
of policy evaluation.

### [Learning Planning Domains from Non-redundant Fully-Observed Traces: Theoretical Foundations and Complexity Analysis](https://ojs.aaai.org/index.php/AAAI/article/view/29980)
We investigate the most basic setting, that is grounded classical planning without negative preconditions or conditional
effects with full observability of the state variables. The given
traces are assumed to be justified in the sense that either no
single action or no set of actions can be removed without
violating correctness of the plan.

### [Making AI Policies Transparent to Humans through Demonstrations]()
Demonstrations are a powerful way of increasing the transparency of AI policies to humans. Though we can approximately model human learning from demonstrations as inverse
reinforcement learning, we note that human learning can differ from algorithmic learning in key ways, e.g. humans are
computationally limited and may sometimes struggle to understand all of the nuances of a demonstration. Unlike related work that provide demonstrations to humans that simply
maximize information gain

### [Fair Multivariate Adaptive Regression Splines for Ensuring Equity and Transparency](https://arxiv.org/abs/2402.15561)
 However, many predictive models are proprietary and inaccessible for evaluation or modification by researchers and practitioners, limiting their accountability and ethical design. Moreover, predictive models are often opaque and incomprehensible to the officials who use them, reducing their trust and utility. Furthermore, predictive models may introduce or exacerbate bias and inequity, as they have done in many sectors of society. Therefore, there is a need for transparent, interpretable, and fair predictive models that can be easily adopted and adapted by different stakeholders. In this paper, we propose a fair predictive model based on multivariate adaptive regression splines(MARS) that incorporates fairness measures in the learning process. MARS is a non-parametric regression model that performs feature selection, handles non-linear relationships, generates interpretable decision rules, and derives optimal splitting criteria on the variables. 

### [Thesis Summary: Operationalizing User-Inclusive Transparency in Artificial Intelligence Systems]()
. We propose the idea of representing an AI system as an amalgamation of the AI Model (algorithms), data (input and output,
including outcomes), and the user interface with visual interpretations (e.g. graphs, Venn diagrams). By designing human controls and feedback mechanisms for AI systems that
allow users to exert control over them we can integrate transparency into existing user interfaces. Our plan is to design
prototypes of transparent user interfaces for AI systems using
well-known usability principles

### [A Submodular Optimization Approach to Accountable Loan Approval](https://ojs.aaai.org/index.php/AAAI/article/view/30310)
In this paper, we outline an automated system for optimizing a rule-based system for approving loan applications, which has been deployed at Hyundai Capital Services (HCS). The main challenge lay in creating a high-quality rule base that is simultaneously simple enough to be interpretable by risk analysts as well as customers, since the approval decision should be accountable. 

### [ESG Accountability Made Easy: DocQA at Your Service](https://arxiv.org/pdf/2311.18481)

### [Beyond Expected Return: Accounting for Policy Reproducibility when Evaluating Reinforcement Learning Algorithms](https://arxiv.org/abs/2312.07178)
Many applications in Reinforcement Learning (RL) usually have noise or stochasticity present in the environment. Beyond their impact on learning, these uncertainties lead the exact same policy to perform differently, i.e. yield different return, from one roll-out to another. Common evaluation procedures in RL summarise the consequent return distributions using solely the expected return, which does not account for the spread of the distribution. Our work defines this spread as the policy reproducibility: the ability of a policy to obtain similar performance when rolled out many times, a crucial property in some real-world applications. We highlight that existing procedures that only use the expected return are limited on two fronts: first an infinite number of return distributions with a wide range of performance-reproducibility trade-offs can have the same expected return, limiting its effectiveness when used for comparing policies; second, the expected return metric does not leave any room for practitioners to choose the best trade-off value for considered applications. 

## [Accountability Layers: Explaining Complex System Failures by Parts](https://ojs.aaai.org/index.php/AAAI/article/view/26806)
This is often done by constructing an ``explainable model'' for a single modality or subsystem. However, this approach fails for complex systems that are made out of multiple parts. In this paper, I discuss how to explain complex system failures. I represent a complex machine as a hierarchical model of introspective sub-systems working together towards a common goal. The subsystems communicate in a common symbolic language. This work creates a set of explanatory accountability layers for trustworthy AI.

## [Monitoring Arithmetic Temporal Properties on Finite Traces](https://arxiv.org/abs/2211.17166)
We study monitoring of linear-time arithmetic properties against finite traces generated by an unknown dynamic system. The monitoring state is determined by considering at once the trace prefix seen so far, and all its possible finite-length, future continuations. This makes monitoring at least as hard as satisfiability and validity. Traces consist of finite sequences of assignments of a fixed set of variables to numerical values. Properties are specified in a logic we call ALTLf, combining LTLf (LTL on finite traces) with linear arithmetic constraints that may carry lookahead, i.e., variables may be compared over multiple instants of the trace. While the monitoring problem for this setting is undecidable in general, we show decidability for (a) properties without lookahead, and (b) properties with lookahead that satisfy the abstract, semantic condition of finite summary, studied before in the context of model checking. We then single out concrete, practically relevant classes of constraints guaranteeing finite summary. Feasibility is witnessed by a prototype implementation.

### [Multispectral Invisible Coating: Laminated Visible-Thermal Physical Attack against Multispectral Object Detectors Using Transparent Low-E Films](https://ojs.aaai.org/index.php/AAAI/article/view/25197)
Despite its crucial competence in safety-related applications, its security against physical attacks is severely understudied. We investigate the vulnerability of multispectral detectors against physical attacks by proposing a new physical method: Multispectral Invisible Coating. Utilizing transparent Low-e films, we realize a laminated visible-thermal physical attack by attaching Low-e films over a visible attack printing. Moreover, we apply our physical method to manufacture a Multispectral Invisible Suit that hides persons from the multiple view angles of Multispectral detectors. To simulate our attack under various surveillance scenes, we constructed a large-scale multispectral pedestrian dataset which we will release in public.

### [A Knowledge Distillation-Based Approach to Enhance Transparency of Classifier Models](https://arxiv.org/abs/2502.15959)
In this study, we propose a Knowledge Distillation (KD)-based approach that aims to enhance the transparency of the AI model in medical image analysis. The initial step is to use traditional CNN to obtain a teacher model and then use KD to simplify the CNN architecture, retain most of the features of the data set, and reduce the number of network layers. It also uses the feature map of the student model to perform hierarchical analysis to identify key features and decision-making processes. This leads to intuitive visual explanations.

### [The Pitfalls of "Security by Obscurity" And What They Mean for Transparent AI](https://arxiv.org/abs/2501.18669)
n computer security, transparency is likewise regarded as a key concept. The security community has for decades pushed back against so-called security by obscurity -- the idea that hiding how a system works protects it from attack -- against significant pressure from industry and other stakeholders. Over the decades, in a community process that is imperfect and ongoing, security researchers and practitioners have gradually built up some norms and practices around how to balance transparency interests with possible negative side effects. This paper asks: What insights can the AI community take from the security community's experience with transparency?
We identify three key themes in the security community's perspective on the benefits of transparency and their approach to balancing transparency against countervailing interests. For each, we investigate parallels and insights relevant to transparency in AI. We then provide a case study discussion on how transparency has shaped the research subfield of anonymization. Finally, shifting our focus from similarities to differences, we highlight key transparency issues where modern AI systems present challenges different from other kinds of security-critical systems, raising interesting open questions for the security and AI communities alike.

## [Making Transparency Advocates: An Educational Approach Towards Better Algorithmic Transparency in Practice](https://arxiv.org/abs/2412.15363)
Over several years, we created an open-source educational workshop on algorithmic transparency and advocacy. We delivered the workshop to professionals across two separate domains to improve their algorithmic transparency literacy and willingness to advocate for change. In the weeks following the workshop, participants applied what they learned, such as speaking up for algorithmic transparency at an organization-wide AI strategy meeting. We also make two broader observations: first, advocacy is not a monolith and can be broken down into different levels. Second, individuals' willingness for advocacy is affected by their professional field. For example, news and media professionals may be more likely to advocate for algorithmic transparency than those working at technology start-ups.

### [TRACE-CS: A Synergistic Approach to Explainable Course Scheduling Using LLMs and Logic](https://arxiv.org/abs/2409.03671)
We present TRACE-cs, a novel hybrid system that combines symbolic reasoning with large language models (LLMs) to address contrastive queries in scheduling problems. TRACE-cs leverages SAT solving techniques to encode scheduling constraints and generate explanations for user queries, while utilizing an LLM to process the user queries into logical clauses as well as refine the explanations generated by the symbolic solver to natural language sentences. By integrating these components, our approach demonstrates the potential of combining symbolic methods with LLMs to create explainable AI agents with correctness guarantees.
