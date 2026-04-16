# How to run
1. insert your own API keys in `rag/.env` file
2. run `/api/main.py` 
3. `cd <project_root>`
4. `streamlit run streamlit_app.py`
5. use web browser for querying

# project structure
``` text
.
├── README.md
├── agents
│   ├── concept_expander.py
│   ├── layout_planner.py
│   ├── quality_refiner.py
│   └── visual_designer.py
├── api
│   ├── __init__.py
│   ├── config.py
│   ├── dependencies.py
│   ├── main.py
│   ├── models
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── routes
│   │   ├── __init__.py
│   │   ├── health.py
│   │   └── poster.py
│   └── services
│       ├── __init__.py
│       ├── poster_service.py
│       └── task_manager.py
├── logs
│   └── agent_execution_time.log
├── models
│   └── best_model.weights.h5
├── outputs
│   ├── posters
│   │   ├── poster_20251119_234116.png
│   └── quality_reports
│       ├── iteration_0
│       │   ├── quality_report_20251122_172315.json
│       ├── iteration_1
│       │   ├── quality_report_20251122_172330.json
│       ├── iteration_2
│       │   └── quality_report_20251205_195133.json
│       └── iteration_3
│           └── quality_report_20251205_195148.json
├── quality_refiner_training
│   ├── inference.py
│   ├── pre_dataset.py
│   ├── quality refiner.py
│   ├── suggestion.py
│   ├── train_0.01.py
│   ├── train_0.1.py
│   └── train_baseline.py
├── rag
│   ├── docs
│   │   └── poster-rag-guide.md
│   └── rag.py
├── requirements.txt
├── streamlit_app.py
├── test_images
│   ├── sample.png
├── test_scripts
│   ├── test_cream_pasta.py
│   ├── test_interface_validation.py
│   └── test_layout_planner.py
├── tools
│   ├── layout_utils.py
│   ├── postermaker_wrapper.py
│   ├── rag_retriever.py
│   ├── style_planner.py
│   └── testing_decorator.py
└── workflow
    └── graph.py
```