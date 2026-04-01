"""
inference.py — Baseline LLM agent for the Traffic Control Environment.

Runs a baseline agent powered by an OpenAI-compatible LLM against all
3 tasks (easy, medium, hard) and prints the grader scores.

Required environment variables:
    API_BASE_URL   — LLM endpoint (e.g. https://api.openai.com/v1)
    MODEL_NAME     — Model identifier (e.g. gpt-4o-mini)
    HF_TOKEN       — Hugging Face token
    OPENAI_API_KEY — API key (same as HF_TOKEN if using HF inference)

Functions:
    build_user_prompt(observation) -> str
        Converts a TrafficObservation into a clear text prompt for the LLM.

    parse_action(response_text) -> dict
        Parses LLM JSON response into an action dict.
        Falls back to KEEP_CURRENT on parse error.

    run_task(env, task_id, client, max_steps) -> float
        Runs one full episode for a given task and returns the grader score.

    main()
        Entry point — initializes OpenAI client, runs all 3 tasks, prints scores.

Usage:
    API_BASE_URL=... MODEL_NAME=... OPENAI_API_KEY=... python inference.py
"""
