import time
import logging
import functools

# Configure system log
logging.basicConfig(
    filename='../logs/agent_execution_time.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def measure_time(node_name: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            result = func(*args, **kwargs)  # Execute the actual function/method

            end_time = time.time()
            duration = end_time - start_time

            log_message = f"Node '{node_name}' finished in {duration:.4f}s"
            print(f"{log_message}")
            logging.info(log_message)

            # Logic to find the 'state' dict among arguments to preserve history
            # We look for the first argument that is a dictionary (the state)
            current_timings = {}
            for arg in args:
                if isinstance(arg, dict):
                    current_timings = arg.get("timings", {}).copy()
                    break
            # Update timings
            current_timings[node_name] = duration
            # Inject into result
            if isinstance(result, dict):
                result["timings"] = current_timings
            return result
        return wrapper
    return decorator