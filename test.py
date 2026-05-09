#!/usr/bin/env python3
"""
Concurrent API Load Tester for Orin Inference Server
Measures response times across multiple concurrent requests
"""

import socket
import requests
import time
import statistics
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json

# ===== CONFIGURATION =====
# Change this to test different models
MODEL_NAME = "qwen3.5:0.8b"

# API Configuration
API_URL = "https://apollo.quocanmeomeo.io.vn/v1/chat/completions"

# Get API key from environment variable
API_KEY = os.environ.get('PASSWORD')
if not API_KEY:
    raise ValueError("PASSWORD environment variable not set. Please run: export PASSWORD='your-api-key'")

# Test configuration
NUM_CONCURRENT_REQUESTS = 20
REQUEST_TIMEOUT = 300  # seconds

# The moderately complicated question
COMPLEX_QUESTION = """
Create a simple code to pull data from a website, code in python3
"""

# ===== FORCE IPv4 (fixes Netbird tunnel issues) =====
original_getaddrinfo = socket.getaddrinfo

def ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    """Force IPv4 resolution to prevent tunnel timeout issues"""
    try:
        return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
    except socket.gaierror:
        return original_getaddrinfo(host, port, family, type, proto, flags)

socket.getaddrinfo = ipv4_only_getaddrinfo

# ===== REQUEST FUNCTION =====
def make_api_request(request_id):
    """Make a single API request and return timing data"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user", 
                "content": COMPLEX_QUESTION
            }
        ],
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    start_time = time.time()
    success = False
    error_message = None
    response_text = None
    
    try:
        response = requests.post(
            API_URL, 
            headers=headers, 
            json=payload, 
            timeout=REQUEST_TIMEOUT
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if response.status_code == 200:
            success = True
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                response_text = response_data['choices'][0].get('message', {}).get('content', '')
        else:
            error_message = f"HTTP {response.status_code}: {response.text[:200]}"
            
    except requests.exceptions.Timeout:
        end_time = time.time()
        elapsed_time = end_time - start_time
        error_message = "Request timeout"
    except requests.exceptions.ConnectionError as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        error_message = f"Connection error: {str(e)[:100]}"
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        error_message = f"Unexpected error: {str(e)[:100]}"
    
    return {
        'request_id': request_id,
        'elapsed_time': elapsed_time,
        'success': success,
        'error': error_message,
        'response_preview': response_text[:200] + "..." if response_text and len(response_text) > 200 else response_text
    }

# ===== STATISTICS CALCULATION =====
def calculate_statistics(times, success_times=None):
    """Calculate statistical metrics from response times"""
    if not times:
        return {}
    
    if success_times is None:
        success_times = times
    
    stats = {
        'total_requests': len(times),
        'successful_requests': len(success_times),
        'failed_requests': len(times) - len(success_times),
        'success_rate': (len(success_times) / len(times)) * 100 if times else 0,
        'min_time': min(times) if times else 0,
        'max_time': max(times) if times else 0,
        'avg_time': statistics.mean(times) if times else 0,
        'median_time': statistics.median(times) if times else 0,
    }
    
    # Add standard deviation if we have enough data points
    if len(times) >= 2:
        stats['std_deviation'] = statistics.stdev(times)
    
    # Add percentiles
    if len(times) >= 1:
        sorted_times = sorted(times)
        stats['p50_time'] = statistics.median(times)
        stats['p90_time'] = sorted_times[int(len(sorted_times) * 0.9)] if len(sorted_times) >= 10 else sorted_times[-1]
        stats['p95_time'] = sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) >= 20 else sorted_times[-1]
        stats['p99_time'] = sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) >= 100 else sorted_times[-1]
    
    return stats

# ===== MAIN EXECUTION =====
def main():
    print("=" * 80)
    print("ORIN API CONCURRENT LOAD TESTER")
    print("=" * 80)
    print(f"API Endpoint: {API_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Concurrent Requests: {NUM_CONCURRENT_REQUESTS}")
    print(f"Timeout: {REQUEST_TIMEOUT} seconds")
    print(f"Question Complexity: High (Quantum Entanglement & Computing)")
    print(f"API Key Source: PASSWORD environment variable {'✓' if API_KEY else '✗'}")
    print("=" * 80)
    print("\nStarting concurrent requests...\n")
    
    # Store results
    results = []
    all_times = []
    successful_times = []
    
    # Start timing the entire batch
    batch_start_time = time.time()
    
    # Execute concurrent requests
    with ThreadPoolExecutor(max_workers=NUM_CONCURRENT_REQUESTS) as executor:
        # Submit all tasks
        future_to_request = {
            executor.submit(make_api_request, i): i 
            for i in range(NUM_CONCURRENT_REQUESTS)
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_request):
            completed += 1
            result = future.result()
            results.append(result)
            all_times.append(result['elapsed_time'])
            
            if result['success']:
                successful_times.append(result['elapsed_time'])
                status = "✓"
            else:
                status = "✗"
            
            # Progress indicator
            print(f"[{completed}/{NUM_CONCURRENT_REQUESTS}] Request {result['request_id']:3d} | "
                  f"{status} | Time: {result['elapsed_time']:6.2f}s | "
                  f"{'Success' if result['success'] else f'Error: {result["error"][:50]}'}")
    
    batch_end_time = time.time()
    total_batch_time = batch_end_time - batch_start_time
    
    # Calculate statistics
    stats = calculate_statistics(all_times, successful_times)
    
    # ===== PRINT RESULTS =====
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 BATCH STATISTICS:")
    print(f"   Total batch time: {total_batch_time:.2f} seconds")
    print(f"   Throughput: {stats['total_requests'] / total_batch_time:.2f} requests/second")
    
    print(f"\n📈 RESPONSE TIME STATISTICS (seconds):")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Successful: {stats['successful_requests']} ({stats['success_rate']:.1f}%)")
    print(f"   Failed: {stats['failed_requests']}")
    print(f"   Minimum time: {stats['min_time']:.3f}s")
    print(f"   Maximum time: {stats['max_time']:.3f}s")
    print(f"   Average time: {stats['avg_time']:.3f}s")
    print(f"   Median time: {stats['median_time']:.3f}s")
    
    if 'std_deviation' in stats:
        print(f"   Std Deviation: {stats['std_deviation']:.3f}s")
    
    if 'p90_time' in stats:
        print(f"\n📊 PERCENTILES:")
        print(f"   P50 (Median): {stats['p50_time']:.3f}s")
        print(f"   P90: {stats['p90_time']:.3f}s")
        print(f"   P95: {stats['p95_time']:.3f}s")
        print(f"   P99: {stats['p99_time']:.3f}s")
    
    # Display sample responses
    print(f"\n💬 SAMPLE RESPONSE PREVIEWS:")
    successful_responses = [r for r in results if r['success']]
    for i, resp in enumerate(successful_responses[:3]):
        print(f"\n   Request {resp['request_id']} (took {resp['elapsed_time']:.2f}s):")
        if resp['response_preview']:
            preview = resp['response_preview'].replace('\n', ' ').strip()
            print(f"   \"{preview[:150]}...\"")
        else:
            print("   [No response content]")
    
    if stats['failed_requests'] > 0:
        print(f"\n⚠️  ERROR SUMMARY:")
        errors = [r['error'] for r in results if not r['success']]
        unique_errors = {}
        for err in errors:
            unique_errors[err] = unique_errors.get(err, 0) + 1
        for err, count in unique_errors.items():
            print(f"   - {err[:100]} ({count} occurrences)")

    print(f"   Success Rate: {stats['success_rate']:.1f}%")
    
    # Export results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"orin_load_test_{timestamp}.json"
    
    export_data = {
        "test_configuration": {
            "api_url": API_URL,
            "model": MODEL_NAME,
            "concurrent_requests": NUM_CONCURRENT_REQUESTS,
            "timeout_seconds": REQUEST_TIMEOUT,
            "timestamp": timestamp
        },
        "statistics": stats,
        "individual_requests": results,
        "total_batch_time": total_batch_time
    }
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("\n" + "=" * 80)
    
    # Return non-zero exit code if too many failures
    if stats['success_rate'] < 50:
        print("\n⚠️  WARNING: Success rate below 50% - test considered FAILED")
        return 1
    else:
        print("\n✅ Test completed successfully")
        return 0

if __name__ == "__main__":
    exit(main())