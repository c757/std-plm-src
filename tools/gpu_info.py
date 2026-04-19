import torch, os, subprocess
print('Python executable:', os.sys.executable)
print('torch version:', torch.__version__)
print('torch cuda available:', torch.cuda.is_available())
print('CUDA device count:', torch.cuda.device_count())
print('PYTORCH_CUDA_ALLOC_CONF:', os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print('--- Device', i, '---')
        try:
            print('name:', torch.cuda.get_device_name(i))
        except Exception as e:
            print('get_device_name failed:', e)
        try:
            print('allocated (bytes):', torch.cuda.memory_allocated(i))
            print('reserved  (bytes):', torch.cuda.memory_reserved(i))
            print('max allocated (bytes):', torch.cuda.max_memory_allocated(i))
        except Exception as e:
            print('memory stats failed:', e)
    try:
        print('\nMemory summary (truncated):')
        print('\n'.join(torch.cuda.memory_summary().splitlines()[:40]))
    except Exception as e:
        print('memory_summary failed:', e)

try:
    print('\n--- nvidia-smi (first 2000 chars) ---')
    out = subprocess.check_output(['nvidia-smi', '-q'], universal_newlines=True, stderr=subprocess.STDOUT)
    print(out[:2000])
except Exception as e:
    print('nvidia-smi failed:', e)
