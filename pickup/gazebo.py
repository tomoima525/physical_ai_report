"""Gazebo service calls with bounded retries for idempotent pose resets."""
import subprocess
import time


def ign_service(world, service, request_type, request):
    attempts = 3 if service == 'set_pose' else 1
    command = ['ign', 'service', '-s', f'/world/{world}/{service}',
               '--reqtype', request_type, '--reptype', 'ignition.msgs.Boolean',
               '--timeout', '5000', '--req', request]
    for attempt in range(1, attempts + 1):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
            details = f'{result.stdout} {result.stderr}'
            if result.returncode == 0 and 'data: true' in result.stdout:
                return
            timed_out = 'timed out' in details.lower() or 'timeout' in details.lower()
        except subprocess.TimeoutExpired as exc:
            details = str(exc)
            timed_out = True
        if not timed_out or attempt == attempts:
            raise RuntimeError(f'Gazebo {service} failed (attempt {attempt}/{attempts}): {details}')
        print(f'Gazebo {service}: timeout; retry {attempt+1}/{attempts} in 1 s', flush=True)
        time.sleep(1)
