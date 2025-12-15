- uv 환경을 사용하여 가상환경을 관리합니다.  
uv를 설치하는 방법은 다음과 같습니다.

1. [uv 공식 다운로드 페이지](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) 에서 다운로드할 수 있습니다.

2. 또는 pip를 사용하여 설치할 수 있습니다.

    ```
    pip install uv
    ```


- pyproject.toml을 참고하여, 가상환경으로 실행하는 방법은 다음과 같습니다.

    ```
    uv run main.py
    ```

- 가상 환경에서 test.py 파일을 실행하여 모델이 정상적으로 작동하는지 확인할 수 있습니다.

    ```
    python test.py "이미지 경로"
    ```

- 또는 가상 환경이 실행되어 있지 않다면 다음과 같이 실행할 수 있습니다.

    ```
    uv run test.py "이미지 경로"
    ```