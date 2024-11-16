from locust import HttpUser, task, between

class FlaskTestUser(HttpUser):
    wait_time = between(1, 2)  # 각 요청 사이의 대기 시간 (초)

    @task
    def test_predict(self):
        # /predict 엔드포인트 테스트
        self.client.post(
            "/predict",
            data={"title": "Test Title", "content": "Test Content"}
        )
