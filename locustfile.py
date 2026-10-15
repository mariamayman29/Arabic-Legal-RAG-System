from locust import HttpUser, between, task


class LegalRagUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def ask_civil_code_question(self):
        payload = {
            "question": "Which entities are explicitly recognized as juristic persons under the Egyptian Civil Code?",
            "stream": False,
        }

        with self.client.post(
            "/ask", json=payload, name="POST /ask", catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")
