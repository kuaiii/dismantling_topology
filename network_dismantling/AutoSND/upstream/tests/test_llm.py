from autosnd.llm import LLMClient, normalize_chat_endpoint


def test_endpoint_normalization():
    assert (
        normalize_chat_endpoint("https://example.test")
        == "https://example.test/v1/chat/completions"
    )
    assert (
        normalize_chat_endpoint("https://example.test/v1/")
        == "https://example.test/v1/chat/completions"
    )
    assert (
        normalize_chat_endpoint("https://example.test/custom/chat/completions")
        == "https://example.test/custom/chat/completions"
    )


def test_custom_transport_receives_key_and_returns_text():
    captured = {}

    def transport(endpoint, headers, payload, timeout):
        captured.update(
            endpoint=endpoint,
            headers=headers,
            payload=payload,
            timeout=timeout,
        )
        return {"choices": [{"message": {"content": "done"}}]}

    client = LLMClient(
        api_url="https://example.test/v1",
        api_key="test-secret",
        model="test-model",
        transport=transport,
    )
    assert client.complete(system="system", user="user") == "done"
    assert captured["headers"]["Authorization"] == "Bearer test-secret"
    assert captured["payload"]["model"] == "test-model"
    assert "seed" not in captured["payload"]


def test_optional_provider_seed_is_sent_only_when_configured():
    captured = {}

    def transport(endpoint, headers, payload, timeout):
        captured.update(payload)
        return {"choices": [{"message": {"content": "done"}}]}

    client = LLMClient(
        api_url="https://example.test/v1",
        api_key="test-secret",
        model="test-model",
        seed=7,
        transport=transport,
    )
    client.complete(system="system", user="user")
    assert captured["seed"] == 7


def test_client_redacts_secrets_from_responses_and_errors():
    api_url = "https://private-gateway.example/v1"
    api_key = "super-secret-key"

    def echo_transport(endpoint, headers, payload, timeout):
        content = f"{endpoint} {headers['Authorization']}"
        return {"choices": [{"message": {"content": content}}]}

    client = LLMClient(
        api_url=api_url,
        api_key=api_key,
        model="test-model",
        transport=echo_transport,
    )
    text = client.complete(system="system", user="user")
    assert api_url not in text
    assert api_key not in text

    def failing_transport(endpoint, headers, payload, timeout):
        raise RuntimeError(f"failed endpoint={endpoint}; headers={headers}")

    client = LLMClient(
        api_url=api_url,
        api_key=api_key,
        model="test-model",
        retries=0,
        transport=failing_transport,
    )
    try:
        client.complete(system="system", user="user")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected a sanitized request failure")
    assert api_url not in message
    assert api_key not in message
    assert "RuntimeError" in message
