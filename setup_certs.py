"""
Установка корневых сертификатов НУЦ Минцифры — разовый шаг перед работой с GigaChat.

Зачем. Хосты GigaChat подписаны Национальным удостоверяющим центром Минцифры, которого
нет в стандартных хранилищах доверия macOS/Python. Поэтому обычный HTTPS-запрос к Сберу
падает на проверке сертификата.

Урок решает это параметром verify=False, то есть просто выключает проверку подлинности
сервера. Работать будет, но так соединение перестаёт защищать от подмены собеседника:
проверка сертификата — единственное, что отличает настоящий сервер Сбера от чужого,
перехватившего трафик. Для учебного скрипта это терпимо, как привычка — нет.

Здесь честный путь: скачиваем публичные корневые сертификаты НУЦ с официального
портала госуслуг и добавляем их к стандартному хранилищу certifi. Получается бандл,
которым проверка проходит полноценно.

Запуск: python setup_certs.py
"""
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi

# Официальный источник корневых сертификатов НУЦ (портал госуслуг)
CA_URLS = (
    "https://gu-st.ru/content/Other/doc/russian_trusted_root_ca.cer",
    "https://gu-st.ru/content/Other/doc/russian_trusted_sub_ca.cer",
)

CERTS_DIR = Path(__file__).parent / "certs"
BUNDLE = CERTS_DIR / "russian_trusted_ca.pem"


def build_bundle() -> Path:
    """Скачать сертификаты НУЦ и склеить их со стандартным бандлом certifi."""
    CERTS_DIR.mkdir(exist_ok=True)
    # качаем с проверкой по обычным CA: сам gu-st.ru подписан общепризнанным центром
    ctx = ssl.create_default_context(cafile=certifi.where())

    parts = [Path(certifi.where()).read_text(encoding="utf-8")]
    for url in CA_URLS:
        print(f"Скачиваю {url.rsplit('/', 1)[-1]} ...")
        with urllib.request.urlopen(url, timeout=30, context=ctx) as response:
            data = response.read().decode("ascii")
        if "BEGIN CERTIFICATE" not in data:
            sys.exit(f"Ответ не похож на сертификат: {url}")
        parts.append(data)

    BUNDLE.write_text("\n".join(parts), encoding="utf-8")
    print(f"\nБандл готов: {BUNDLE}")
    print("Это certifi + корневой и промежуточный сертификаты НУЦ Минцифры.")
    return BUNDLE


def verify_hosts(bundle: Path) -> bool:
    """Проверить, что с этим бандлом TLS-соединение с хостами GigaChat проходит."""
    import socket

    ctx = ssl.create_default_context(cafile=str(bundle))
    hosts = (("ngw.devices.sberbank.ru", 9443), ("gigachat.devices.sberbank.ru", 443))
    ok = True
    print("\nПроверка TLS с этим бандлом:")
    for host, port in hosts:
        try:
            with socket.create_connection((host, port), timeout=15) as sock:
                with ctx.wrap_socket(sock, server_hostname=host) as tls:
                    issuer = dict(x[0] for x in tls.getpeercert()["issuer"])
                    print(f"  ✅ {host}:{port} — сертификат подтверждён")
                    print(f"     издатель: {issuer.get('commonName', '?')}")
        except Exception as e:
            ok = False
            print(f"  ❌ {host}:{port} — {type(e).__name__}: {e}")
    return ok


if __name__ == "__main__":
    bundle = build_bundle()
    if verify_hosts(bundle):
        print("\nГотово: GigaChat будет работать с полноценной проверкой сертификата.")
    else:
        print(
            "\nПроверка не прошла. Соединение с Сбером возможно только в обход проверки:\n"
            "  GIGACHAT_INSECURE=1 python app.py --provider gigachat\n"
            "Это путь урока (verify=False) — рабочий, но незащищённый от подмены сервера."
        )
