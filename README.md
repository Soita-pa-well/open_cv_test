## Описание

### Реализована ручка по адресу http://127.0.0.1:8000/uploadimage/. Через Swagger загружаем изображение и отправляем POST-запрос. В ответе получаем словарь, в значении возвращается превалирующий цвет на изображении.

## Запуск

```pip install requariements.txt```
```uvicorn app:app```