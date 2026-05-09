import re

try:
    from num2words import num2words
except Exception:
    num2words = None


class TextProcessor:
    """Chuẩn hóa và chia đoạn văn bản tiếng Việt cho TTS."""

    ABBREVIATIONS = {
        "TP.HCM": "thành phố Hồ Chí Minh",
        "TP.HN": "thành phố Hà Nội",
        "GS.TS": "giáo sư tiến sĩ",
        "PGS.TS": "phó giáo sư tiến sĩ",
        "TS.": "tiến sĩ",
        "ThS.": "thạc sĩ",
        "BS.": "bác sĩ",
        "KS.": "kỹ sư",
        "Tr.": "trang",
        "đ/c": "đồng chí",
        "k/g": "không giới hạn",
        "tu lv": "tu luyện",
        "đkt": "đại kết giới",
    }

    def normalize(self, text: str) -> str:
        """Chuẩn hóa viết tắt, số, dấu câu và khoảng trắng."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        text = text.strip()
        if not text:
            return ""

        for short, full in self.ABBREVIATIONS.items():
            text = text.replace(short, full)

        text = re.sub(r"\.\.\.+", "…", text)
        text = re.sub(r"--+", "—", text)
        text = self._convert_numbers(text)
        text = re.sub(r"\s+", " ", text).strip()

        if text and text[-1] not in ".!?…":
            text += "."
        return text

    def split_chunks(self, text: str, max_chars: int = 150) -> list[str]:
        """Chia văn bản thành chunks không vượt quá max_chars."""
        if max_chars < 20:
            raise ValueError("max_chars must be >= 20")

        text = re.sub(r"\s+", " ", text.strip())
        if not text:
            return []

        sentences = self._split_sentences(text)
        chunks = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > max_chars:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._split_long_sentence(sentence, max_chars))
                continue

            candidate = sentence if not current else current + " " + sentence
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence

        if current:
            chunks.append(current.strip())
        return chunks

    def _convert_numbers(self, text: str) -> str:
        """Chuyển số độc lập sang chữ, giữ nguyên năm 1000-2099."""
        def repl(match):
            num = match.group(0)
            value = int(num)

            if 1000 <= value <= 2099:
                return num

            if num2words is None:
                basic = {
                    0: "không", 1: "một", 2: "hai", 3: "ba", 4: "bốn",
                    5: "năm", 6: "sáu", 7: "bảy", 8: "tám", 9: "chín",
                    10: "mười",
                }
                return basic.get(value, num)

            try:
                return num2words(value, lang="vi")
            except Exception:
                return num

        return re.sub(r"(?<![\w/.-])\d+(?![\w/.-])", repl, text)

    def _split_sentences(self, text: str) -> list[str]:
        """Tách câu nhưng không tách khi đang trong ngoặc kép."""
        parts = []
        start = 0
        quote_open = False

        for i, ch in enumerate(text):
            if ch in '"“”':
                quote_open = not quote_open
            if ch in ".!?…\n" and not quote_open:
                end = i + 1
                parts.append(text[start:end].strip())
                start = end

        tail = text[start:].strip()
        if tail:
            parts.append(tail)
        return parts or [text]

    def _split_long_sentence(self, text: str, max_chars: int) -> list[str]:
        """Chia câu dài theo khoảng trắng, giữ nguyên thứ tự từ."""
        words = text.split()
        chunks = []
        current = ""

        for word in words:
            if len(word) > max_chars:
                if current:
                    chunks.append(current.strip())
                    current = ""
                for i in range(0, len(word), max_chars):
                    chunks.append(word[i:i + max_chars])
                continue

            candidate = word if not current else current + " " + word
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                current = word

        if current:
            chunks.append(current.strip())
        return chunks


if __name__ == "__main__":
    tp = TextProcessor()

    assert "thành phố Hồ Chí Minh" in tp.normalize("Đến TP.HCM hôm nay")

    result = tp.normalize("Có 3 người đến")
    assert "ba" in result.lower()

    result = tp.normalize("Năm 2025 là năm quan trọng")
    assert "2025" in result

    chunks = tp.split_chunks("A " * 200, max_chars=150)
    assert all(len(c) <= 150 for c in chunks)

    text = 'Tiêu Viêm nói: "Ta sẽ trở thành mạnh nhất!" rồi bước đi.'
    chunks = tp.split_chunks(text)
    joined = " ".join(chunks)
    assert "mạnh nhất" in joined

    print("✅ TextProcessor tests passed")
