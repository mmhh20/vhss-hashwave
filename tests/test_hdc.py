import unittest

from hashwave.hdc import AssociativeClassifier, build_demo_classifier, normalize_fa, tokenize
from hashwave.hypervector import HyperVector, bundle


class HDCTests(unittest.TestCase):
    def test_hypervector_bind_inverse(self):
        a = HyperVector.from_label("a", 512)
        b = HyperVector.from_label("b", 512)
        self.assertEqual(a.bind(b).bind(b), a)
        self.assertEqual(a.similarity(a), 1.0)

    def test_permutation_roundtrip(self):
        a = HyperVector.from_label("a", 511)
        self.assertEqual(a.permute(0), a)
        self.assertEqual(a.permute(17).permute(-17), a)

    def test_bundle_order_independent(self):
        vectors = [HyperVector.from_label(str(i), 256) for i in range(5)]
        self.assertEqual(bundle(vectors), bundle(reversed(vectors)))

    def test_hypervector_validation(self):
        with self.assertRaises(ValueError):
            HyperVector(0, 0)
        with self.assertRaises(ValueError):
            HyperVector(1 << 10, 10)
        with self.assertRaises(ValueError):
            bundle([])

    def test_persian_normalization(self):
        self.assertEqual(normalize_fa("رأی‌گیری، با كاربَر"), "رایگیری با کاربر")
        self.assertEqual(tokenize("ان‌اف‌تی برای عکس"), ["انفتی", "برای", "دارایی"])

    def test_classifier_training_errors(self):
        model = AssociativeClassifier(dimension=512)
        with self.assertRaises(ValueError):
            model.fit()
        with self.assertRaises(ValueError):
            model.add("", "test")

    def test_demo_classifier_suite(self):
        model = build_demo_classifier()
        tests = [
            ("token", "ارزی بساز که تعدادش محدود باشد"),
            ("token", "سکه قابل سوزاندن با عرضه نهایی"),
            ("token", "امکان ضرب توکن جدید وجود نداشته باشد"),
            ("token", "یک رمزارز با سقف ثابت می خواهم"),
            ("nft", "برای عکس های من مجموعه دیجیتال یکتا بساز"),
            ("nft", "هر تصویر یک دارایی غیر قابل تعویض باشد"),
            ("nft", "قرارداد مالکیت آثار هنری"),
            ("nft", "ان اف تی برای تصویرها ایجاد کن"),
            ("voting", "کاربران بتوانند به پیشنهادها رأی بدهند"),
            ("voting", "یک سیستم نظرسنجی روی زنجیره"),
            ("voting", "تصمیم اعضا با شمارش آرا مشخص شود"),
            ("voting", "هر کاربر حق رای داشته باشد"),
        ]
        correct = 0
        for expected, text in tests:
            prediction, _ = model.predict(text)
            correct += prediction == expected
        self.assertGreaterEqual(correct, 10)


if __name__ == "__main__":
    unittest.main()
