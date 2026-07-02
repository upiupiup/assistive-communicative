"""
Filkom UB - Domain Vocabulary Map
Diekstrak dari Buku Panduan Kurikulum Teknik Informatika 2024

Digunakan untuk:
1. TTER (Tech Term Error Rate) evaluation per domain
2. Panduan sampling dataset fine-tuning
3. Opsional: initial_prompt injection per konteks mata kuliah
"""

VOCAB_MAP = {
    "programming": {
        "label": "Pemrograman",
        "courses": [
            "Pemrograman Dasar",
            "Pemrograman Lanjut",
            "Pemrograman Aplikasi Perangkat Bergerak",
            "Pengembangan Aplikasi Web",
            "Pemrograman Gim",
            "Pemrograman Sistem Interaktif",
        ],
        "terms": [
            # Dasar
            "algoritma",
            "flowchart",
            "pseudocode",
            "debugging",
            "tipe data",
            "variabel",
            "konstanta",
            "operator",
            "rekursif",
            "array",
            "fungsi",
            "prosedur",
            "method",
            "class",
            "object",
            "constructor",
            "encapsulation",
            "enkapsulasi",
            "identifier",
            "looping",
            "iterasi",
            # OOP
            "inheritance",
            "pewarisan",
            "polymorphism",
            "polimorfisme",
            "abstract class",
            "interface",
            "exception handling",
            "generic",
            "override",
            "overload",
            "thread",
            "SOLID",
            "UML",
            "class diagram",
            # Mobile/Web
            "activity",
            "fragment",
            "intent",
            "widget",
            "layout",
            "adapter",
            "API",
            "HTML",
            "CSS",
            "javascript",
            "PHP",
            "AJAX",
            "MVC",
            "session",
            "cookies",
            "client side",
            "server side",
            # Game/Interactive
            "game engine",
            "collision detection",
            "shader",
            "rendering",
            "state chart",
            "lambda",
            "higher order function",
            "coroutine",
            "observer pattern",
            "asinkron",
            "reaktif",
        ],
    },
    "algorithms": {
        "label": "Algoritma dan Struktur Data",
        "courses": ["Algoritma dan Struktur Data", "Desain dan Analisis Algoritma"],
        "terms": [
            "abstract data type",
            "ADT",
            "linked list",
            "singly linked list",
            "doubly linked list",
            "circular",
            "stack",
            "queue",
            "priority queue",
            "binary tree",
            "heap",
            "BST",
            "AVL tree",
            "graph",
            "hashing",
            "hashtable",
            "dictionary",
            "sorting",
            "searching",
            "notasi asimtotik",
            "brute force",
            "greedy",
            "divide and conquer",
            "decrease and conquer",
            "dynamic programming",
            "backtracking",
            "order of growth",
            "kompleksitas",
            "efisiensi",
        ],
    },
    "systems": {
        "label": "Sistem Komputer",
        "courses": [
            "Sistem Operasi",
            "Arsitektur dan Organisasi Komputer",
            "Sistem Komputasi Terdistribusi",
        ],
        "terms": [
            # OS
            "proses",
            "thread",
            "penjadwalan",
            "preemptive",
            "non preemptive",
            "manajemen memori",
            "memori virtual",
            "sistem file",
            "virtualisasi",
            "proteksi",
            "system call",
            "multiprosesor",
            "deadlock",
            "sinkronisasi",
            "kernel",
            "scheduling",
            "semaphore",
            # Architecture
            "prosesor",
            "ISA",
            "instruction set",
            "ALU",
            "aritmatika biner",
            "register",
            "cache",
            "bus",
            "CPU",
            "peripheral",
            # Distributed
            "paralelisme",
            "distributed",
            "blockchain",
            "map reduce",
            "high availability",
            "konkurensi",
            "atomicity",
            "message passing",
        ],
    },
    "networks": {
        "label": "Jaringan Komputer",
        "courses": [
            "Jaringan Komputer",
            "Jaringan Komputer Lanjut",
            "Jaringan Nirkabel",
            "Jaringan Sensor Nirkabel",
            "Arsitektur Jaringan Modern",
            "Penyediaan dan Otomasi Layanan",
        ],
        "terms": [
            # Fundamentals
            "protokol",
            "TCP",
            "UDP",
            "HTTP",
            "DNS",
            "routing",
            "forwarding",
            "switching",
            "LAN",
            "bandwidth",
            "packet",
            "latency",
            # Advanced
            "IPv4",
            "IPv6",
            "VLAN",
            "RIP",
            "OSPF",
            "BGP",
            "distance vector",
            "link state",
            "overlay network",
            "QoS",
            # Wireless
            "wireless",
            "WiFi",
            "Bluetooth",
            "ZigBee",
            "LoRa",
            "WSN",
            "IoT",
            "modulasi",
            "multiplexing",
            "seluler",
            "MAC",
            # Modern/DevOps-Net
            "SDN",
            "NFV",
            "software defined networking",
            "network function virtualization",
            "segment routing",
            "infrastructure as code",
            "IaC",
            "Ansible",
            "Terraform",
            "CI CD",
            "orkestrasi",
            "cloud",
        ],
    },
    "ai_ml": {
        "label": "Kecerdasan Artifisial dan Machine Learning",
        "courses": [
            "Kecerdasan Artifisial",
            "Kecerdasan Artifisial Lanjut",
            "Jaringan Saraf Tiruan",
            "Deep Learning",
            "Pemrosesan Bahasa Alami",
            "Penerapan Kecerdasan Artifisial",
            "Machine Learning Operations",
        ],
        "terms": [
            # Classical AI
            "searching",
            "heuristic",
            "algoritma genetika",
            "knowledge based",
            "uncertainty",
            "KNN",
            "naive bayes",
            "decision tree",
            "SVM",
            "K-means",
            "clustering",
            "klasifikasi",
            # Neural Networks
            "neuron",
            "perceptron",
            "backpropagation",
            "self organizing maps",
            "learning vector quantization",
            "extreme learning machine",
            # Deep Learning
            "feedforward",
            "convolutional neural network",
            "CNN",
            "RNN",
            "LSTM",
            "GRU",
            "transformer",
            "autoencoder",
            "GAN",
            "generative adversarial network",
            "gradient descent",
            "Adam",
            "hyperparameter",
            "overfitting",
            "underfitting",
            "fine tuning",
            "transfer learning",
            # NLP
            "NLP",
            "tokenisasi",
            "POS tagging",
            "parsing",
            "NER",
            "named entity recognition",
            "word embedding",
            "language model",
            "n-gram",
            "sentiment analysis",
            "attention mechanism",
            "encoder decoder",
            "pre-trained model",
            # MLOps
            "MLOps",
            "pipeline",
            "versioning",
            "MLflow",
            "DVC",
            "Docker",
            "containerization",
            "monitoring",
            "Prometheus",
            "Grafana",
            "deployment",
            "drift",
            "model registry",
        ],
    },
    "data_science": {
        "label": "Data Science dan Matematika",
        "courses": [
            "Statistika dan Teori Peluang",
            "Statistika Inferensi",
            "Aljabar Linear",
            "Metode Numerik",
            "Kalkulus",
            "Analitik Big Data",
            "Rekayasa Data dan Visualisasi",
            "Pemerolehan Informasi dan Penambangan Teks",
            "Pengolahan Citra dan Visi Komputer Digital",
        ],
        "terms": [
            # Stats/Math
            "probabilitas",
            "distribusi",
            "teorema bayes",
            "sampling",
            "varian",
            "mean",
            "median",
            "hipotesis",
            "confidence interval",
            "maximum likelihood",
            "uji t",
            "uji z",
            "chi square",
            "ANOVA",
            "regresi",
            "matriks",
            "determinan",
            "nilai eigen",
            "vektor eigen",
            "transformasi linear",
            "integral",
            "turunan",
            "diferensial",
            # Big Data
            "Hadoop",
            "MapReduce",
            "Spark",
            "NoSQL",
            "ETL",
            "ELT",
            "batch processing",
            "stream processing",
            "data lake",
            # Visualization
            "Tableau",
            "Power BI",
            "D3",
            "dashboarding",
            "visualisasi",
            # IR/Text Mining
            "information retrieval",
            "text mining",
            "TF-IDF",
            "BM25",
            "vector space model",
            "topic modelling",
            "summarization",
            "query expansion",
            # Computer Vision
            "segmentasi",
            "ekstraksi fitur",
            "histogram",
            "morfologi",
            "konvolusi",
            "smoothing",
            "sharpening",
            "tekstur",
            "deteksi objek",
            "pengenalan objek",
        ],
    },
    "software_engineering": {
        "label": "Rekayasa Perangkat Lunak",
        "courses": [
            "Rekayasa Perangkat Lunak",
            "Analisis dan Perancangan Sistem",
            "Pola Perancangan",
            "Penjaminan Kualitas Perangkat Lunak",
            "Pengujian Perangkat Lunak",
            "Enterprise Application Integration",
            "Arsitektur Berbasis Layanan",
            "Perancangan Pengalaman Pengguna",
            "Interaksi Manusia dan Komputer",
        ],
        "terms": [
            # SDLC
            "SDLC",
            "analisis kebutuhan",
            "perancangan",
            "deployment",
            "pemeliharaan",
            "manajemen proyek",
            "waterfall",
            "agile",
            "scrum",
            # Modeling
            "use case",
            "activity diagram",
            "sequence diagram",
            "requirement engineering",
            "UML",
            "class diagram",
            # Design Patterns
            "design pattern",
            "creational pattern",
            "structural pattern",
            "behavioral pattern",
            "singleton",
            "factory",
            "observer",
            "decorator",
            "facade",
            # Quality/Testing
            "software quality assurance",
            "ISO 9126",
            "CMMI",
            "root cause analysis",
            "defect prevention",
            "maturity model",
            "unit testing",
            "white box",
            "black box",
            "regression testing",
            "stress test",
            "load test",
            "usability testing",
            "boundary value",
            # Architecture
            "monolitik",
            "SOA",
            "microservices",
            "service oriented",
            "API gateway",
            "event based",
            "dekomposisi",
            # UX
            "user experience",
            "usability",
            "prototyping",
            "wireframe",
            "human centered design",
        ],
    },
    "security": {
        "label": "Keamanan",
        "courses": [
            "Keamanan Informasi",
            "Keamanan Jaringan",
            "Keamanan Aplikasi Berbasis Web",
            "DevSecOps",
            "Forensik Digital",
        ],
        "terms": [
            # Crypto/Core
            "kriptografi",
            "enkripsi",
            "confidentiality",
            "integrity",
            "availability",
            "autentikasi",
            "otorisasi",
            "access control",
            "public key",
            "private key",
            "sertifikat digital",
            # Network Security
            "firewall",
            "IDS",
            "IPS",
            "VPN",
            "malware",
            "intrusion detection",
            "intrusion prevention",
            "symmetric encryption",
            # Web Security
            "XSS",
            "CSRF",
            "SQL injection",
            "XXE",
            "denial of service",
            "DoS",
            "cross site scripting",
            "penetration testing",
            "vulnerability",
            # DevSecOps
            "Jenkins",
            "Kubernetes",
            "container",
            "pipeline keamanan",
            # Forensics
            "forensik digital",
            "akuisisi data",
            "bukti digital",
            "anti forensic",
            "investigasi",
        ],
    },
}


def get_all_terms() -> list[str]:
    """Flatten semua terms dari semua cluster."""
    all_terms = []
    for cluster in VOCAB_MAP.values():
        all_terms.extend(cluster["terms"])
    return list(set(all_terms))


def get_terms_by_cluster(cluster_key: str) -> list[str]:
    """Ambil terms untuk satu cluster tertentu."""
    return VOCAB_MAP.get(cluster_key, {}).get("terms", [])


def compute_tter_per_cluster(hypothesis: str, reference: str) -> dict:
    """
    Hitung TTER (Tech Term Error Rate) per domain cluster.

    Returns dict: {cluster_key: {"tter": float, "missed": list, "correct": list}}
    """
    hyp = hypothesis.lower()
    ref = reference.lower()
    results = {}

    for key, cluster in VOCAB_MAP.items():
        terms_in_ref = [t for t in cluster["terms"] if t in ref]
        if not terms_in_ref:
            continue
        correct = [t for t in terms_in_ref if t in hyp]
        missed = [t for t in terms_in_ref if t not in hyp]
        tter = len(missed) / len(terms_in_ref) * 100

        results[key] = {
            "cluster_label": cluster["label"],
            "tter": round(tter, 2),
            "total_terms": len(terms_in_ref),
            "correct": correct,
            "missed": missed,
        }

    return results


def get_domain_prompt(cluster_key: str, max_terms: int = 20) -> str:
    """
    Generate initial_prompt untuk Whisper berdasarkan domain cluster.
    Gunakan sebelum transcribe() kalau konteks mata kuliah diketahui.
    """
    cluster = VOCAB_MAP.get(cluster_key)
    if not cluster:
        return ""
    terms = cluster["terms"][:max_terms]
    return (
        f"Presentasi mata kuliah {cluster['label']} di Fakultas Ilmu Komputer "
        f"Universitas Brawijaya. Istilah teknis: {', '.join(terms)}."
    )


SAMPLING_STRATEGY = {
    "recommended_total_sessions": 20,
    "distribution": {
        "ai_ml": {"sessions": 5, "reason": "vocab paling dense, prioritas utama riset"},
        "programming": {
            "sessions": 4,
            "reason": "paling sering, banyak code-switching",
        },
        "networks": {
            "sessions": 3,
            "reason": "jargon spesifik, beda dari cluster lain",
        },
        "systems": {"sessions": 3, "reason": "OS + distributed, vocab teknis hardware"},
        "software_engineering": {
            "sessions": 3,
            "reason": "RPL + testing, banyak istilah proses",
        },
        "data_science": {"sessions": 2, "reason": "math verbal + data terms"},
    },
    "notes": [
        "Security sengaja tidak dimasukkan karena vocab banyak overlap dengan networks",
        "Algorithms overlap banyak dengan programming, bisa di-cover dari sesi programming",
        "Pastikan tiap sesi minimal 30 menit audio untuk data yang cukup",
        "Rekam kondisi kelas nyata (noise AC, reverb, mic jauh) bukan kondisi ideal",
    ],
}


if __name__ == "__main__":
    print("=" * 60)
    print("FILKOM VOCAB MAP - SUMMARY")
    print("=" * 60)
    total = 0
    for key, cluster in VOCAB_MAP.items():
        n = len(cluster["terms"])
        total += n
        print(f"  {cluster['label']:45s} → {n:3d} terms")
    print(f"  {'TOTAL':45s} → {total:3d} terms")

    print()
    print("SAMPLING STRATEGY:")
    for cluster_key, info in SAMPLING_STRATEGY["distribution"].items():
        label = VOCAB_MAP[cluster_key]["label"]
        print(f"  {info['sessions']} sesi  {label}")
    print()
    print("Contoh domain prompt untuk AI/ML:")
    print(get_domain_prompt("ai_ml"))
