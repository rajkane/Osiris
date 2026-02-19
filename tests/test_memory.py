from utils.memory import MemoryManager


def test_get_memory_usage_mb():
    usage = MemoryManager.get_memory_usage_mb()
    assert isinstance(usage, float)
    assert usage > 0


def test_check_memory_limit_true():
    # Should be True for a very high limit
    assert MemoryManager.check_memory_limit(100000)


def test_check_memory_limit_false():
    # Should be False for a very low limit
    assert not MemoryManager.check_memory_limit(0.00001)
