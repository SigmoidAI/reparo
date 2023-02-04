class NonNumericDataError(Exception):
    pass

class NoSuchDistanceMetricError(Exception):
    pass

class AllSamplesHaveNaNsError(Exception):
    pass

class SampleFullOfNaNs(Exception):
    pass

class OnlyOneColumnSelectedError(Exception):
    pass