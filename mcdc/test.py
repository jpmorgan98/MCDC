class A:
    x: str

print([x for x in dir(A) if x[:2] != '__'])
