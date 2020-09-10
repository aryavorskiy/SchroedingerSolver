def plain_mesh(bounds, **kwargs):
    begin, end = bounds
    step = 1
    if 'step' in kwargs:
        step = kwargs['step']
    elif 'count' in kwargs:
        step = (end - begin) / (kwargs['count'] + 1)
    else:
        raise NameError('Cannot invoke plain_mesh with such kwargs')
    mesh = []
    current = begin
    while current <= end:
        mesh.append(current)
        current += step
    return mesh
