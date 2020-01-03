import mypackage.geometry as geo


class Profile:
    def __init__(self, shapes):

        if not isinstance(shapes, list):
            shapes = [shapes]

        if not all(isinstance(shape, geo.Shape2D) for shape in shapes):
            raise ValueError(
                "Only instances or lists of Shape2d opbjects are accepted.")

        self.shapes = shapes
