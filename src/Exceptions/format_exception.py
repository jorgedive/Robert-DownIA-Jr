class FormatException(Exception):
    def __init__(self, pattern):
        mensaje = f"Invalid Format. Object must follow {pattern} pattern."
        self.mensaje = mensaje
        super().__init__(self.mensaje)
