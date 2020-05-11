from dragonfly import *
from dragonfly_daanzu_tools import noise_recognition

noise_grammar = Grammar('noise_recognition')
noise_grammar.add_rule(MappingRule(
    name = 'noise sink',
    mapping = {
        '<dictation>': ActionBase(),
        },
    extras = [ Dictation("dictation") ],
    context = FuncContext(lambda: noise_recognition.any_active()),
    ))

scroll_grammar = Grammar('noise_recognition_scroll')
scroll_grammar.add_rule(MappingRule(
    name = 'scroll',
    mapping = {
        'enable (scroll | scrolling)': Function(lambda: noise_recognition.setup('scroll', noise_recognition.make_hmm_recognizer(action=Mouse("wheeldown")))),
        'disable (scroll | scrolling)': Function(lambda: noise_recognition.destroy('scroll')),
        },
    ))

noise_grammar.load()
scroll_grammar.load()

def unload():
    noise_grammar.unload()
    scroll_grammar.unload()
