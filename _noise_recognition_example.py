from dragonfly import *
import dragonfly_daanzu_tools

grammar = Grammar('noise_recognition')



grammar.add_rule(MappingRule(
    name = 'scroll',
    mapping = {
        'enable (scroll | scrolling)': Function(lambda: dragonfly_daanzu_tools.setup('scroll', make_hmm_recognizer(action=Mouse("wheeldown")))),
        'disable (scroll | scrolling)': Function(lambda: dragonfly_daanzu_tools.destroy('scroll')),
        },
    ))

grammar.load()
