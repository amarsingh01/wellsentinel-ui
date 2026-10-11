# WellSentinel UI

Live early-warning web app for oil well undesirable events. Front end for the
[wellsentinel](https://github.com/amarsingh01/wellsentinel) ML core.

Built for **AMD Developer Hackathon ACT III**, Track 1 (Intelligent Industry).

## What the operator sees

- Live sensor traces for the streaming well.
- A risk gauge: probability that an undesirable event starts soon.
- Lead time: estimated hours of warning.
- A plain-language explanation and a recommended action.
- A human review gate. Nothing is actioned automatically.

## Stack

Streamlit. Chosen because the judges name it, and it gives a working product fast
without spending build time on front-end plumbing.

## Status

Scaffold. Not yet wired to the model service.
