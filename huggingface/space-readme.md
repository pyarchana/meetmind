---
title: MeetMind
emoji: 🎙️
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 8080
pinned: false
license: mit
short_description: A real time AI meeting agent that tracks what was decided
---

# MeetMind

An AI participant in a work meeting. Everyone in the room knows it is there.

It listens, and while people talk it records what the group settled, who
committed to what, and which questions were raised and left hanging. Ask it
what was decided about pricing and it answers from the meeting, not from
general knowledge.

Speak to it, or type if you cannot. Share your screen and ask what is on it.

## Running it here

This Space needs a `GOOGLE_API_KEY` with Gemini Live access, set as a Space
secret. Without one the container refuses to start, which is deliberate: a
server that boots happily and then fails on the first connection is harder to
diagnose than one that says what is missing.

The key is on the free tier, so if the Space has been busy you may hit a rate
limit. That is the demo running out of quota, not the app breaking.

## Source

[github.com/pyarchana/meetmind](https://github.com/pyarchana/meetmind), MIT.

The repo also carries the measurement work behind it: a latency benchmark, a
104 question answer quality eval, and offline measurements of the audio
transport tradeoffs.
