from lieink.basic_object import BasicLie as _BasicLie
from lieink.containers import Container, LieContainer

LieContainer.UPDATE_CONTAINER_TYPES(None, Container)
LieContainer.UPDATE_CONTAINER_TYPES(_BasicLie, LieContainer)
