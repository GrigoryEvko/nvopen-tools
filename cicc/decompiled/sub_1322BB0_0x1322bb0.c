// Function: sub_1322BB0
// Address: 0x1322bb0
//
void *__fastcall sub_1322BB0(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  void *v5; // r13

  if ( pthread_mutex_trylock(&stru_4F96C00) )
  {
    sub_130AD90((__int64)&xmmword_4F96BC0);
    byte_4F96C28 = 1;
  }
  ++*((_QWORD *)&xmmword_4F96BF0 + 1);
  if ( a1 != (_QWORD)xmmword_4F96BF0 )
  {
    ++qword_4F96BE8;
    *(_QWORD *)&xmmword_4F96BF0 = a1;
  }
  v5 = &unk_4983620;
  if ( a4 - 4096 > 1 && *(unsigned int *)(qword_4F96BA0 + 8) < a4 )
    v5 = 0;
  byte_4F96C28 = 0;
  pthread_mutex_unlock(&stru_4F96C00);
  return v5;
}
