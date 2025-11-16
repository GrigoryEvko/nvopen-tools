// Function: sub_132D060
// Address: 0x132d060
//
void *__fastcall sub_132D060(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  unsigned __int64 v5; // rdx
  void *v6; // r12
  __int64 v7; // rdx

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
  if ( a4 != 4096 )
  {
    if ( a4 == 4097 )
    {
      v7 = 1;
      goto LABEL_12;
    }
    v5 = *(unsigned int *)(qword_4F96BA0 + 8);
    if ( a4 != v5 )
    {
      v6 = 0;
      if ( a4 >= v5 )
        goto LABEL_14;
      v7 = (unsigned int)(a4 + 2);
      if ( v7 == 0xFFFFFFFFLL )
        goto LABEL_14;
      goto LABEL_12;
    }
  }
  v7 = 0;
LABEL_12:
  v6 = &unk_4980FA0;
  if ( !*(_BYTE *)(*(_QWORD *)(qword_4F96BA0 + 8 * v7 + 24) + 4LL) )
    v6 = 0;
LABEL_14:
  byte_4F96C28 = 0;
  pthread_mutex_unlock(&stru_4F96C00);
  return v6;
}
