// Function: sub_1338930
// Address: 0x1338930
//
__int64 __fastcall sub_1338930(__int64 a1, __int64 a2, __int64 a3, char **a4, _QWORD *a5, const char **a6, __int64 a7)
{
  unsigned __int64 v10; // rax
  __int64 v11; // r14
  unsigned int v12; // r12d

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
  v10 = *(_QWORD *)(a2 + 8);
  if ( v10 > 0xFFFFFFFF )
    goto LABEL_19;
  if ( v10 == 4096 || *(_DWORD *)(qword_4F96BA0 + 8) <= (unsigned int)v10 )
    goto LABEL_17;
  v11 = qword_50579C0[(unsigned int)v10];
  if ( !v11 )
  {
LABEL_19:
    v12 = 14;
    goto LABEL_18;
  }
  if ( a4 && a5 )
  {
    if ( *a5 != 8 )
      goto LABEL_17;
    sub_1316F80(qword_50579C0[(unsigned int)v10], *a4);
  }
  v12 = 0;
  if ( a6 )
  {
    if ( a7 == 8 && *a6 )
    {
      sub_1316FC0(v11, *a6);
      goto LABEL_18;
    }
LABEL_17:
    v12 = 22;
  }
LABEL_18:
  byte_4F96C28 = 0;
  pthread_mutex_unlock(&stru_4F96C00);
  return v12;
}
