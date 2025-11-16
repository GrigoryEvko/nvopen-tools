// Function: sub_1314B60
// Address: 0x1314b60
//
__int64 __fastcall sub_1314B60(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        volatile signed __int64 *a4,
        __int64 a5,
        char a6,
        unsigned int a7)
{
  pthread_mutex_t *v8; // r14
  int v10; // r9d
  unsigned int v11; // r15d
  int v13; // eax
  __int64 v14; // r8
  volatile signed __int64 *v15; // rcx
  __int64 v16; // [rsp+0h] [rbp-50h]

  v8 = (pthread_mutex_t *)(a3 + 64);
  if ( (_BYTE)a7 )
  {
    v13 = pthread_mutex_trylock((pthread_mutex_t *)(a3 + 64));
    v14 = a5;
    if ( v13 )
    {
      sub_130AD90(a3);
      *(_BYTE *)(a3 + 104) = 1;
      v14 = a5;
    }
    ++*(_QWORD *)(a3 + 56);
    if ( a1 != *(_QWORD *)(a3 + 48) )
    {
      ++*(_QWORD *)(a3 + 40);
      *(_QWORD *)(a3 + 48) = a1;
    }
    v15 = a4;
    v11 = 0;
    sub_130C610(a1, a2 + 10672, a3, v15, v14, 1);
    *(_BYTE *)(a3 + 104) = 0;
    pthread_mutex_unlock(v8);
  }
  else if ( pthread_mutex_trylock((pthread_mutex_t *)(a3 + 64)) )
  {
    *(_BYTE *)(a3 + 104) = 1;
    return 1;
  }
  else
  {
    ++*(_QWORD *)(a3 + 56);
    if ( a1 != *(_QWORD *)(a3 + 48) )
    {
      ++*(_QWORD *)(a3 + 40);
      *(_QWORD *)(a3 + 48) = a1;
    }
    if ( a6 )
      v10 = 0;
    else
      v10 = (byte_5260DD0[0] == 0) + 1;
    v11 = sub_130C6A0(a1, a2 + 10672, a3, a4, a5, v10);
    if ( (_BYTE)v11 )
    {
      v16 = *(_QWORD *)(a3 + 1768);
      *(_BYTE *)(a3 + 104) = 0;
      pthread_mutex_unlock(v8);
      v11 = byte_5260DD0[0];
      if ( byte_5260DD0[0] )
      {
        v11 = a7;
        if ( !a6 )
        {
          v11 = 0;
          sub_1314970(a1, a2, a3, v16);
        }
      }
    }
    else
    {
      *(_BYTE *)(a3 + 104) = 0;
      pthread_mutex_unlock(v8);
    }
  }
  return v11;
}
