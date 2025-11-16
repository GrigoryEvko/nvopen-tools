// Function: sub_13114E0
// Address: 0x13114e0
//
int __fastcall sub_13114E0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  pthread_mutex_t *v4; // r15
  _QWORD *v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // rax
  __int64 v10; // rdx
  __int64 v12; // [rsp+8h] [rbp-38h]

  v4 = (pthread_mutex_t *)(a4 + 10472);
  a2[5] = a4;
  v12 = a4 + 10408;
  if ( pthread_mutex_trylock((pthread_mutex_t *)(a4 + 10472)) )
  {
    sub_130AD90(v12);
    *(_BYTE *)(a4 + 10512) = 1;
  }
  ++*(_QWORD *)(a4 + 10464);
  if ( a1 != *(_QWORD *)(a4 + 10456) )
  {
    ++*(_QWORD *)(a4 + 10448);
    *(_QWORD *)(a4 + 10456) = a1;
  }
  *a2 = a2;
  v7 = a2;
  a2[1] = a2;
  v8 = *(_QWORD *)(a4 + 10392);
  if ( v8 )
  {
    *a2 = *(_QWORD *)(v8 + 8);
    *(_QWORD *)(*(_QWORD *)(a4 + 10392) + 8LL) = a2;
    a2[1] = *(_QWORD *)a2[1];
    **(_QWORD **)(*(_QWORD *)(a4 + 10392) + 8LL) = *(_QWORD *)(a4 + 10392);
    *(_QWORD *)a2[1] = a2;
    v7 = (_QWORD *)*a2;
  }
  *(_QWORD *)(a4 + 10392) = v7;
  v9 = a2 + 2;
  a2[2] = a2 + 2;
  a2[3] = a2 + 2;
  a2[4] = a3 + 8;
  v10 = *(_QWORD *)(a4 + 10400);
  if ( v10 )
  {
    a2[2] = *(_QWORD *)(v10 + 8);
    *(_QWORD *)(*(_QWORD *)(a4 + 10400) + 8LL) = v9;
    a2[3] = *(_QWORD *)a2[3];
    **(_QWORD **)(*(_QWORD *)(a4 + 10400) + 8LL) = *(_QWORD *)(a4 + 10400);
    *(_QWORD *)a2[3] = v9;
    v9 = (_QWORD *)a2[2];
  }
  *(_QWORD *)(a4 + 10400) = v9;
  *(_BYTE *)(a4 + 10512) = 0;
  return pthread_mutex_unlock(v4);
}
