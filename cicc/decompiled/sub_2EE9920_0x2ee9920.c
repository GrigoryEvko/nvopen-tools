// Function: sub_2EE9920
// Address: 0x2ee9920
//
_DWORD *__fastcall sub_2EE9920(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const void *v6; // r12
  _DWORD *result; // rax
  unsigned __int64 v8; // r14
  _DWORD *v9; // rdx
  _DWORD *i; // rsi
  unsigned __int64 v11; // rdx
  __int64 v12; // r8
  unsigned __int64 v13; // r15
  __int64 v14; // rcx
  _DWORD *v15; // rax
  _DWORD *j; // rdx
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // r14
  __int64 v19; // rcx
  _DWORD *k; // rdx
  __int64 v21; // [rsp+8h] [rbp-38h]

  v6 = (const void *)(a1 + 424);
  *(_QWORD *)a1 = &unk_4A2A380;
  result = (_DWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 16) = 0x400000000LL;
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 408) = a1 + 424;
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = a1 + 440;
  *(_QWORD *)(a1 + 432) = 0;
  v8 = *(unsigned int *)(a2 + 344);
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 392) = 0;
  *(_DWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 440) = a2;
  if ( !v8 )
    return result;
  v9 = (_DWORD *)(a1 + 24);
  if ( v8 > 4 )
  {
    sub_2EE9700(a1 + 8, v8, (__int64)result, a4, a5, a6);
    result = *(_DWORD **)(a1 + 8);
    v9 = &result[22 * *(unsigned int *)(a1 + 16)];
  }
  for ( i = &result[22 * v8]; i != v9; v9 += 22 )
  {
    if ( v9 )
    {
      memset(v9, 0, 0x58u);
      v9[6] = -1;
      v9[7] = -1;
      *((_QWORD *)v9 + 5) = v9 + 14;
      v9[13] = 4;
    }
  }
  result = *(_DWORD **)(a1 + 440);
  v11 = *(unsigned int *)(a1 + 416);
  *(_DWORD *)(a1 + 16) = v8;
  v12 = (unsigned int)result[22];
  v13 = v12 * (unsigned int)result[86];
  if ( v13 != v11 )
  {
    if ( v13 >= v11 )
    {
      if ( v13 > *(unsigned int *)(a1 + 420) )
      {
        v21 = (unsigned int)result[22];
        sub_C8D5F0(a1 + 408, v6, v13, 4u, v12, a6);
        v11 = *(unsigned int *)(a1 + 416);
        v12 = v21;
      }
      v14 = *(_QWORD *)(a1 + 408);
      v15 = (_DWORD *)(v14 + 4 * v11);
      for ( j = (_DWORD *)(v14 + 4 * v13); j != v15; ++v15 )
      {
        if ( v15 )
          *v15 = 0;
      }
      *(_DWORD *)(a1 + 416) = v13;
      result = *(_DWORD **)(a1 + 440);
      v17 = *(unsigned int *)(a1 + 432);
      goto LABEL_17;
    }
    *(_DWORD *)(a1 + 416) = v13;
  }
  v17 = *(unsigned int *)(a1 + 432);
LABEL_17:
  v18 = v12 * (unsigned int)result[86];
  if ( v18 != v17 )
  {
    if ( v18 >= v17 )
    {
      if ( v18 > *(unsigned int *)(a1 + 436) )
      {
        sub_C8D5F0((__int64)v6, (const void *)(a1 + 440), v18, 4u, v12, a6);
        v17 = *(unsigned int *)(a1 + 432);
      }
      v19 = *(_QWORD *)(a1 + 424);
      result = (_DWORD *)(v19 + 4 * v17);
      for ( k = (_DWORD *)(v19 + 4 * v18); k != result; ++result )
      {
        if ( result )
          *result = 0;
      }
    }
    *(_DWORD *)(a1 + 432) = v18;
  }
  return result;
}
