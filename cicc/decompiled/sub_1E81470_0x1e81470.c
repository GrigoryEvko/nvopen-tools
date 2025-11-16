// Function: sub_1E81470
// Address: 0x1e81470
//
_DWORD *__fastcall sub_1E81470(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  const void *v6; // r13
  _DWORD *result; // rax
  unsigned __int64 v8; // r12
  _DWORD *v9; // rdx
  _DWORD *i; // rsi
  unsigned __int64 v11; // rdx
  __int64 v12; // r8
  unsigned __int64 v13; // r15
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // r12
  __int64 v16; // rcx
  _DWORD *k; // rdx
  __int64 v18; // rcx
  _DWORD *v19; // rax
  _DWORD *j; // rdx
  __int64 v21; // [rsp+8h] [rbp-38h]

  v6 = (const void *)(a1 + 424);
  *(_QWORD *)a1 = &unk_49FCE08;
  result = (_DWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 16) = 0x400000000LL;
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 408) = a1 + 424;
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = a1 + 440;
  *(_QWORD *)(a1 + 432) = 0;
  v8 = *(unsigned int *)(a2 + 560);
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 392) = 0;
  *(_DWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 440) = a2;
  if ( v8 )
  {
    v9 = (_DWORD *)(a1 + 24);
    if ( v8 > 4 )
    {
      sub_1E811D0((unsigned __int64 *)(a1 + 8), v8);
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
    v12 = (unsigned int)result[80];
    v13 = v12 * (unsigned int)result[140];
    if ( v13 >= v11 )
    {
      if ( v13 > v11 )
      {
        if ( v13 > *(unsigned int *)(a1 + 420) )
        {
          v21 = (unsigned int)result[80];
          sub_16CD150(a1 + 408, v6, v13, 4, v12, a6);
          v11 = *(unsigned int *)(a1 + 416);
          v12 = v21;
        }
        v18 = *(_QWORD *)(a1 + 408);
        v19 = (_DWORD *)(v18 + 4 * v11);
        for ( j = (_DWORD *)(v18 + 4 * v13); j != v19; ++v19 )
        {
          if ( v19 )
            *v19 = 0;
        }
        *(_DWORD *)(a1 + 416) = v13;
        result = *(_DWORD **)(a1 + 440);
      }
      v14 = *(unsigned int *)(a1 + 432);
    }
    else
    {
      *(_DWORD *)(a1 + 416) = v13;
      v14 = *(unsigned int *)(a1 + 432);
    }
    v15 = v12 * (unsigned int)result[140];
    if ( v15 < v14 )
      goto LABEL_11;
    if ( v15 > v14 )
    {
      if ( v15 > *(unsigned int *)(a1 + 436) )
      {
        sub_16CD150((__int64)v6, (const void *)(a1 + 440), v15, 4, v12, a6);
        v14 = *(unsigned int *)(a1 + 432);
      }
      v16 = *(_QWORD *)(a1 + 424);
      result = (_DWORD *)(v16 + 4 * v14);
      for ( k = (_DWORD *)(v16 + 4 * v15); k != result; ++result )
      {
        if ( result )
          *result = 0;
      }
LABEL_11:
      *(_DWORD *)(a1 + 432) = v15;
    }
  }
  return result;
}
