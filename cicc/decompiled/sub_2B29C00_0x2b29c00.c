// Function: sub_2B29C00
// Address: 0x2b29c00
//
_QWORD *__fastcall sub_2B29C00(_QWORD *a1, __int64 *a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v8; // rax
  const void *v9; // rsi
  unsigned __int64 v10; // r14
  _QWORD *i; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  int v14; // eax
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rsi

  v6 = 48LL * a3;
  v8 = *a2;
  v9 = a1 + 2;
  v10 = *(unsigned int *)(v8 + v6 + 8);
  *a1 = a1 + 2;
  a1[1] = 0x800000000LL;
  if ( v10 )
  {
    i = a1 + 2;
    if ( v10 > 8 )
    {
      sub_C8D5F0((__int64)a1, v9, v10, 8u, a5, a6);
      v12 = *a1 + 8 * v10;
      for ( i = (_QWORD *)(*a1 + 8LL * *((unsigned int *)a1 + 2)); (_QWORD *)v12 != i; ++i )
      {
LABEL_4:
        if ( i )
          *i = 0;
      }
    }
    else
    {
      v12 = (__int64)v9 + 8 * v10;
      if ( (const void *)v12 != v9 )
        goto LABEL_4;
    }
    *((_DWORD *)a1 + 2) = v10;
  }
  v13 = *a2;
  v14 = *(_DWORD *)(*a2 + 8);
  if ( v14 )
  {
    v15 = (unsigned int)(v14 - 1);
    v16 = 0;
    v17 = 8 * v15;
    while ( 1 )
    {
      *(_QWORD *)(*a1 + v16) = *(_QWORD *)(*(_QWORD *)(v13 + v6) + 2 * v16);
      if ( v16 == v17 )
        break;
      v13 = *a2;
      v16 += 8;
    }
  }
  return a1;
}
