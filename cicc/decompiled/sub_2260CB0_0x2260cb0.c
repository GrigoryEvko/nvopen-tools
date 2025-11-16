// Function: sub_2260CB0
// Address: 0x2260cb0
//
void __fastcall sub_2260CB0(const void **a1, const void **a2, __int64 a3)
{
  size_t v4; // r12
  const void *v5; // r13
  int v6; // eax
  int v7; // eax
  __int64 v8; // rax
  unsigned int v9; // r14d
  size_t v10; // r12
  const void *v11; // r13
  int v12; // eax
  int v13; // eax
  __int64 v14; // rax
  unsigned int v15; // eax
  size_t v16; // r12
  const void *v17; // r13
  const __m128i *j; // r15
  __int64 v19; // rax
  const void *v20; // r14
  int v21; // eax
  int v22; // eax
  __int64 v23; // rax
  int v24; // eax
  int v25; // eax
  __int64 *v26; // [rsp+10h] [rbp-50h]
  const __m128i *i; // [rsp+18h] [rbp-48h]
  size_t v28; // [rsp+20h] [rbp-40h]
  unsigned int v29; // [rsp+2Ch] [rbp-34h]

  if ( a1 != a2 && a1 + 2 != a2 )
  {
    for ( i = (const __m128i *)(a1 + 2); a2 != (const void **)i; ++i )
    {
      while ( 1 )
      {
        v4 = i->m128i_u64[1];
        v5 = (const void *)i->m128i_i64[0];
        v6 = sub_C92610();
        v7 = sub_C92860((__int64 *)a3, v5, v4, v6);
        if ( v7 == -1 || (v8 = *(_QWORD *)a3 + 8LL * v7, v8 == *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8)) )
          v9 = 0;
        else
          v9 = *(_DWORD *)(*(_QWORD *)v8 + 8LL);
        v10 = (size_t)a1[1];
        v11 = *a1;
        v12 = sub_C92610();
        v13 = sub_C92860((__int64 *)a3, v11, v10, v12);
        if ( v13 == -1 || (v14 = *(_QWORD *)a3 + 8LL * v13, v14 == *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8)) )
          v15 = 0;
        else
          v15 = *(_DWORD *)(*(_QWORD *)v14 + 8LL);
        v16 = i->m128i_u64[1];
        v17 = (const void *)i->m128i_i64[0];
        if ( v15 >= v9 )
          break;
        if ( a1 != (const void **)i )
          memmove(a1 + 2, a1, (char *)i - (char *)a1);
        *a1 = v17;
        a1[1] = (const void *)v16;
        if ( a2 == (const void **)++i )
          return;
      }
      for ( j = i; ; j[1] = _mm_loadu_si128(j) )
      {
        v26 = (__int64 *)j;
        v24 = sub_C92610();
        v25 = sub_C92860((__int64 *)a3, v17, v16, v24);
        if ( v25 == -1 || (v19 = *(_QWORD *)a3 + 8LL * v25, v19 == *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8)) )
          v29 = 0;
        else
          v29 = *(_DWORD *)(*(_QWORD *)v19 + 8LL);
        v20 = (const void *)j[-1].m128i_i64[0];
        v28 = j[-1].m128i_u64[1];
        v21 = sub_C92610();
        v22 = sub_C92860((__int64 *)a3, v20, v28, v21);
        if ( v22 == -1 )
          break;
        v23 = *(_QWORD *)a3 + 8LL * v22;
        if ( v23 == *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8) )
          break;
        --j;
        if ( *(_DWORD *)(*(_QWORD *)v23 + 8LL) >= v29 )
          goto LABEL_24;
LABEL_20:
        ;
      }
      --j;
      if ( v29 )
        goto LABEL_20;
LABEL_24:
      *v26 = (__int64)v17;
      v26[1] = v16;
    }
  }
}
