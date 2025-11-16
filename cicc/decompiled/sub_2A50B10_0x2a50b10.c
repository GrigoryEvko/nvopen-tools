// Function: sub_2A50B10
// Address: 0x2a50b10
//
__int64 __fastcall sub_2A50B10(__int64 a1, __int64 a2, __m128i *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i *v6; // r14
  __int64 v9; // rax
  char v10; // dl
  __m128i *v12; // rcx
  unsigned __int64 v13; // rsi
  __m128i *v14; // r13
  __int64 v15; // rdx
  __m128i *v16; // rax
  __int8 v17; // di
  _QWORD *v18; // rdi
  const __m128i *v19; // r15
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __m128i *v26; // rax
  const void *v27; // rsi
  char *v28; // r14

  v6 = a3;
  if ( *(_QWORD *)(a2 + 136) )
  {
    v9 = sub_2A4DEC0(a2 + 96, a3);
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v9;
    *(_BYTE *)(a1 + 16) = v10;
    return a1;
  }
  v12 = *(__m128i **)a2;
  v13 = *(unsigned int *)(a2 + 8);
  v14 = (__m128i *)((char *)v12 + 40 * v13);
  if ( v12 == v14 )
  {
    v18 = (_QWORD *)(a2 + 96);
    if ( v13 > 1 )
    {
LABEL_20:
      *(_DWORD *)(a2 + 8) = 0;
      v22 = sub_2A4DEC0((__int64)v18, v6);
      *(_BYTE *)(a1 + 8) = 0;
      *(_QWORD *)a1 = v22;
      *(_BYTE *)(a1 + 16) = 1;
      return a1;
    }
  }
  else
  {
    v15 = a3->m128i_i64[0];
    v16 = v12;
    do
    {
      if ( v15 == v16->m128i_i64[0] )
      {
        v17 = v16[1].m128i_i8[8];
        if ( v17 == v6[1].m128i_i8[8]
          && (!v17 || v16->m128i_i64[1] == v6->m128i_i64[1] && v16[1].m128i_i64[0] == v6[1].m128i_i64[0])
          && v16[2].m128i_i64[0] == v6[2].m128i_i64[0] )
        {
          *(_BYTE *)(a1 + 8) = 1;
          *(_QWORD *)a1 = v16;
          *(_BYTE *)(a1 + 16) = 0;
          return a1;
        }
      }
      v16 = (__m128i *)((char *)v16 + 40);
    }
    while ( v14 != v16 );
    if ( v13 > 1 )
    {
      v18 = (_QWORD *)(a2 + 96);
      v19 = v12;
      do
      {
        v21 = sub_F1BF80(v18, a2 + 104, (__int64)v19);
        if ( v20 )
          sub_2A4D550((__int64)v18, v21, v20, v19);
        v19 = (const __m128i *)((char *)v19 + 40);
      }
      while ( v14 != v19 );
      goto LABEL_20;
    }
  }
  v23 = v13 + 1;
  if ( v13 + 1 > *(unsigned int *)(a2 + 12) )
  {
    v27 = (const void *)(a2 + 16);
    if ( v12 > v6 || v14 <= v6 )
    {
      sub_C8D5F0(a2, v27, v23, 0x28u, a5, a6);
      v14 = (__m128i *)(*(_QWORD *)a2 + 40LL * *(unsigned int *)(a2 + 8));
    }
    else
    {
      v28 = (char *)((char *)v6 - (char *)v12);
      sub_C8D5F0(a2, v27, v23, 0x28u, a5, a6);
      v6 = (__m128i *)&v28[*(_QWORD *)a2];
      v14 = (__m128i *)(*(_QWORD *)a2 + 40LL * *(unsigned int *)(a2 + 8));
    }
  }
  *v14 = _mm_loadu_si128(v6);
  v14[1] = _mm_loadu_si128(v6 + 1);
  v14[2].m128i_i64[0] = v6[2].m128i_i64[0];
  v24 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  *(_DWORD *)(a2 + 8) = v24;
  v25 = 5 * v24;
  v26 = *(__m128i **)a2;
  *(_BYTE *)(a1 + 8) = 1;
  *(_BYTE *)(a1 + 16) = 1;
  *(_QWORD *)a1 = &v26->m128i_i64[v25 - 5];
  return a1;
}
