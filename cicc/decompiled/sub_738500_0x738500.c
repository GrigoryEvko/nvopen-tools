// Function: sub_738500
// Address: 0x738500
//
__int64 *__fastcall sub_738500(__int64 *a1, __int64 a2, __int64 a3, const __m128i **a4, int a5)
{
  int v9; // esi
  __int64 v10; // r8
  unsigned int v11; // r12d
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  const __m128i **v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __m128i *v20; // rax
  const __m128i *v21; // r14
  const __m128i *v22; // r12
  const __m128i *v23; // rdx
  __m128i *v24; // rsi
  __int64 v26; // rdx
  __int64 v28; // [rsp+18h] [rbp-38h]

  v9 = *(_DWORD *)(a2 + 8);
  v10 = *(_QWORD *)a2;
  *a1 = 0;
  a1[1] = 0;
  v11 = v9 & a5;
  a1[2] = 0;
  v28 = v10;
  v12 = sub_823970(0);
  a1[1] = 0;
  v13 = v12;
  *a1 = v12;
  v14 = v11;
  if ( !*(_QWORD *)(*(_QWORD *)a2 + 32LL * v11) )
  {
LABEL_15:
    sub_729010((__int64 *)a2, v11, a3, (__int64)a4);
    return a1;
  }
  while ( 1 )
  {
    v15 = (const __m128i **)(v28 + 32 * v14);
    if ( *v15 == (const __m128i *)a3 )
      break;
    v11 = v9 & (v11 + 1);
    v14 = v11;
    if ( !*(_QWORD *)(*(_QWORD *)a2 + 32LL * v11) )
      goto LABEL_15;
  }
  if ( a1 != (__int64 *)(v15 + 1) )
  {
    sub_823A00(v13, 0);
    v16 = (__int64)v15[3];
    v17 = (__int64)v15[1];
    v15[3] = 0;
    v15[1] = 0;
    a1[2] = v16;
    v18 = (__int64)v15[2];
    *a1 = v17;
    v19 = 0;
    a1[1] = v18;
    v20 = 0;
    v15[2] = 0;
    v21 = a4[2];
    v22 = *a4;
    if ( !v21 )
      return a1;
    goto LABEL_7;
  }
  v22 = *a4;
  v21 = a4[2];
  v20 = (__m128i *)v15[1];
  if ( v21 != v15[3] )
  {
    v19 = (__int64)v15[2];
LABEL_7:
    if ( v19 < (__int64)v21 )
    {
      v15[3] = 0;
      sub_738450(v15 + 1, v21);
      v20 = (__m128i *)v15[1];
    }
    if ( (__int64)v21 > 0 )
    {
      v23 = v22;
      v24 = (__m128i *)((char *)v20 + 24 * (_QWORD)v21);
      do
      {
        if ( v20 )
        {
          *v20 = _mm_loadu_si128(v23);
          v20[1].m128i_i64[0] = v23[1].m128i_i64[0];
        }
        v20 = (__m128i *)((char *)v20 + 24);
        v23 = (const __m128i *)((char *)v23 + 24);
      }
      while ( v20 != v24 );
    }
    v15[3] = v21;
    return a1;
  }
  if ( (__int64)v21 > 0 )
  {
    v26 = 0;
    do
    {
      *(__m128i *)((char *)v20 + v26 * 8) = _mm_loadu_si128((const __m128i *)((char *)v22 + v26 * 8));
      v20[1].m128i_i64[v26] = v22[1].m128i_i64[v26];
      v26 += 3;
    }
    while ( v26 != 3LL * (_QWORD)v21 );
  }
  return a1;
}
