// Function: sub_A2B8A0
// Address: 0xa2b8a0
//
__m128i *__fastcall sub_A2B8A0(__int64 a1, __m128i *a2)
{
  unsigned int v4; // r14d
  int v5; // esi
  __m128i *v6; // rcx
  int v7; // eax
  __m128i v8; // xmm0
  __int64 v10; // r15
  unsigned int v11; // r14d
  int v12; // eax
  const void *v13; // rdi
  size_t v14; // rdx
  int v15; // r9d
  unsigned int i; // r8d
  __m128i *v17; // r13
  const void *v18; // rsi
  int v19; // eax
  int v20; // eax
  unsigned int v21; // r8d
  size_t v22; // [rsp+8h] [rbp-68h]
  __m128i *v23; // [rsp+20h] [rbp-50h]
  int v24; // [rsp+28h] [rbp-48h]
  unsigned int v25; // [rsp+2Ch] [rbp-44h]
  __m128i *v26; // [rsp+38h] [rbp-38h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    v26 = 0;
LABEL_3:
    v5 = 2 * v4;
    goto LABEL_4;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = v4 - 1;
  v12 = sub_C94890(a2->m128i_i64[0], a2->m128i_i64[1]);
  v13 = (const void *)a2->m128i_i64[0];
  v14 = a2->m128i_u64[1];
  v6 = 0;
  v15 = 1;
  for ( i = v11 & v12; ; i = v11 & v21 )
  {
    v17 = (__m128i *)(v10 + 24LL * i);
    v18 = (const void *)v17->m128i_i64[0];
    if ( v17->m128i_i64[0] == -1 )
      break;
    if ( v18 == (const void *)-2LL )
    {
      if ( v13 == (const void *)-2LL )
        return v17 + 1;
    }
    else
    {
      if ( v14 != v17->m128i_i64[1] )
        goto LABEL_25;
      v23 = v6;
      v24 = v15;
      v25 = i;
      if ( !v14 )
        return v17 + 1;
      v22 = v14;
      v19 = memcmp(v13, v18, v14);
      v14 = v22;
      i = v25;
      v15 = v24;
      v6 = v23;
      if ( !v19 )
        return v17 + 1;
    }
    if ( v18 == (const void *)-2LL && !v6 )
      v6 = v17;
LABEL_25:
    v21 = v15 + i;
    ++v15;
  }
  if ( v13 == (const void *)-1LL )
    return v17 + 1;
  v20 = *(_DWORD *)(a1 + 16);
  v4 = *(_DWORD *)(a1 + 24);
  if ( !v6 )
    v6 = (__m128i *)(v10 + 24LL * i);
  ++*(_QWORD *)a1;
  v7 = v20 + 1;
  v26 = v6;
  if ( 4 * v7 >= 3 * v4 )
    goto LABEL_3;
  if ( v4 - (v7 + *(_DWORD *)(a1 + 20)) <= v4 >> 3 )
  {
    v5 = v4;
LABEL_4:
    sub_A2B260(a1, v5);
    sub_A19D80(a1, a2, &v26);
    v6 = v26;
    v7 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v7;
  if ( v6->m128i_i64[0] != -1 )
    --*(_DWORD *)(a1 + 20);
  v8 = _mm_loadu_si128(a2);
  v6[1].m128i_i64[0] = 0;
  *v6 = v8;
  return v6 + 1;
}
