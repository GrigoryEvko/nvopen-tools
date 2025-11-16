// Function: sub_39DB780
// Address: 0x39db780
//
void __fastcall sub_39DB780(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v4; // r14
  unsigned __int64 v5; // rsi
  __int64 v6; // rbx
  _QWORD *v7; // rdx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  bool v10; // cf
  unsigned __int64 v11; // rax
  _QWORD *v12; // rbx
  unsigned __int64 v13; // r11
  __int64 v14; // r12
  const __m128i *v15; // rbx
  __int64 v16; // rax
  __m128i v17; // xmm1
  __int64 v18; // rdx
  __m128i v19; // xmm2
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  const __m128i *v22; // rdx
  unsigned __int64 v23; // r12
  __int64 v24; // rax
  unsigned __int64 v25; // [rsp-58h] [rbp-58h]
  __int64 v26; // [rsp-50h] [rbp-50h]
  unsigned __int64 v27; // [rsp-48h] [rbp-48h]
  __int64 v28; // [rsp-40h] [rbp-40h]
  __int64 v29; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v4 = a1[1];
  v5 = *a1;
  v6 = v4 - *a1;
  v27 = 0xEEEEEEEEEEEEEEEFLL * (v6 >> 3);
  if ( 0xEEEEEEEEEEEEEEEFLL * ((__int64)(a1[2] - v4) >> 3) >= a2 )
  {
    v7 = (_QWORD *)a1[1];
    v8 = a2;
    do
    {
      if ( v7 )
      {
        memset(v7, 0, 0x78u);
        v7[3] = v7 + 5;
        v7[9] = v7 + 11;
      }
      v7 += 15;
      --v8;
    }
    while ( v8 );
    a1[1] = v4 + 120 * a2;
    return;
  }
  if ( 0x111111111111111LL - v27 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v9 = 0xEEEEEEEEEEEEEEEFLL * ((__int64)(v4 - *a1) >> 3);
  if ( v27 < a2 )
    v9 = a2;
  v10 = __CFADD__(v27, v9);
  v11 = v27 + v9;
  if ( v10 )
  {
    v23 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v11 )
    {
      v25 = 0;
      v26 = 0;
      goto LABEL_15;
    }
    if ( v11 > 0x111111111111111LL )
      v11 = 0x111111111111111LL;
    v23 = 120 * v11;
  }
  v24 = sub_22077B0(v23);
  v4 = a1[1];
  v5 = *a1;
  v26 = v24;
  v25 = v24 + v23;
LABEL_15:
  v12 = (_QWORD *)(v26 + v6);
  v13 = a2;
  do
  {
    if ( v12 )
    {
      memset(v12, 0, 0x78u);
      v12[3] = v12 + 5;
      v12[9] = v12 + 11;
    }
    v12 += 15;
    --v13;
  }
  while ( v13 );
  if ( v5 != v4 )
  {
    v14 = v26;
    v15 = (const __m128i *)(v5 + 40);
    v16 = v5 + 88;
    while ( 1 )
    {
      if ( v14 )
      {
        *(__m128i *)v14 = _mm_loadu_si128((const __m128i *)((char *)v15 - 40));
        *(_QWORD *)(v14 + 16) = v15[-2].m128i_i64[1];
        *(_QWORD *)(v14 + 24) = v14 + 40;
        v22 = (const __m128i *)v15[-1].m128i_i64[0];
        if ( v15 == v22 )
        {
          *(__m128i *)(v14 + 40) = _mm_loadu_si128(v15);
        }
        else
        {
          *(_QWORD *)(v14 + 24) = v22;
          *(_QWORD *)(v14 + 40) = v15->m128i_i64[0];
        }
        *(_QWORD *)(v14 + 32) = v15[-1].m128i_i64[1];
        v17 = _mm_loadu_si128(v15 + 1);
        v15[-1].m128i_i64[0] = (__int64)v15;
        v15[-1].m128i_i64[1] = 0;
        v15->m128i_i8[0] = 0;
        *(_QWORD *)(v14 + 72) = v14 + 88;
        *(__m128i *)(v14 + 56) = v17;
        v18 = v15[2].m128i_i64[0];
        if ( v16 == v18 )
        {
          *(__m128i *)(v14 + 88) = _mm_loadu_si128(v15 + 3);
        }
        else
        {
          *(_QWORD *)(v14 + 72) = v18;
          *(_QWORD *)(v14 + 88) = v15[3].m128i_i64[0];
        }
        *(_QWORD *)(v14 + 80) = v15[2].m128i_i64[1];
        v19 = _mm_loadu_si128(v15 + 4);
        v15[2].m128i_i64[0] = v16;
        v15[2].m128i_i64[1] = 0;
        v15[3].m128i_i8[0] = 0;
        *(__m128i *)(v14 + 104) = v19;
      }
      v20 = v15[2].m128i_u64[0];
      if ( v16 != v20 )
      {
        v28 = v16;
        j_j___libc_free_0(v20);
        v16 = v28;
      }
      v21 = v15[-1].m128i_u64[0];
      if ( v15 != (const __m128i *)v21 )
      {
        v29 = v16;
        j_j___libc_free_0(v21);
        v16 = v29;
      }
      v14 += 120;
      v16 += 120;
      if ( (const __m128i *)v4 == &v15[5] )
        break;
      v15 = (const __m128i *)((char *)v15 + 120);
    }
    v4 = *a1;
  }
  if ( v4 )
    j_j___libc_free_0(v4);
  *a1 = v26;
  a1[1] = v26 + 120 * (a2 + v27);
  a1[2] = v25;
}
