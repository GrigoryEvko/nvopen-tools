// Function: sub_29B7DB0
// Address: 0x29b7db0
//
void __fastcall sub_29B7DB0(unsigned __int64 *a1, __int64 *a2, __int64 *a3, __int64 *a4)
{
  __m128i *v5; // rbx
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rax
  unsigned __int64 v9; // r13
  __int64 v10; // r14
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rdi
  bool v13; // cf
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // r8
  __int64 v17; // r15
  __int64 v18; // r14
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __m128i *v22; // rdx
  const __m128i *v23; // rax
  unsigned __int64 v24; // r8
  __int64 *v25; // [rsp+0h] [rbp-50h]
  __int64 *v26; // [rsp+8h] [rbp-48h]
  unsigned __int64 v27; // [rsp+10h] [rbp-40h]
  unsigned __int64 v28; // [rsp+18h] [rbp-38h]
  unsigned __int64 v29; // [rsp+18h] [rbp-38h]

  v5 = (__m128i *)a1[1];
  if ( v5 != (__m128i *)a1[2] )
  {
    if ( v5 )
    {
      v6 = *a2;
      v7 = *a3;
      v8 = *a4;
      v5[1].m128i_i8[8] = 0;
      v5->m128i_i64[0] = v6;
      v5->m128i_i64[1] = v7;
      v5[1].m128i_i64[0] = v8;
      v5[2].m128i_i64[0] = 0;
      v5 = (__m128i *)a1[1];
    }
    a1[1] = (unsigned __int64)&v5[2].m128i_u64[1];
    return;
  }
  v9 = *a1;
  v10 = (__int64)v5->m128i_i64 - *a1;
  v11 = 0xCCCCCCCCCCCCCCCDLL * (v10 >> 3);
  if ( v11 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v12 = 1;
  if ( v11 )
    v12 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)((__int64)v5->m128i_i64 - v9) >> 3);
  v13 = __CFADD__(v12, v11);
  v14 = v12 - 0x3333333333333333LL * ((__int64)((__int64)v5->m128i_i64 - v9) >> 3);
  if ( v13 )
  {
    v24 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_25:
    v25 = a4;
    v26 = a3;
    v29 = v24;
    v17 = sub_22077B0(v24);
    a3 = v26;
    a4 = v25;
    v16 = v17 + v29;
    v15 = v17 + 40;
    goto LABEL_11;
  }
  if ( v14 )
  {
    if ( v14 > 0x333333333333333LL )
      v14 = 0x333333333333333LL;
    v24 = 40 * v14;
    goto LABEL_25;
  }
  v15 = 40;
  v16 = 0;
  v17 = 0;
LABEL_11:
  v18 = v17 + v10;
  if ( v18 )
  {
    v19 = *a3;
    v20 = *a4;
    *(_BYTE *)(v18 + 24) = 0;
    v21 = *a2;
    *(_QWORD *)(v18 + 32) = 0;
    *(_QWORD *)(v18 + 8) = v19;
    *(_QWORD *)v18 = v21;
    *(_QWORD *)(v18 + 16) = v20;
  }
  if ( v5 != (__m128i *)v9 )
  {
    v22 = (__m128i *)v17;
    v23 = (const __m128i *)v9;
    do
    {
      if ( v22 )
      {
        *v22 = _mm_loadu_si128(v23);
        v22[1] = _mm_loadu_si128(v23 + 1);
        v22[2].m128i_i64[0] = v23[2].m128i_i64[0];
      }
      v23 = (const __m128i *)((char *)v23 + 40);
      v22 = (__m128i *)((char *)v22 + 40);
    }
    while ( v5 != v23 );
    v15 = v17 + 8 * (((unsigned __int64)&v5[-3].m128i_u64[1] - v9) >> 3) + 80;
  }
  if ( v9 )
  {
    v27 = v15;
    v28 = v16;
    j_j___libc_free_0(v9);
    v15 = v27;
    v16 = v28;
  }
  *a1 = v17;
  a1[1] = v15;
  a1[2] = v16;
}
