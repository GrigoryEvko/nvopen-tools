// Function: sub_262E5B0
// Address: 0x262e5b0
//
void __fastcall sub_262E5B0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v4; // r13
  const __m128i *v5; // rbx
  __int64 v6; // r15
  unsigned __int64 v7; // r12
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rax
  bool v11; // cf
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rcx
  __int8 *v14; // rax
  __int64 v15; // r15
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rcx
  __int64 v18; // rax
  unsigned __int64 v19; // [rsp-50h] [rbp-50h]
  __m128i *v20; // [rsp-48h] [rbp-48h]
  unsigned __int64 v21; // [rsp-40h] [rbp-40h]
  unsigned __int64 v22; // [rsp-40h] [rbp-40h]
  unsigned __int64 v23; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v2 = a2;
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(const __m128i **)a1;
  v6 = v4 - *(_QWORD *)a1;
  v7 = 0xCCCCCCCCCCCCCCCDLL * (v6 >> 3);
  if ( a2 <= 0xCCCCCCCCCCCCCCCDLL * ((__int64)(*(_QWORD *)(a1 + 16) - v4) >> 3) )
  {
    v8 = *(_QWORD *)(a1 + 8);
    v9 = a2;
    do
    {
      if ( v8 )
      {
        *(_QWORD *)(v8 + 32) = 0;
        *(_QWORD *)(v8 + 16) = 0;
        *(_QWORD *)(v8 + 24) = 0;
        *(_OWORD *)v8 = 0;
      }
      v8 += 40;
      --v9;
    }
    while ( v9 );
    *(_QWORD *)(a1 + 8) = v4 + 40 * a2;
    return;
  }
  if ( 0x333333333333333LL - v7 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v10 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v4 - *(_QWORD *)a1) >> 3);
  if ( a2 >= v7 )
    v10 = a2;
  v11 = __CFADD__(v7, v10);
  v12 = v7 + v10;
  if ( v11 )
  {
    v17 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_33:
    v23 = v17;
    v18 = sub_22077B0(v17);
    v4 = *(_QWORD *)(a1 + 8);
    v5 = *(const __m128i **)a1;
    v2 = a2;
    v20 = (__m128i *)v18;
    v19 = v18 + v23;
    goto LABEL_14;
  }
  if ( v12 )
  {
    if ( v12 > 0x333333333333333LL )
      v12 = 0x333333333333333LL;
    v17 = 40 * v12;
    goto LABEL_33;
  }
  v19 = 0;
  v20 = 0;
LABEL_14:
  v13 = v2;
  v14 = &v20->m128i_i8[v6];
  do
  {
    if ( v14 )
    {
      *((_QWORD *)v14 + 4) = 0;
      *((_QWORD *)v14 + 2) = 0;
      *((_QWORD *)v14 + 3) = 0;
      *(_OWORD *)v14 = 0;
    }
    v14 += 40;
    --v13;
  }
  while ( v13 );
  if ( v5 != (const __m128i *)v4 )
  {
    v15 = (__int64)v20;
    while ( 1 )
    {
      while ( v15 )
      {
        *(__m128i *)v15 = _mm_loadu_si128(v5);
        *(_QWORD *)(v15 + 16) = v5[1].m128i_i64[0];
        *(_QWORD *)(v15 + 24) = v5[1].m128i_i64[1];
        *(_QWORD *)(v15 + 32) = v5[2].m128i_i64[0];
        v5[2].m128i_i64[0] = 0;
        v5[1].m128i_i64[0] = 0;
LABEL_21:
        v5 = (const __m128i *)((char *)v5 + 40);
        v15 += 40;
        if ( v5 == (const __m128i *)v4 )
          goto LABEL_25;
      }
      v16 = v5[1].m128i_u64[0];
      if ( !v16 )
        goto LABEL_21;
      v5 = (const __m128i *)((char *)v5 + 40);
      v21 = v2;
      v15 = 40;
      j_j___libc_free_0(v16);
      v2 = v21;
      if ( v5 == (const __m128i *)v4 )
      {
LABEL_25:
        v4 = *(_QWORD *)a1;
        break;
      }
    }
  }
  if ( v4 )
  {
    v22 = v2;
    j_j___libc_free_0(v4);
    v2 = v22;
  }
  *(_QWORD *)a1 = v20;
  *(_QWORD *)(a1 + 8) = (char *)v20 + 40 * v7 + 40 * v2;
  *(_QWORD *)(a1 + 16) = v19;
}
