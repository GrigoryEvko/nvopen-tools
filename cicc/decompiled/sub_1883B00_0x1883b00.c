// Function: sub_1883B00
// Address: 0x1883b00
//
void __fastcall sub_1883B00(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rdx
  const __m128i *v4; // r13
  const __m128i *v5; // rbx
  __int64 v6; // r15
  unsigned __int64 v7; // r12
  _QWORD *v8; // rax
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rax
  bool v11; // cf
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rcx
  _QWORD *v14; // rax
  __int64 v15; // r15
  __int64 v16; // rdi
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // [rsp-50h] [rbp-50h]
  __m128i *v21; // [rsp-48h] [rbp-48h]
  unsigned __int64 v22; // [rsp-40h] [rbp-40h]
  unsigned __int64 v23; // [rsp-40h] [rbp-40h]
  __int64 v24; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v2 = a2;
  v4 = *(const __m128i **)(a1 + 8);
  v5 = *(const __m128i **)a1;
  v6 = (__int64)v4->m128i_i64 - *(_QWORD *)a1;
  v7 = 0xCCCCCCCCCCCCCCCDLL * (v6 >> 3);
  if ( 0xCCCCCCCCCCCCCCCDLL * ((__int64)(*(_QWORD *)(a1 + 16) - (_QWORD)v4) >> 3) >= a2 )
  {
    v8 = *(_QWORD **)(a1 + 8);
    v9 = a2;
    do
    {
      if ( v8 )
      {
        *v8 = 0;
        v8[1] = 0;
        v8[2] = 0;
        v8[3] = 0;
        v8[4] = 0;
      }
      v8 += 5;
      --v9;
    }
    while ( v9 );
    *(_QWORD *)(a1 + 8) = (char *)v4 + 40 * a2;
    return;
  }
  if ( 0x333333333333333LL - v7 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v10 = a2;
  if ( v7 >= a2 )
    v10 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)v4->m128i_i64 - *(_QWORD *)a1) >> 3);
  v11 = __CFADD__(v7, v10);
  v12 = v7 + v10;
  if ( v11 )
  {
    v18 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_33:
    v24 = v18;
    v19 = sub_22077B0(v18);
    v4 = *(const __m128i **)(a1 + 8);
    v5 = *(const __m128i **)a1;
    v2 = a2;
    v21 = (__m128i *)v19;
    v20 = v19 + v24;
    goto LABEL_14;
  }
  if ( v12 )
  {
    if ( v12 > 0x333333333333333LL )
      v12 = 0x333333333333333LL;
    v18 = 40 * v12;
    goto LABEL_33;
  }
  v20 = 0;
  v21 = 0;
LABEL_14:
  v13 = v2;
  v14 = (__int64 *)((char *)v21->m128i_i64 + v6);
  do
  {
    if ( v14 )
    {
      *v14 = 0;
      v14[1] = 0;
      v14[2] = 0;
      v14[3] = 0;
      v14[4] = 0;
    }
    v14 += 5;
    --v13;
  }
  while ( v13 );
  if ( v5 != v4 )
  {
    v15 = (__int64)v21;
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
        if ( v5 == v4 )
          goto LABEL_25;
      }
      v16 = v5[1].m128i_i64[0];
      v17 = v5[2].m128i_i64[0] - v16;
      if ( !v16 )
        goto LABEL_21;
      v5 = (const __m128i *)((char *)v5 + 40);
      v22 = v2;
      v15 = 40;
      j_j___libc_free_0(v16, v17);
      v2 = v22;
      if ( v5 == v4 )
      {
LABEL_25:
        v4 = *(const __m128i **)a1;
        break;
      }
    }
  }
  if ( v4 )
  {
    v23 = v2;
    j_j___libc_free_0(v4, *(_QWORD *)(a1 + 16) - (_QWORD)v4);
    v2 = v23;
  }
  *(_QWORD *)a1 = v21;
  *(_QWORD *)(a1 + 8) = (char *)v21 + 40 * v7 + 40 * v2;
  *(_QWORD *)(a1 + 16) = v20;
}
