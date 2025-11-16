// Function: sub_39DC770
// Address: 0x39dc770
//
void __fastcall sub_39DC770(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rbx
  __int64 v6; // r15
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rax
  bool v11; // cf
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // r15
  const __m128i *i; // rbx
  __m128i v17; // xmm2
  unsigned __int64 v18; // rdi
  const __m128i *v19; // rax
  unsigned __int64 v20; // rcx
  __int64 v21; // rax
  unsigned __int64 v22; // [rsp-50h] [rbp-50h]
  __int64 v23; // [rsp-48h] [rbp-48h]
  unsigned __int64 v24; // [rsp-40h] [rbp-40h]
  unsigned __int64 v25; // [rsp-40h] [rbp-40h]
  unsigned __int64 v26; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v2 = a2;
  v4 = a1[1];
  v5 = *a1;
  v6 = v4 - *a1;
  v7 = 0xCCCCCCCCCCCCCCCDLL * (v6 >> 4);
  if ( 0xCCCCCCCCCCCCCCCDLL * ((__int64)(a1[2] - v4) >> 4) >= a2 )
  {
    v8 = a1[1];
    v9 = a2;
    do
    {
      if ( v8 )
      {
        *(_OWORD *)(v8 + 16) = 0;
        *(_OWORD *)(v8 + 32) = 0;
        *(_QWORD *)(v8 + 24) = v8 + 40;
        *(_BYTE *)(v8 + 40) = 0;
        *(_OWORD *)v8 = 0;
        *(_OWORD *)(v8 + 48) = 0;
        *(_OWORD *)(v8 + 64) = 0;
      }
      v8 += 80LL;
      --v9;
    }
    while ( v9 );
    a1[1] = v4 + 80 * a2;
    return;
  }
  if ( 0x199999999999999LL - v7 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v10 = a2;
  if ( v7 >= a2 )
    v10 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v4 - *a1) >> 4);
  v11 = __CFADD__(v7, v10);
  v12 = v7 + v10;
  if ( v11 )
  {
    v20 = 0x7FFFFFFFFFFFFFD0LL;
  }
  else
  {
    if ( !v12 )
    {
      v22 = 0;
      v23 = 0;
      goto LABEL_14;
    }
    if ( v12 > 0x199999999999999LL )
      v12 = 0x199999999999999LL;
    v20 = 80 * v12;
  }
  v26 = v20;
  v21 = sub_22077B0(v20);
  v4 = a1[1];
  v5 = *a1;
  v2 = a2;
  v23 = v21;
  v22 = v21 + v26;
LABEL_14:
  v13 = v2;
  v14 = v6 + v23;
  do
  {
    if ( v14 )
    {
      *(_OWORD *)(v14 + 16) = 0;
      *(_OWORD *)(v14 + 32) = 0;
      *(_QWORD *)(v14 + 24) = v14 + 40;
      *(_BYTE *)(v14 + 40) = 0;
      *(_OWORD *)v14 = 0;
      *(_OWORD *)(v14 + 48) = 0;
      *(_OWORD *)(v14 + 64) = 0;
    }
    v14 += 80;
    --v13;
  }
  while ( v13 );
  if ( v5 != v4 )
  {
    v15 = v23;
    for ( i = (const __m128i *)(v5 + 40); ; i += 5 )
    {
      if ( v15 )
      {
        *(__m128i *)v15 = _mm_loadu_si128((const __m128i *)((char *)i - 40));
        *(_QWORD *)(v15 + 16) = i[-2].m128i_i64[1];
        *(_QWORD *)(v15 + 24) = v15 + 40;
        v19 = (const __m128i *)i[-1].m128i_i64[0];
        if ( i == v19 )
        {
          *(__m128i *)(v15 + 40) = _mm_loadu_si128(i);
        }
        else
        {
          *(_QWORD *)(v15 + 24) = v19;
          *(_QWORD *)(v15 + 40) = i->m128i_i64[0];
        }
        *(_QWORD *)(v15 + 32) = i[-1].m128i_i64[1];
        v17 = _mm_loadu_si128(i + 1);
        i[-1].m128i_i64[0] = (__int64)i;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        *(__m128i *)(v15 + 56) = v17;
        *(_DWORD *)(v15 + 72) = i[2].m128i_i32[0];
        *(_BYTE *)(v15 + 76) = i[2].m128i_i8[4];
      }
      v18 = i[-1].m128i_u64[0];
      if ( i != (const __m128i *)v18 )
      {
        v24 = v2;
        j_j___libc_free_0(v18);
        v2 = v24;
      }
      v15 += 80;
      if ( (unsigned __int64 *)v4 == &i[2].m128i_u64[1] )
        break;
    }
    v4 = *a1;
  }
  if ( v4 )
  {
    v25 = v2;
    j_j___libc_free_0(v4);
    v2 = v25;
  }
  *a1 = v23;
  a1[1] = v23 + 80 * (v7 + v2);
  a1[2] = v22;
}
