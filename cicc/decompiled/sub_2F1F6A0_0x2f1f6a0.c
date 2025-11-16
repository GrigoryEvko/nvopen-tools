// Function: sub_2F1F6A0
// Address: 0x2f1f6a0
//
void __fastcall sub_2F1F6A0(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // r14
  unsigned __int64 v4; // rbx
  __int64 v5; // r15
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rcx
  __int64 v8; // rax
  bool v9; // cf
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // r15
  const __m128i *i; // rbx
  __m128i v15; // xmm1
  unsigned __int64 v16; // rdi
  const __m128i *v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // [rsp-50h] [rbp-50h]
  unsigned __int64 v21; // [rsp-50h] [rbp-50h]
  unsigned __int64 v22; // [rsp-48h] [rbp-48h]
  __int64 v23; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v3 = a1[1];
  v4 = *a1;
  v5 = v3 - *a1;
  v22 = v5 >> 6;
  if ( a2 <= (__int64)(a1[2] - v3) >> 6 )
  {
    v6 = a1[1];
    v7 = a2;
    do
    {
      if ( v6 )
      {
        *(_OWORD *)v6 = 0;
        *(_OWORD *)(v6 + 16) = 0;
        *(_QWORD *)(v6 + 8) = v6 + 24;
        *(_BYTE *)(v6 + 24) = 0;
        *(_OWORD *)(v6 + 32) = 0;
        *(_OWORD *)(v6 + 48) = 0;
      }
      v6 += 64LL;
      --v7;
    }
    while ( v7 );
    a1[1] = v3 + (a2 << 6);
    return;
  }
  if ( 0x1FFFFFFFFFFFFFFLL - v22 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v8 = (__int64)(a1[1] - *a1) >> 6;
  if ( a2 >= v22 )
    v8 = a2;
  v9 = __CFADD__(v22, v8);
  v10 = v22 + v8;
  if ( v9 )
  {
    v18 = 0x7FFFFFFFFFFFFFC0LL;
  }
  else
  {
    if ( !v10 )
    {
      v20 = 0;
      v23 = 0;
      goto LABEL_14;
    }
    if ( v10 > 0x1FFFFFFFFFFFFFFLL )
      v10 = 0x1FFFFFFFFFFFFFFLL;
    v18 = v10 << 6;
  }
  v21 = v18;
  v19 = sub_22077B0(v18);
  v3 = a1[1];
  v23 = v19;
  v4 = *a1;
  v20 = v19 + v21;
LABEL_14:
  v11 = a2;
  v12 = v5 + v23;
  do
  {
    if ( v12 )
    {
      *(_OWORD *)v12 = 0;
      *(_OWORD *)(v12 + 16) = 0;
      *(_QWORD *)(v12 + 8) = v12 + 24;
      *(_BYTE *)(v12 + 24) = 0;
      *(_OWORD *)(v12 + 32) = 0;
      *(_OWORD *)(v12 + 48) = 0;
    }
    v12 += 64;
    --v11;
  }
  while ( v11 );
  if ( v4 != v3 )
  {
    v13 = v23;
    for ( i = (const __m128i *)(v4 + 24); ; i += 4 )
    {
      if ( v13 )
      {
        *(_QWORD *)v13 = i[-2].m128i_i64[1];
        *(_QWORD *)(v13 + 8) = v13 + 24;
        v17 = (const __m128i *)i[-1].m128i_i64[0];
        if ( i == v17 )
        {
          *(__m128i *)(v13 + 24) = _mm_loadu_si128(i);
        }
        else
        {
          *(_QWORD *)(v13 + 8) = v17;
          *(_QWORD *)(v13 + 24) = i->m128i_i64[0];
        }
        *(_QWORD *)(v13 + 16) = i[-1].m128i_i64[1];
        v15 = _mm_loadu_si128(i + 1);
        i[-1].m128i_i64[0] = (__int64)i;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        *(__m128i *)(v13 + 40) = v15;
        *(_DWORD *)(v13 + 56) = i[2].m128i_i32[0];
      }
      v16 = i[-1].m128i_u64[0];
      if ( i != (const __m128i *)v16 )
        j_j___libc_free_0(v16);
      v13 += 64;
      if ( (unsigned __int64 *)v3 == &i[2].m128i_u64[1] )
        break;
    }
    v3 = *a1;
  }
  if ( v3 )
    j_j___libc_free_0(v3);
  *a1 = v23;
  a1[1] = v23 + ((a2 + v22) << 6);
  a1[2] = v20;
}
