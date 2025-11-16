// Function: sub_2F1B880
// Address: 0x2f1b880
//
void __fastcall sub_2F1B880(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r13
  __int64 v3; // rbx
  _QWORD *v4; // rdx
  unsigned __int64 v5; // rax
  bool v6; // cf
  unsigned __int64 v7; // rax
  _QWORD *v8; // rbx
  const __m128i *v9; // r14
  __int64 v10; // rbx
  unsigned __int64 v11; // r13
  const __m128i *v12; // rdx
  __m128i v13; // xmm1
  __int64 v14; // rdx
  __m128i v15; // xmm2
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 *v18; // r12
  unsigned __int64 *v19; // r15
  unsigned __int64 v20; // r12
  unsigned __int64 v21; // [rsp+0h] [rbp-60h]
  unsigned __int64 v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+10h] [rbp-50h]
  unsigned __int64 v25; // [rsp+20h] [rbp-40h]
  const __m128i *v26; // [rsp+28h] [rbp-38h]

  v25 = a2;
  if ( !a2 )
    return;
  v2 = *a1;
  v26 = (const __m128i *)a1[1];
  v3 = (__int64)v26->m128i_i64 - *a1;
  v22 = 0x8E38E38E38E38E39LL * (v3 >> 4);
  if ( a2 <= 0x8E38E38E38E38E39LL * ((__int64)(a1[2] - (_QWORD)v26) >> 4) )
  {
    v4 = (_QWORD *)a1[1];
    do
    {
      if ( v4 )
      {
        memset(v4, 0, 0x90u);
        v4[3] = v4 + 5;
        v4[9] = v4 + 11;
      }
      v4 += 18;
      --a2;
    }
    while ( a2 );
    a1[1] = (unsigned __int64)&v26[9 * v25];
    return;
  }
  if ( 0xE38E38E38E38E3LL - v22 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v5 = 0x8E38E38E38E38E39LL * ((__int64)((__int64)v26->m128i_i64 - *a1) >> 4);
  if ( a2 >= v22 )
    v5 = a2;
  v6 = __CFADD__(v22, v5);
  v7 = v22 + v5;
  if ( v6 )
  {
    v20 = 0x7FFFFFFFFFFFFFB0LL;
  }
  else
  {
    if ( !v7 )
    {
      v21 = 0;
      v23 = 0;
      goto LABEL_15;
    }
    if ( v7 > 0xE38E38E38E38E3LL )
      v7 = 0xE38E38E38E38E3LL;
    v20 = 144 * v7;
  }
  v23 = sub_22077B0(v20);
  v2 = *a1;
  v21 = v23 + v20;
  v26 = (const __m128i *)a1[1];
LABEL_15:
  v8 = (_QWORD *)(v23 + v3);
  do
  {
    if ( v8 )
    {
      memset(v8, 0, 0x90u);
      v8[3] = v8 + 5;
      v8[9] = v8 + 11;
    }
    v8 += 18;
    --a2;
  }
  while ( a2 );
  if ( v26 != (const __m128i *)v2 )
  {
    v9 = (const __m128i *)(v2 + 40);
    v10 = v23;
    v11 = v2 + 88;
    while ( 1 )
    {
      if ( v10 )
      {
        *(__m128i *)v10 = _mm_loadu_si128((const __m128i *)((char *)v9 - 40));
        *(_QWORD *)(v10 + 16) = v9[-2].m128i_i64[1];
        *(_QWORD *)(v10 + 24) = v10 + 40;
        v12 = (const __m128i *)v9[-1].m128i_i64[0];
        if ( v12 == v9 )
        {
          *(__m128i *)(v10 + 40) = _mm_loadu_si128(v9);
        }
        else
        {
          *(_QWORD *)(v10 + 24) = v12;
          *(_QWORD *)(v10 + 40) = v9->m128i_i64[0];
        }
        *(_QWORD *)(v10 + 32) = v9[-1].m128i_i64[1];
        v13 = _mm_loadu_si128(v9 + 1);
        v9[-1].m128i_i64[0] = (__int64)v9;
        v9[-1].m128i_i64[1] = 0;
        v9->m128i_i8[0] = 0;
        *(_QWORD *)(v10 + 72) = v10 + 88;
        *(__m128i *)(v10 + 56) = v13;
        v14 = v9[2].m128i_i64[0];
        if ( v14 == v11 )
        {
          *(__m128i *)(v10 + 88) = _mm_loadu_si128(v9 + 3);
        }
        else
        {
          *(_QWORD *)(v10 + 72) = v14;
          *(_QWORD *)(v10 + 88) = v9[3].m128i_i64[0];
        }
        *(_QWORD *)(v10 + 80) = v9[2].m128i_i64[1];
        v15 = _mm_loadu_si128(v9 + 4);
        v9[2].m128i_i64[0] = v11;
        v9[2].m128i_i64[1] = 0;
        v9[3].m128i_i8[0] = 0;
        *(__m128i *)(v10 + 104) = v15;
        *(_QWORD *)(v10 + 120) = v9[5].m128i_i64[0];
        *(_QWORD *)(v10 + 128) = v9[5].m128i_i64[1];
        *(_QWORD *)(v10 + 136) = v9[6].m128i_i64[0];
        v9[6].m128i_i64[0] = 0;
        v9[5].m128i_i64[1] = 0;
        v9[5].m128i_i64[0] = 0;
      }
      else
      {
        v18 = (unsigned __int64 *)v9[5].m128i_i64[1];
        v19 = (unsigned __int64 *)v9[5].m128i_i64[0];
        if ( v18 != v19 )
        {
          do
          {
            if ( (unsigned __int64 *)*v19 != v19 + 2 )
              j_j___libc_free_0(*v19);
            v19 += 6;
          }
          while ( v18 != v19 );
          v19 = (unsigned __int64 *)v9[5].m128i_i64[0];
        }
        if ( v19 )
          j_j___libc_free_0((unsigned __int64)v19);
      }
      v16 = v9[2].m128i_u64[0];
      if ( v16 != v11 )
        j_j___libc_free_0(v16);
      v17 = v9[-1].m128i_u64[0];
      if ( (const __m128i *)v17 != v9 )
        j_j___libc_free_0(v17);
      v10 += 144;
      v11 += 144LL;
      if ( v26 == (const __m128i *)&v9[6].m128i_u64[1] )
        break;
      v9 += 9;
    }
    v2 = *a1;
  }
  if ( v2 )
    j_j___libc_free_0(v2);
  *a1 = v23;
  a1[1] = v23 + 144 * (v22 + v25);
  a1[2] = v21;
}
