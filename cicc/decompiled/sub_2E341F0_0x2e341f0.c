// Function: sub_2E341F0
// Address: 0x2e341f0
//
unsigned __int64 __fastcall sub_2E341F0(unsigned __int64 *a1, const __m128i *a2, const __m128i *a3)
{
  const __m128i *v5; // rsi
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int8 *v11; // rdx
  unsigned __int64 v12; // rbx
  __m128i *v13; // r15
  __m128i *v14; // rdx
  __m128i v15; // xmm1
  __m128i *v16; // rdx
  const __m128i *v17; // rax
  unsigned __int64 v19; // rbx
  __int64 v20; // rax
  const __m128i *v21; // [rsp+0h] [rbp-50h]
  size_t v22; // [rsp+10h] [rbp-40h]
  unsigned __int64 v23; // [rsp+18h] [rbp-38h]

  v5 = (const __m128i *)a1[1];
  v6 = *a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v5->m128i_i64 - *a1) >> 3);
  if ( v7 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v5->m128i_i64 - v6) >> 3);
  v9 = __CFADD__(v8, v7);
  v10 = v8 - 0x5555555555555555LL * ((__int64)((__int64)v5->m128i_i64 - v6) >> 3);
  v11 = &a2->m128i_i8[-v6];
  if ( v9 )
  {
    v19 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v23 = 0;
      v12 = 24;
      v13 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x555555555555555LL )
      v10 = 0x555555555555555LL;
    v19 = 24 * v10;
  }
  v21 = a3;
  v20 = sub_22077B0(v19);
  v11 = &a2->m128i_i8[-v6];
  v13 = (__m128i *)v20;
  a3 = v21;
  v23 = v19 + v20;
  v12 = v20 + 24;
LABEL_7:
  v14 = (__m128i *)&v11[(_QWORD)v13];
  if ( v14 )
  {
    v15 = _mm_loadu_si128(a3);
    v14[1].m128i_i64[0] = a3[1].m128i_i64[0];
    *v14 = v15;
  }
  if ( a2 != (const __m128i *)v6 )
  {
    v16 = v13;
    v17 = (const __m128i *)v6;
    do
    {
      if ( v16 )
      {
        *v16 = _mm_loadu_si128(v17);
        v16[1].m128i_i64[0] = v17[1].m128i_i64[0];
      }
      v17 = (const __m128i *)((char *)v17 + 24);
      v16 = (__m128i *)((char *)v16 + 24);
    }
    while ( v17 != a2 );
    v12 = (unsigned __int64)&v13[3] + 8 * (((unsigned __int64)&a2[-2].m128i_u64[1] - v6) >> 3);
  }
  if ( a2 != v5 )
  {
    v22 = 8 * ((unsigned __int64)((char *)v5 - (char *)a2 - 24) >> 3) + 24;
    memcpy((void *)v12, a2, v22);
    v12 += v22;
  }
  if ( v6 )
    j_j___libc_free_0(v6);
  *a1 = (unsigned __int64)v13;
  a1[1] = v12;
  a1[2] = v23;
  return v23;
}
