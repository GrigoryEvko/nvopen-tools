// Function: sub_931E30
// Address: 0x931e30
//
__int64 __fastcall sub_931E30(const __m128i **a1, const __m128i *a2, const __m128i *a3)
{
  const __m128i *v5; // rcx
  const __m128i *v6; // r15
  __int64 v7; // rax
  __int64 v8; // rdx
  bool v9; // cf
  unsigned __int64 v10; // rax
  char *v11; // rdx
  __int64 v12; // rbx
  __m128i *v13; // r14
  __m128i *v14; // rax
  __m128i v15; // xmm3
  __m128i v16; // xmm4
  __m128i v17; // xmm5
  __m128i *v18; // rdx
  const __m128i *v19; // rax
  __int64 v21; // rbx
  __int64 v22; // rax
  size_t v23; // [rsp+10h] [rbp-40h]
  const __m128i *v24; // [rsp+10h] [rbp-40h]
  __int64 v25; // [rsp+18h] [rbp-38h]

  v5 = a1[1];
  v6 = *a1;
  v7 = 0x6DB6DB6DB6DB6DB7LL * (((char *)v5 - (char *)*a1) >> 3);
  if ( v7 == 0x249249249249249LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0x6DB6DB6DB6DB6DB7LL * (((char *)v5 - (char *)v6) >> 3);
  v9 = __CFADD__(v8, v7);
  v10 = v8 + v7;
  v11 = (char *)((char *)a2 - (char *)v6);
  if ( v9 )
  {
    v21 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v25 = 0;
      v12 = 56;
      v13 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x249249249249249LL )
      v10 = 0x249249249249249LL;
    v21 = 56 * v10;
  }
  v24 = a1[1];
  v22 = sub_22077B0(v21);
  v5 = v24;
  v11 = (char *)((char *)a2 - (char *)v6);
  v13 = (__m128i *)v22;
  v25 = v21 + v22;
  v12 = v22 + 56;
LABEL_7:
  v14 = (__m128i *)&v11[(_QWORD)v13];
  if ( &v11[(_QWORD)v13] )
  {
    v15 = _mm_loadu_si128(a3);
    v16 = _mm_loadu_si128(a3 + 1);
    v17 = _mm_loadu_si128(a3 + 2);
    v14[3].m128i_i64[0] = a3[3].m128i_i64[0];
    *v14 = v15;
    v14[1] = v16;
    v14[2] = v17;
  }
  if ( a2 != v6 )
  {
    v18 = v13;
    v19 = v6;
    do
    {
      if ( v18 )
      {
        *v18 = _mm_loadu_si128(v19);
        v18[1] = _mm_loadu_si128(v19 + 1);
        v18[2] = _mm_loadu_si128(v19 + 2);
        v18[3].m128i_i64[0] = v19[3].m128i_i64[0];
      }
      v19 = (const __m128i *)((char *)v19 + 56);
      v18 = (__m128i *)((char *)v18 + 56);
    }
    while ( a2 != v19 );
    v12 = (__int64)&v13[7].m128i_i64[7
                                   * ((0xDB6DB6DB6DB6DB7LL
                                     * ((unsigned __int64)((char *)&a2[-4].m128i_u64[1] - (char *)v6) >> 3))
                                    & 0x1FFFFFFFFFFFFFFFLL)];
  }
  if ( a2 != v5 )
  {
    v23 = 56
        * (((0xDB6DB6DB6DB6DB7LL * ((unsigned __int64)((char *)v5 - (char *)a2 - 56) >> 3)) & 0x1FFFFFFFFFFFFFFFLL) + 1);
    memcpy((void *)v12, a2, v23);
    v12 += v23;
  }
  if ( v6 )
    j_j___libc_free_0(v6, (char *)a1[2] - (char *)v6);
  *a1 = v13;
  a1[1] = (const __m128i *)v12;
  a1[2] = (const __m128i *)v25;
  return v25;
}
