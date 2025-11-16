// Function: sub_FDDD10
// Address: 0xfddd10
//
__int64 __fastcall sub_FDDD10(const __m128i **a1, const __m128i *a2)
{
  const __m128i *v3; // rsi
  const __m128i *v4; // r14
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx
  bool v7; // cf
  unsigned __int64 v8; // rax
  char *v9; // rdx
  unsigned __int64 v10; // rbx
  __m128i *v11; // r15
  char *v12; // rax
  __m128i *v13; // rdx
  const __m128i *v14; // rax
  __int64 v16; // rbx
  __int64 v17; // rax
  size_t v18; // [rsp+10h] [rbp-40h]
  __int64 v19; // [rsp+18h] [rbp-38h]

  v3 = a1[1];
  v4 = *a1;
  v5 = 0xAAAAAAAAAAAAAAABLL * (((char *)v3 - (char *)*a1) >> 3);
  if ( v5 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v5 )
    v6 = 0xAAAAAAAAAAAAAAABLL * (((char *)a1[1] - (char *)*a1) >> 3);
  v7 = __CFADD__(v6, v5);
  v8 = v6 - 0x5555555555555555LL * (((char *)a1[1] - (char *)*a1) >> 3);
  v9 = (char *)((char *)a2 - (char *)v4);
  if ( v7 )
  {
    v16 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v8 )
    {
      v19 = 0;
      v10 = 24;
      v11 = 0;
      goto LABEL_7;
    }
    if ( v8 > 0x555555555555555LL )
      v8 = 0x555555555555555LL;
    v16 = 24 * v8;
  }
  v17 = sub_22077B0(v16);
  v9 = (char *)((char *)a2 - (char *)v4);
  v11 = (__m128i *)v17;
  v19 = v16 + v17;
  v10 = v17 + 24;
LABEL_7:
  v12 = &v9[(_QWORD)v11];
  if ( &v9[(_QWORD)v11] )
  {
    *(_QWORD *)v12 = 0;
    *((_WORD *)v12 + 4) = 0;
    *((_QWORD *)v12 + 2) = 0;
  }
  if ( a2 != v4 )
  {
    v13 = v11;
    v14 = v4;
    do
    {
      if ( v13 )
      {
        *v13 = _mm_loadu_si128(v14);
        v13[1].m128i_i64[0] = v14[1].m128i_i64[0];
      }
      v14 = (const __m128i *)((char *)v14 + 24);
      v13 = (__m128i *)((char *)v13 + 24);
    }
    while ( v14 != a2 );
    v10 = (unsigned __int64)&v11[3] + 8 * ((unsigned __int64)((char *)&a2[-2].m128i_u64[1] - (char *)v4) >> 3);
  }
  if ( a2 != v3 )
  {
    v18 = 8 * ((unsigned __int64)((char *)v3 - (char *)a2 - 24) >> 3) + 24;
    memcpy((void *)v10, a2, v18);
    v10 += v18;
  }
  if ( v4 )
    j_j___libc_free_0(v4, (char *)a1[2] - (char *)v4);
  *a1 = v11;
  a1[1] = (const __m128i *)v10;
  a1[2] = (const __m128i *)v19;
  return v19;
}
