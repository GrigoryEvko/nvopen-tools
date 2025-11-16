// Function: sub_1369B80
// Address: 0x1369b80
//
void __fastcall sub_1369B80(const __m128i **a1, unsigned __int64 a2)
{
  const __m128i *v4; // rdi
  const __m128i *v5; // rsi
  const __m128i *v6; // rdx
  __int64 v7; // rbx
  unsigned __int64 v8; // r14
  const __m128i *v9; // rax
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rax
  bool v12; // cf
  unsigned __int64 v13; // rax
  const __m128i *v14; // r8
  __m128i *v15; // r15
  __int8 *v16; // rax
  unsigned __int64 v17; // rcx
  __m128i *v18; // rax
  __int64 v19; // r8
  __int64 v20; // rax
  const __m128i *v21; // [rsp-40h] [rbp-40h]
  __int64 v22; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v4 = a1[1];
  v5 = *a1;
  v6 = *a1;
  v7 = (char *)v4 - (char *)*a1;
  v8 = 0xAAAAAAAAAAAAAAABLL * (v7 >> 3);
  if ( 0xAAAAAAAAAAAAAAABLL * (((char *)a1[2] - (char *)v4) >> 3) >= a2 )
  {
    v9 = v4;
    v10 = a2;
    do
    {
      if ( v9 )
      {
        v9->m128i_i64[0] = 0;
        v9->m128i_i16[4] = 0;
        v9[1].m128i_i64[0] = 0;
      }
      v9 = (const __m128i *)((char *)v9 + 24);
      --v10;
    }
    while ( v10 );
    a1[1] = (const __m128i *)((char *)v4 + 24 * a2);
    return;
  }
  if ( 0x555555555555555LL - v8 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v11 = a2;
  if ( v8 >= a2 )
    v11 = 0xAAAAAAAAAAAAAAABLL * (((char *)v4 - (char *)*a1) >> 3);
  v12 = __CFADD__(v8, v11);
  v13 = v8 + v11;
  if ( v12 )
  {
    v19 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v13 )
    {
      v14 = 0;
      v15 = 0;
      goto LABEL_15;
    }
    if ( v13 > 0x555555555555555LL )
      v13 = 0x555555555555555LL;
    v19 = 24 * v13;
  }
  v22 = v19;
  v20 = sub_22077B0(v19);
  v5 = *a1;
  v4 = a1[1];
  v15 = (__m128i *)v20;
  v6 = *a1;
  v14 = (const __m128i *)(v20 + v22);
LABEL_15:
  v16 = &v15->m128i_i8[v7];
  v17 = a2;
  do
  {
    if ( v16 )
    {
      *(_QWORD *)v16 = 0;
      *((_WORD *)v16 + 4) = 0;
      *((_QWORD *)v16 + 2) = 0;
    }
    v16 += 24;
    --v17;
  }
  while ( v17 );
  if ( v5 != v4 )
  {
    v18 = v15;
    do
    {
      if ( v18 )
      {
        *v18 = _mm_loadu_si128(v6);
        v18[1].m128i_i64[0] = v6[1].m128i_i64[0];
      }
      v6 = (const __m128i *)((char *)v6 + 24);
      v18 = (__m128i *)((char *)v18 + 24);
    }
    while ( v6 != v4 );
    v4 = v5;
  }
  if ( v4 )
  {
    v21 = v14;
    j_j___libc_free_0(v4, (char *)a1[2] - (char *)v4);
    v14 = v21;
  }
  *a1 = v15;
  a1[2] = v14;
  a1[1] = (__m128i *)((char *)v15 + 24 * v8 + 24 * a2);
}
