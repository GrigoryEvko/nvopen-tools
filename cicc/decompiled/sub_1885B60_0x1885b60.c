// Function: sub_1885B60
// Address: 0x1885b60
//
void __fastcall sub_1885B60(__m128i **a1, unsigned __int64 a2)
{
  const __m128i *v3; // r9
  const __m128i *v4; // rbx
  __int64 v5; // r15
  __m128i *v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int64 v11; // r13
  unsigned __int64 v12; // rdx
  __int8 *v13; // rax
  __m128i *v14; // r15
  const __m128i *i; // rbx
  const __m128i *v16; // rax
  const __m128i *v17; // rdi
  __int64 v18; // r13
  __int64 v19; // rax
  __m128i *v20; // [rsp-50h] [rbp-50h]
  unsigned __int64 v21; // [rsp-48h] [rbp-48h]
  const __m128i *v22; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return;
  v3 = a1[1];
  v4 = *a1;
  v5 = (char *)v3 - (char *)*a1;
  v21 = v5 >> 5;
  if ( ((char *)a1[2] - (char *)v3) >> 5 >= a2 )
  {
    v6 = a1[1];
    v7 = a2;
    do
    {
      if ( v6 )
      {
        v6->m128i_i64[1] = 0;
        v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
        v6[1].m128i_i8[0] = 0;
      }
      v6 += 2;
      --v7;
    }
    while ( v7 );
    a1[1] = (__m128i *)&v3[2 * a2];
    return;
  }
  if ( 0x3FFFFFFFFFFFFFFLL - v21 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v8 = ((char *)a1[1] - (char *)*a1) >> 5;
  if ( v21 < a2 )
    v8 = a2;
  v9 = __CFADD__(v21, v8);
  v10 = v21 + v8;
  if ( v9 )
  {
    v18 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v10 )
    {
      v20 = 0;
      v11 = 0;
      goto LABEL_15;
    }
    if ( v10 > 0x3FFFFFFFFFFFFFFLL )
      v10 = 0x3FFFFFFFFFFFFFFLL;
    v18 = 32 * v10;
  }
  v19 = sub_22077B0(v18);
  v3 = a1[1];
  v4 = *a1;
  v20 = (__m128i *)v19;
  v11 = v19 + v18;
LABEL_15:
  v12 = a2;
  v13 = &v20->m128i_i8[v5];
  do
  {
    if ( v13 )
    {
      *((_QWORD *)v13 + 1) = 0;
      *(_QWORD *)v13 = v13 + 16;
      v13[16] = 0;
    }
    v13 += 32;
    --v12;
  }
  while ( v12 );
  if ( v4 != v3 )
  {
    v14 = v20;
    for ( i = v4 + 1; ; i += 2 )
    {
      if ( v14 )
      {
        v14->m128i_i64[0] = (__int64)v14[1].m128i_i64;
        v16 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v16 == i )
        {
          v14[1] = _mm_loadu_si128(i);
        }
        else
        {
          v14->m128i_i64[0] = (__int64)v16;
          v14[1].m128i_i64[0] = i->m128i_i64[0];
        }
        v14->m128i_i64[1] = i[-1].m128i_i64[1];
        i[-1].m128i_i64[0] = (__int64)i;
      }
      else
      {
        v17 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v17 != i )
        {
          v22 = v3;
          j_j___libc_free_0(v17, i->m128i_i64[0] + 1);
          v3 = v22;
        }
      }
      v14 += 2;
      if ( v3 == &i[1] )
        break;
    }
    v3 = *a1;
  }
  if ( v3 )
    j_j___libc_free_0(v3, (char *)a1[2] - (char *)v3);
  a1[2] = (__m128i *)v11;
  *a1 = v20;
  a1[1] = &v20[2 * a2 + 2 * v21];
}
