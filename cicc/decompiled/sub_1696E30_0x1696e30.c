// Function: sub_1696E30
// Address: 0x1696e30
//
__int64 *__fastcall sub_1696E30(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, char a5)
{
  __int64 *v5; // r14
  __int64 *i; // rbx
  __m128i *v7; // rsi
  __m128i *v8; // rdi
  _BYTE *v9; // rax
  __int64 v10; // rdx
  char v11; // al
  __m128i *v12; // rbx
  __m128i *v13; // r12
  __m128i *v18; // [rsp+20h] [rbp-70h] BYREF
  __m128i *v19; // [rsp+28h] [rbp-68h]
  __m128i *v20; // [rsp+30h] [rbp-60h]
  __m128i v21; // [rsp+40h] [rbp-50h] BYREF
  __m128i v22[4]; // [rsp+50h] [rbp-40h] BYREF

  v5 = &a2[a3];
  v18 = 0;
  v19 = 0;
  v20 = 0;
  if ( a2 != v5 )
  {
    for ( i = a2; v5 != i; ++i )
    {
      v9 = (_BYTE *)sub_1694C30(*i);
      v21.m128i_i64[0] = (__int64)v22;
      if ( v9 )
      {
        sub_1693C00(v21.m128i_i64, v9, (__int64)&v9[v10]);
        v7 = v19;
        if ( v19 != v20 )
          goto LABEL_4;
      }
      else
      {
        v21.m128i_i64[1] = 0;
        v7 = v19;
        v22[0].m128i_i8[0] = 0;
        if ( v19 != v20 )
        {
LABEL_4:
          v8 = (__m128i *)v21.m128i_i64[0];
          if ( v7 )
          {
            v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
            if ( (__m128i *)v21.m128i_i64[0] == v22 )
            {
              v7[1] = _mm_load_si128(v22);
            }
            else
            {
              v7->m128i_i64[0] = v21.m128i_i64[0];
              v7[1].m128i_i64[0] = v22[0].m128i_i64[0];
            }
            v21.m128i_i64[0] = (__int64)v22;
            v8 = v22;
            v7->m128i_i64[1] = v21.m128i_i64[1];
            v7 = v19;
            v21.m128i_i64[1] = 0;
            v22[0].m128i_i8[0] = 0;
          }
          v19 = v7 + 2;
          goto LABEL_9;
        }
      }
      sub_8F99A0(&v18, v7, &v21);
      v8 = (__m128i *)v21.m128i_i64[0];
LABEL_9:
      if ( v8 != v22 )
        j_j___libc_free_0(v8, v22[0].m128i_i64[0] + 1);
    }
  }
  v11 = sub_16BA240();
  sub_1696580(a1, v18, ((char *)v19 - (char *)v18) >> 5, v11 & a5, a4);
  v12 = v19;
  v13 = v18;
  if ( v19 != v18 )
  {
    do
    {
      if ( (__m128i *)v13->m128i_i64[0] != &v13[1] )
        j_j___libc_free_0(v13->m128i_i64[0], v13[1].m128i_i64[0] + 1);
      v13 += 2;
    }
    while ( v12 != v13 );
    v13 = v18;
  }
  if ( v13 )
    j_j___libc_free_0(v13, (char *)v20 - (char *)v13);
  return a1;
}
