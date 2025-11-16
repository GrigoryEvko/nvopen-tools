// Function: sub_23667F0
// Address: 0x23667f0
//
__m128i **__fastcall sub_23667F0(__m128i **a1, const __m128i **a2, __int64 a3)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rbx
  __m128i *v5; // r15
  const __m128i *v6; // r14
  unsigned __int64 v7; // rbx
  __m128i *v8; // r13
  const __m128i *v9; // rbx
  const __m128i *v10; // r12
  __m128i **v12; // [rsp+0h] [rbp-60h]
  const __m128i *v13; // [rsp+8h] [rbp-58h]
  __m128i *v14; // [rsp+10h] [rbp-50h]
  const __m128i *v15; // [rsp+18h] [rbp-48h]
  const __m128i *v16; // [rsp+20h] [rbp-40h]

  v3 = (char *)a2[1] - (char *)*a2;
  v12 = a1;
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( v3 )
  {
    if ( v3 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_29:
      sub_4261EA(a1, a2, a3);
    a1 = (__m128i **)v3;
    v14 = (__m128i *)sub_22077B0(v3);
  }
  else
  {
    v14 = 0;
  }
  *v12 = v14;
  v12[1] = v14;
  v12[2] = (__m128i *)((char *)v14 + v3);
  v13 = a2[1];
  if ( *a2 != v13 )
  {
    v15 = *a2;
    do
    {
      if ( v14 )
      {
        *v14 = _mm_loadu_si128(v15);
        v4 = v15[1].m128i_i64[1] - v15[1].m128i_i64[0];
        v14[1].m128i_i64[0] = 0;
        v14[1].m128i_i64[1] = 0;
        v14[2].m128i_i64[0] = 0;
        if ( v4 )
        {
          if ( v4 > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_29;
          a1 = (__m128i **)v4;
          v5 = (__m128i *)sub_22077B0(v4);
        }
        else
        {
          v5 = 0;
        }
        v14[1].m128i_i64[0] = (__int64)v5;
        v14[1].m128i_i64[1] = (__int64)v5;
        v14[2].m128i_i64[0] = (__int64)v5->m128i_i64 + v4;
        a3 = v15[1].m128i_i64[1];
        v16 = (const __m128i *)a3;
        if ( a3 != v15[1].m128i_i64[0] )
        {
          v6 = (const __m128i *)v15[1].m128i_i64[0];
          do
          {
            if ( v5 )
            {
              *v5 = _mm_loadu_si128(v6);
              v7 = v6[1].m128i_i64[1] - v6[1].m128i_i64[0];
              v5[1].m128i_i64[0] = 0;
              v5[1].m128i_i64[1] = 0;
              v5[2].m128i_i64[0] = 0;
              if ( v7 )
              {
                if ( v7 > 0x7FFFFFFFFFFFFFF8LL )
                  goto LABEL_29;
                a1 = (__m128i **)v7;
                v8 = (__m128i *)sub_22077B0(v7);
              }
              else
              {
                v7 = 0;
                v8 = 0;
              }
              v5[1].m128i_i64[0] = (__int64)v8;
              v5[1].m128i_i64[1] = (__int64)v8;
              v5[2].m128i_i64[0] = (__int64)v8->m128i_i64 + v7;
              v9 = (const __m128i *)v6[1].m128i_i64[1];
              a3 = v6[1].m128i_i64[0];
              if ( v9 != (const __m128i *)a3 )
              {
                v10 = (const __m128i *)v6[1].m128i_i64[0];
                do
                {
                  if ( v8 )
                  {
                    a2 = (const __m128i **)&v10[1];
                    a1 = (__m128i **)&v8[1];
                    *v8 = _mm_loadu_si128(v10);
                    sub_23667F0(&v8[1], &v10[1]);
                  }
                  v10 = (const __m128i *)((char *)v10 + 40);
                  v8 = (__m128i *)((char *)v8 + 40);
                }
                while ( v9 != v10 );
              }
              v5[1].m128i_i64[1] = (__int64)v8;
            }
            v5 = (__m128i *)((char *)v5 + 40);
            v6 = (const __m128i *)((char *)v6 + 40);
          }
          while ( v16 != v6 );
        }
        v14[1].m128i_i64[1] = (__int64)v5;
      }
      v15 = (const __m128i *)((char *)v15 + 40);
      v14 = (__m128i *)((char *)v14 + 40);
    }
    while ( v13 != v15 );
  }
  v12[1] = v14;
  return v12;
}
