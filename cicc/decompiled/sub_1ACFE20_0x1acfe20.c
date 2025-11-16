// Function: sub_1ACFE20
// Address: 0x1acfe20
//
void __fastcall sub_1ACFE20(const void **a1, const void **a2)
{
  __m128i *v2; // r12
  int v3; // eax
  const void **v4; // rbx
  size_t v5; // r15
  const void *v6; // r14
  size_t v7; // rbx
  const void *v8; // rsi
  __m128i *i; // rbx
  int v10; // eax
  __m128i v11; // xmm0
  size_t v12; // r13
  const void *v13; // rsi

  if ( a1 != a2 )
  {
    v2 = (__m128i *)(a1 + 2);
    if ( a2 != a1 + 2 )
    {
      do
      {
        while ( 1 )
        {
          v5 = v2->m128i_u64[1];
          v6 = (const void *)v2->m128i_i64[0];
          v7 = (size_t)a1[1];
          v8 = *a1;
          if ( v5 <= v7 )
            break;
          if ( !v7 )
            goto LABEL_15;
          v3 = memcmp((const void *)v2->m128i_i64[0], v8, (size_t)a1[1]);
          if ( v3 )
            goto LABEL_14;
LABEL_7:
          if ( v5 < v7 )
            goto LABEL_8;
LABEL_15:
          for ( i = v2; ; i[1] = v11 )
          {
            v12 = i[-1].m128i_u64[1];
            v13 = (const void *)i[-1].m128i_i64[0];
            if ( v5 > v12 )
              break;
            if ( v5 )
            {
              v10 = memcmp(v6, v13, v5);
              if ( v10 )
                goto LABEL_24;
            }
            if ( v5 == v12 )
              goto LABEL_25;
LABEL_19:
            if ( v5 >= v12 )
              goto LABEL_25;
LABEL_20:
            v11 = _mm_loadu_si128(--i);
          }
          if ( !v12 )
            goto LABEL_25;
          v10 = memcmp(v6, v13, i[-1].m128i_u64[1]);
          if ( !v10 )
            goto LABEL_19;
LABEL_24:
          if ( v10 < 0 )
            goto LABEL_20;
LABEL_25:
          i->m128i_i64[0] = (__int64)v6;
          i->m128i_i64[1] = v5;
          if ( a2 == (const void **)++v2 )
            return;
        }
        if ( !v5 || (v3 = memcmp((const void *)v2->m128i_i64[0], v8, v2->m128i_u64[1])) == 0 )
        {
          if ( v5 == v7 )
            goto LABEL_15;
          goto LABEL_7;
        }
LABEL_14:
        if ( v3 >= 0 )
          goto LABEL_15;
LABEL_8:
        v4 = (const void **)&v2[1];
        if ( a1 != (const void **)v2 )
          memmove(a1 + 2, a1, (char *)v2 - (char *)a1);
        ++v2;
        *a1 = v6;
        a1[1] = (const void *)v5;
      }
      while ( a2 != v4 );
    }
  }
}
