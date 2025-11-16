// Function: sub_1AD17F0
// Address: 0x1ad17f0
//
const void **__fastcall sub_1AD17F0(const void **a1, const void **a2)
{
  const void **v2; // rbx
  const void *v3; // rcx
  size_t v4; // r14
  size_t v5; // r13
  const void *v6; // rdi
  const void **v7; // r12
  int v8; // eax
  const __m128i *v9; // rbx
  const void *v11; // [rsp+0h] [rbp-40h]

  if ( a2 != a1 )
  {
    v2 = a1 + 2;
    if ( a2 != a1 + 2 )
    {
      v3 = *a1;
      v4 = (size_t)a1[1];
      while ( 1 )
      {
        v5 = v4;
        v4 = (size_t)v2[1];
        v6 = v3;
        v7 = v2 - 2;
        v3 = *v2;
        if ( v4 == v5 )
        {
          if ( !v4 )
            break;
          v11 = *v2;
          v8 = memcmp(v6, *v2, (size_t)v2[1]);
          v3 = v11;
          if ( !v8 )
            break;
        }
        v2 += 2;
        if ( a2 == v2 )
          return a2;
      }
      if ( a2 != v7 )
      {
        v9 = (const __m128i *)(v7 + 4);
        if ( a2 == v7 + 4 )
          return v7 + 2;
        while ( v5 == v9->m128i_i64[1] )
        {
          if ( !v5 )
            goto LABEL_12;
          if ( memcmp(v6, (const void *)v9->m128i_i64[0], v5) )
            break;
          if ( ++v9 == (const __m128i *)a2 )
            return v7 + 2;
LABEL_13:
          v6 = *v7;
          v5 = (size_t)v7[1];
        }
        v7 += 2;
        *(__m128i *)v7 = _mm_loadu_si128(v9);
LABEL_12:
        if ( ++v9 == (const __m128i *)a2 )
          return v7 + 2;
        goto LABEL_13;
      }
    }
  }
  return a2;
}
