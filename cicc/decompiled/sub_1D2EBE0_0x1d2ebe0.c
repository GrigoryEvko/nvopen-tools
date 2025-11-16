// Function: sub_1D2EBE0
// Address: 0x1d2ebe0
//
__m128i **sub_1D2EBE0()
{
  __m128i **v0; // rax
  __m128i **v1; // r14
  __m128i *v2; // rax
  const __m128i *v3; // rdi
  __m128i *v4; // rcx
  __m128i *v5; // r12
  const __m128i *v6; // rdx
  __m128i *v7; // rcx
  __m128i *v8; // rax
  int v9; // ebx
  __m128i v11; // [rsp+0h] [rbp-30h] BYREF

  v0 = (__m128i **)sub_22077B0(24);
  v1 = v0;
  if ( v0 )
  {
    *v0 = 0;
    v0[1] = 0;
    v0[2] = 0;
    v2 = (__m128i *)sub_22077B0(1840);
    v3 = *v1;
    v4 = v1[1];
    v5 = v2;
    v6 = *v1;
    if ( v4 != *v1 )
    {
      v7 = (__m128i *)((char *)v2 + (char *)v4 - (char *)v3);
      do
      {
        if ( v2 )
          *v2 = _mm_loadu_si128(v6);
        ++v2;
        ++v6;
      }
      while ( v7 != v2 );
    }
    if ( v3 )
      j_j___libc_free_0(v3, (char *)v1[2] - (char *)v3);
    v8 = v5 + 115;
    *v1 = v5;
    v9 = 0;
    v1[1] = v5;
    for ( v1[2] = v5 + 115; ; v8 = v1[2] )
    {
      v11.m128i_i8[0] = v9;
      v11.m128i_i64[1] = 0;
      if ( v5 == v8 )
      {
        ++v9;
        sub_1D2EA60((const __m128i **)v1, v5, &v11);
        if ( v9 == 115 )
          return v1;
      }
      else
      {
        if ( v5 )
        {
          *v5 = _mm_loadu_si128(&v11);
          v5 = v1[1];
        }
        ++v9;
        v1[1] = v5 + 1;
        if ( v9 == 115 )
          return v1;
      }
      v5 = v1[1];
    }
  }
  return v1;
}
