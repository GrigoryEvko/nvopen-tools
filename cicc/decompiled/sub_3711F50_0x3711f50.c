// Function: sub_3711F50
// Address: 0x3711f50
//
void __fastcall sub_3711F50(
        __m128i *src,
        __m128i *a2,
        unsigned __int8 (__fastcall *a3)(__m128i *, __int8 *),
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  const __m128i *v7; // rbx
  char i; // al
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __m128i v14; // xmm1
  __int64 v15; // rax
  __m128i v16; // xmm3
  __int16 v17; // ax
  const __m128i *v18; // rdi
  __m128i v19; // [rsp-68h] [rbp-68h] BYREF
  __m128i v20; // [rsp-58h] [rbp-58h] BYREF
  __int64 v21; // [rsp-48h] [rbp-48h]

  if ( src != a2 )
  {
    v7 = (__m128i *)((char *)src + 40);
    if ( a2 != (__m128i *)&src[2].m128i_u64[1] )
    {
      for ( i = ((__int64 (__fastcall *)(const __m128i *, __m128i *, unsigned __int8 (__fastcall *)(__m128i *, __int8 *), __int64, __int64, __int64, __int64, __int64, __int64, __int64))a3)(
                  v7,
                  src,
                  a3,
                  a4,
                  a5,
                  a6,
                  v19.m128i_i64[0],
                  v19.m128i_i64[1],
                  v20.m128i_i64[0],
                  v20.m128i_i64[1]);
            ;
            i = ((__int64 (__fastcall *)(const __m128i *, __m128i *, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64))a3)(
                  v7,
                  src,
                  v10,
                  v11,
                  v12,
                  v13,
                  v19.m128i_i64[0],
                  v19.m128i_i64[1],
                  v20.m128i_i64[0],
                  v20.m128i_i64[1]) )
      {
        if ( i )
        {
          v14 = _mm_loadu_si128(v7 + 1);
          v15 = v7[2].m128i_i64[0];
          v19 = _mm_loadu_si128(v7);
          v21 = v15;
          v20 = v14;
          if ( src != v7 )
            memmove(&src[2].m128i_u64[1], src, (char *)v7 - (char *)src);
          v16 = _mm_loadu_si128(&v20);
          v7 = (const __m128i *)((char *)v7 + 40);
          v17 = v21;
          *src = _mm_loadu_si128(&v19);
          src[2].m128i_i16[0] = v17;
          src[1] = v16;
          if ( a2 == v7 )
            return;
        }
        else
        {
          v18 = v7;
          v7 = (const __m128i *)((char *)v7 + 40);
          sub_3711EC0(v18, a3);
          if ( a2 == v7 )
            return;
        }
      }
    }
  }
}
