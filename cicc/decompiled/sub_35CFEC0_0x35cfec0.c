// Function: sub_35CFEC0
// Address: 0x35cfec0
//
void __fastcall sub_35CFEC0(__m128i *src, __m128i *a2)
{
  __m128i *v2; // r10
  unsigned __int8 v3; // dl
  unsigned __int8 v4; // al
  __int64 v5; // rdx
  __int64 v6; // rax
  __m128i *v7; // r12
  __m128i v8; // xmm1
  __int64 v9; // rax
  __m128i v10; // xmm2
  __m128i v11; // xmm3
  __int8 v12; // al
  __m128i v13; // [rsp-58h] [rbp-58h] BYREF
  __m128i v14; // [rsp-48h] [rbp-48h] BYREF
  __int64 v15; // [rsp-38h] [rbp-38h]

  if ( src != a2 )
  {
    v2 = (__m128i *)((char *)src + 40);
    if ( a2 != (__m128i *)&src[2].m128i_u64[1] )
    {
      do
      {
        v3 = src[2].m128i_i32[0] != 2;
        v4 = v2[2].m128i_i32[0] != 2;
        if ( v3 < v4
          || v3 == v4
          && ((v5 = src[1].m128i_i64[1] + src[1].m128i_i64[0], v6 = v2[1].m128i_i64[1] + v2[1].m128i_i64[0], v5 < v6)
           || v5 == v6 && src->m128i_i32[0] < v2->m128i_i32[0]) )
        {
          v8 = _mm_loadu_si128(v2 + 1);
          v7 = (__m128i *)((char *)v2 + 40);
          v9 = v2[2].m128i_i64[0];
          v13 = _mm_loadu_si128(v2);
          v15 = v9;
          v14 = v8;
          if ( src != v2 )
            memmove(&src[2].m128i_u64[1], src, (char *)v2 - (char *)src);
          v10 = _mm_loadu_si128(&v13);
          v11 = _mm_loadu_si128(&v14);
          src[2].m128i_i32[0] = v15;
          v12 = BYTE4(v15);
          *src = v10;
          src[2].m128i_i8[4] = v12;
          src[1] = v11;
        }
        else
        {
          v7 = (__m128i *)((char *)v2 + 40);
          sub_35CFE10(v2->m128i_i32);
        }
        v2 = v7;
      }
      while ( a2 != v7 );
    }
  }
}
