// Function: sub_39F0D00
// Address: 0x39f0d00
//
unsigned __int64 __fastcall sub_39F0D00(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  unsigned __int64 *v4; // rdi
  __m128i *v5; // rsi
  unsigned __int64 result; // rax
  __m128i v7; // [rsp+0h] [rbp-20h] BYREF
  unsigned __int64 v8; // [rsp+10h] [rbp-10h]

  v4 = *(unsigned __int64 **)(a1 + 264);
  v7.m128i_i64[0] = a2;
  v7.m128i_i64[1] = a3;
  v5 = (__m128i *)v4[261];
  v8 = a4;
  if ( v5 == (__m128i *)v4[262] )
    return sub_39F0B50(v4 + 260, v5, &v7);
  if ( v5 )
  {
    *v5 = _mm_loadu_si128(&v7);
    result = v8;
    v5[1].m128i_i64[0] = v8;
    v5 = (__m128i *)v4[261];
  }
  v4[261] = (unsigned __int64)&v5[1].m128i_u64[1];
  return result;
}
