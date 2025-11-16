// Function: sub_2044510
// Address: 0x2044510
//
__int64 __fastcall sub_2044510(const __m128i *a1)
{
  const __m128i *v1; // rbx
  __int64 v2; // rax
  __int64 v3; // r12
  __m128i v4; // xmm0
  __m128i v5; // xmm1
  __int64 v6; // rax
  __m128i *v7; // r13
  __m128i v8; // xmm5
  __int64 result; // rax
  __m128i v10; // [rsp+0h] [rbp-50h] BYREF
  __m128i v11; // [rsp+10h] [rbp-40h] BYREF
  __int64 v12; // [rsp+20h] [rbp-30h]

  v1 = a1;
  v12 = a1[2].m128i_i64[0];
  v2 = a1->m128i_i64[1];
  v10 = _mm_loadu_si128(a1);
  v11 = _mm_loadu_si128(a1 + 1);
  v3 = v2 + 24;
  while ( 1 )
  {
    v6 = v1[-2].m128i_i64[0];
    v7 = (__m128i *)v1;
    v1 = (const __m128i *)((char *)v1 - 40);
    if ( (int)sub_16AEA10(v3, v6 + 24) >= 0 )
      break;
    v4 = _mm_loadu_si128(v1);
    v5 = _mm_loadu_si128(v1 + 1);
    v1[4].m128i_i32[2] = v1[2].m128i_i32[0];
    *(__m128i *)((char *)v1 + 40) = v4;
    *(__m128i *)((char *)v1 + 56) = v5;
  }
  v8 = _mm_loadu_si128(&v11);
  result = (unsigned int)v12;
  *v7 = _mm_loadu_si128(&v10);
  v7[2].m128i_i32[0] = result;
  v7[1] = v8;
  return result;
}
