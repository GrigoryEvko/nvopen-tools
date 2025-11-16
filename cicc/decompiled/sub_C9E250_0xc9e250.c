// Function: sub_C9E250
// Address: 0xc9e250
//
__int64 __fastcall sub_C9E250(__int64 a1)
{
  __m128i v1; // xmm1
  __int64 result; // rax
  __m128i v3; // [rsp+0h] [rbp-40h] BYREF
  __m128i v4; // [rsp+10h] [rbp-30h] BYREF
  __int64 v5; // [rsp+20h] [rbp-20h]

  *(_WORD *)(a1 + 144) = 257;
  sub_C9E0E0((double *)v3.m128i_i64, 1);
  v1 = _mm_loadu_si128(&v4);
  result = v5;
  *(__m128i *)(a1 + 40) = _mm_loadu_si128(&v3);
  *(_QWORD *)(a1 + 72) = result;
  *(__m128i *)(a1 + 56) = v1;
  return result;
}
