// Function: sub_1070780
// Address: 0x1070780
//
__int64 __fastcall sub_1070780(const __m128i *a1)
{
  __int64 v1; // rbx
  __int64 v2; // rax
  __m128i v3; // xmm0
  __int64 v4; // rax
  __int64 result; // rax
  __m128i v6; // [rsp+0h] [rbp-40h] BYREF
  __int64 v7; // [rsp+10h] [rbp-30h]

  v1 = (__int64)&a1[-2].m128i_i64[1];
  v2 = a1[1].m128i_i64[0];
  v6 = _mm_loadu_si128(a1);
  v7 = v2;
  while ( sub_10704F0((__int64)&v6, v1) )
  {
    v3 = _mm_loadu_si128((const __m128i *)v1);
    v4 = *(_QWORD *)(v1 + 16);
    v1 -= 24;
    *(_QWORD *)(v1 + 64) = v4;
    *(__m128i *)(v1 + 48) = v3;
  }
  result = v7;
  *(__m128i *)(v1 + 24) = _mm_loadu_si128(&v6);
  *(_QWORD *)(v1 + 40) = result;
  return result;
}
