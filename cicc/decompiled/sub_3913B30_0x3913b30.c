// Function: sub_3913B30
// Address: 0x3913b30
//
__int64 __fastcall sub_3913B30(const __m128i *a1)
{
  __int8 *v1; // rbx
  __int64 v2; // rax
  __m128i v3; // xmm0
  __int64 v4; // rax
  __int64 result; // rax
  __m128i v6; // [rsp+0h] [rbp-40h] BYREF
  __int64 v7; // [rsp+10h] [rbp-30h]

  v1 = &a1[-2].m128i_i8[8];
  v2 = a1[1].m128i_i64[0];
  v6 = _mm_loadu_si128(a1);
  v7 = v2;
  while ( sub_3913890(&v6, v1) )
  {
    v3 = _mm_loadu_si128((const __m128i *)v1);
    v4 = *((_QWORD *)v1 + 2);
    v1 -= 24;
    *((_QWORD *)v1 + 8) = v4;
    *((__m128i *)v1 + 3) = v3;
  }
  result = v7;
  *(__m128i *)(v1 + 24) = _mm_loadu_si128(&v6);
  *((_QWORD *)v1 + 5) = result;
  return result;
}
