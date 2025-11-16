// Function: sub_1516E80
// Address: 0x1516e80
//
__int64 __fastcall sub_1516E80(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdi
  __m128i *v4; // rsi
  __int64 result; // rax
  __m128i v6; // [rsp+0h] [rbp-10h] BYREF

  v3 = *a1;
  v6.m128i_i64[0] = a2;
  v6.m128i_i64[1] = a3;
  v4 = *(__m128i **)(v3 + 640);
  if ( v4 == *(__m128i **)(v3 + 648) )
    return sub_1516D00((const __m128i **)(v3 + 632), v4, &v6);
  if ( v4 )
  {
    *v4 = _mm_loadu_si128(&v6);
    v4 = *(__m128i **)(v3 + 640);
  }
  *(_QWORD *)(v3 + 640) = v4 + 1;
  return result;
}
