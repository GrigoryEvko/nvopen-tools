// Function: sub_1362810
// Address: 0x1362810
//
__int64 *__fastcall sub_1362810(__int64 a1, __int64 a2)
{
  char v2; // r8
  __int64 *result; // rax
  __int64 v4; // rdx
  __int64 *v5; // [rsp+8h] [rbp-18h] BYREF

  v2 = sub_1361B70(a1, (__int64 *)a2, &v5);
  result = v5;
  if ( !v2 )
  {
    result = sub_1362710(a1, (__int64 *)a2, v5);
    *(__m128i *)result = _mm_loadu_si128((const __m128i *)a2);
    *((__m128i *)result + 1) = _mm_loadu_si128((const __m128i *)(a2 + 16));
    result[4] = *(_QWORD *)(a2 + 32);
    *(__m128i *)(result + 5) = _mm_loadu_si128((const __m128i *)(a2 + 40));
    *(__m128i *)(result + 7) = _mm_loadu_si128((const __m128i *)(a2 + 56));
    v4 = *(_QWORD *)(a2 + 72);
    *((_BYTE *)result + 80) = 0;
    result[9] = v4;
  }
  return result;
}
