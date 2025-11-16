// Function: sub_3184E50
// Address: 0x3184e50
//
__m128i *__fastcall sub_3184E50(__m128i *a1, __int64 a2)
{
  __m128i v2; // xmm0
  __m128i v4; // [rsp+0h] [rbp-30h] BYREF
  __int64 v5; // [rsp+10h] [rbp-20h]

  (*(void (__fastcall **)(__m128i *, __int64, _QWORD, _QWORD))(*(_QWORD *)a2 + 16LL))(&v4, a2, 0, 0);
  v2 = _mm_loadu_si128(&v4);
  a1[1].m128i_i64[0] = v5;
  *a1 = v2;
  return a1;
}
