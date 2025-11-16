// Function: sub_A56220
// Address: 0xa56220
//
__int64 __fastcall sub_A56220(const __m128i *a1, __int64 a2)
{
  void (__fastcall *v2)(__m128i *, __int64, __int64); // rax
  __int64 v3; // rdx
  __m128i v4; // xmm0
  __m128i v5; // xmm1
  __int64 (__fastcall *v6)(_QWORD, _QWORD, _QWORD); // rcx
  __int64 result; // rax
  __m128i v8; // [rsp+0h] [rbp-30h] BYREF
  __int64 (__fastcall *v9)(_QWORD, _QWORD, _QWORD); // [rsp+10h] [rbp-20h]
  __int64 v10; // [rsp+18h] [rbp-18h]

  v2 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a2 + 16);
  v9 = 0;
  v3 = v10;
  if ( v2 )
  {
    v2(&v8, a2, 2);
    v3 = *(_QWORD *)(a2 + 24);
    v2 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a2 + 16);
  }
  v4 = _mm_loadu_si128(&v8);
  v5 = _mm_loadu_si128(a1 + 2);
  v6 = (__int64 (__fastcall *)(_QWORD, _QWORD, _QWORD))a1[3].m128i_i64[0];
  a1[3].m128i_i64[0] = (__int64)v2;
  result = a1[3].m128i_i64[1];
  v8 = v5;
  v9 = v6;
  v10 = result;
  a1[3].m128i_i64[1] = v3;
  a1[2] = v4;
  if ( v6 )
    return v6(&v8, &v8, 3);
  return result;
}
