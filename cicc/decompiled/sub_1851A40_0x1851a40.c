// Function: sub_1851A40
// Address: 0x1851a40
//
__int64 (__fastcall *__fastcall sub_1851A40(__int64 a1, __int64 a2, __m128i *a3))(_QWORD, _QWORD, _QWORD)
{
  __m128i v3; // xmm0
  __int64 (__fastcall *result)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v5; // rcx
  __m128i v6; // xmm1
  __int64 v7; // rsi
  __m128i v8; // [rsp+0h] [rbp-20h] BYREF
  __int64 (__fastcall *v9)(_QWORD, _QWORD, _QWORD); // [rsp+10h] [rbp-10h]
  __int64 v10; // [rsp+18h] [rbp-8h]

  v3 = _mm_loadu_si128(a3);
  result = (__int64 (__fastcall *)(_QWORD, _QWORD, _QWORD))a3[1].m128i_i64[0];
  v5 = a3[1].m128i_i64[1];
  v6 = _mm_loadu_si128(&v8);
  a3[1].m128i_i64[0] = 0;
  v7 = v10;
  v9 = result;
  v10 = v5;
  a3[1].m128i_i64[1] = v7;
  *a3 = v6;
  v8 = v3;
  if ( result )
    return (__int64 (__fastcall *)(_QWORD, _QWORD, _QWORD))result(&v8, &v8, 3);
  return result;
}
