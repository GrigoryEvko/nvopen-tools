// Function: sub_72C890
// Address: 0x72c890
//
__m128i *__fastcall sub_72C890(unsigned __int8 a1, const char *a2, const char *a3, __int64 a4)
{
  __m128i *result; // rax
  int v7; // [rsp+1Ch] [rbp-44h] BYREF
  __m128i v8[4]; // [rsp+20h] [rbp-40h] BYREF

  sub_724C70(a4, 4);
  *(_QWORD *)(a4 + 128) = sub_72C6F0(a1);
  sub_70AFD0(a1, a2, v8, &v7);
  *(__m128i *)*(_QWORD *)(a4 + 176) = _mm_loadu_si128(v8);
  sub_70AFD0(a1, a3, v8, &v7);
  result = *(__m128i **)(a4 + 176);
  result[1] = _mm_loadu_si128(v8);
  return result;
}
