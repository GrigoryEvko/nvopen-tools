// Function: sub_6EA060
// Address: 0x6ea060
//
__m128i *__fastcall sub_6EA060(__int64 a1, __int64 a2)
{
  __m128i *result; // rax
  __m128i v3[23]; // [rsp+0h] [rbp-170h] BYREF

  sub_6E9FE0(a2, v3);
  result = sub_6E3700(v3, 0);
  *(_QWORD *)(a1 + 120) = result;
  return result;
}
