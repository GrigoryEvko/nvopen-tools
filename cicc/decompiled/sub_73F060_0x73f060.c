// Function: sub_73F060
// Address: 0x73f060
//
__m128i *__fastcall sub_73F060(__int64 a1, __m128i *a2)
{
  __m128i *result; // rax
  __m128i *v3; // [rsp+8h] [rbp-18h] BYREF

  v3 = a2;
  sub_73EE70((__int64 *)(a1 + 160), &v3);
  result = v3;
  *(_QWORD *)(a1 + 168) = v3;
  return result;
}
