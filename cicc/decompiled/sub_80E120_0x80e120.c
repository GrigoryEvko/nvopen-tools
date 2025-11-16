// Function: sub_80E120
// Address: 0x80e120
//
_QWORD *__fastcall sub_80E120(_QWORD *a1)
{
  _QWORD *v1; // rax
  const __m128i *v3; // [rsp+8h] [rbp-18h] BYREF

  v3 = (const __m128i *)sub_724DC0();
  v1 = sub_72BA30(5u);
  sub_72BB40((__int64)v1, v3);
  sub_80D8A0(v3, 0, 0, a1);
  return sub_724E30((__int64)&v3);
}
