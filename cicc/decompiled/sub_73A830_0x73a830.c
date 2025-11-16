// Function: sub_73A830
// Address: 0x73a830
//
_QWORD *__fastcall sub_73A830(__int64 a1, unsigned __int8 a2)
{
  _QWORD *v2; // r12
  const __m128i *v4; // [rsp+8h] [rbp-18h] BYREF

  v4 = (const __m128i *)sub_724DC0();
  sub_72BAF0((__int64)v4, a1, a2);
  v2 = sub_73A720(v4, a1);
  sub_724E30((__int64)&v4);
  return v2;
}
