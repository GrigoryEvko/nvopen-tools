// Function: sub_7F0190
// Address: 0x7f0190
//
__m128i *__fastcall sub_7F0190(int a1, __int64 a2)
{
  __int64 v2; // r12
  const __m128i *v3; // r13
  _BYTE *v4; // r14
  char i; // al
  __int64 v6; // rax
  __m128i *v7; // r12

  v2 = a2;
  v3 = (const __m128i *)sub_724D50(10);
  v4 = sub_724D50(1);
  for ( i = *(_BYTE *)(a2 + 140); i == 12; i = *(_BYTE *)(v2 + 140) )
    v2 = *(_QWORD *)(v2 + 160);
  if ( (unsigned __int8)(i - 9) > 2u
    || (v6 = *(_QWORD *)(v2 + 160)) == 0
    || *(_QWORD *)(v6 + 112)
    || !(unsigned int)sub_8D2780(*(_QWORD *)(v6 + 120)) )
  {
    sub_685390(0xBABu, dword_4F07508, v2);
  }
  v3[8].m128i_i64[0] = v2;
  *((_QWORD *)v4 + 16) = *(_QWORD *)(*(_QWORD *)(v2 + 160) + 120LL);
  sub_620D80((_WORD *)v4 + 88, a1);
  sub_72A690((__int64)v4, (__int64)v3, 0, 0);
  v7 = (__m128i *)sub_73A720(v3, (__int64)v3);
  sub_7EE560(v7, 0);
  return v7;
}
