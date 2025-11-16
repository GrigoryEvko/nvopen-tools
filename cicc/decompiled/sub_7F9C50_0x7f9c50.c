// Function: sub_7F9C50
// Address: 0x7f9c50
//
_QWORD *__fastcall sub_7F9C50(__int64 a1, __int64 a2, _QWORD *a3, _QWORD *a4)
{
  _QWORD *result; // rax
  __int64 v7; // rax
  _QWORD *v8; // rax
  const __m128i *v9; // [rsp+8h] [rbp-28h] BYREF

  *a4 = 0;
  *a3 = 0;
  sub_7E3EE0(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL));
  result = (_QWORD *)((*(_BYTE *)(a1 + 205) & 0x1Cu) - 8);
  if ( (((*(_BYTE *)(a1 + 205) & 0x1C) - 8) & 0xF4) == 0 )
  {
    result = *(_QWORD **)(*(_QWORD *)(a1 + 40) + 32LL);
    if ( (result[22] & 0x10) != 0 )
    {
      v9 = (const __m128i *)sub_724DC0();
      v7 = sub_7E1DF0();
      sub_72BB40(v7, v9);
      v8 = sub_73A720(v9, (__int64)v9);
      *a3 = v8;
      *a4 = v8;
      return sub_724E30((__int64)&v9);
    }
  }
  return result;
}
