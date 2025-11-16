// Function: sub_7F9BA0
// Address: 0x7f9ba0
//
_QWORD *__fastcall sub_7F9BA0(__int64 a1, __int64 a2, _QWORD *a3, _QWORD *a4)
{
  const __m128i *v6; // rax
  __int64 v8; // rax
  _QWORD *v9; // rax
  const __m128i *v10; // [rsp+8h] [rbp-28h] BYREF

  v6 = (const __m128i *)sub_724DC0();
  *a4 = 0;
  v10 = v6;
  *a3 = 0;
  sub_7E3EE0(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL));
  if ( (((*(_BYTE *)(a1 + 205) & 0x1C) - 8) & 0xF4) == 0
    && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL) + 176LL) & 0x10) != 0 )
  {
    v8 = sub_7E1DF0();
    sub_72BB40(v8, v10);
    v9 = sub_73A720(v10, (__int64)v10);
    *a4 = v9;
    *a3 = v9;
  }
  return sub_724E30((__int64)&v10);
}
