// Function: sub_7E0E90
// Address: 0x7e0e90
//
_QWORD *__fastcall sub_7E0E90(__int64 a1, unsigned __int8 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rax
  _QWORD *v5; // r12
  int v7; // [rsp+4h] [rbp-1Ch] BYREF
  const __m128i *v8; // [rsp+8h] [rbp-18h] BYREF

  v3 = a1;
  v8 = (const __m128i *)sub_724DC0();
  sub_72BAF0((__int64)v8, a1, a2);
  v4 = sub_8D6540(v8[8].m128i_i64[0]);
  if ( v4 != v8[8].m128i_i64[0] )
  {
    v3 = v4;
    sub_712540(v8, v4, 1, 0, &v7, dword_4F07508);
  }
  v5 = sub_73A720(v8, v3);
  sub_724E30((__int64)&v8);
  return v5;
}
