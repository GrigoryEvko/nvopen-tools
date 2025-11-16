// Function: sub_7DF9B0
// Address: 0x7df9b0
//
_QWORD *__fastcall sub_7DF9B0(__m128i *a1, unsigned __int8 a2, __int64 a3)
{
  _QWORD *v4; // rax
  const __m128i *v5; // rsi
  char v7; // [rsp+3h] [rbp-2Dh] BYREF
  int v8; // [rsp+4h] [rbp-2Ch] BYREF
  const __m128i *v9; // [rsp+8h] [rbp-28h] BYREF

  v9 = (const __m128i *)sub_724DC0();
  sub_724C70((__int64)v9, 0);
  v4 = sub_72BA30(a2);
  v5 = v9;
  v9[8].m128i_i64[0] = (__int64)v4;
  sub_710080(a1, (__int64)v5, 1, &v8, &v7);
  if ( v8 )
  {
    if ( a3 )
      sub_685360(0x4E6u, dword_4F07508, a3);
    else
      sub_6851C0(0x4E7u, dword_4F07508);
  }
  sub_72A510(v9, a1);
  return sub_724E30((__int64)&v9);
}
