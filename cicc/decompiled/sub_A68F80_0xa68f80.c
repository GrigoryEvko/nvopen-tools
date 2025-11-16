// Function: sub_A68F80
// Address: 0xa68f80
//
void __fastcall sub_A68F80(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  const __m128i *v6; // r13
  __int64 v7; // r13
  __int64 v8; // rsi
  _BYTE v9[112]; // [rsp+10h] [rbp-4F0h] BYREF
  _BYTE v10[416]; // [rsp+80h] [rbp-480h] BYREF
  _BYTE v11[736]; // [rsp+220h] [rbp-2E0h] BYREF

  memset(v10, 0, 0x198u);
  v6 = sub_A56340(a3, a2);
  if ( !v6 )
  {
    v7 = *(_QWORD *)(a1 + 48);
    if ( v10[400] )
    {
      v10[400] = 0;
      sub_A552A0((__int64)v10, a2);
    }
    v8 = v7;
    v6 = (const __m128i *)v10;
    sub_A55A10((__int64)v10, v8, 0);
    v10[400] = 1;
  }
  sub_A54BD0((__int64)v9, a2);
  sub_A685A0((__int64)v11, (__int64)v9, (__int64)v6, *(_QWORD *)(a1 + 48), 0, a4, 0);
  sub_A530C0((__int64)v11, a1);
  sub_A555E0((__int64)v11);
  sub_A54D10((__int64)v9, a1);
  if ( v10[400] )
  {
    v10[400] = 0;
    sub_A552A0((__int64)v10, a1);
  }
}
