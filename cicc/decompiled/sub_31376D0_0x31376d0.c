// Function: sub_31376D0
// Address: 0x31376d0
//
__int64 __fastcall sub_31376D0(__int64 a1, __int64 *a2, _DWORD *a3)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // r15
  __int64 v7; // r12
  __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *a2;
  v5 = a2[3];
  v6 = *(_QWORD *)(v4 + 72);
  v9[0] = v5;
  if ( v5 )
    sub_B96E90((__int64)v9, v5, 1);
  v7 = sub_3137510(a1, (__int64)v9, a3, v6);
  if ( v9[0] )
    sub_B91220((__int64)v9, v9[0]);
  return v7;
}
