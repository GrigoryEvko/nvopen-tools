// Function: sub_3936550
// Address: 0x3936550
//
__int64 __fastcall sub_3936550(unsigned int *a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // r12
  __int64 (__fastcall **v5)(); // [rsp+0h] [rbp-50h] BYREF
  __int64 v6; // [rsp+8h] [rbp-48h]
  __int64 v7; // [rsp+10h] [rbp-40h]
  __int64 v8; // [rsp+18h] [rbp-38h]
  int v9; // [rsp+20h] [rbp-30h]
  __int64 v10; // [rsp+28h] [rbp-28h]

  v9 = 1;
  v8 = 0;
  v7 = 0;
  v6 = 0;
  v5 = off_4A3EE88;
  v10 = 0;
  sub_39363F0(a1, (__int64)&v5, a2, a3);
  v5 = off_4A3EE88;
  v3 = v10 + v8 - v6;
  if ( v8 != v6 )
    sub_16E7BA0((__int64 *)&v5);
  sub_16E7960((__int64)&v5);
  return v3;
}
