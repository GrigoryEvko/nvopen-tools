// Function: sub_12536B0
// Address: 0x12536b0
//
__int64 __fastcall sub_12536B0(unsigned int *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v5[2]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v6; // [rsp+10h] [rbp-50h]
  __int64 v7; // [rsp+18h] [rbp-48h]
  __int64 v8; // [rsp+20h] [rbp-40h]
  __int64 v9; // [rsp+28h] [rbp-38h]
  __int64 v10; // [rsp+30h] [rbp-30h]

  v9 = 0x100000000LL;
  v5[1] = 0;
  v6 = 0;
  v7 = 0;
  v8 = 0;
  v5[0] = (__int64)off_49E67B0;
  v10 = 0;
  sub_1253560(a1, a2, (__int64)v5, a3);
  v5[0] = (__int64)off_49E67B0;
  v3 = v10 + v8 - v6;
  if ( v8 != v6 )
    sub_CB5AE0(v5);
  sub_CB5840((__int64)v5);
  return v3;
}
