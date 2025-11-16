// Function: sub_8D70E0
// Address: 0x8d70e0
//
__int64 __fastcall sub_8D70E0(_BYTE *a1, int a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  _QWORD v6[12]; // [rsp+0h] [rbp-F0h] BYREF
  int v7; // [rsp+60h] [rbp-90h]
  __int64 v8; // [rsp+88h] [rbp-68h]
  int v9; // [rsp+90h] [rbp-60h]

  if ( (a1[25] & 3) == 0 )
    return *(_QWORD *)a1;
  sub_76C7C0((__int64)v6);
  v9 = a2;
  v7 = 1;
  v6[0] = sub_8D6DC0;
  sub_76CDC0(a1, (__int64)v6, v3, v4, v5);
  return v8;
}
