// Function: sub_67B370
// Address: 0x67b370
//
__int64 __fastcall sub_67B370(_DWORD *a1)
{
  __int64 v1; // rcx
  _QWORD v3[3]; // [rsp+0h] [rbp-50h] BYREF
  int v4; // [rsp+18h] [rbp-38h]
  _BYTE v5[4]; // [rsp+1Ch] [rbp-34h] BYREF
  int v6; // [rsp+20h] [rbp-30h]
  _BYTE v7[4]; // [rsp+24h] [rbp-2Ch] BYREF
  _BYTE v8[4]; // [rsp+28h] [rbp-28h] BYREF
  int v9; // [rsp+2Ch] [rbp-24h]
  unsigned int v10; // [rsp+30h] [rbp-20h]
  _BYTE v11[24]; // [rsp+38h] [rbp-18h] BYREF

  v3[1] = 0x100000000LL;
  v3[0] = 0;
  v3[2] = 0x100000000LL;
  v4 = 0;
  v10 = dword_4F06650[0];
  v6 = 1;
  v9 = 1;
  sub_7BDB60(1);
  sub_866940(1, v5, v11, v7, v8);
  sub_67A900(v3, 66, 1, v1);
  *a1 = v4;
  sub_679880((__int64)v3);
  return v3[0];
}
