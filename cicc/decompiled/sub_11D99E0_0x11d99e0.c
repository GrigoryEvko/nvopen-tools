// Function: sub_11D99E0
// Address: 0x11d99e0
//
__int64 __fastcall sub_11D99E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  unsigned int v5; // [rsp+8h] [rbp-58h]
  _QWORD v6[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v7[2]; // [rsp+20h] [rbp-40h] BYREF
  _BYTE v8[32]; // [rsp+30h] [rbp-30h] BYREF
  __int16 v9; // [rsp+50h] [rbp-10h]

  v7[0] = *(_QWORD *)(a1 + 8);
  v3 = *(_QWORD *)(a2 + 8);
  v6[0] = a1;
  v7[1] = v3;
  v9 = 257;
  v6[1] = a2;
  return sub_B33D10(a3, 0x11Du, (__int64)v7, 2, (int)v6, 2, v5, (__int64)v8);
}
