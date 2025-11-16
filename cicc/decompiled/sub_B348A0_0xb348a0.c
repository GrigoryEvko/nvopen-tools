// Function: sub_B348A0
// Address: 0xb348a0
//
__int64 __fastcall sub_B348A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v5; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v6; // [rsp+8h] [rbp-48h]
  int v7; // [rsp+Ch] [rbp-44h]
  _QWORD v8[2]; // [rsp+10h] [rbp-40h] BYREF
  _BYTE v9[32]; // [rsp+20h] [rbp-30h] BYREF
  __int16 v10; // [rsp+40h] [rbp-10h]

  v10 = 257;
  v3 = *(_QWORD *)(a3 + 8);
  v7 = 0;
  v5 = v3;
  v8[0] = a2;
  v8[1] = a3;
  return sub_B33D10(a1, 0x185u, (__int64)&v5, 1, (int)v8, 2, v6, (__int64)v9);
}
