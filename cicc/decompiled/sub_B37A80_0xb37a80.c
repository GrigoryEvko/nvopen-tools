// Function: sub_B37A80
// Address: 0xb37a80
//
__int64 __fastcall sub_B37A80(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v7; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-68h]
  int v9; // [rsp+Ch] [rbp-64h]
  _QWORD v10[2]; // [rsp+10h] [rbp-60h] BYREF
  _BYTE v11[32]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v12; // [rsp+40h] [rbp-30h]

  v4 = sub_BCB2D0(*(_QWORD *)(a1 + 72));
  v10[0] = a2;
  v10[1] = sub_ACD640(v4, a3, 0);
  v5 = *(_QWORD *)(a2 + 8);
  v12 = 257;
  v7 = v5;
  v9 = 0;
  return sub_B33D10(a1, 0xCFu, (__int64)&v7, 1, (int)v10, 2, v8, (__int64)v11);
}
