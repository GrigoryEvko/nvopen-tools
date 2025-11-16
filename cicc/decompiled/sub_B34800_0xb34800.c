// Function: sub_B34800
// Address: 0xb34800
//
__int64 __fastcall sub_B34800(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v5; // [rsp+8h] [rbp-48h] BYREF
  __int64 v6; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v7; // [rsp+18h] [rbp-38h]
  int v8; // [rsp+1Ch] [rbp-34h]
  _BYTE v9[32]; // [rsp+20h] [rbp-30h] BYREF
  __int16 v10; // [rsp+40h] [rbp-10h]

  v3 = *(_QWORD *)(a3 + 8);
  v8 = 0;
  v6 = v3;
  v10 = 257;
  v5 = a3;
  return sub_B33D10(a1, a2, (__int64)&v6, 1, (int)&v5, 1, v7, (__int64)v9);
}
