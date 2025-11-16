// Function: sub_A7F8B0
// Address: 0xa7f8b0
//
__int64 __fastcall sub_A7F8B0(__int64 a1, __int64 a2, int a3)
{
  int v5; // edx
  __int64 v6; // rdx
  __int64 v7; // rax
  unsigned int v9; // [rsp+8h] [rbp-58h]
  _QWORD v10[2]; // [rsp+10h] [rbp-50h] BYREF
  _BYTE v11[32]; // [rsp+20h] [rbp-40h] BYREF
  __int16 v12; // [rsp+40h] [rbp-20h]

  v5 = *(_DWORD *)(a2 + 4);
  v12 = 257;
  v6 = v5 & 0x7FFFFFF;
  v10[0] = *(_QWORD *)(a2 - 32 * v6);
  v10[1] = *(_QWORD *)(a2 + 32 * (1 - v6));
  v7 = sub_B33D10(a1, a3, 0, 0, (unsigned int)v10, 2, v9, (__int64)v11);
  return sub_A7EE20(
           a1,
           *(_BYTE **)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
           v7,
           *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
}
