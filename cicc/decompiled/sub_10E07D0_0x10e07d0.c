// Function: sub_10E07D0
// Address: 0x10e07d0
//
__int64 __fastcall sub_10E07D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rcx
  __int64 v7; // rdx
  _QWORD v9[2]; // [rsp+20h] [rbp-20h] BYREF
  _QWORD v10[2]; // [rsp+30h] [rbp-10h] BYREF

  v6 = *(_QWORD *)(a2 + 8);
  v10[1] = a3;
  v7 = *(_QWORD *)(a3 + 8);
  v10[0] = a2;
  v9[0] = v6;
  v9[1] = v7;
  return sub_B33D10(a1, 0xD1u, (__int64)v9, 2, (int)v10, 2, a4, a5);
}
