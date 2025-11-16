// Function: sub_33CA4F0
// Address: 0x33ca4f0
//
__int64 __fastcall sub_33CA4F0(__int64 a1, __int64 *a2, unsigned int a3, unsigned int a4)
{
  __int64 v7; // [rsp+0h] [rbp-40h] BYREF
  int v8; // [rsp+8h] [rbp-38h]
  __int64 v9; // [rsp+10h] [rbp-30h] BYREF
  int v10; // [rsp+18h] [rbp-28h]

  sub_C440A0((__int64)&v9, a2 + 2, a3, a4);
  sub_C440A0((__int64)&v7, a2, a3, a4);
  *(_DWORD *)(a1 + 8) = v8;
  *(_QWORD *)a1 = v7;
  *(_DWORD *)(a1 + 24) = v10;
  *(_QWORD *)(a1 + 16) = v9;
  return a1;
}
