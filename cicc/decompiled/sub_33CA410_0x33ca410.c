// Function: sub_33CA410
// Address: 0x33ca410
//
__int64 __fastcall sub_33CA410(__int64 a1, const void **a2, unsigned int a3)
{
  __int64 v5; // [rsp+0h] [rbp-40h] BYREF
  int v6; // [rsp+8h] [rbp-38h]
  __int64 v7; // [rsp+10h] [rbp-30h] BYREF
  int v8; // [rsp+18h] [rbp-28h]

  sub_C449B0((__int64)&v7, a2 + 2, a3);
  sub_C449B0((__int64)&v5, a2, a3);
  *(_DWORD *)(a1 + 8) = v6;
  *(_QWORD *)a1 = v5;
  *(_DWORD *)(a1 + 24) = v8;
  *(_QWORD *)(a1 + 16) = v7;
  return a1;
}
