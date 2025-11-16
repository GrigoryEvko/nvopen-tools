// Function: sub_1287ED0
// Address: 0x1287ed0
//
__int64 __fastcall sub_1287ED0(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD v9[4]; // [rsp+0h] [rbp-90h] BYREF
  __int64 v10[14]; // [rsp+20h] [rbp-70h] BYREF

  sub_1286D80((__int64)v10, *a1, a2, a4, a5);
  sub_1287CD0(
    (__int64)v9,
    *a1,
    (_DWORD *)(a2 + 36),
    v5,
    v6,
    v7,
    v10[0],
    (_BYTE *)v10[1],
    v10[2],
    v10[3],
    v10[4],
    v10[5]);
  return v9[0];
}
