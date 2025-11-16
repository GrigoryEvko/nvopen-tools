// Function: sub_2E236E0
// Address: 0x2e236e0
//
__int64 __fastcall sub_2E236E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD v7[21]; // [rsp+0h] [rbp-F0h] BYREF
  int v8; // [rsp+A8h] [rbp-48h] BYREF
  __int64 v9; // [rsp+B0h] [rbp-40h]
  int *v10; // [rsp+B8h] [rbp-38h]
  int *v11; // [rsp+C0h] [rbp-30h]
  __int64 v12; // [rsp+C8h] [rbp-28h]

  v8 = 0;
  memset(v7, 0, 0xA0u);
  v7[12] = 1;
  v7[3] = &v7[5];
  v7[4] = 0x400000000LL;
  v7[9] = &v7[11];
  v7[13] = &v7[19];
  v7[14] = 1;
  v7[17] = 1065353216;
  v9 = 0;
  v10 = &v8;
  v11 = &v8;
  v12 = 0;
  sub_2E23270(a1 + 200, (__int64)v7, a3, 0, a5, a6);
  sub_2E22D50((__int64)v7);
  sub_2E23240((__int64 *)(a1 + 200), a2);
  return 0;
}
