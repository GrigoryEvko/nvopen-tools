// Function: sub_2E237C0
// Address: 0x2e237c0
//
void __fastcall sub_2E237C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD v6[21]; // [rsp+0h] [rbp-E0h] BYREF
  int v7; // [rsp+A8h] [rbp-38h] BYREF
  __int64 v8; // [rsp+B0h] [rbp-30h]
  int *v9; // [rsp+B8h] [rbp-28h]
  int *v10; // [rsp+C0h] [rbp-20h]
  __int64 v11; // [rsp+C8h] [rbp-18h]

  v7 = 0;
  memset(v6, 0, 0xA0u);
  v6[12] = 1;
  v6[3] = &v6[5];
  v6[4] = 0x400000000LL;
  v6[9] = &v6[11];
  v6[13] = &v6[19];
  v6[14] = 1;
  v6[17] = 1065353216;
  v8 = 0;
  v9 = &v7;
  v10 = &v7;
  v11 = 0;
  sub_2E23270(a1 + 200, (__int64)v6, a3, 0, a1, a6);
  sub_2E22D50((__int64)v6);
}
