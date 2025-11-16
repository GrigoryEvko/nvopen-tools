// Function: sub_2C2B410
// Address: 0x2c2b410
//
__int64 __fastcall sub_2C2B410(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  _QWORD v10[16]; // [rsp+20h] [rbp-330h] BYREF
  _QWORD v11[16]; // [rsp+A0h] [rbp-2B0h] BYREF
  _QWORD v12[15]; // [rsp+120h] [rbp-230h] BYREF
  __int16 v13; // [rsp+198h] [rbp-1B8h]
  _QWORD v14[15]; // [rsp+1A0h] [rbp-1B0h] BYREF
  __int16 v15; // [rsp+218h] [rbp-138h]
  _QWORD v16[15]; // [rsp+220h] [rbp-130h] BYREF
  __int16 v17; // [rsp+298h] [rbp-B8h]
  _QWORD v18[15]; // [rsp+2A0h] [rbp-B0h] BYREF
  __int16 v19; // [rsp+318h] [rbp-38h]

  sub_2ABCC20(v11, a2 + 120, a3, a4, a5, a6);
  sub_2C2B3B0(v16, v11);
  sub_2C2B3B0(v18, v16);
  sub_2C2B3B0(v14, v18);
  sub_2AB1B50((__int64)v18);
  v15 = 256;
  sub_2AB1B50((__int64)v16);
  sub_2ABCC20(v10, a2, v6, v7, v8, (__int64)v10);
  sub_2C2B3B0(v16, v10);
  sub_2C2B3B0(v18, v16);
  sub_2C2B3B0(v12, v18);
  sub_2AB1B50((__int64)v18);
  v13 = 256;
  sub_2AB1B50((__int64)v16);
  sub_2C2B3B0(v18, v14);
  v19 = v15;
  sub_2C2B3B0(v16, v12);
  v17 = v13;
  sub_2C2B3B0((_QWORD *)a1, v16);
  *(_WORD *)(a1 + 120) = v17;
  sub_2C2B3B0((_QWORD *)(a1 + 128), v18);
  *(_WORD *)(a1 + 248) = v19;
  sub_2AB1B50((__int64)v16);
  sub_2AB1B50((__int64)v18);
  sub_2AB1B50((__int64)v12);
  sub_2AB1B50((__int64)v10);
  sub_2AB1B50((__int64)v14);
  sub_2AB1B50((__int64)v11);
  return a1;
}
