// Function: sub_276A0C0
// Address: 0x276a0c0
//
_QWORD *__fastcall sub_276A0C0(__int64 *a1, _QWORD *a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rsi
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // r15
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // r11
  __int64 v12; // r10
  __int64 v13; // r9
  __int64 v14; // r8
  __int64 v15; // rax
  _QWORD v17[4]; // [rsp+0h] [rbp-B0h] BYREF
  _QWORD v18[4]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v19[4]; // [rsp+40h] [rbp-70h] BYREF
  _QWORD v20[10]; // [rsp+60h] [rbp-50h] BYREF

  v3 = ((__int64)(a2[6] - a2[7]) >> 3) + ((((__int64)(a2[9] - a2[5]) >> 3) - 1) << 6);
  v4 = a2[4] - a2[2];
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  a1[5] = 0;
  a1[6] = 0;
  a1[7] = 0;
  a1[8] = 0;
  a1[9] = 0;
  sub_2768EA0(a1, v3 + (v4 >> 3));
  v5 = a2[2];
  v6 = a2[3];
  v7 = a2[4];
  v8 = a1[2];
  v9 = a1[3];
  v19[3] = a2[9];
  v10 = a1[4];
  v11 = a1[5];
  v18[2] = v7;
  v12 = a2[6];
  v13 = a2[7];
  v20[0] = v8;
  v14 = a2[8];
  v15 = a2[5];
  v20[1] = v9;
  v18[0] = v5;
  v18[1] = v6;
  v20[2] = v10;
  v20[3] = v11;
  v19[0] = v12;
  v19[1] = v13;
  v19[2] = v14;
  v18[3] = v15;
  return sub_2769DA0(v17, (__int64)v18, v19, (__int64)v20);
}
