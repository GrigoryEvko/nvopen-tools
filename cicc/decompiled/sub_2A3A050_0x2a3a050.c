// Function: sub_2A3A050
// Address: 0x2a3a050
//
__int64 __fastcall sub_2A3A050(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  _BYTE v4[32]; // [rsp+0h] [rbp-230h] BYREF
  _QWORD v5[7]; // [rsp+20h] [rbp-210h] BYREF
  char v6; // [rsp+58h] [rbp-1D8h] BYREF
  char *v7; // [rsp+60h] [rbp-1D0h]
  __int64 v8; // [rsp+68h] [rbp-1C8h]
  char v9; // [rsp+70h] [rbp-1C0h] BYREF
  char *v10; // [rsp+A0h] [rbp-190h]
  __int64 v11; // [rsp+A8h] [rbp-188h]
  char v12; // [rsp+B0h] [rbp-180h] BYREF
  char *v13; // [rsp+D0h] [rbp-160h]
  __int64 v14; // [rsp+D8h] [rbp-158h]
  char v15; // [rsp+E0h] [rbp-150h] BYREF
  char *v16; // [rsp+130h] [rbp-100h]
  __int64 v17; // [rsp+138h] [rbp-F8h]
  char v18; // [rsp+140h] [rbp-F0h] BYREF
  char *v19; // [rsp+1E0h] [rbp-50h]
  __int64 v20; // [rsp+1E8h] [rbp-48h]
  char v21; // [rsp+1F0h] [rbp-40h] BYREF
  __int16 v22; // [rsp+200h] [rbp-30h]
  __int64 v23; // [rsp+208h] [rbp-28h]

  v1 = sub_B43CC0(a1);
  memset(v5, 0, 32);
  v21 = 0;
  v5[4] = &v6;
  v7 = &v9;
  v8 = 0x600000000LL;
  v10 = &v12;
  v11 = 0x400000000LL;
  v13 = &v15;
  v14 = 0xA00000000LL;
  v16 = &v18;
  v17 = 0x800000000LL;
  v19 = &v21;
  v5[5] = 0;
  v5[6] = 8;
  v20 = 0;
  v22 = 768;
  v23 = 0;
  sub_AE1EA0((__int64)v5, v1);
  sub_B4CED0((__int64)v4, a1, (__int64)v5);
  v2 = sub_CA1930(v4);
  sub_AE4030(v5, a1);
  return v2;
}
