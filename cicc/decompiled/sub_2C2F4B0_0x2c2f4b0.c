// Function: sub_2C2F4B0
// Address: 0x2c2f4b0
//
_QWORD *__fastcall sub_2C2F4B0(_QWORD *a1, __int64 a2)
{
  _QWORD v3[3]; // [rsp+0h] [rbp-230h] BYREF
  int v4; // [rsp+18h] [rbp-218h]
  char v5; // [rsp+1Ch] [rbp-214h]
  char v6; // [rsp+20h] [rbp-210h] BYREF
  unsigned __int64 v7[4]; // [rsp+60h] [rbp-1D0h] BYREF
  _QWORD v8[16]; // [rsp+80h] [rbp-1B0h] BYREF
  _QWORD v9[16]; // [rsp+100h] [rbp-130h] BYREF
  _QWORD v10[3]; // [rsp+180h] [rbp-B0h] BYREF
  char v11; // [rsp+198h] [rbp-98h]

  v5 = 1;
  memset(v8, 0, 0x78u);
  LODWORD(v8[2]) = 8;
  v8[1] = &v8[4];
  BYTE4(v8[3]) = 1;
  v3[0] = 0;
  v3[1] = &v6;
  v3[2] = 8;
  v4 = 0;
  memset(v7, 0, 24);
  sub_AE6EC0((__int64)v3, a2);
  v10[0] = a2;
  v11 = 0;
  sub_2C2F460(v7, (__int64)v10);
  sub_2C2B3B0(v10, v8);
  sub_2C2B3B0(v9, v3);
  sub_2C2B3B0(a1, v9);
  sub_2C2B3B0(a1 + 15, v10);
  sub_2AB1B50((__int64)v9);
  sub_2AB1B50((__int64)v10);
  sub_2AB1B50((__int64)v3);
  sub_2AB1B50((__int64)v8);
  return a1;
}
