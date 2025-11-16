// Function: sub_2AC67F0
// Address: 0x2ac67f0
//
_QWORD *__fastcall sub_2AC67F0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD v14[3]; // [rsp+10h] [rbp-230h] BYREF
  int v15; // [rsp+28h] [rbp-218h]
  char v16; // [rsp+2Ch] [rbp-214h]
  _BYTE v17[64]; // [rsp+30h] [rbp-210h] BYREF
  unsigned __int64 v18; // [rsp+70h] [rbp-1D0h] BYREF
  __int64 v19; // [rsp+78h] [rbp-1C8h]
  __int64 v20; // [rsp+80h] [rbp-1C0h]
  _QWORD v21[16]; // [rsp+90h] [rbp-1B0h] BYREF
  _BYTE v22[32]; // [rsp+110h] [rbp-130h] BYREF
  _BYTE v23[64]; // [rsp+130h] [rbp-110h] BYREF
  unsigned __int64 v24; // [rsp+170h] [rbp-D0h]
  __int64 v25; // [rsp+178h] [rbp-C8h]
  __int64 v26; // [rsp+180h] [rbp-C0h]
  _QWORD v27[3]; // [rsp+190h] [rbp-B0h] BYREF
  char v28; // [rsp+1A8h] [rbp-98h]
  _BYTE v29[64]; // [rsp+1B0h] [rbp-90h] BYREF
  __int64 v30; // [rsp+1F0h] [rbp-50h]
  __int64 v31; // [rsp+1F8h] [rbp-48h]
  __int64 v32; // [rsp+200h] [rbp-40h]

  v14[1] = v17;
  memset(v21, 0, 0x78u);
  v21[1] = &v21[4];
  LODWORD(v21[2]) = 8;
  BYTE4(v21[3]) = 1;
  v14[0] = 0;
  v14[2] = 8;
  v15 = 0;
  v16 = 1;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  sub_AE6EC0((__int64)v14, a2);
  v27[0] = a2;
  v28 = 0;
  sub_2AC67A0(&v18, (__int64)v27);
  sub_C8CF70((__int64)v27, v29, 8, (__int64)&v21[4], (__int64)v21);
  v3 = v21[12];
  memset(&v21[12], 0, 24);
  v30 = v3;
  v31 = v21[13];
  v32 = v21[14];
  sub_C8CF70((__int64)v22, v23, 8, (__int64)v17, (__int64)v14);
  v4 = v18;
  v18 = 0;
  v24 = v4;
  v5 = v19;
  v19 = 0;
  v25 = v5;
  v6 = v20;
  v20 = 0;
  v26 = v6;
  sub_C8CF70((__int64)a1, a1 + 4, 8, (__int64)v23, (__int64)v22);
  v7 = v24;
  v24 = 0;
  a1[12] = v7;
  v8 = v25;
  v25 = 0;
  a1[13] = v8;
  v9 = v26;
  v26 = 0;
  a1[14] = v9;
  sub_C8CF70((__int64)(a1 + 15), a1 + 19, 8, (__int64)v29, (__int64)v27);
  v10 = v30;
  v30 = 0;
  a1[27] = v10;
  v11 = v31;
  v31 = 0;
  a1[28] = v11;
  v12 = v32;
  v32 = 0;
  a1[29] = v12;
  sub_2AB1B50((__int64)v22);
  sub_2AB1B50((__int64)v27);
  sub_2AB1B50((__int64)v14);
  sub_2AB1B50((__int64)v21);
  return a1;
}
