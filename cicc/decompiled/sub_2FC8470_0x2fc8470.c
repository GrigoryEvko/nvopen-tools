// Function: sub_2FC8470
// Address: 0x2fc8470
//
_QWORD *__fastcall sub_2FC8470(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r8
  __int64 v6; // r9
  bool v7; // al
  _QWORD *v8; // rsi
  _QWORD *v9; // rdx
  _QWORD v11[12]; // [rsp+0h] [rbp-5B0h] BYREF
  char v12; // [rsp+60h] [rbp-550h] BYREF
  char *v13; // [rsp+A0h] [rbp-510h]
  __int64 v14; // [rsp+A8h] [rbp-508h]
  char v15; // [rsp+B0h] [rbp-500h] BYREF
  char *v16; // [rsp+130h] [rbp-480h]
  __int64 v17; // [rsp+138h] [rbp-478h]
  char v18; // [rsp+140h] [rbp-470h] BYREF
  __int64 v19; // [rsp+440h] [rbp-170h]
  __int64 v20; // [rsp+448h] [rbp-168h]
  char *v21; // [rsp+450h] [rbp-160h]
  __int64 v22; // [rsp+458h] [rbp-158h]
  char v23; // [rsp+460h] [rbp-150h] BYREF
  _QWORD *v24; // [rsp+480h] [rbp-130h]
  __int64 v25; // [rsp+488h] [rbp-128h]
  _QWORD v26[5]; // [rsp+490h] [rbp-120h] BYREF
  char v27; // [rsp+4B8h] [rbp-F8h] BYREF
  char *v28; // [rsp+4F8h] [rbp-B8h]
  __int64 v29; // [rsp+500h] [rbp-B0h]
  char v30; // [rsp+508h] [rbp-A8h] BYREF
  int v31; // [rsp+538h] [rbp-78h]
  _BYTE *v32; // [rsp+540h] [rbp-70h]
  __int64 v33; // [rsp+548h] [rbp-68h]
  _BYTE v34[48]; // [rsp+550h] [rbp-60h] BYREF
  int v35; // [rsp+580h] [rbp-30h]

  v13 = &v15;
  v26[2] = sub_2EB2140(a4, &qword_5025C20, a3) + 8;
  v11[10] = &v12;
  v14 = 0x1000000000LL;
  v16 = &v18;
  v17 = 0x1000000000LL;
  v26[3] = &v27;
  v11[11] = 0x800000000LL;
  v21 = &v23;
  v22 = 0x400000000LL;
  v26[4] = 0x800000000LL;
  v28 = &v30;
  v32 = v34;
  v24 = v26;
  v29 = 0x600000000LL;
  v33 = 0x600000000LL;
  memset(v11, 0, 80);
  v19 = 0;
  v20 = 0;
  v25 = 0;
  v26[0] = 0;
  v26[1] = 1;
  v31 = 0;
  v35 = 0;
  v7 = sub_2FC4FC0((__int64)v11, a3, (__int64)v34, (__int64)v26, v5, v6);
  v8 = a1 + 4;
  v9 = a1 + 10;
  if ( v7 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v8;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v9;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[1] = v8;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[7] = v9;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    a1[4] = &qword_4F82400;
    *a1 = 1;
  }
  sub_2FBF560((__int64)v11);
  return a1;
}
