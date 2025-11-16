// Function: sub_E1D0B0
// Address: 0xe1d0b0
//
_BYTE *__fastcall sub_E1D0B0(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  _BYTE *v3; // r13
  __int64 v4; // rsi
  _BYTE *v5; // rax
  _QWORD *i; // rdi
  _QWORD *v7; // rdx
  _QWORD v9[4]; // [rsp+20h] [rbp-13A0h] BYREF
  int v10; // [rsp+40h] [rbp-1380h]
  _QWORD v11[2]; // [rsp+50h] [rbp-1370h] BYREF
  _BYTE *v12; // [rsp+60h] [rbp-1360h]
  _BYTE *v13; // [rsp+68h] [rbp-1358h]
  _QWORD *v14; // [rsp+70h] [rbp-1350h]
  _BYTE v15[256]; // [rsp+78h] [rbp-1348h] BYREF
  _QWORD v16[3]; // [rsp+178h] [rbp-1248h] BYREF
  _BYTE v17[256]; // [rsp+190h] [rbp-1230h] BYREF
  _QWORD v18[3]; // [rsp+290h] [rbp-1130h] BYREF
  _OWORD v19[4]; // [rsp+2A8h] [rbp-1118h] BYREF
  _QWORD v20[3]; // [rsp+2E8h] [rbp-10D8h] BYREF
  _OWORD v21[2]; // [rsp+300h] [rbp-10C0h] BYREF
  _QWORD v22[3]; // [rsp+320h] [rbp-10A0h] BYREF
  _OWORD v23[2]; // [rsp+338h] [rbp-1088h] BYREF
  __int16 v24; // [rsp+358h] [rbp-1068h] BYREF
  char v25; // [rsp+35Ah] [rbp-1066h]
  __int64 v26; // [rsp+360h] [rbp-1060h]
  __int64 v27; // [rsp+368h] [rbp-1058h]
  int v28; // [rsp+370h] [rbp-1050h]
  _QWORD v29[512]; // [rsp+380h] [rbp-1040h] BYREF
  _QWORD *v30; // [rsp+1380h] [rbp-40h]

  v3 = 0;
  if ( a1 )
  {
    v11[0] = a2;
    v14 = v16;
    memset(v15, 0, sizeof(v15));
    v16[2] = v18;
    memset(v17, 0, sizeof(v17));
    v11[1] = a1 + a2;
    v4 = a3;
    v18[0] = v19;
    v18[1] = v19;
    v18[2] = v20;
    v20[0] = v21;
    v20[1] = v21;
    v20[2] = v22;
    v22[2] = &v24;
    v24 = 1;
    memset(v19, 0, sizeof(v19));
    memset(v21, 0, sizeof(v21));
    memset(v23, 0, sizeof(v23));
    v12 = v15;
    v13 = v15;
    v16[0] = v17;
    v16[1] = v17;
    v22[0] = v23;
    v22[1] = v23;
    v25 = 0;
    v26 = -1;
    v27 = 0;
    v28 = 0;
    v29[0] = 0;
    v29[1] = 0;
    v30 = v29;
    v5 = (_BYTE *)sub_E1CEE0((__int64)v11, a3);
    v3 = v5;
    if ( v5 )
    {
      memset(v9, 0, 24);
      v9[3] = -1;
      v10 = 1;
      sub_E15BE0(v5, (__int64)v9);
      v4 = 0;
      sub_E14360((__int64)v9, 0);
      v3 = (_BYTE *)v9[0];
    }
LABEL_4:
    for ( i = v30; i; i = v7 )
    {
      v7 = (_QWORD *)*i;
      v30 = (_QWORD *)*i;
      if ( i != v29 )
      {
        _libc_free(i, v4);
        goto LABEL_4;
      }
    }
    if ( (_OWORD *)v22[0] != v23 )
      _libc_free(v22[0], v4);
    if ( (_OWORD *)v20[0] != v21 )
      _libc_free(v20[0], v4);
    if ( (_OWORD *)v18[0] != v19 )
      _libc_free(v18[0], v4);
    if ( (_BYTE *)v16[0] != v17 )
      _libc_free(v16[0], v4);
    if ( v12 != v15 )
      _libc_free(v12, v4);
  }
  return v3;
}
