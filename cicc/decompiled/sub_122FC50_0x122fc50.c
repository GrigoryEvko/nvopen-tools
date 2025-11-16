// Function: sub_122FC50
// Address: 0x122fc50
//
__int64 __fastcall sub_122FC50(__int64 **a1, _QWORD *a2, __int64 *a3)
{
  int v3; // r13d
  int v6; // eax
  int v7; // edx
  unsigned int v8; // r13d
  __int64 v9; // rax
  __int64 v10; // rsi
  const char *v11; // rdi
  unsigned __int64 v13; // rsi
  int v14; // r14d
  _QWORD *v15; // r12
  __int64 *v16; // [rsp+8h] [rbp-D8h]
  __int64 v17; // [rsp+10h] [rbp-D0h]
  __int64 v18; // [rsp+18h] [rbp-C8h]
  __int64 v19; // [rsp+28h] [rbp-B8h] BYREF
  _BYTE v20[32]; // [rsp+30h] [rbp-B0h] BYREF
  __int16 v21; // [rsp+50h] [rbp-90h]
  const char *v22; // [rsp+60h] [rbp-80h] BYREF
  __int64 v23; // [rsp+68h] [rbp-78h]
  _BYTE v24[112]; // [rsp+70h] [rbp-70h] BYREF

  v19 = 0;
  v6 = sub_120AFE0((__int64)a1, 58, "expected 'within' after cleanuppad");
  if ( (_BYTE)v6 )
    return 1;
  v7 = *((_DWORD *)a1 + 60);
  LOBYTE(v3) = v7 != 510;
  LOBYTE(v6) = v7 != 55;
  v8 = v6 & v3;
  LOBYTE(v8) = (v7 != 504) & v8;
  if ( (_BYTE)v8 )
  {
    v13 = (unsigned __int64)a1[29];
    v24[17] = 1;
    v22 = "expected scope value for cleanuppad";
    v24[16] = 3;
    sub_11FD800((__int64)(a1 + 22), v13, (__int64)&v22, 1);
    return v8;
  }
  v9 = sub_BCB190(*a1);
  if ( (unsigned __int8)sub_1224B80(a1, v9, &v19, a3) )
    return 1;
  v10 = (__int64)&v22;
  v22 = v24;
  v23 = 0x800000000LL;
  v8 = sub_122F930((__int64)a1, (__int64)&v22, a3);
  if ( (_BYTE)v8 )
  {
    v11 = v22;
    if ( v22 != v24 )
LABEL_6:
      _libc_free(v11, v10);
  }
  else
  {
    v21 = 257;
    v16 = (__int64 *)v22;
    v14 = v23 + 1;
    v17 = (unsigned int)v23;
    v10 = (unsigned int)(v23 + 1);
    v18 = v19;
    v15 = sub_BD2C40(72, v10);
    if ( v15 )
    {
      v10 = 51;
      sub_B4C840((__int64)v15, 51, v18, v16, v17, v14 & 0x7FFFFFF, (__int64)v20, 0, 0);
    }
    v11 = v22;
    *a2 = v15;
    if ( v11 != v24 )
      goto LABEL_6;
  }
  return v8;
}
