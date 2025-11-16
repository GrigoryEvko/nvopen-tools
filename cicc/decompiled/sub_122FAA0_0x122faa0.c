// Function: sub_122FAA0
// Address: 0x122faa0
//
__int64 __fastcall sub_122FAA0(__int64 **a1, _QWORD *a2, __int64 *a3)
{
  unsigned int v3; // r13d
  __int64 v6; // rax
  unsigned __int64 v8; // rsi
  __int64 v9; // rsi
  int v10; // r14d
  _QWORD *v11; // r12
  __int64 *v12; // [rsp+8h] [rbp-D8h]
  __int64 v13; // [rsp+10h] [rbp-D0h]
  __int64 v14; // [rsp+18h] [rbp-C8h]
  __int64 v15; // [rsp+28h] [rbp-B8h] BYREF
  _BYTE v16[32]; // [rsp+30h] [rbp-B0h] BYREF
  __int16 v17; // [rsp+50h] [rbp-90h]
  const char *v18; // [rsp+60h] [rbp-80h] BYREF
  __int64 v19; // [rsp+68h] [rbp-78h]
  _BYTE v20[112]; // [rsp+70h] [rbp-70h] BYREF

  v15 = 0;
  if ( (unsigned __int8)sub_120AFE0((__int64)a1, 58, "expected 'within' after catchpad") )
    return 1;
  LOBYTE(v3) = *((_DWORD *)a1 + 60) != 510 && *((_DWORD *)a1 + 60) != 504;
  if ( (_BYTE)v3 )
  {
    v8 = (unsigned __int64)a1[29];
    v20[17] = 1;
    v18 = "expected scope value for catchpad";
    v20[16] = 3;
    sub_11FD800((__int64)(a1 + 22), v8, (__int64)&v18, 1);
    return v3;
  }
  v6 = sub_BCB190(*a1);
  if ( (unsigned __int8)sub_1224B80(a1, v6, &v15, a3) )
  {
    return 1;
  }
  else
  {
    v9 = (__int64)&v18;
    v18 = v20;
    v19 = 0x800000000LL;
    v3 = sub_122F930((__int64)a1, (__int64)&v18, a3);
    if ( !(_BYTE)v3 )
    {
      v17 = 257;
      v12 = (__int64 *)v18;
      v10 = v19 + 1;
      v13 = (unsigned int)v19;
      v9 = (unsigned int)(v19 + 1);
      v14 = v15;
      v11 = sub_BD2C40(72, v9);
      if ( v11 )
      {
        v9 = 52;
        sub_B4C840((__int64)v11, 52, v14, v12, v13, v10 & 0x7FFFFFF, (__int64)v16, 0, 0);
      }
      *a2 = v11;
    }
    if ( v18 != v20 )
      _libc_free(v18, v9);
  }
  return v3;
}
