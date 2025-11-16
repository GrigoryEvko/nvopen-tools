// Function: sub_DAC8D0
// Address: 0xdac8d0
//
__int64 __fastcall sub_DAC8D0(__int64 a1, _BYTE *a2)
{
  __int64 v2; // rsi
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 result; // rax
  __int64 v7; // rdi
  _BYTE *v8; // [rsp-168h] [rbp-168h] BYREF
  __int64 v9; // [rsp-160h] [rbp-160h]
  _BYTE v10[64]; // [rsp-158h] [rbp-158h] BYREF
  __int64 v11; // [rsp-118h] [rbp-118h] BYREF
  _QWORD *v12; // [rsp-110h] [rbp-110h]
  __int64 v13; // [rsp-108h] [rbp-108h]
  int v14; // [rsp-100h] [rbp-100h]
  char v15; // [rsp-FCh] [rbp-FCh]
  _BYTE *v16; // [rsp-F8h] [rbp-F8h] BYREF
  _QWORD v17[2]; // [rsp-B8h] [rbp-B8h] BYREF
  _QWORD v18[21]; // [rsp-A8h] [rbp-A8h] BYREF

  if ( *a2 > 0x1Cu )
  {
    v12 = &v16;
    v9 = 0x800000000LL;
    v18[0] = a2;
    v17[1] = 0x1000000001LL;
    v16 = a2;
    v13 = 0x100000008LL;
    v17[0] = v18;
    v8 = v10;
    v14 = 0;
    v15 = 1;
    v11 = 1;
    sub_D988C0(a1, (__int64)v17, (__int64)&v11, (__int64)&v8);
    v2 = (__int64)v8;
    result = sub_DAB940(a1, (__int64)v8, (unsigned int)v9, v3, v4, v5);
    if ( v8 != v10 )
      result = _libc_free(v8, v2);
    if ( v15 )
    {
      v7 = v17[0];
      if ( (_QWORD *)v17[0] == v18 )
        return result;
    }
    else
    {
      result = _libc_free(v12, v2);
      v7 = v17[0];
      if ( (_QWORD *)v17[0] == v18 )
        return result;
    }
    return _libc_free(v7, v2);
  }
  return result;
}
