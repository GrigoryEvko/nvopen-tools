// Function: sub_C19720
// Address: 0xc19720
//
void __fastcall sub_C19720(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rsi
  __int64 v4; // rbx
  __int64 i; // r12
  __int64 v6; // rsi
  __int64 v7; // rax
  int v8; // edx
  __int64 v9; // [rsp+18h] [rbp-E8h]
  __int64 v11; // [rsp+30h] [rbp-D0h]
  char *v12; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v13; // [rsp+40h] [rbp-C0h]
  _BYTE v14[56]; // [rsp+48h] [rbp-B8h] BYREF
  __int64 v15; // [rsp+80h] [rbp-80h] BYREF
  _BYTE *v16; // [rsp+88h] [rbp-78h] BYREF
  __int64 v17; // [rsp+90h] [rbp-70h]
  _BYTE v18[104]; // [rsp+98h] [rbp-68h] BYREF

  v3 = a2 - a1;
  if ( v3 > 72 )
  {
    v9 = 0x8E38E38E38E38E39LL * (v3 >> 3);
    v4 = (v9 - 2) / 2;
    for ( i = a1 + 72 * v4 + 8; ; i -= 72 )
    {
      v7 = *(_QWORD *)(i - 8);
      v13 = 0xC00000000LL;
      v8 = *(_DWORD *)(i + 8);
      v11 = v7;
      v12 = v14;
      if ( v8 )
      {
        sub_C15E20((__int64)&v12, (char **)i);
        v16 = v18;
        v17 = 0xC00000000LL;
        v15 = v11;
        if ( (_DWORD)v13 )
          sub_C15E20((__int64)&v16, &v12);
      }
      else
      {
        v15 = v7;
        v16 = v18;
        v17 = 0xC00000000LL;
      }
      v6 = v4;
      sub_C19490(a1, v4, v9, &v15, *a3);
      if ( v16 != v18 )
        _libc_free(v16, v4);
      if ( !v4 )
        break;
      --v4;
      if ( v12 != v14 )
        _libc_free(v12, v6);
    }
    if ( v12 != v14 )
      _libc_free(v12, v4);
  }
}
