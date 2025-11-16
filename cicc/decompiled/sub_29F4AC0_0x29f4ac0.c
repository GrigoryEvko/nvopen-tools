// Function: sub_29F4AC0
// Address: 0x29f4ac0
//
void __fastcall sub_29F4AC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char *a5, __int64 a6)
{
  __int64 v6; // rsi
  char *v7; // rsi
  __int64 v8; // rbx
  __int64 i; // r12
  bool v10; // zf
  __int64 v11; // rdx
  __int64 v12; // rcx
  char *v13; // [rsp-178h] [rbp-178h] BYREF
  __int64 v14; // [rsp-170h] [rbp-170h]
  __int64 v15; // [rsp-168h] [rbp-168h]
  _BYTE v16[136]; // [rsp-160h] [rbp-160h] BYREF
  _BYTE *v17; // [rsp-D8h] [rbp-D8h] BYREF
  __int64 v18; // [rsp-D0h] [rbp-D0h]
  __int64 v19; // [rsp-C8h] [rbp-C8h]
  _BYTE v20[192]; // [rsp-C0h] [rbp-C0h] BYREF

  v6 = a2 - a1;
  if ( v6 > 152 )
  {
    v7 = (char *)(0x86BCA1AF286BCA1BLL * (v6 >> 3));
    v8 = (__int64)(v7 - 2) / 2;
    for ( i = a1 + 152 * v8; ; i -= 152 )
    {
      v10 = *(_QWORD *)(i + 8) == 0;
      v14 = 0;
      v15 = 128;
      v13 = v16;
      if ( v10 )
      {
        v17 = v20;
        v18 = 0;
        v19 = 128;
      }
      else
      {
        sub_29F3DD0((__int64)&v13, (char **)i, a3, a4, (__int64)a5, a6);
        v17 = v20;
        v18 = 0;
        v19 = 128;
        if ( v14 )
          sub_29F3DD0((__int64)&v17, &v13, v11, v12, (__int64)a5, a6);
      }
      sub_29F46E0(a1, v8, v7, (__int64)&v17, a5, a6);
      if ( v17 != v20 )
        _libc_free((unsigned __int64)v17);
      if ( !v8 )
        break;
      --v8;
      if ( v13 != v16 )
        _libc_free((unsigned __int64)v13);
    }
    if ( v13 != v16 )
      _libc_free((unsigned __int64)v13);
  }
}
