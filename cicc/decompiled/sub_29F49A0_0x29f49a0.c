// Function: sub_29F49A0
// Address: 0x29f49a0
//
void __fastcall sub_29F49A0(char **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  bool v7; // zf
  __int64 v8; // rdx
  char *v9; // r8
  __int64 v10; // r9
  char *v11; // [rsp+10h] [rbp-170h] BYREF
  __int64 v12; // [rsp+18h] [rbp-168h]
  __int64 v13; // [rsp+20h] [rbp-160h]
  _BYTE v14[136]; // [rsp+28h] [rbp-158h] BYREF
  unsigned __int64 v15[3]; // [rsp+B0h] [rbp-D0h] BYREF
  _BYTE v16[184]; // [rsp+C8h] [rbp-B8h] BYREF

  v7 = *(_QWORD *)(a3 + 8) == 0;
  v11 = v14;
  v12 = 0;
  v13 = 128;
  if ( !v7 )
    sub_29F3DD0((__int64)&v11, (char **)a3, a3, a4, a5, a6);
  sub_29F3DD0(a3, a1, a3, a4, a5, a6);
  v15[0] = (unsigned __int64)v16;
  v15[1] = 0;
  v15[2] = 128;
  if ( v12 )
    sub_29F3DD0((__int64)v15, &v11, v8, (__int64)v15, (__int64)v9, v10);
  sub_29F46E0((__int64)a1, 0, (char *)(0x86BCA1AF286BCA1BLL * ((a2 - (__int64)a1) >> 3)), (__int64)v15, v9, v10);
  if ( (_BYTE *)v15[0] != v16 )
    _libc_free(v15[0]);
  if ( v11 != v14 )
    _libc_free((unsigned __int64)v11);
}
