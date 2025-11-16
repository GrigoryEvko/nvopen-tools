// Function: sub_33CEDC0
// Address: 0x33cedc0
//
__int64 __fastcall sub_33CEDC0(__int64 a1, int a2, unsigned __int64 a3, int a4, unsigned __int64 *a5, __int64 a6)
{
  _QWORD *v6; // r12
  __int64 v9; // [rsp-D0h] [rbp-D0h] BYREF
  __int64 v10; // [rsp-C8h] [rbp-C8h] BYREF
  int v11; // [rsp-C0h] [rbp-C0h]
  unsigned __int64 v12[2]; // [rsp-B8h] [rbp-B8h] BYREF
  _BYTE v13[168]; // [rsp-A8h] [rbp-A8h] BYREF

  if ( *(_WORD *)(a3 + 16LL * (unsigned int)(a4 - 1)) == 262 )
    return 0;
  v12[1] = 0x2000000000LL;
  v12[0] = (unsigned __int64)v13;
  sub_33C9670((__int64)v12, a2, a3, a5, a6, 0);
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v6 = sub_33CCCF0(a1, (__int64)v12, (__int64)&v10, &v9);
  if ( v10 )
    sub_B91220((__int64)&v10, v10);
  if ( v6 )
  {
    if ( (_BYTE *)v12[0] != v13 )
      _libc_free(v12[0]);
    return 1;
  }
  else
  {
    if ( (_BYTE *)v12[0] != v13 )
      _libc_free(v12[0]);
    return 0;
  }
}
