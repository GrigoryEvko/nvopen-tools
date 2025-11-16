// Function: sub_1916670
// Address: 0x1916670
//
__int64 __fastcall sub_1916670(__int64 a1, int a2, unsigned int a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r12d
  _BYTE v7[24]; // [rsp+0h] [rbp-50h] BYREF
  char *v8; // [rsp+18h] [rbp-38h]
  char v9; // [rsp+28h] [rbp-28h] BYREF

  sub_1913110((__int64)v7, a1, a2, a3, a4, a5);
  v5 = sub_1911DB0(a1, (__int64)v7);
  if ( v8 != &v9 )
    _libc_free((unsigned __int64)v8);
  return v5;
}
