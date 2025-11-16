// Function: sub_D9B720
// Address: 0xd9b720
//
__int64 __fastcall sub_D9B720(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  char v8[8]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v9; // [rsp+8h] [rbp-58h]
  char *v10; // [rsp+10h] [rbp-50h]
  __int64 v11; // [rsp+18h] [rbp-48h]
  int v12; // [rsp+20h] [rbp-40h]
  char v13; // [rsp+24h] [rbp-3Ch]
  char v14; // [rsp+28h] [rbp-38h] BYREF

  v10 = &v14;
  v8[0] = 1;
  v9 = 0;
  v11 = 4;
  v12 = 0;
  v13 = 1;
  sub_D9B3F0(a1, (__int64)v8, a3, a4, a5, a6);
  LOBYTE(v6) = HIDWORD(v11) == v12;
  if ( !v13 )
    _libc_free(v10, v8);
  return v6;
}
