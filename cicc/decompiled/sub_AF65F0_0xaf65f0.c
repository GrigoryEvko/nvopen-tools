// Function: sub_AF65F0
// Address: 0xaf65f0
//
__int64 __fastcall sub_AF65F0(__int64 a1, char a2, __int64 a3, char a4)
{
  __int64 v6; // rsi
  unsigned int v7; // r15d
  _BYTE *v8; // r12
  size_t v9; // rdx
  void *s1; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v12; // [rsp+8h] [rbp-A8h]
  _BYTE v13[48]; // [rsp+10h] [rbp-A0h] BYREF
  void *s2; // [rsp+40h] [rbp-70h] BYREF
  __int64 v15; // [rsp+48h] [rbp-68h]
  _BYTE v16[96]; // [rsp+50h] [rbp-60h] BYREF

  v12 = 0x600000000LL;
  s1 = v13;
  sub_AF6390((__int64)&s1, a1, a2);
  v6 = a3;
  v15 = 0x600000000LL;
  v7 = 0;
  s2 = v16;
  sub_AF6390((__int64)&s2, a3, a4);
  v8 = s2;
  if ( (unsigned int)v12 == (unsigned __int64)(unsigned int)v15 )
  {
    v9 = 8LL * (unsigned int)v12;
    v7 = 1;
    if ( v9 )
    {
      v6 = (__int64)s2;
      LOBYTE(v7) = memcmp(s1, s2, v9) == 0;
    }
  }
  if ( v8 != v16 )
    _libc_free(v8, v6);
  if ( s1 != v13 )
    _libc_free(s1, v6);
  return v7;
}
