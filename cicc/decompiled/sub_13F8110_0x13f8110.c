// Function: sub_13F8110
// Address: 0x13f8110
//
__int64 __fastcall sub_13F8110(
        unsigned __int64 a1,
        unsigned int a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned int v6; // r12d
  __int64 v8; // [rsp+8h] [rbp-140h] BYREF
  _BYTE *v9; // [rsp+10h] [rbp-138h]
  _BYTE *v10; // [rsp+18h] [rbp-130h]
  __int64 v11; // [rsp+20h] [rbp-128h]
  int v12; // [rsp+28h] [rbp-120h]
  _BYTE v13[272]; // [rsp+30h] [rbp-118h] BYREF

  v9 = v13;
  v8 = 0;
  v10 = v13;
  v11 = 32;
  v12 = 0;
  v6 = sub_13F7530(a1, a2, a3, a4, a5, a6, (__int64)&v8);
  if ( v10 != v9 )
    _libc_free((unsigned __int64)v10);
  return v6;
}
