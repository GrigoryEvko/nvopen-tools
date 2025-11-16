// Function: sub_29B77F0
// Address: 0x29b77f0
//
__int64 __fastcall sub_29B77F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v4; // [rsp+0h] [rbp-70h] BYREF
  __int64 v5; // [rsp+8h] [rbp-68h]
  __int64 v6; // [rsp+10h] [rbp-60h]
  __int64 v7; // [rsp+18h] [rbp-58h]
  __int64 *v8; // [rsp+20h] [rbp-50h]
  __int64 v9; // [rsp+28h] [rbp-48h]
  __int64 v10; // [rsp+30h] [rbp-40h] BYREF
  __int64 v11; // [rsp+38h] [rbp-38h]
  __int64 v12; // [rsp+40h] [rbp-30h]
  __int64 v13; // [rsp+48h] [rbp-28h]
  _BYTE *v14; // [rsp+50h] [rbp-20h]
  __int64 v15; // [rsp+58h] [rbp-18h]
  _BYTE v16[16]; // [rsp+60h] [rbp-10h] BYREF

  v4 = 0;
  v5 = 0;
  v6 = 0;
  v7 = 0;
  v8 = &v10;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  v14 = v16;
  v15 = 0;
  v2 = sub_29B4E00(a1, a2, (__int64)&v4, (__int64)&v10);
  if ( v14 != v16 )
    _libc_free((unsigned __int64)v14);
  sub_C7D6A0(v11, 8LL * (unsigned int)v13, 8);
  if ( v8 != &v10 )
    _libc_free((unsigned __int64)v8);
  sub_C7D6A0(v5, 8LL * (unsigned int)v7, 8);
  return v2;
}
