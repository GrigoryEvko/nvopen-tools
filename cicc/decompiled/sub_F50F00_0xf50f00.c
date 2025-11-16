// Function: sub_F50F00
// Address: 0xf50f00
//
__int64 __fastcall sub_F50F00(__int64 a1)
{
  __int64 *v2; // rbx
  __int64 *v3; // r14
  __int64 v4; // rdi
  __int64 *v5; // rdi
  __int64 *v6; // r14
  int v7; // eax
  __int64 *v8; // rbx
  __int64 v9; // rdi
  __int64 *v11; // [rsp+0h] [rbp-60h] BYREF
  __int64 v12; // [rsp+8h] [rbp-58h]
  _BYTE v13[16]; // [rsp+10h] [rbp-50h] BYREF
  __int64 *v14; // [rsp+20h] [rbp-40h] BYREF
  __int64 v15; // [rsp+28h] [rbp-38h]
  _BYTE v16[48]; // [rsp+30h] [rbp-30h] BYREF

  v12 = 0x100000000LL;
  v15 = 0x100000000LL;
  v11 = (__int64 *)v13;
  v14 = (__int64 *)v16;
  sub_AE7A50((__int64)&v11, a1, (__int64)&v14);
  v2 = v11;
  v3 = &v11[(unsigned int)v12];
  if ( v11 != v3 )
  {
    do
    {
      v4 = *v2++;
      sub_F507F0(v4);
    }
    while ( v3 != v2 );
  }
  v5 = v14;
  v6 = &v14[(unsigned int)v15];
  v7 = v15;
  v8 = v14;
  if ( v6 != v14 )
  {
    do
    {
      v9 = *v8++;
      sub_B13710(v9);
    }
    while ( v6 != v8 );
    v7 = v15;
    v5 = v14;
  }
  LOBYTE(v6) = ((unsigned int)v12 | v7) != 0;
  if ( v5 != (__int64 *)v16 )
    _libc_free(v5, a1);
  if ( v11 != (__int64 *)v13 )
    _libc_free(v11, a1);
  return (unsigned int)v6;
}
