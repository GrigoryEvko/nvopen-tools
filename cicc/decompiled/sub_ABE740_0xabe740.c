// Function: sub_ABE740
// Address: 0xabe740
//
__int64 __fastcall sub_ABE740(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  char v4; // [rsp+1Fh] [rbp-81h] BYREF
  __int64 v5; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v6; // [rsp+28h] [rbp-78h]
  __int64 v7; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v8; // [rsp+38h] [rbp-68h]
  __int64 v9; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v10; // [rsp+48h] [rbp-58h]
  __int64 v11; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v12; // [rsp+58h] [rbp-48h]
  __int64 v13; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v14; // [rsp+68h] [rbp-38h]

  if ( sub_AAF7D0(a1) || sub_AAF7D0(a2) )
  {
    return 2;
  }
  else
  {
    sub_AB0A00((__int64)&v5, a1);
    sub_AB0910((__int64)&v7, a1);
    sub_AB0A00((__int64)&v9, a2);
    sub_AB0910((__int64)&v11, a2);
    sub_C49BE0(&v13, &v5, &v9, &v4);
    if ( v14 > 0x40 && v13 )
      j_j___libc_free_0_0(v13);
    v2 = 1;
    if ( !v4 )
    {
      sub_C49BE0(&v13, &v7, &v11, &v4);
      if ( v14 > 0x40 && v13 )
        j_j___libc_free_0_0(v13);
      v2 = (v4 == 0) + 2;
    }
    if ( v12 > 0x40 && v11 )
      j_j___libc_free_0_0(v11);
    if ( v10 > 0x40 && v9 )
      j_j___libc_free_0_0(v9);
    if ( v8 > 0x40 && v7 )
      j_j___libc_free_0_0(v7);
    if ( v6 > 0x40 && v5 )
      j_j___libc_free_0_0(v5);
  }
  return v2;
}
