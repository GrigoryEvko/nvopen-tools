// Function: sub_ABE130
// Address: 0xabe130
//
__int64 __fastcall sub_ABE130(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v4; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v5; // [rsp+8h] [rbp-68h]
  __int64 v6; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v7; // [rsp+18h] [rbp-58h]
  __int64 v8; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v9; // [rsp+28h] [rbp-48h]
  __int64 v10; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v11; // [rsp+38h] [rbp-38h]

  if ( sub_AAF7D0(a1) || sub_AAF7D0(a2) )
  {
    return 2;
  }
  else
  {
    sub_AB0A00((__int64)&v4, a1);
    sub_AB0910((__int64)&v6, a1);
    sub_AB0A00((__int64)&v8, a2);
    sub_AB0910((__int64)&v10, a2);
    v2 = 0;
    if ( (int)sub_C49970(&v6, &v8) >= 0 )
      v2 = ((int)sub_C49970(&v4, &v10) >= 0) + 2;
    if ( v11 > 0x40 && v10 )
      j_j___libc_free_0_0(v10);
    if ( v9 > 0x40 && v8 )
      j_j___libc_free_0_0(v8);
    if ( v7 > 0x40 && v6 )
      j_j___libc_free_0_0(v6);
    if ( v5 > 0x40 && v4 )
      j_j___libc_free_0_0(v4);
  }
  return v2;
}
