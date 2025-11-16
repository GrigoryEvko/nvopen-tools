// Function: sub_1590F80
// Address: 0x1590f80
//
__int64 __fastcall sub_1590F80(__int64 a1, unsigned int a2, __int64 a3)
{
  int v4; // eax
  __int64 v6; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v7; // [rsp+8h] [rbp-38h]
  __int64 v8; // [rsp+10h] [rbp-30h]
  unsigned int v9; // [rsp+18h] [rbp-28h]

  v4 = sub_15FF0F0(a2);
  sub_158AE10((__int64)&v6, v4, a3);
  sub_1590E70(a1, (__int64)&v6);
  if ( v9 > 0x40 && v8 )
    j_j___libc_free_0_0(v8);
  if ( v7 > 0x40 && v6 )
    j_j___libc_free_0_0(v6);
  return a1;
}
