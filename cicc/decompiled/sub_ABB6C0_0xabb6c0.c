// Function: sub_ABB6C0
// Address: 0xabb6c0
//
__int64 __fastcall sub_ABB6C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v5; // [rsp+8h] [rbp-38h]
  __int64 v6; // [rsp+10h] [rbp-30h]
  unsigned int v7; // [rsp+18h] [rbp-28h]

  sub_ABB300((__int64)&v4, a3);
  sub_AB2160(a1, a2, (__int64)&v4, 0);
  if ( v7 > 0x40 && v6 )
    j_j___libc_free_0_0(v6);
  if ( v5 > 0x40 && v4 )
    j_j___libc_free_0_0(v4);
  return a1;
}
