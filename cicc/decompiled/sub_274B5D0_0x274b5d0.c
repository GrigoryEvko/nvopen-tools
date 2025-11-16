// Function: sub_274B5D0
// Address: 0x274b5d0
//
__int64 __fastcall sub_274B5D0(__int64 a1)
{
  char v1; // r12
  unsigned __int64 v3; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v4; // [rsp+8h] [rbp-48h]
  unsigned __int64 v5; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v6; // [rsp+18h] [rbp-38h]
  unsigned __int64 v7; // [rsp+20h] [rbp-30h]
  unsigned int v8; // [rsp+28h] [rbp-28h]

  v4 = *(_DWORD *)(a1 + 8);
  if ( v4 > 0x40 )
    sub_C43690((__int64)&v3, 0, 0);
  else
    v3 = 0;
  sub_AADBC0((__int64)&v5, (__int64 *)&v3);
  v1 = sub_ABB410((__int64 *)a1, 41, (__int64 *)&v5);
  if ( v8 > 0x40 && v7 )
    j_j___libc_free_0_0(v7);
  if ( v6 > 0x40 && v5 )
    j_j___libc_free_0_0(v5);
  if ( v4 > 0x40 && v3 )
    j_j___libc_free_0_0(v3);
  return (unsigned int)(v1 == 0) + 1;
}
