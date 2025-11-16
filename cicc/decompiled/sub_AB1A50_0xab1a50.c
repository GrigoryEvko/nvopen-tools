// Function: sub_AB1A50
// Address: 0xab1a50
//
__int64 __fastcall sub_AB1A50(__int64 a1, int a2, __int64 a3)
{
  __int64 v4; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v5; // [rsp+8h] [rbp-48h]
  __int64 v6; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v7; // [rsp+18h] [rbp-38h]
  __int64 v8; // [rsp+20h] [rbp-30h]
  unsigned int v9; // [rsp+28h] [rbp-28h]

  v5 = *(_DWORD *)(a3 + 8);
  if ( v5 > 0x40 )
    sub_C43780(&v4, a3);
  else
    v4 = *(_QWORD *)a3;
  sub_AADBC0((__int64)&v6, &v4);
  sub_AB15A0(a1, a2, (__int64)&v6);
  if ( v9 > 0x40 && v8 )
    j_j___libc_free_0_0(v8);
  if ( v7 > 0x40 && v6 )
    j_j___libc_free_0_0(v6);
  if ( v5 > 0x40 && v4 )
    j_j___libc_free_0_0(v4);
  return a1;
}
