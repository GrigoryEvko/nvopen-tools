// Function: sub_AB3450
// Address: 0xab3450
//
__int64 __fastcall sub_AB3450(__int64 a1, unsigned int a2, __int64 a3, int a4)
{
  __int64 v6; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v7; // [rsp+8h] [rbp-58h]
  __int64 v8; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v9; // [rsp+18h] [rbp-48h]
  __int64 v10; // [rsp+20h] [rbp-40h]
  unsigned int v11; // [rsp+28h] [rbp-38h]

  v7 = *(_DWORD *)(a3 + 8);
  if ( v7 > 0x40 )
    sub_C43780(&v6, a3);
  else
    v6 = *(_QWORD *)a3;
  sub_AADBC0((__int64)&v8, &v6);
  sub_AB28E0(a1, a2, (__int64)&v8, a4);
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  if ( v9 > 0x40 && v8 )
    j_j___libc_free_0_0(v8);
  if ( v7 > 0x40 && v6 )
    j_j___libc_free_0_0(v6);
  return a1;
}
