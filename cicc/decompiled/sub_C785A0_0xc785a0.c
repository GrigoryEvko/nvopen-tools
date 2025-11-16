// Function: sub_C785A0
// Address: 0xc785a0
//
__int64 __fastcall sub_C785A0(__int64 a1, __int64 a2, __int64 a3)
{
  const void *v5; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v6; // [rsp+8h] [rbp-58h]
  __int64 v7; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v8; // [rsp+18h] [rbp-48h]
  const void *v9; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v10; // [rsp+28h] [rbp-38h]
  __int64 v11; // [rsp+30h] [rbp-30h] BYREF
  unsigned int v12; // [rsp+38h] [rbp-28h]

  v10 = *(_DWORD *)(a3 + 8);
  if ( v10 > 0x40 )
    sub_C43780((__int64)&v9, (const void **)a3);
  else
    v9 = *(const void **)a3;
  v12 = *(_DWORD *)(a3 + 24);
  if ( v12 > 0x40 )
    sub_C43780((__int64)&v11, (const void **)(a3 + 16));
  else
    v11 = *(_QWORD *)(a3 + 16);
  v6 = *(_DWORD *)(a2 + 8);
  if ( v6 > 0x40 )
    sub_C43780((__int64)&v5, (const void **)a2);
  else
    v5 = *(const void **)a2;
  v8 = *(_DWORD *)(a2 + 24);
  if ( v8 > 0x40 )
    sub_C43780((__int64)&v7, (const void **)(a2 + 16));
  else
    v7 = *(_QWORD *)(a2 + 16);
  sub_C6F890(a1, (unsigned __int64 *)&v5, (unsigned __int64 *)&v9, 1u);
  if ( v8 > 0x40 && v7 )
    j_j___libc_free_0_0(v7);
  if ( v6 > 0x40 && v5 )
    j_j___libc_free_0_0(v5);
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  return a1;
}
