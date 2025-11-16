// Function: sub_274B690
// Address: 0x274b690
//
__int64 __fastcall sub_274B690(__int64 a1, __int64 *a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rdx
  int v4; // r13d
  unsigned int v5; // eax
  unsigned int v6; // r12d
  unsigned __int64 v8; // [rsp+0h] [rbp-80h] BYREF
  unsigned int v9; // [rsp+8h] [rbp-78h]
  unsigned __int64 v10; // [rsp+10h] [rbp-70h]
  unsigned int v11; // [rsp+18h] [rbp-68h]
  unsigned __int64 v12; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v13; // [rsp+28h] [rbp-58h]
  unsigned __int64 v14; // [rsp+30h] [rbp-50h]
  unsigned int v15; // [rsp+38h] [rbp-48h]
  unsigned __int64 v16; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v17; // [rsp+48h] [rbp-38h]
  unsigned __int64 v18; // [rsp+50h] [rbp-30h]
  unsigned int v19; // [rsp+58h] [rbp-28h]

  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v2 = *(__int64 **)(a1 - 8);
  else
    v2 = (__int64 *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  sub_22CEA30((__int64)&v8, a2, v2, 0);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v3 = *(_QWORD *)(a1 - 8);
  else
    v3 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  sub_22CEA30((__int64)&v12, a2, (__int64 *)(v3 + 32), 0);
  v4 = sub_B5B690(a1);
  v5 = sub_B5B5E0(a1);
  sub_AB28E0((__int64)&v16, v5, (__int64)&v12, v4);
  v6 = sub_AB1BB0((__int64)&v16, (__int64)&v8);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  if ( v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  if ( v9 > 0x40 && v8 )
    j_j___libc_free_0_0(v8);
  return v6;
}
