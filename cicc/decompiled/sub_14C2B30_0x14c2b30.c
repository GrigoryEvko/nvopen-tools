// Function: sub_14C2B30
// Address: 0x14c2b30
//
__int64 __fastcall sub_14C2B30(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // rdx
  unsigned int v11; // r12d
  __int64 v12; // rax
  __int64 v14; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v15; // [rsp+8h] [rbp-68h]
  __int64 v16; // [rsp+10h] [rbp-60h]
  unsigned int v17; // [rsp+18h] [rbp-58h]
  __int64 v18; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v19; // [rsp+28h] [rbp-48h]
  __int64 v20; // [rsp+30h] [rbp-40h]
  unsigned int v21; // [rsp+38h] [rbp-38h]

  sub_14C2530((__int64)&v14, a1, a3, 0, a4, a5, a6, 0);
  sub_14C2530((__int64)&v18, a2, a3, 0, a4, a5, a6, 0);
  if ( v17 > 0x40 )
    v10 = *(_QWORD *)(v16 + 8LL * ((v17 - 1) >> 6));
  else
    v10 = v16;
  v11 = 1;
  if ( (v10 & (1LL << ((unsigned __int8)v17 - 1))) != 0 )
  {
    v12 = v18;
    if ( v19 > 0x40 )
      v12 = *(_QWORD *)(v18 + 8LL * ((v19 - 1) >> 6));
    v11 = ((v12 & (1LL << ((unsigned __int8)v19 - 1))) != 0) + 1;
  }
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  return v11;
}
