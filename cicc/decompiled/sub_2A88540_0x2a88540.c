// Function: sub_2A88540
// Address: 0x2a88540
//
__int64 __fastcall sub_2A88540(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  char v6; // al
  __int64 v9; // r14
  unsigned __int8 *v10; // r15
  __int64 v11; // r13
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int8 *v14; // [rsp+0h] [rbp-50h]
  unsigned __int64 v16; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-38h]

  v6 = *(_BYTE *)(a1 + 8);
  if ( (unsigned __int8)(v6 - 15) <= 1u || v6 == 18 )
    return 0xFFFFFFFFLL;
  v17 = sub_AE43F0(a5, *(_QWORD *)(a3 + 8));
  if ( v17 > 0x40 )
    sub_C43690((__int64)&v16, 0, 0);
  else
    v16 = 0;
  v14 = sub_BD45C0((unsigned __int8 *)a3, a5, (__int64)&v16, 1, 0, 0, 0, 0);
  if ( v17 > 0x40 )
  {
    v9 = *(_QWORD *)v16;
    j_j___libc_free_0_0(v16);
  }
  else
  {
    v9 = 0;
    if ( v17 )
      v9 = (__int64)(v16 << (64 - (unsigned __int8)v17)) >> (64 - (unsigned __int8)v17);
  }
  v17 = sub_AE43F0(a5, *(_QWORD *)(a2 + 8));
  if ( v17 > 0x40 )
    sub_C43690((__int64)&v16, 0, 0);
  else
    v16 = 0;
  v10 = sub_BD45C0((unsigned __int8 *)a2, a5, (__int64)&v16, 1, 0, 0, 0, 0);
  if ( v17 > 0x40 )
  {
    v11 = *(_QWORD *)v16;
    j_j___libc_free_0_0(v16);
  }
  else
  {
    v11 = 0;
    if ( v17 )
      v11 = (__int64)(v16 << (64 - (unsigned __int8)v17)) >> (64 - (unsigned __int8)v17);
  }
  if ( v10 != v14 )
    return 0xFFFFFFFFLL;
  v12 = sub_9208B0(a5, a1);
  if ( (((unsigned __int8)v12 | (unsigned __int8)a4) & 7) != 0 )
    return 0xFFFFFFFFLL;
  v13 = v12 >> 3;
  if ( v9 > v11 || (__int64)(v9 + (a4 >> 3)) < (__int64)(v11 + v13) )
    return 0xFFFFFFFFLL;
  else
    return (unsigned int)(v11 - v9);
}
