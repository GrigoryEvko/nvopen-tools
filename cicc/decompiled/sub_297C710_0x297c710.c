// Function: sub_297C710
// Address: 0x297c710
//
__int64 __fastcall sub_297C710(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rsi
  unsigned int v5; // r12d
  __int64 v6; // rdi
  _QWORD *v7; // rax
  __int64 v8; // rax
  __int64 v9; // r12
  unsigned __int64 v11; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-58h]
  unsigned __int64 v13; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-48h]
  unsigned __int64 v15; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-38h]
  unsigned __int64 v17; // [rsp+30h] [rbp-30h] BYREF
  unsigned int v18; // [rsp+38h] [rbp-28h]

  v3 = *(_QWORD *)(a1 + 16);
  v12 = *(_DWORD *)(v3 + 32);
  if ( v12 > 0x40 )
    sub_C43780((__int64)&v11, (const void **)(v3 + 24));
  else
    v11 = *(_QWORD *)(v3 + 24);
  v4 = *(_QWORD *)(a2 + 16);
  v14 = *(_DWORD *)(v4 + 32);
  if ( v14 > 0x40 )
    sub_C43780((__int64)&v13, (const void **)(v4 + 24));
  else
    v13 = *(_QWORD *)(v4 + 24);
  sub_297C660(&v11, &v13);
  v18 = v12;
  if ( v12 > 0x40 )
    sub_C43780((__int64)&v17, (const void **)&v11);
  else
    v17 = v11;
  sub_C46B40((__int64)&v17, (__int64 *)&v13);
  v5 = v18;
  v6 = *(_QWORD *)(a1 + 32);
  v16 = v18;
  v15 = v17;
  v7 = (_QWORD *)sub_BD5C60(v6);
  v8 = sub_BCCE00(v7, v5);
  v9 = sub_AD8D80(v8, (__int64)&v15);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  return v9;
}
