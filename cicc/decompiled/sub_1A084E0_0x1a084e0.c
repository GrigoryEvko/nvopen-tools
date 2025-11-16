// Function: sub_1A084E0
// Address: 0x1a084e0
//
__int64 __fastcall sub_1A084E0(__int64 a1, __int64 a2, unsigned __int64 *a3, __int64 a4, __int64 *a5)
{
  unsigned int v5; // r13d
  const void **v7; // r15
  unsigned int v8; // edx
  int v11; // eax
  __int64 v12; // rax
  __int64 v14; // rsi
  __int64 *v15; // r10
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  bool v18; // cc
  __int64 *v19; // r10
  __int64 *v20; // [rsp+0h] [rbp-70h]
  unsigned int v21; // [rsp+8h] [rbp-68h]
  unsigned int v22; // [rsp+8h] [rbp-68h]
  unsigned __int64 v24; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v25; // [rsp+28h] [rbp-48h]
  unsigned __int64 v26; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v27; // [rsp+38h] [rbp-38h]

  v5 = *((unsigned __int8 *)a3 + 36);
  if ( !(_BYTE)v5 )
    return v5;
  v7 = (const void **)(a3 + 2);
  v8 = *((_DWORD *)a3 + 6);
  if ( v8 <= 0x40 )
  {
    if ( !a3[2] )
      return 0;
  }
  else
  {
    v21 = v8;
    v11 = sub_16A57B0((__int64)v7);
    v8 = v21;
    if ( v21 == v11 )
      return 0;
  }
  v12 = *(_QWORD *)(*a3 + 8);
  if ( !v12 || *(_QWORD *)(v12 + 8) )
    return 0;
  if ( v8 <= 0x40 )
  {
    v14 = a3[2];
    if ( v14 != *(_QWORD *)a4 )
      return 0;
    v15 = (__int64 *)a3[1];
    goto LABEL_14;
  }
  v22 = v8;
  if ( !sub_16A5220((__int64)v7, (const void **)a4) )
    return 0;
  v19 = (__int64 *)a3[1];
  v25 = v22;
  v20 = v19;
  sub_16A4FD0((__int64)&v24, v7);
  v8 = v25;
  v15 = v20;
  if ( v25 <= 0x40 )
  {
    v14 = v24;
LABEL_14:
    v16 = ~v14 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v8);
    v24 = v16;
    goto LABEL_15;
  }
  sub_16A8F40((__int64 *)&v24);
  v8 = v25;
  v16 = v24;
  v15 = v20;
LABEL_15:
  v27 = v8;
  v26 = v16;
  v25 = 0;
  v17 = sub_19FF2B0(a2, v15, (__int64)&v26);
  v18 = v27 <= 0x40;
  *a5 = v17;
  if ( !v18 && v26 )
    j_j___libc_free_0_0(v26);
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  if ( *(_DWORD *)(a4 + 8) > 0x40u )
    sub_16A8F00((__int64 *)a4, (__int64 *)v7);
  else
    *(_QWORD *)a4 ^= a3[2];
  if ( *(_BYTE *)(*a3 + 16) > 0x17u )
  {
    v26 = *a3;
    sub_1A062A0(a1 + 64, &v26);
  }
  return v5;
}
