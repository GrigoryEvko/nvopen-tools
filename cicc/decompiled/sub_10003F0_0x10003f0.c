// Function: sub_10003F0
// Address: 0x10003f0
//
__int64 __fastcall sub_10003F0(__int64 a1, __int64 a2, __int64 a3)
{
  char v3; // r13
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 v6; // r14
  __int64 v7; // rdi
  _BYTE *v8; // r8
  char v9; // r13
  _BYTE *v11; // rax
  char v12; // r13
  __int64 v13; // rdx
  _BYTE *v14; // rax
  __int64 v15; // [rsp+8h] [rbp-98h]
  __int64 v16; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-88h]
  __int64 v18; // [rsp+20h] [rbp-80h]
  unsigned int v19; // [rsp+28h] [rbp-78h]
  __int64 v20; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v21; // [rsp+38h] [rbp-68h]
  __int64 v22; // [rsp+40h] [rbp-60h]
  unsigned int v23; // [rsp+48h] [rbp-58h]
  __int64 v24; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v25; // [rsp+58h] [rbp-48h]
  __int64 v26; // [rsp+60h] [rbp-40h]
  unsigned int v27; // [rsp+68h] [rbp-38h]

  v3 = a3;
  v4 = a1;
  v5 = *(_QWORD *)(a1 - 32);
  v6 = v5 + 24;
  if ( *(_BYTE *)v5 != 17 )
  {
    v13 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 8LL) - 17;
    if ( (unsigned int)v13 > 1 )
      return 0;
    if ( *(_BYTE *)v5 > 0x15u )
      return 0;
    v14 = sub_AD7630(v5, 0, v13);
    if ( !v14 || *v14 != 17 )
      return 0;
    v6 = (__int64)(v14 + 24);
  }
  v7 = *(_QWORD *)(a2 - 32);
  v8 = (_BYTE *)(v7 + 24);
  if ( *(_BYTE *)v7 != 17 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17 > 1 )
      return 0;
    if ( *(_BYTE *)v7 > 0x15u )
      return 0;
    v11 = sub_AD7630(v7, 0, a3);
    if ( !v11 )
      return 0;
    v8 = v11 + 24;
    if ( *v11 != 17 )
      return 0;
  }
  v15 = (__int64)v8;
  sub_AB1A50((__int64)&v16, *(_WORD *)(v4 + 2) & 0x3F, v6);
  sub_AB1A50((__int64)&v20, *(_WORD *)(a2 + 2) & 0x3F, v15);
  if ( v3 )
  {
    sub_AB2160((__int64)&v24, (__int64)&v16, (__int64)&v20, 0);
    v12 = sub_AAF7D0((__int64)&v24);
    if ( v27 > 0x40 && v26 )
      j_j___libc_free_0_0(v26);
    if ( v25 > 0x40 && v24 )
      j_j___libc_free_0_0(v24);
    if ( v12 )
    {
      v4 = sub_AD6450(*(_QWORD *)(v4 + 8));
      goto LABEL_12;
    }
    if ( (unsigned __int8)sub_AB1BB0((__int64)&v16, (__int64)&v20) )
      goto LABEL_48;
    if ( (unsigned __int8)sub_AB1BB0((__int64)&v20, (__int64)&v16) )
      goto LABEL_12;
    goto LABEL_45;
  }
  sub_AB3510((__int64)&v24, (__int64)&v16, (__int64)&v20, 0);
  v9 = sub_AAF760((__int64)&v24);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  if ( v9 )
  {
    v4 = sub_AD6400(*(_QWORD *)(v4 + 8));
    goto LABEL_12;
  }
  if ( !(unsigned __int8)sub_AB1BB0((__int64)&v16, (__int64)&v20) )
  {
    if ( (unsigned __int8)sub_AB1BB0((__int64)&v20, (__int64)&v16) )
    {
LABEL_48:
      v4 = a2;
      goto LABEL_12;
    }
LABEL_45:
    v4 = 0;
  }
LABEL_12:
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  return v4;
}
