// Function: sub_1479410
// Address: 0x1479410
//
__int64 __fastcall sub_1479410(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int16 v3; // ax
  __int64 *v5; // rbx
  __int64 v6; // rax
  __int64 *v7; // r12
  __int64 *v8; // rbx
  __int64 v9; // rax
  __int64 *v10; // rbx
  __int64 v11; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-88h]
  __int64 v13; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v14; // [rsp+28h] [rbp-78h]
  __int64 v15; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v16; // [rsp+38h] [rbp-68h]
  __int64 v17; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v18; // [rsp+48h] [rbp-58h]
  __int64 v19; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v20; // [rsp+58h] [rbp-48h]
  __int64 v21; // [rsp+60h] [rbp-40h]
  unsigned int v22; // [rsp+68h] [rbp-38h]

  v2 = 0;
  if ( *(_QWORD *)(a2 + 40) != 2 )
    return v2;
  v3 = *(_WORD *)(a2 + 26);
  if ( (v3 & 4) != 0 )
  {
    if ( (v3 & 2) != 0 )
      return v2;
    goto LABEL_5;
  }
  v8 = sub_1477920(a1, a2, 1u);
  v12 = *((_DWORD *)v8 + 2);
  if ( v12 > 0x40 )
    sub_16A4FD0(&v11, v8);
  else
    v11 = *v8;
  v14 = *((_DWORD *)v8 + 6);
  if ( v14 > 0x40 )
    sub_16A4FD0(&v13, v8 + 2);
  else
    v13 = v8[2];
  v9 = sub_13A5BC0((_QWORD *)a2, a1);
  v10 = sub_1477920(a1, v9, 1u);
  v16 = *((_DWORD *)v10 + 2);
  if ( v16 > 0x40 )
    sub_16A4FD0(&v15, v10);
  else
    v15 = *v10;
  v18 = *((_DWORD *)v10 + 6);
  if ( v18 > 0x40 )
    sub_16A4FD0(&v17, v10 + 2);
  else
    v17 = v10[2];
  sub_1591060(&v19, 11, &v15, 2);
  v2 = 4 * ((unsigned __int8)sub_158BB40(&v19, &v11) != 0);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  if ( (*(_WORD *)(a2 + 26) & 2) == 0 )
  {
LABEL_5:
    v5 = sub_1477920(a1, a2, 0);
    v12 = *((_DWORD *)v5 + 2);
    if ( v12 > 0x40 )
      sub_16A4FD0(&v11, v5);
    else
      v11 = *v5;
    v14 = *((_DWORD *)v5 + 6);
    if ( v14 > 0x40 )
      sub_16A4FD0(&v13, v5 + 2);
    else
      v13 = v5[2];
    v6 = sub_13A5BC0((_QWORD *)a2, a1);
    v7 = sub_1477920(a1, v6, 0);
    v16 = *((_DWORD *)v7 + 2);
    if ( v16 > 0x40 )
      sub_16A4FD0(&v15, v7);
    else
      v15 = *v7;
    v18 = *((_DWORD *)v7 + 6);
    if ( v18 > 0x40 )
      sub_16A4FD0(&v17, v7 + 2);
    else
      v17 = v7[2];
    sub_1591060(&v19, 11, &v15, 1);
    if ( (unsigned __int8)sub_158BB40(&v19, &v11) )
      v2 |= 2u;
    if ( v22 > 0x40 && v21 )
      j_j___libc_free_0_0(v21);
    if ( v20 > 0x40 && v19 )
      j_j___libc_free_0_0(v19);
    if ( v18 > 0x40 && v17 )
      j_j___libc_free_0_0(v17);
    if ( v16 > 0x40 && v15 )
      j_j___libc_free_0_0(v15);
    if ( v14 > 0x40 && v13 )
      j_j___libc_free_0_0(v13);
    if ( v12 > 0x40 && v11 )
      j_j___libc_free_0_0(v11);
  }
  return v2;
}
