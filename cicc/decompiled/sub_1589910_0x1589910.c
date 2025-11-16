// Function: sub_1589910
// Address: 0x1589910
//
__int64 __fastcall sub_1589910(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // r14d
  int v8; // eax
  unsigned int v9; // ecx
  bool v10; // zf
  unsigned int v11; // eax
  unsigned int v12; // r15d
  char v13; // cl
  unsigned int v14; // r15d
  unsigned int v15; // edx
  __int64 v16; // r8
  __int64 v17; // rdx
  __int64 v18; // rdx
  unsigned int v19; // eax
  unsigned __int64 v20; // rdi
  unsigned int v21; // r15d
  _QWORD *v22; // rdx
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  unsigned __int64 *v25; // rsi
  __int64 v26; // rdx
  char v27; // cl
  unsigned __int64 v28; // rdx
  unsigned int v29; // eax
  __int64 v30; // [rsp+8h] [rbp-A8h]
  __int64 v31; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v32; // [rsp+10h] [rbp-A0h] BYREF
  unsigned int v33; // [rsp+18h] [rbp-98h]
  unsigned __int64 v34; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v35; // [rsp+28h] [rbp-88h]
  unsigned __int64 v36; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v37; // [rsp+38h] [rbp-78h]
  unsigned __int64 v38; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v39; // [rsp+48h] [rbp-68h]
  unsigned __int64 v40; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v41; // [rsp+58h] [rbp-58h]
  unsigned __int64 v42; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v43; // [rsp+68h] [rbp-48h]
  unsigned __int64 v44; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v45; // [rsp+78h] [rbp-38h]

  v6 = *(_DWORD *)(a3 + 8);
  if ( v6 > 0x40 )
  {
    v8 = sub_16A57B0(a3);
    if ( v6 - v8 <= 0x40 && !**(_QWORD **)a3 || v8 == v6 - 1 )
      goto LABEL_3;
  }
  else if ( *(_QWORD *)a3 <= 1u )
  {
LABEL_3:
    sub_15897D0(a1, *(_DWORD *)(a2 + 4), 1);
    return a1;
  }
  v9 = *(_DWORD *)(a2 + 4);
  v10 = *(_BYTE *)a2 == 0;
  v33 = 1;
  v32 = 0;
  v35 = 1;
  v11 = v9;
  v34 = 0;
  if ( v10 )
  {
    v12 = v9 - 1;
    v45 = v9;
    v13 = v9 - 1;
    if ( v11 > 0x40 )
    {
      v31 = 1LL << v13;
      sub_16A4EF0(&v44, 0, 0);
      if ( v45 <= 0x40 )
        v44 |= v31;
      else
        *(_QWORD *)(v44 + 8LL * (v12 >> 6)) |= v31;
      if ( v33 > 0x40 && v32 )
        j_j___libc_free_0_0(v32);
      v11 = *(_DWORD *)(a2 + 4);
    }
    else
    {
      v44 = 1LL << v13;
    }
    v14 = v11 - 1;
    v32 = v44;
    v15 = v45;
    v45 = v11;
    v33 = v15;
    v16 = ~(1LL << ((unsigned __int8)v11 - 1));
    if ( v11 > 0x40 )
    {
      v30 = ~(1LL << ((unsigned __int8)v11 - 1));
      sub_16A4EF0(&v44, -1, 1);
      v16 = v30;
      if ( v45 > 0x40 )
      {
        *(_QWORD *)(v44 + 8LL * (v14 >> 6)) &= v30;
        goto LABEL_13;
      }
    }
    else
    {
      v44 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v11;
    }
    v44 &= v16;
  }
  else
  {
    v45 = v9;
    if ( v9 > 0x40 )
    {
      sub_16A4EF0(&v44, 0, 0);
      v9 = *(_DWORD *)(a2 + 4);
      v33 = v45;
      v32 = v44;
      v45 = v9;
      if ( v9 > 0x40 )
      {
        sub_16A4EF0(&v44, -1, 1);
        goto LABEL_13;
      }
    }
    else
    {
      v33 = v9;
    }
    v44 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v9;
  }
LABEL_13:
  if ( v35 > 0x40 && v34 )
    j_j___libc_free_0_0(v34);
  v10 = *(_BYTE *)a2 == 0;
  v34 = v44;
  v35 = v45;
  if ( !v10 )
  {
    v37 = 1;
    v36 = 0;
    v39 = 1;
    v38 = 0;
    sub_16AEB70(&v44, &v32, a3, 2);
    if ( v37 > 0x40 && v36 )
      j_j___libc_free_0_0(v36);
    v36 = v44;
    v37 = v45;
    sub_16AEB70(&v44, &v34, a3, 0);
    if ( v39 > 0x40 )
    {
LABEL_21:
      if ( v38 )
        j_j___libc_free_0_0(v38);
    }
LABEL_23:
    v10 = *(_BYTE *)a2 == 0;
    v17 = *(unsigned int *)(a2 + 4);
    v38 = v44;
    v39 = v45;
    if ( v10 )
    {
      sub_16A5E20(&v44, &v36, v17);
      if ( v37 > 0x40 && v36 )
        j_j___libc_free_0_0(v36);
      v26 = *(unsigned int *)(a2 + 4);
      v36 = v44;
      v37 = v45;
      sub_16A5E20(&v44, &v38, v26);
      if ( v39 > 0x40 )
        goto LABEL_28;
    }
    else
    {
      sub_16A5DD0(&v44, &v36, v17);
      if ( v37 > 0x40 && v36 )
        j_j___libc_free_0_0(v36);
      v18 = *(unsigned int *)(a2 + 4);
      v36 = v44;
      v37 = v45;
      sub_16A5DD0(&v44, &v38, v18);
      if ( v39 > 0x40 )
      {
LABEL_28:
        if ( v38 )
          j_j___libc_free_0_0(v38);
      }
    }
    v38 = v44;
    v39 = v45;
    v41 = v45;
    if ( v45 > 0x40 )
      sub_16A4FD0(&v40, &v38);
    else
      v40 = v38;
    sub_16A7490(&v40, 1);
    v19 = v41;
    v41 = 0;
    v43 = v19;
    v42 = v40;
    v45 = v37;
    if ( v37 > 0x40 )
      sub_16A4FD0(&v44, &v36);
    else
      v44 = v36;
    sub_15898E0(a1, (__int64)&v44, (__int64 *)&v42);
    if ( v45 > 0x40 && v44 )
      j_j___libc_free_0_0(v44);
    if ( v43 > 0x40 && v42 )
      j_j___libc_free_0_0(v42);
    if ( v41 > 0x40 && v40 )
      j_j___libc_free_0_0(v40);
    if ( v39 > 0x40 && v38 )
      j_j___libc_free_0_0(v38);
    if ( v37 <= 0x40 )
      goto LABEL_49;
    v20 = v36;
    if ( !v36 )
      goto LABEL_49;
    goto LABEL_48;
  }
  v21 = *(_DWORD *)(a3 + 8);
  if ( v21 <= 0x40 )
  {
    v24 = *(_QWORD *)a3;
    if ( *(_QWORD *)a3 != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v21) )
    {
      v37 = 1;
      v36 = 0;
      v23 = 1LL << ((unsigned __int8)v21 - 1);
      v39 = 1;
      v38 = 0;
LABEL_63:
      if ( (v24 & v23) != 0 )
      {
        sub_16AECF0(&v44, &v34, a3, 2);
        if ( v37 > 0x40 && v36 )
          j_j___libc_free_0_0(v36);
        v25 = &v32;
        v36 = v44;
        v37 = v45;
      }
      else
      {
        sub_16AECF0(&v44, &v32, a3, 2);
        if ( v37 > 0x40 && v36 )
          j_j___libc_free_0_0(v36);
        v25 = &v34;
        v36 = v44;
        v37 = v45;
      }
      sub_16AECF0(&v44, v25, a3, 0);
      if ( v39 > 0x40 )
        goto LABEL_21;
      goto LABEL_23;
    }
  }
  else if ( v21 != (unsigned int)sub_16A58F0(a3) )
  {
    v22 = *(_QWORD **)a3;
    v37 = 1;
    v23 = 1LL << ((unsigned __int8)v21 - 1);
    v36 = 0;
    v39 = 1;
    v38 = 0;
    v24 = v22[(v21 - 1) >> 6];
    goto LABEL_63;
  }
  v45 = v33;
  if ( v33 > 0x40 )
    sub_16A4FD0(&v44, &v32);
  else
    v44 = v32;
  v27 = v35;
  v41 = v35;
  if ( v35 > 0x40 )
  {
    sub_16A4FD0(&v40, &v34);
    v27 = v41;
    if ( v41 > 0x40 )
    {
      sub_16A8F40(&v40);
      goto LABEL_94;
    }
    v28 = v40;
  }
  else
  {
    v28 = v34;
  }
  v40 = ~v28 & (0xFFFFFFFFFFFFFFFFLL >> -v27);
LABEL_94:
  sub_16A7400(&v40);
  v29 = v41;
  v41 = 0;
  v43 = v29;
  v42 = v40;
  sub_15898E0(a1, (__int64)&v42, (__int64 *)&v44);
  if ( v43 > 0x40 && v42 )
    j_j___libc_free_0_0(v42);
  if ( v41 > 0x40 && v40 )
    j_j___libc_free_0_0(v40);
  if ( v45 <= 0x40 )
    goto LABEL_49;
  v20 = v44;
  if ( !v44 )
    goto LABEL_49;
LABEL_48:
  j_j___libc_free_0_0(v20);
LABEL_49:
  if ( v35 > 0x40 && v34 )
    j_j___libc_free_0_0(v34);
  if ( v33 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  return a1;
}
