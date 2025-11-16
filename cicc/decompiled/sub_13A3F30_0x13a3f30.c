// Function: sub_13A3F30
// Address: 0x13a3f30
//
__int64 __fastcall sub_13A3F30(unsigned int a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  unsigned int v7; // r13d
  unsigned int v8; // eax
  unsigned int v9; // eax
  char v10; // cl
  bool v11; // al
  char v12; // cl
  unsigned __int64 v13; // rdx
  unsigned int v14; // eax
  bool v15; // cc
  bool v16; // al
  char v17; // cl
  _QWORD *v18; // r15
  unsigned int v19; // r13d
  unsigned int v20; // r13d
  unsigned __int64 v22; // rdx
  unsigned int v23; // eax
  unsigned __int64 v28; // [rsp+40h] [rbp-E0h] BYREF
  unsigned int v29; // [rsp+48h] [rbp-D8h]
  unsigned __int64 v30; // [rsp+50h] [rbp-D0h] BYREF
  unsigned int v31; // [rsp+58h] [rbp-C8h]
  unsigned __int64 v32; // [rsp+60h] [rbp-C0h] BYREF
  unsigned int v33; // [rsp+68h] [rbp-B8h]
  unsigned __int64 v34; // [rsp+70h] [rbp-B0h] BYREF
  unsigned int v35; // [rsp+78h] [rbp-A8h]
  unsigned __int64 v36; // [rsp+80h] [rbp-A0h] BYREF
  unsigned int v37; // [rsp+88h] [rbp-98h]
  unsigned __int64 v38; // [rsp+90h] [rbp-90h] BYREF
  unsigned int v39; // [rsp+98h] [rbp-88h]
  unsigned __int64 v40; // [rsp+A0h] [rbp-80h] BYREF
  unsigned int v41; // [rsp+A8h] [rbp-78h]
  __int64 v42; // [rsp+B0h] [rbp-70h] BYREF
  unsigned int v43; // [rsp+B8h] [rbp-68h]
  unsigned __int64 v44; // [rsp+C0h] [rbp-60h] BYREF
  unsigned int v45; // [rsp+C8h] [rbp-58h]
  unsigned __int64 v46; // [rsp+D0h] [rbp-50h] BYREF
  unsigned int v47; // [rsp+D8h] [rbp-48h]
  unsigned __int64 v48; // [rsp+E0h] [rbp-40h] BYREF
  unsigned int v49; // [rsp+E8h] [rbp-38h]

  v29 = a1;
  if ( a1 > 0x40 )
  {
    sub_16A4EF0(&v28, 1, 1);
    v31 = a1;
    sub_16A4EF0(&v30, 0, 1);
    v33 = a1;
    sub_16A4EF0(&v32, 0, 1);
    v35 = a1;
    sub_16A4EF0(&v34, 1, 1);
  }
  else
  {
    v31 = a1;
    v30 = 0;
    v33 = a1;
    v35 = a1;
    v28 = (0xFFFFFFFFFFFFFFFFLL >> -(char)a1) & 1;
    v32 = 0;
    v34 = v28;
  }
  sub_13A3E40((__int64)&v36, a2);
  sub_13A3E40((__int64)&v38, a3);
  v41 = v37;
  if ( v37 > 0x40 )
  {
    sub_16A4FD0(&v40, &v36);
    v43 = v37;
    if ( v37 > 0x40 )
    {
      sub_16A4FD0(&v42, &v36);
      goto LABEL_6;
    }
  }
  else
  {
    v43 = v37;
    v40 = v36;
  }
  v42 = v36;
LABEL_6:
  sub_16AE5C0(&v36, &v38, &v40, &v42);
LABEL_7:
  v7 = v43;
  while ( v43 > 0x40 )
  {
    if ( v7 - (unsigned int)sub_16A57B0(&v42) <= 0x40 && !*(_QWORD *)v42 )
      goto LABEL_36;
LABEL_9:
    sub_16A7B50(&v48, &v40, &v30);
    if ( v49 > 0x40 )
      sub_16A8F40(&v48);
    else
      v48 = ~v48 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v49);
    sub_16A7400(&v48);
    sub_16A7200(&v48, &v28);
    v8 = v49;
    v45 = v49;
    v44 = v48;
    if ( v29 <= 0x40 && v31 <= 0x40 )
    {
      v29 = v31;
      v28 = v30 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v31);
      if ( v49 > 0x40 )
        goto LABEL_15;
    }
    else
    {
      sub_16A51C0(&v28, &v30);
      if ( v31 > 0x40 || (v8 = v45, v45 > 0x40) )
      {
LABEL_15:
        sub_16A51C0(&v30, &v44);
        goto LABEL_16;
      }
    }
    v31 = v8;
    v30 = v44 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v8);
LABEL_16:
    sub_16A7B50(&v48, &v40, &v34);
    if ( v49 > 0x40 )
      sub_16A8F40(&v48);
    else
      v48 = ~v48 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v49);
    sub_16A7400(&v48);
    sub_16A7200(&v48, &v32);
    v9 = v49;
    v47 = v49;
    v46 = v48;
    if ( v33 <= 0x40 && v35 <= 0x40 )
    {
      v33 = v35;
      v32 = v34 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v35);
      if ( v49 <= 0x40 )
        goto LABEL_52;
    }
    else
    {
      sub_16A51C0(&v32, &v34);
      if ( v35 <= 0x40 )
      {
        v9 = v47;
        if ( v47 <= 0x40 )
        {
LABEL_52:
          v35 = v9;
          v34 = v46 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v9);
          goto LABEL_23;
        }
      }
    }
    sub_16A51C0(&v34, &v46);
LABEL_23:
    if ( v37 <= 0x40 && v39 <= 0x40 )
    {
      v37 = v39;
      v36 = v38 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v39);
    }
    else
    {
      sub_16A51C0(&v36, &v38);
      if ( v39 > 0x40 )
        goto LABEL_27;
    }
    if ( v43 <= 0x40 )
    {
      v39 = v43;
      v38 = v42 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v43);
      goto LABEL_28;
    }
LABEL_27:
    sub_16A51C0(&v38, &v42);
LABEL_28:
    sub_16AE5C0(&v36, &v38, &v40, &v42);
    if ( v47 > 0x40 && v46 )
      j_j___libc_free_0_0(v46);
    if ( v45 <= 0x40 || !v44 )
      goto LABEL_7;
    j_j___libc_free_0_0(v44);
    v7 = v43;
  }
  if ( v42 )
    goto LABEL_9;
LABEL_36:
  if ( *(_DWORD *)(a5 + 8) <= 0x40u && (v10 = v39, v39 <= 0x40) )
  {
    *(_DWORD *)(a5 + 8) = v39;
    *(_QWORD *)a5 = v38 & (0xFFFFFFFFFFFFFFFFLL >> -v10);
  }
  else
  {
    sub_16A51C0(a5, &v38);
  }
  v11 = sub_13A3940(a2, 0);
  v12 = v31;
  if ( !v11 )
  {
    v49 = v31;
    if ( v31 > 0x40 )
      sub_16A4FD0(&v48, &v30);
    else
      v48 = v30;
    if ( *(_DWORD *)(a6 + 8) > 0x40u && *(_QWORD *)a6 )
    {
      j_j___libc_free_0_0(*(_QWORD *)a6);
      *(_QWORD *)a6 = v48;
      *(_DWORD *)(a6 + 8) = v49;
    }
    else
    {
      *(_QWORD *)a6 = v48;
      *(_DWORD *)(a6 + 8) = v49;
    }
    goto LABEL_59;
  }
  v47 = v31;
  if ( v31 <= 0x40 )
  {
    v13 = v30;
LABEL_42:
    v46 = ~v13 & (0xFFFFFFFFFFFFFFFFLL >> -v12);
    goto LABEL_43;
  }
  sub_16A4FD0(&v46, &v30);
  v12 = v47;
  if ( v47 <= 0x40 )
  {
    v13 = v46;
    goto LABEL_42;
  }
  sub_16A8F40(&v46);
LABEL_43:
  sub_16A7400(&v46);
  v14 = v47;
  v47 = 0;
  v15 = *(_DWORD *)(a6 + 8) <= 0x40u;
  v49 = v14;
  v48 = v46;
  if ( v15 || !*(_QWORD *)a6 )
  {
    *(_QWORD *)a6 = v46;
    *(_DWORD *)(a6 + 8) = v14;
  }
  else
  {
    j_j___libc_free_0_0(*(_QWORD *)a6);
    *(_QWORD *)a6 = v48;
    *(_DWORD *)(a6 + 8) = v49;
  }
  sub_135E100((__int64 *)&v46);
LABEL_59:
  v16 = sub_13A3940(a3, 0);
  v17 = v35;
  if ( v16 )
  {
    v49 = v35;
    if ( v35 > 0x40 )
      sub_16A4FD0(&v48, &v34);
    else
      v48 = v34;
    if ( *(_DWORD *)(a7 + 8) > 0x40u && *(_QWORD *)a7 )
    {
      j_j___libc_free_0_0(*(_QWORD *)a7);
      *(_QWORD *)a7 = v48;
      *(_DWORD *)(a7 + 8) = v49;
    }
    else
    {
      *(_QWORD *)a7 = v48;
      *(_DWORD *)(a7 + 8) = v49;
    }
    goto LABEL_65;
  }
  v47 = v35;
  if ( v35 > 0x40 )
  {
    sub_16A4FD0(&v46, &v34);
    v17 = v47;
    if ( v47 > 0x40 )
    {
      sub_16A8F40(&v46);
      goto LABEL_107;
    }
    v22 = v46;
  }
  else
  {
    v22 = v34;
  }
  v46 = ~v22 & (0xFFFFFFFFFFFFFFFFLL >> -v17);
LABEL_107:
  sub_16A7400(&v46);
  v23 = v47;
  v47 = 0;
  v15 = *(_DWORD *)(a7 + 8) <= 0x40u;
  v49 = v23;
  v48 = v46;
  if ( v15 )
  {
    *(_QWORD *)a7 = v46;
    *(_DWORD *)(a7 + 8) = v23;
  }
  else if ( *(_QWORD *)a7 )
  {
    j_j___libc_free_0_0(*(_QWORD *)a7);
    *(_QWORD *)a7 = v48;
    *(_DWORD *)(a7 + 8) = v49;
  }
  else
  {
    *(_QWORD *)a7 = v46;
    *(_DWORD *)(a7 + 8) = v23;
  }
  sub_135E100((__int64 *)&v46);
LABEL_65:
  sub_16AB4D0(&v48, a4, a5);
  if ( v43 > 0x40 && v42 )
    j_j___libc_free_0_0(v42);
  v18 = (_QWORD *)v48;
  v19 = v49;
  v42 = v48;
  v43 = v49;
  if ( v49 <= 0x40 )
  {
    v20 = 1;
    if ( v48 )
      goto LABEL_73;
    goto LABEL_102;
  }
  v15 = v19 - (unsigned int)sub_16A57B0(&v42) <= 0x40;
  v20 = 1;
  if ( v15 && !*v18 )
  {
LABEL_102:
    v20 = 0;
    sub_16A9F90(&v48, a4, a5);
    sub_13A3610((__int64 *)&v40, (__int64 *)&v48);
    sub_135E100((__int64 *)&v48);
    sub_16A7C10(a6, &v40);
    sub_16A7C10(a7, &v40);
    if ( v43 <= 0x40 )
      goto LABEL_73;
  }
  if ( v42 )
    j_j___libc_free_0_0(v42);
LABEL_73:
  if ( v41 > 0x40 && v40 )
    j_j___libc_free_0_0(v40);
  if ( v39 > 0x40 && v38 )
    j_j___libc_free_0_0(v38);
  if ( v37 > 0x40 && v36 )
    j_j___libc_free_0_0(v36);
  if ( v35 > 0x40 && v34 )
    j_j___libc_free_0_0(v34);
  if ( v33 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  return v20;
}
