// Function: sub_158C3A0
// Address: 0x158c3a0
//
__int64 __fastcall sub_158C3A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  unsigned int v6; // eax
  unsigned int v7; // eax
  unsigned int v9; // eax
  unsigned int v10; // eax
  __int64 v11; // r13
  unsigned int v12; // eax
  __int64 v13; // r12
  unsigned int v14; // eax
  unsigned int v15; // eax
  int v16; // eax
  __int64 v17; // r10
  unsigned int v18; // r12d
  unsigned int v19; // r13d
  unsigned int v20; // eax
  __int64 v21; // rdi
  unsigned int v22; // eax
  unsigned int v23; // eax
  __int64 v24; // rbx
  unsigned int v25; // eax
  __int64 v26; // rdx
  int v27; // ebx
  __int64 v28; // [rsp+0h] [rbp-C0h]
  __int64 v29; // [rsp+8h] [rbp-B8h]
  __int64 v30; // [rsp+8h] [rbp-B8h]
  unsigned int v31; // [rsp+14h] [rbp-ACh]
  unsigned int v32; // [rsp+14h] [rbp-ACh]
  unsigned int v33; // [rsp+18h] [rbp-A8h]
  unsigned int v34; // [rsp+18h] [rbp-A8h]
  __int64 v35; // [rsp+28h] [rbp-98h]
  __int64 v36; // [rsp+28h] [rbp-98h]
  __int64 v37; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v38; // [rsp+38h] [rbp-88h]
  __int64 v39; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v40; // [rsp+48h] [rbp-78h]
  __int64 v41; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v42; // [rsp+58h] [rbp-68h]
  __int64 v43; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v44; // [rsp+68h] [rbp-58h]
  __int64 v45; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v46; // [rsp+78h] [rbp-48h]
  __int64 v47; // [rsp+80h] [rbp-40h] BYREF
  unsigned int v48; // [rsp+88h] [rbp-38h]

  v4 = a2;
  if ( sub_158A0B0(a2) || sub_158A120(a3) )
  {
    v6 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v6;
    if ( v6 > 0x40 )
      sub_16A4FD0(a1, a2);
    else
      *(_QWORD *)a1 = *(_QWORD *)a2;
    v7 = *(_DWORD *)(a2 + 24);
    *(_DWORD *)(a1 + 24) = v7;
    if ( v7 <= 0x40 )
    {
LABEL_6:
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
      return a1;
    }
LABEL_70:
    sub_16A4FD0(a1 + 16, a2 + 16);
    return a1;
  }
  if ( sub_158A0B0(a3) || sub_158A120(a2) )
  {
    v9 = *(_DWORD *)(a3 + 8);
    *(_DWORD *)(a1 + 8) = v9;
    if ( v9 > 0x40 )
      sub_16A4FD0(a1, a3);
    else
      *(_QWORD *)a1 = *(_QWORD *)a3;
    v10 = *(_DWORD *)(a3 + 24);
    *(_DWORD *)(a1 + 24) = v10;
    if ( v10 > 0x40 )
      sub_16A4FD0(a1 + 16, a3 + 16);
    else
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a3 + 16);
    return a1;
  }
  if ( !sub_158A670(a2) && sub_158A670(a3) )
  {
    sub_158C3A0(a1, a3, a2);
    return a1;
  }
  v11 = a2 + 16;
  if ( !sub_158A670(a2) && !sub_158A670(a3) )
  {
    v35 = a3 + 16;
    if ( (int)sub_16A9900(a3 + 16, a2) >= 0 && (int)sub_16A9900(a2 + 16, a3) >= 0 )
    {
      if ( (int)sub_16A9900(a3, a2) < 0 )
        a2 = a3;
      v38 = *(_DWORD *)(a2 + 8);
      if ( v38 > 0x40 )
        sub_16A4FD0(&v37, a2);
      else
        v37 = *(_QWORD *)a2;
      v42 = *(_DWORD *)(a3 + 24);
      if ( v42 > 0x40 )
        sub_16A4FD0(&v41, v35);
      else
        v41 = *(_QWORD *)(a3 + 16);
      sub_16A7800(&v41, 1);
      v12 = v42;
      v13 = v41;
      v42 = 0;
      v33 = v12;
      v44 = v12;
      v14 = *(_DWORD *)(v4 + 24);
      v43 = v41;
      v46 = v14;
      if ( v14 > 0x40 )
        sub_16A4FD0(&v45, v4 + 16);
      else
        v45 = *(_QWORD *)(v4 + 16);
      sub_16A7800(&v45, 1);
      v15 = v46;
      v46 = 0;
      v47 = v45;
      v29 = v45;
      v31 = v15;
      v48 = v15;
      v16 = sub_16A9900(&v43, &v47);
      v17 = v29;
      if ( v16 > 0 )
        v11 = v35;
      v40 = *(_DWORD *)(v11 + 8);
      if ( v40 > 0x40 )
      {
        sub_16A4FD0(&v39, v11);
        v17 = v29;
      }
      else
      {
        v39 = *(_QWORD *)v11;
      }
      if ( v31 > 0x40 && v17 )
        j_j___libc_free_0_0(v17);
      if ( v46 > 0x40 && v45 )
        j_j___libc_free_0_0(v45);
      if ( v33 > 0x40 && v13 )
        j_j___libc_free_0_0(v13);
      if ( v42 > 0x40 && v41 )
        j_j___libc_free_0_0(v41);
      v18 = v38;
      if ( v38 <= 0x40 )
      {
        if ( v37 )
          goto LABEL_52;
      }
      else if ( v18 != (unsigned int)sub_16A57B0(&v37) )
      {
        goto LABEL_52;
      }
      v19 = v40;
      if ( v40 > 0x40 )
      {
        if ( v19 != (unsigned int)sub_16A57B0(&v39) )
          goto LABEL_52;
LABEL_151:
        sub_15897D0(a1, *(_DWORD *)(v4 + 8), 1);
        goto LABEL_58;
      }
      if ( !v39 )
        goto LABEL_151;
LABEL_52:
      v20 = v40;
      v40 = 0;
      v48 = v20;
      v46 = v18;
      v47 = v39;
      v38 = 0;
      v45 = v37;
      sub_15898E0(a1, (__int64)&v45, &v47);
      if ( v46 > 0x40 && v45 )
        j_j___libc_free_0_0(v45);
      if ( v48 > 0x40 && v47 )
        j_j___libc_free_0_0(v47);
LABEL_58:
      if ( v40 > 0x40 && v39 )
        j_j___libc_free_0_0(v39);
      if ( v38 > 0x40 )
      {
        v21 = v37;
        if ( v37 )
        {
LABEL_63:
          j_j___libc_free_0_0(v21);
          return a1;
        }
      }
      return a1;
    }
    v48 = *(_DWORD *)(a3 + 8);
    if ( v48 > 0x40 )
      sub_16A4FD0(&v47, a3);
    else
      v47 = *(_QWORD *)a3;
    sub_16A7590(&v47, a2 + 16);
    v34 = v48;
    v42 = v48;
    v30 = v47;
    v41 = v47;
    v48 = *(_DWORD *)(a2 + 8);
    if ( v48 > 0x40 )
      sub_16A4FD0(&v47, a2);
    else
      v47 = *(_QWORD *)a2;
    sub_16A7590(&v47, v35);
    v32 = v48;
    v44 = v48;
    v28 = v47;
    v43 = v47;
    if ( (int)sub_16A9900(&v41, &v43) < 0 )
    {
      v48 = *(_DWORD *)(a3 + 24);
      if ( v48 > 0x40 )
        sub_16A4FD0(&v47, v35);
      else
        v47 = *(_QWORD *)(a3 + 16);
      v46 = *(_DWORD *)(a2 + 8);
      if ( v46 > 0x40 )
        sub_16A4FD0(&v45, a2);
      else
        v45 = *(_QWORD *)a2;
    }
    else
    {
      v48 = *(_DWORD *)(a2 + 24);
      if ( v48 > 0x40 )
        sub_16A4FD0(&v47, a2 + 16);
      else
        v47 = *(_QWORD *)(a2 + 16);
      v46 = *(_DWORD *)(a3 + 8);
      if ( v46 > 0x40 )
      {
        sub_16A4FD0(&v45, a3);
        sub_15898E0(a1, (__int64)&v45, &v47);
        goto LABEL_106;
      }
      v45 = *(_QWORD *)a3;
    }
    sub_15898E0(a1, (__int64)&v45, &v47);
LABEL_106:
    if ( v46 > 0x40 && v45 )
      j_j___libc_free_0_0(v45);
    if ( v48 > 0x40 && v47 )
      j_j___libc_free_0_0(v47);
    if ( v32 > 0x40 && v28 )
      j_j___libc_free_0_0(v28);
    if ( v34 > 0x40 )
    {
      v21 = v30;
      if ( v30 )
        goto LABEL_63;
    }
    return a1;
  }
  if ( sub_158A670(a3) )
  {
    if ( (int)sub_16A9900(a3, a2 + 16) > 0 )
    {
      v24 = a3 + 16;
      if ( (int)sub_16A9900(a2, a3 + 16) > 0 )
      {
        if ( (int)sub_16A9900(a3, a2) < 0 )
          a2 = a3;
        v42 = *(_DWORD *)(a2 + 8);
        if ( v42 > 0x40 )
          sub_16A4FD0(&v41, a2);
        else
          v41 = *(_QWORD *)a2;
        if ( (int)sub_16A9900(a3 + 16, v4 + 16) <= 0 )
          v24 = v4 + 16;
        v25 = *(_DWORD *)(v24 + 8);
        v44 = v25;
        if ( v25 > 0x40 )
        {
          sub_16A4FD0(&v43, v24);
          v25 = v44;
          v26 = v43;
        }
        else
        {
          v26 = *(_QWORD *)v24;
          v43 = *(_QWORD *)v24;
        }
        v48 = v25;
        v47 = v26;
        v46 = v42;
        v44 = 0;
        v45 = v41;
        v42 = 0;
        sub_15898E0(a1, (__int64)&v45, &v47);
        if ( v46 > 0x40 && v45 )
          j_j___libc_free_0_0(v45);
        if ( v48 > 0x40 && v47 )
          j_j___libc_free_0_0(v47);
        if ( v44 > 0x40 && v43 )
          j_j___libc_free_0_0(v43);
        if ( v42 > 0x40 )
        {
          v21 = v41;
          if ( v41 )
            goto LABEL_63;
        }
        return a1;
      }
    }
LABEL_94:
    sub_15897D0(a1, *(_DWORD *)(a2 + 8), 1);
    return a1;
  }
  v36 = a3 + 16;
  if ( (int)sub_16A9900(a3 + 16, a2 + 16) <= 0 || (int)sub_16A9900(a3, a2) >= 0 )
  {
    v22 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v22;
    if ( v22 > 0x40 )
      sub_16A4FD0(a1, a2);
    else
      *(_QWORD *)a1 = *(_QWORD *)a2;
    v23 = *(_DWORD *)(a2 + 24);
    *(_DWORD *)(a1 + 24) = v23;
    if ( v23 <= 0x40 )
      goto LABEL_6;
    goto LABEL_70;
  }
  if ( (int)sub_16A9900(a3, a2 + 16) <= 0 && (int)sub_16A9900(a2, v36) <= 0 )
    goto LABEL_94;
  v27 = sub_16A9900(a2 + 16, a3);
  if ( v27 <= 0 )
  {
    if ( (int)sub_16A9900(v36, a2) <= 0 )
    {
      sub_13A38D0((__int64)&v47, a3);
      sub_16A7590(&v47, a2 + 16);
      v42 = v48;
      v41 = v47;
      sub_13A38D0((__int64)&v47, a2);
      sub_16A7590(&v47, v36);
      v44 = v48;
      v43 = v47;
      if ( (int)sub_16A9900(&v41, &v43) < 0 )
      {
        sub_13A38D0((__int64)&v47, v36);
        sub_13A38D0((__int64)&v45, a2);
      }
      else
      {
        sub_13A38D0((__int64)&v47, a2 + 16);
        sub_13A38D0((__int64)&v45, a3);
      }
      sub_15898E0(a1, (__int64)&v45, &v47);
      sub_135E100(&v45);
      sub_135E100(&v47);
      sub_135E100(&v43);
      sub_135E100(&v41);
      return a1;
    }
    if ( v27 && (int)sub_16A9900(a2, v36) < 0 )
    {
      sub_13A38D0((__int64)&v47, v11);
      sub_13A38D0((__int64)&v45, a3);
      sub_15898E0(a1, (__int64)&v45, &v47);
      sub_135E100(&v45);
      sub_135E100(&v47);
      return a1;
    }
  }
  v48 = *(_DWORD *)(a3 + 24);
  if ( v48 > 0x40 )
    sub_16A4FD0(&v47, v36);
  else
    v47 = *(_QWORD *)(a3 + 16);
  v46 = *(_DWORD *)(a2 + 8);
  if ( v46 > 0x40 )
    sub_16A4FD0(&v45, a2);
  else
    v45 = *(_QWORD *)a2;
  sub_15898E0(a1, (__int64)&v45, &v47);
  if ( v46 > 0x40 && v45 )
    j_j___libc_free_0_0(v45);
  if ( v48 > 0x40 )
  {
    v21 = v47;
    if ( v47 )
      goto LABEL_63;
  }
  return a1;
}
