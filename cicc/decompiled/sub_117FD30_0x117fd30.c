// Function: sub_117FD30
// Address: 0x117fd30
//
__int64 __fastcall sub_117FD30(_QWORD **a1, __int64 a2, char a3)
{
  __int64 v3; // rax
  bool v4; // zf
  unsigned int v5; // r13d
  _BYTE **v7; // rcx
  _BYTE *v8; // r14
  unsigned int v9; // eax
  char v10; // dl
  unsigned __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rcx
  _QWORD *v17; // rcx
  __int64 v18; // rax
  int v19; // eax
  unsigned int v20; // ebx
  int v21; // eax
  int v22; // eax
  unsigned int v23; // ebx
  unsigned __int64 v24; // r12
  int v25; // eax
  unsigned int v26; // eax
  unsigned __int64 v27; // r15
  unsigned int v28; // r14d
  bool v29; // al
  char v30; // bl
  unsigned int v31; // ebx
  int v32; // eax
  char v33; // r8
  unsigned __int64 v34; // r15
  unsigned int v35; // r14d
  bool v36; // al
  char v37; // bl
  bool v38; // al
  const void **v39; // r14
  unsigned int v40; // ebx
  bool v41; // al
  __int64 v43; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v44; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v45; // [rsp+28h] [rbp-98h] BYREF
  __int64 v46; // [rsp+30h] [rbp-90h] BYREF
  const void **v47; // [rsp+38h] [rbp-88h] BYREF
  __int64 *v48; // [rsp+40h] [rbp-80h] BYREF
  int v49; // [rsp+48h] [rbp-78h] BYREF
  char v50; // [rsp+4Ch] [rbp-74h]
  unsigned __int64 v51; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v52; // [rsp+58h] [rbp-68h]
  int *v53; // [rsp+60h] [rbp-60h] BYREF
  __int64 *v54; // [rsp+68h] [rbp-58h]
  const void ***v55; // [rsp+70h] [rbp-50h] BYREF
  char v56; // [rsp+78h] [rbp-48h]
  __int64 *v57; // [rsp+80h] [rbp-40h]
  __int64 *v58; // [rsp+88h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 8);
  v4 = *(_BYTE *)a2 == 86;
  v49 = 42;
  v50 = 0;
  v43 = v3;
  v53 = &v49;
  v54 = &v46;
  v55 = &v47;
  v57 = &v44;
  v56 = 0;
  v58 = &v45;
  if ( !v4 )
    return 0;
  v7 = (*(_BYTE *)(a2 + 7) & 0x40) != 0
     ? *(_BYTE ***)(a2 - 8)
     : (_BYTE **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v8 = *v7;
  if ( **v7 != 82 )
    return 0;
  if ( !*((_QWORD *)v8 - 8) )
    return 0;
  v46 = *((_QWORD *)v8 - 8);
  v9 = sub_991580((__int64)&v55, *((_QWORD *)v8 - 4));
  v10 = a3;
  v5 = v9;
  if ( !(_BYTE)v9 )
    return 0;
  if ( v53 )
  {
    v11 = sub_B53900((__int64)v8);
    v12 = (__int64)v53;
    v10 = a3;
    *v53 = v11;
    *(_BYTE *)(v12 + 4) = BYTE4(v11);
  }
  v13 = (*(_BYTE *)(a2 + 7) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v14 = *(_QWORD *)(v13 + 32);
  if ( !v14 )
    return 0;
  *v57 = v14;
  v15 = (*(_BYTE *)(a2 + 7) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v16 = *(_QWORD *)(v15 + 64);
  if ( !v16 )
    return 0;
  *v58 = v16;
  v17 = *a1;
  v48 = &v43;
  v18 = v46;
  if ( *v17 == v46 )
  {
    if ( !v10 )
    {
      if ( v49 == 40 )
      {
        v52 = *((_DWORD *)v47 + 2);
        if ( v52 > 0x40 )
          sub_C43780((__int64)&v51, v47);
        else
          v51 = (unsigned __int64)*v47;
        sub_C46A40((__int64)&v51, 1);
        v27 = v51;
        v28 = v52;
        v52 = 0;
        LODWORD(v54) = v28;
        v53 = (int *)v51;
        v29 = v51 == 0;
        if ( v28 > 0x40 )
          v29 = v28 == (unsigned int)sub_C444A0((__int64)&v53);
        if ( v29 )
          goto LABEL_50;
        v30 = v27 == 1;
        if ( v28 > 0x40 )
          v30 = v28 - 1 == (unsigned int)sub_C444A0((__int64)&v53);
        if ( v30 )
LABEL_50:
          v30 = sub_117FBB0(&v48, v44, v45);
        if ( v28 > 0x40 && v27 )
          j_j___libc_free_0_0(v27);
        if ( v52 > 0x40 && v51 )
          j_j___libc_free_0_0(v51);
        if ( v30 )
          return v5;
        v17 = *a1;
        v18 = v46;
      }
      goto LABEL_20;
    }
LABEL_29:
    v22 = v49;
    if ( v49 != 40 )
    {
LABEL_30:
      if ( v22 != 38 )
        return 0;
      v52 = *((_DWORD *)v47 + 2);
      if ( v52 > 0x40 )
        sub_C43780((__int64)&v51, v47);
      else
        v51 = (unsigned __int64)*v47;
      sub_C46A40((__int64)&v51, 1);
      v23 = v52;
      v24 = v51;
      v52 = 0;
      LODWORD(v54) = v23;
      v53 = (int *)v51;
      if ( v23 <= 0x40 )
      {
        if ( v51 > 1 )
          return 0;
        v5 = sub_117FBB0(&v48, v45, v44);
        v26 = v52;
      }
      else
      {
        v25 = sub_C444A0((__int64)&v53);
        if ( v23 == v25 || v25 == v23 - 1 )
          v5 = sub_117FBB0(&v48, v45, v44);
        else
          v5 = 0;
        if ( v24 )
          j_j___libc_free_0_0(v24);
        v26 = v52;
      }
      if ( v26 > 0x40 && v51 )
        j_j___libc_free_0_0(v51);
      return v5;
    }
    v31 = *((_DWORD *)v47 + 2);
    if ( v31 <= 0x40 )
    {
      if ( !*v47 )
      {
LABEL_61:
        v33 = sub_117FBB0(&v48, v44, v45);
        v22 = v49;
        if ( v33 )
          return v5;
        goto LABEL_30;
      }
      v38 = *v47 == (const void *)1;
    }
    else
    {
      v32 = sub_C444A0((__int64)v47);
      if ( v31 == v32 )
        goto LABEL_61;
      v38 = v31 - 1 == v32;
    }
    if ( !v38 )
      return 0;
    goto LABEL_61;
  }
  if ( v46 != *a1[1] )
    return 0;
  if ( v10 )
    goto LABEL_29;
LABEL_20:
  if ( *v17 != v18 || v49 != 38 )
  {
LABEL_21:
    if ( *a1[1] != v18 )
      return 0;
    v19 = v49;
    if ( v49 != 40 )
      goto LABEL_23;
    v39 = v47;
    v40 = *((_DWORD *)v47 + 2);
    if ( v40 <= 0x40 )
    {
      if ( !*v47 )
      {
LABEL_91:
        if ( (unsigned __int8)sub_117FBB0(&v48, v45, v44) )
          return v5;
        if ( *a1[1] == v46 )
        {
          v19 = v49;
LABEL_23:
          if ( v19 == 38 )
          {
            sub_9865C0((__int64)&v51, (__int64)v47);
            sub_C46A40((__int64)&v51, 1);
            v20 = v52;
            v52 = 0;
            LODWORD(v54) = v20;
            v53 = (int *)v51;
            if ( v20 <= 0x40 )
            {
              if ( !v51 )
                goto LABEL_26;
              LOBYTE(v5) = v51 == 1;
            }
            else
            {
              v21 = sub_C444A0((__int64)&v53);
              if ( v20 == v21 )
              {
LABEL_26:
                v5 = sub_117FBB0(&v48, v44, v45);
LABEL_27:
                sub_969240((__int64 *)&v53);
                sub_969240((__int64 *)&v51);
                return v5;
              }
              LOBYTE(v5) = v20 - 1 == v21;
            }
            if ( !(_BYTE)v5 )
              goto LABEL_27;
            goto LABEL_26;
          }
        }
        return 0;
      }
      v41 = *v47 == (const void *)1;
    }
    else
    {
      if ( v40 == (unsigned int)sub_C444A0((__int64)v47) )
        goto LABEL_91;
      v41 = v40 - 1 == (unsigned int)sub_C444A0((__int64)v39);
    }
    if ( !v41 )
      return 0;
    goto LABEL_91;
  }
  v52 = *((_DWORD *)v47 + 2);
  if ( v52 > 0x40 )
    sub_C43780((__int64)&v51, v47);
  else
    v51 = (unsigned __int64)*v47;
  sub_C46A40((__int64)&v51, 2);
  v34 = v51;
  v35 = v52;
  v52 = 0;
  LODWORD(v54) = v35;
  v53 = (int *)v51;
  v36 = v51 == 0;
  if ( v35 > 0x40 )
    v36 = v35 == (unsigned int)sub_C444A0((__int64)&v53);
  if ( v36 )
    goto LABEL_71;
  v37 = v34 == 1;
  if ( v35 > 0x40 )
    v37 = v35 - 1 == (unsigned int)sub_C444A0((__int64)&v53);
  if ( v37 )
LABEL_71:
    v37 = sub_117FBB0(&v48, v45, v44);
  if ( v35 > 0x40 && v34 )
    j_j___libc_free_0_0(v34);
  if ( v52 > 0x40 && v51 )
    j_j___libc_free_0_0(v51);
  if ( !v37 )
  {
    v18 = v46;
    goto LABEL_21;
  }
  return v5;
}
