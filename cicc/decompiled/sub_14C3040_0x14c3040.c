// Function: sub_14C3040
// Address: 0x14c3040
//
__int64 __fastcall sub_14C3040(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  unsigned int v11; // eax
  __int64 *v12; // r10
  unsigned int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r10
  __int64 v17; // r8
  unsigned int v18; // esi
  __int64 v19; // r8
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rdi
  __int64 v23; // rdx
  unsigned int v24; // r15d
  __int64 v25; // rax
  unsigned __int64 v26; // r8
  unsigned int v27; // edx
  __int64 v28; // rdi
  unsigned __int64 v29; // rsi
  __int64 v30; // rax
  unsigned int v31; // edx
  __int64 v32; // rax
  bool v33; // r15
  __int64 v34; // rdi
  __int64 v35; // rdx
  bool v36; // r15
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rdx
  char v41; // al
  unsigned int v42; // esi
  __int64 v43; // rcx
  __int64 v44; // rdx
  unsigned int v45; // eax
  __int64 v46; // r15
  __int64 v47; // rsi
  __int64 v48; // rdx
  __int64 v49; // rax
  unsigned int v50; // edx
  __int64 v51; // rax
  char v52; // [rsp+0h] [rbp-C0h]
  __int64 v53; // [rsp+10h] [rbp-B0h] BYREF
  unsigned int v54; // [rsp+18h] [rbp-A8h]
  __int64 v55; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v56; // [rsp+28h] [rbp-98h]
  __int64 v57; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v58; // [rsp+38h] [rbp-88h]
  __int64 v59; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v60; // [rsp+48h] [rbp-78h]
  __int64 v61; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v62; // [rsp+58h] [rbp-68h]
  __int64 v63; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v64; // [rsp+68h] [rbp-58h]
  __int64 v65; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v66; // [rsp+78h] [rbp-48h]
  __int64 v67; // [rsp+80h] [rbp-40h] BYREF
  unsigned int v68; // [rsp+88h] [rbp-38h]

  if ( a3 && (*(_BYTE *)(a3 + 17) & 4) != 0 )
    return 2;
  v11 = sub_14C23D0(a1, a4, 0, a5, a6, a7);
  v12 = (__int64 *)a1;
  if ( v11 > 1 )
  {
    v13 = sub_14C23D0((__int64)a2, a4, 0, a5, a6, a7);
    v12 = (__int64 *)a1;
    if ( v13 > 1 )
      return 2;
  }
  sub_14C2530((__int64)&v57, v12, a4, 0, a5, a6, a7, 0);
  sub_14C2530((__int64)&v61, a2, a4, 0, a5, a6, a7, 0);
  v14 = 1LL << ((unsigned __int8)v60 - 1);
  if ( v60 <= 0x40 )
    v15 = v59;
  else
    v15 = *(_QWORD *)(v59 + 8LL * ((v60 - 1) >> 6));
  v16 = v14 & v15;
  if ( (v14 & v15) != 0 )
  {
    v17 = v61;
    if ( v62 > 0x40 )
      v17 = *(_QWORD *)(v61 + 8LL * ((v62 - 1) >> 6));
    if ( (v17 & (1LL << ((unsigned __int8)v62 - 1))) != 0 )
      goto LABEL_67;
  }
  v18 = v58;
  v19 = v57;
  v20 = 1LL << ((unsigned __int8)v58 - 1);
  if ( v58 > 0x40 )
    v21 = *(_QWORD *)(v57 + 8LL * ((v58 - 1) >> 6));
  else
    v21 = v57;
  if ( (v21 & v20) == 0 )
  {
    v46 = 1LL << ((unsigned __int8)v62 - 1);
    if ( v62 > 0x40 )
    {
      if ( (*(_QWORD *)(v61 + 8LL * ((v62 - 1) >> 6)) & v46) != 0 )
        goto LABEL_16;
    }
    else if ( (v61 & v46) != 0 )
    {
      goto LABEL_16;
    }
    if ( !v16 )
    {
      v47 = v63;
      if ( v64 > 0x40 )
        v47 = *(_QWORD *)(v63 + 8LL * ((v64 - 1) >> 6));
      if ( (v47 & (1LL << ((unsigned __int8)v64 - 1))) == 0 )
        goto LABEL_31;
    }
    v56 = v60;
    if ( v60 > 0x40 )
    {
      sub_16A4FD0(&v55, &v59);
      v48 = ~(1LL << ((unsigned __int8)v56 - 1));
      if ( v56 > 0x40 )
      {
        *(_QWORD *)(v55 + 8LL * ((v56 - 1) >> 6)) &= v48;
LABEL_81:
        v66 = v64;
        if ( v64 > 0x40 )
        {
          sub_16A4FD0(&v65, &v63);
          v49 = ~(1LL << ((unsigned __int8)v66 - 1));
          if ( v66 > 0x40 )
          {
            *(_QWORD *)(v65 + 8LL * ((v66 - 1) >> 6)) &= v49;
LABEL_84:
            sub_16A7200(&v65, &v55);
            v50 = v66;
            v66 = 0;
            v51 = 1LL << ((unsigned __int8)v50 - 1);
            if ( v50 <= 0x40 )
            {
              v33 = (v51 & v65) != 0;
            }
            else
            {
              v33 = (*(_QWORD *)(v65 + 8LL * ((v50 - 1) >> 6)) & v51) != 0;
              if ( v65 )
              {
                j_j___libc_free_0_0(v65);
                if ( v66 > 0x40 )
                {
                  if ( v65 )
                    j_j___libc_free_0_0(v65);
                }
              }
            }
            if ( v56 > 0x40 )
            {
              v34 = v55;
              if ( v55 )
                goto LABEL_29;
            }
            goto LABEL_30;
          }
        }
        else
        {
          v65 = v63;
          v49 = ~(1LL << ((unsigned __int8)v64 - 1));
        }
        v65 &= v49;
        goto LABEL_84;
      }
    }
    else
    {
      v48 = ~v14;
      v55 = v59;
    }
    v55 &= v48;
    goto LABEL_81;
  }
  v22 = v63;
  v23 = 1LL << ((unsigned __int8)v64 - 1);
  if ( v64 > 0x40 )
  {
    if ( (*(_QWORD *)(v63 + 8LL * ((v64 - 1) >> 6)) & v23) != 0 )
    {
      v24 = 2;
LABEL_70:
      if ( v22 )
        j_j___libc_free_0_0(v22);
      goto LABEL_46;
    }
  }
  else
  {
    v24 = 2;
    if ( (v23 & v63) != 0 )
      goto LABEL_46;
  }
LABEL_16:
  v66 = v58;
  if ( v58 <= 0x40 )
  {
LABEL_17:
    v25 = ~v20;
    v54 = v18;
    v26 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v18) & ~v19;
    goto LABEL_18;
  }
  sub_16A4FD0(&v65, &v57);
  v18 = v66;
  if ( v66 <= 0x40 )
  {
    v19 = v65;
    v20 = 1LL << ((unsigned __int8)v66 - 1);
    goto LABEL_17;
  }
  sub_16A8F40(&v65);
  v26 = v65;
  v54 = v66;
  v53 = v65;
  v25 = ~(1LL << ((unsigned __int8)v66 - 1));
  if ( v66 > 0x40 )
  {
    *(_QWORD *)(v65 + 8LL * ((v66 - 1) >> 6)) &= v25;
    v27 = v62;
    v66 = v62;
    if ( v62 <= 0x40 )
      goto LABEL_19;
    goto LABEL_108;
  }
LABEL_18:
  v27 = v62;
  v53 = v26 & v25;
  v66 = v62;
  if ( v62 <= 0x40 )
  {
LABEL_19:
    v28 = v61;
LABEL_20:
    v56 = v27;
    v29 = ~v28 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v27);
    v30 = ~(1LL << ((unsigned __int8)v27 - 1));
LABEL_21:
    v55 = v29 & v30;
    goto LABEL_22;
  }
LABEL_108:
  sub_16A4FD0(&v65, &v61);
  v27 = v66;
  if ( v66 <= 0x40 )
  {
    v28 = v65;
    goto LABEL_20;
  }
  sub_16A8F40(&v65);
  v29 = v65;
  v56 = v66;
  v55 = v65;
  v30 = ~(1LL << ((unsigned __int8)v66 - 1));
  if ( v66 <= 0x40 )
    goto LABEL_21;
  *(_QWORD *)(v65 + 8LL * ((v66 - 1) >> 6)) &= v30;
LABEL_22:
  sub_16A7200(&v55, &v53);
  v31 = v56;
  v56 = 0;
  v32 = 1LL << ((unsigned __int8)v31 - 1);
  if ( v31 <= 0x40 )
  {
    v33 = (v32 & v55) == 0;
  }
  else
  {
    v33 = (*(_QWORD *)(v55 + 8LL * ((v31 - 1) >> 6)) & v32) == 0;
    if ( v55 )
    {
      j_j___libc_free_0_0(v55);
      if ( v56 > 0x40 )
      {
        if ( v55 )
          j_j___libc_free_0_0(v55);
      }
    }
  }
  if ( v54 > 0x40 )
  {
    v34 = v53;
    if ( v53 )
LABEL_29:
      j_j___libc_free_0_0(v34);
  }
LABEL_30:
  if ( v33 )
  {
LABEL_67:
    v45 = v64;
    v24 = 2;
    goto LABEL_68;
  }
LABEL_31:
  if ( a3 )
  {
    v35 = v57;
    if ( v58 > 0x40 )
      v35 = *(_QWORD *)(v57 + 8LL * ((v58 - 1) >> 6));
    v36 = 1;
    if ( (v35 & (1LL << ((unsigned __int8)v58 - 1))) == 0 )
    {
      v37 = v61;
      if ( v62 > 0x40 )
        v37 = *(_QWORD *)(v61 + 8LL * ((v62 - 1) >> 6));
      v36 = (v37 & (1LL << ((unsigned __int8)v62 - 1))) != 0;
    }
    v38 = v59;
    if ( v60 > 0x40 )
      v38 = *(_QWORD *)(v59 + 8LL * ((v60 - 1) >> 6));
    if ( (v38 & (1LL << ((unsigned __int8)v60 - 1))) != 0 )
      goto LABEL_58;
    v39 = v63;
    if ( v64 > 0x40 )
      v39 = *(_QWORD *)(v63 + 8LL * ((v64 - 1) >> 6));
    if ( (v39 & (1LL << ((unsigned __int8)v64 - 1))) != 0 )
    {
LABEL_58:
      v41 = 1;
    }
    else
    {
      if ( !v36 )
      {
        v24 = 1;
        if ( v64 <= 0x40 )
          goto LABEL_46;
        goto LABEL_69;
      }
      v41 = 0;
    }
    v52 = v41;
    sub_14C2530((__int64)&v65, (__int64 *)a3, a4, 0, a5, a6, a7, 0);
    v42 = v66;
    if ( v66 > 0x40 )
      v43 = *(_QWORD *)(v65 + 8LL * ((v66 - 1) >> 6));
    else
      v43 = v65;
    if ( (v43 & (1LL << ((unsigned __int8)v66 - 1))) != 0 && v36 )
      goto LABEL_66;
    v44 = 1LL << ((unsigned __int8)v68 - 1);
    if ( v68 > 0x40 )
    {
      if ( (*(_QWORD *)(v67 + 8LL * ((v68 - 1) >> 6)) & v44) != 0 && v52 )
        goto LABEL_66;
      if ( v67 )
      {
        j_j___libc_free_0_0(v67);
        v42 = v66;
      }
    }
    else if ( (v44 & v67) != 0 && v52 )
    {
LABEL_66:
      sub_135E100(&v67);
      sub_135E100(&v65);
      goto LABEL_67;
    }
    if ( v42 > 0x40 && v65 )
      j_j___libc_free_0_0(v65);
  }
  v45 = v64;
  v24 = 1;
LABEL_68:
  if ( v45 > 0x40 )
  {
LABEL_69:
    v22 = v63;
    goto LABEL_70;
  }
LABEL_46:
  if ( v62 > 0x40 && v61 )
    j_j___libc_free_0_0(v61);
  if ( v60 > 0x40 && v59 )
    j_j___libc_free_0_0(v59);
  if ( v58 > 0x40 && v57 )
    j_j___libc_free_0_0(v57);
  return v24;
}
