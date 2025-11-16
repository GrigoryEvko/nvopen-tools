// Function: sub_14BB210
// Address: 0x14bb210
//
__int64 __fastcall sub_14BB210(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r11
  char v10; // al
  char v11; // al
  __int64 v12; // r15
  char v13; // al
  __int64 v14; // rdx
  char v15; // al
  __int64 v16; // rax
  __int64 v17; // rcx
  char v18; // al
  __int64 v19; // rax
  char v20; // al
  char v21; // al
  __int64 v22; // r15
  char v23; // al
  __int64 v24; // r15
  char v25; // al
  __int64 v26; // rax
  __int64 v27; // rcx
  char v28; // al
  __int64 v29; // rax
  char v30; // al
  char v31; // al
  __int64 v32; // rdx
  unsigned __int64 v33; // r12
  unsigned __int64 v34; // r12
  __int64 v36; // rax
  __int64 v37; // r15
  char v38; // cl
  __int64 v39; // rax
  char v40; // al
  __int64 v41; // rax
  char v42; // al
  char v43; // al
  char v44; // al
  __int64 v45; // rax
  char v46; // al
  __int64 v47; // rax
  char v48; // al
  __int64 v49; // rax
  char v50; // al
  char v51; // al
  __int64 v52; // rax
  char v53; // al
  char v54; // al
  char v55; // al
  __int64 v56; // rax
  char v57; // al
  __int64 v58; // rax
  char v59; // al
  __int64 v60; // rax
  char v61; // al
  __int64 v62; // rax
  char v63; // al
  char v64; // al
  __int64 v65; // rax
  char v66; // al
  __int64 v67; // rax
  char v68; // al
  __int64 v69; // rax
  char v70; // al
  __int64 v71; // [rsp+0h] [rbp-C0h]
  __int64 v72; // [rsp+8h] [rbp-B8h]
  __int64 v73; // [rsp+8h] [rbp-B8h]
  __int64 v74; // [rsp+8h] [rbp-B8h]
  __int64 v75; // [rsp+10h] [rbp-B0h]
  __int64 v76; // [rsp+10h] [rbp-B0h]
  __int64 v77; // [rsp+10h] [rbp-B0h]
  __int64 v78; // [rsp+10h] [rbp-B0h]
  __int64 v79; // [rsp+10h] [rbp-B0h]
  __int64 v80; // [rsp+10h] [rbp-B0h]
  __int64 v81; // [rsp+10h] [rbp-B0h]
  __int64 v82; // [rsp+10h] [rbp-B0h]
  __int64 v83; // [rsp+10h] [rbp-B0h]
  __int64 v84; // [rsp+10h] [rbp-B0h]
  __int64 v85; // [rsp+10h] [rbp-B0h]
  __int64 v86; // [rsp+10h] [rbp-B0h]
  __int64 v87; // [rsp+10h] [rbp-B0h]
  __int64 v88; // [rsp+10h] [rbp-B0h]
  __int64 v89; // [rsp+10h] [rbp-B0h]
  __int64 v90; // [rsp+10h] [rbp-B0h]
  __int64 v91; // [rsp+10h] [rbp-B0h]
  __int64 v92; // [rsp+10h] [rbp-B0h]
  __int64 v94; // [rsp+28h] [rbp-98h] BYREF
  unsigned __int64 v95; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v96; // [rsp+38h] [rbp-88h]
  unsigned __int64 v97; // [rsp+40h] [rbp-80h] BYREF
  int v98; // [rsp+48h] [rbp-78h]
  unsigned __int64 v99; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v100; // [rsp+58h] [rbp-68h]
  __int64 v101; // [rsp+60h] [rbp-60h]
  unsigned int v102; // [rsp+68h] [rbp-58h]
  unsigned __int64 v103; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v104; // [rsp+78h] [rbp-48h]
  __int64 v105; // [rsp+80h] [rbp-40h]
  unsigned int v106; // [rsp+88h] [rbp-38h]

  v6 = a2;
  v10 = *(_BYTE *)(a1 + 16);
  v103 = (unsigned __int64)&v94;
  if ( v10 != 50 )
  {
    if ( v10 != 5 || *(_WORD *)(a1 + 18) != 26 )
      goto LABEL_4;
    v36 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    v37 = *(_QWORD *)(a1 - 24 * v36);
    v38 = *(_BYTE *)(v37 + 16);
    if ( v38 == 52 )
    {
      if ( *(_QWORD *)(v37 - 48) )
      {
        v94 = *(_QWORD *)(v37 - 48);
        v64 = sub_14A9710(*(_QWORD *)(v37 - 24));
        v6 = a2;
        if ( v64 )
          goto LABEL_33;
      }
      v65 = *(_QWORD *)(v37 - 24);
      if ( v65 )
      {
        v90 = v6;
        *(_QWORD *)v103 = v65;
        v66 = sub_14A9710(*(_QWORD *)(v37 - 48));
        v6 = v90;
        if ( v66 )
          goto LABEL_33;
      }
    }
    else
    {
      if ( v38 != 5 || *(_WORD *)(v37 + 18) != 28 )
      {
LABEL_66:
        v80 = v6;
        v43 = sub_14B2B20((_QWORD **)&v103, *(_QWORD *)(a1 + 24 * (1 - v36)));
        v6 = v80;
        if ( !v43 )
          goto LABEL_4;
LABEL_33:
        v31 = *(_BYTE *)(v6 + 16);
        if ( v31 == 50 )
        {
          if ( v94 == *(_QWORD *)(v6 - 48) || v94 == *(_QWORD *)(v6 - 24) )
            goto LABEL_21;
          v12 = *(_QWORD *)(v6 - 48);
          v103 = (unsigned __int64)&v94;
          v13 = *(_BYTE *)(v12 + 16);
          if ( v13 != 52 )
            goto LABEL_6;
LABEL_68:
          if ( *(_QWORD *)(v12 - 48) )
          {
            v94 = *(_QWORD *)(v12 - 48);
            v81 = v6;
            v44 = sub_14A9710(*(_QWORD *)(v12 - 24));
            v6 = v81;
            if ( v44 )
              goto LABEL_16;
          }
          v45 = *(_QWORD *)(v12 - 24);
          if ( v45 )
          {
            v82 = v6;
            *(_QWORD *)v103 = v45;
            v46 = sub_14A9710(*(_QWORD *)(v12 - 48));
            v6 = v82;
            if ( v46 )
              goto LABEL_16;
          }
          goto LABEL_8;
        }
        if ( v31 != 5 )
          goto LABEL_80;
        if ( *(_WORD *)(v6 + 18) == 26
          && (v94 == *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF))
           || v94 == *(_QWORD *)(v6 + 24 * (1LL - (*(_DWORD *)(v6 + 20) & 0xFFFFFFF)))) )
        {
          goto LABEL_21;
        }
        v103 = (unsigned __int64)&v94;
LABEL_38:
        if ( *(_WORD *)(v6 + 18) != 26 )
          goto LABEL_39;
        v86 = v6;
        v54 = sub_14B2B20((_QWORD **)&v103, *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF)));
        v6 = v86;
        if ( !v54 )
        {
          v55 = sub_14B2B20((_QWORD **)&v103, *(_QWORD *)(v86 + 24 * (1LL - (*(_DWORD *)(v86 + 20) & 0xFFFFFFF))));
          v6 = v86;
          if ( !v55 )
            goto LABEL_39;
        }
        goto LABEL_16;
      }
      v39 = *(_DWORD *)(v37 + 20) & 0xFFFFFFF;
      if ( *(_QWORD *)(v37 - 24 * v39) )
      {
        v94 = *(_QWORD *)(v37 - 24LL * (*(_DWORD *)(v37 + 20) & 0xFFFFFFF));
        v40 = sub_14A9880(*(_QWORD *)(v37 + 24 * (1 - v39)));
        v6 = a2;
        if ( v40 )
          goto LABEL_33;
        v39 = *(_DWORD *)(v37 + 20) & 0xFFFFFFF;
      }
      v41 = *(_QWORD *)(v37 + 24 * (1 - v39));
      if ( v41 )
      {
        v79 = v6;
        *(_QWORD *)v103 = v41;
        v42 = sub_14A9880(*(_QWORD *)(v37 - 24LL * (*(_DWORD *)(v37 + 20) & 0xFFFFFFF)));
        v6 = v79;
        if ( v42 )
          goto LABEL_33;
      }
    }
    v36 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    goto LABEL_66;
  }
  v22 = *(_QWORD *)(a1 - 48);
  v23 = *(_BYTE *)(v22 + 16);
  if ( v23 == 52 )
  {
    if ( *(_QWORD *)(v22 - 48) )
    {
      v94 = *(_QWORD *)(v22 - 48);
      v51 = sub_14A9710(*(_QWORD *)(v22 - 24));
      v6 = a2;
      if ( v51 )
        goto LABEL_33;
    }
    v52 = *(_QWORD *)(v22 - 24);
    if ( v52 )
    {
      v85 = v6;
      *(_QWORD *)v103 = v52;
      v53 = sub_14A9710(*(_QWORD *)(v22 - 48));
      v6 = v85;
      if ( v53 )
        goto LABEL_33;
    }
  }
  else if ( v23 == 5 && *(_WORD *)(v22 + 18) == 28 )
  {
    v56 = *(_DWORD *)(v22 + 20) & 0xFFFFFFF;
    if ( *(_QWORD *)(v22 - 24 * v56) )
    {
      v94 = *(_QWORD *)(v22 - 24LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF));
      v57 = sub_14A9880(*(_QWORD *)(v22 + 24 * (1 - v56)));
      v6 = a2;
      if ( v57 )
        goto LABEL_33;
      v56 = *(_DWORD *)(v22 + 20) & 0xFFFFFFF;
    }
    v58 = *(_QWORD *)(v22 + 24 * (1 - v56));
    if ( v58 )
    {
      v87 = v6;
      *(_QWORD *)v103 = v58;
      v59 = sub_14A9880(*(_QWORD *)(v22 - 24LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF)));
      v6 = v87;
      if ( v59 )
        goto LABEL_33;
    }
  }
  v24 = *(_QWORD *)(a1 - 24);
  v25 = *(_BYTE *)(v24 + 16);
  if ( v25 == 52 )
  {
    v47 = *(_QWORD *)(v24 - 48);
    if ( v47 )
    {
      v83 = v6;
      *(_QWORD *)v103 = v47;
      v48 = sub_14A9710(*(_QWORD *)(v24 - 24));
      v6 = v83;
      if ( v48 )
        goto LABEL_33;
    }
    v49 = *(_QWORD *)(v24 - 24);
    if ( v49 )
    {
      v84 = v6;
      *(_QWORD *)v103 = v49;
      v50 = sub_14A9710(*(_QWORD *)(v24 - 48));
      v6 = v84;
      if ( v50 )
        goto LABEL_33;
    }
  }
  else if ( v25 == 5 && *(_WORD *)(v24 + 18) == 28 )
  {
    v26 = *(_DWORD *)(v24 + 20) & 0xFFFFFFF;
    v27 = *(_QWORD *)(v24 - 24 * v26);
    if ( v27 )
    {
      v77 = v6;
      *(_QWORD *)v103 = v27;
      v28 = sub_14A9880(*(_QWORD *)(v24 + 24 * (1LL - (*(_DWORD *)(v24 + 20) & 0xFFFFFFF))));
      v6 = v77;
      if ( v28 )
        goto LABEL_33;
      v26 = *(_DWORD *)(v24 + 20) & 0xFFFFFFF;
    }
    v29 = *(_QWORD *)(v24 + 24 * (1 - v26));
    if ( v29 )
    {
      v78 = v6;
      *(_QWORD *)v103 = v29;
      v30 = sub_14A9880(*(_QWORD *)(v24 - 24LL * (*(_DWORD *)(v24 + 20) & 0xFFFFFFF)));
      v6 = v78;
      if ( v30 )
        goto LABEL_33;
    }
  }
LABEL_4:
  v11 = *(_BYTE *)(v6 + 16);
  v103 = (unsigned __int64)&v94;
  if ( v11 != 50 )
  {
    if ( v11 != 5 )
      goto LABEL_80;
    goto LABEL_38;
  }
  v12 = *(_QWORD *)(v6 - 48);
  v13 = *(_BYTE *)(v12 + 16);
  if ( v13 == 52 )
    goto LABEL_68;
LABEL_6:
  if ( v13 != 5 || *(_WORD *)(v12 + 18) != 28 )
    goto LABEL_8;
  v67 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
  if ( *(_QWORD *)(v12 - 24 * v67) )
  {
    v94 = *(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
    v91 = v6;
    v68 = sub_14A9880(*(_QWORD *)(v12 + 24 * (1 - v67)));
    v6 = v91;
    if ( v68 )
      goto LABEL_16;
    v67 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
  }
  v69 = *(_QWORD *)(v12 + 24 * (1 - v67));
  if ( !v69
    || (v92 = v6,
        *(_QWORD *)v103 = v69,
        v70 = sub_14A9880(*(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF))),
        v6 = v92,
        !v70) )
  {
LABEL_8:
    v14 = *(_QWORD *)(v6 - 24);
    v15 = *(_BYTE *)(v14 + 16);
    if ( v15 == 52 )
    {
      v60 = *(_QWORD *)(v14 - 48);
      if ( v60 )
      {
        v88 = v6;
        v74 = *(_QWORD *)(v6 - 24);
        *(_QWORD *)v103 = v60;
        v61 = sub_14A9710(*(_QWORD *)(v14 - 24));
        v6 = v88;
        if ( v61 )
          goto LABEL_16;
        v14 = v74;
      }
      v62 = *(_QWORD *)(v14 - 24);
      if ( !v62 )
        goto LABEL_80;
      v89 = v6;
      *(_QWORD *)v103 = v62;
      v63 = sub_14A9710(*(_QWORD *)(v14 - 48));
      v6 = v89;
      if ( !v63 )
        goto LABEL_80;
    }
    else
    {
      if ( v15 != 5 )
        goto LABEL_80;
      if ( *(_WORD *)(v14 + 18) != 28 )
        goto LABEL_39;
      v16 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
      v17 = *(_QWORD *)(v14 - 24 * v16);
      if ( v17 )
      {
        v75 = v6;
        v72 = *(_QWORD *)(v6 - 24);
        *(_QWORD *)v103 = v17;
        v18 = sub_14A9880(*(_QWORD *)(v14 + 24 * (1LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF))));
        v6 = v75;
        if ( v18 )
          goto LABEL_16;
        v14 = v72;
        v16 = *(_DWORD *)(v72 + 20) & 0xFFFFFFF;
      }
      v19 = *(_QWORD *)(v14 + 24 * (1 - v16));
      if ( !v19 )
        goto LABEL_80;
      v76 = v6;
      *(_QWORD *)v103 = v19;
      v20 = sub_14A9880(*(_QWORD *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF)));
      v6 = v76;
      if ( !v20 )
        goto LABEL_80;
    }
  }
LABEL_16:
  v21 = *(_BYTE *)(a1 + 16);
  if ( v21 == 50 )
  {
    if ( v94 == *(_QWORD *)(a1 - 48) || v94 == *(_QWORD *)(a1 - 24) )
      goto LABEL_21;
LABEL_80:
    v32 = *(_QWORD *)a1;
    if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 16 )
      goto LABEL_40;
    goto LABEL_81;
  }
  if ( v21 != 5 )
    goto LABEL_80;
  if ( *(_WORD *)(a1 + 18) == 26
    && (v94 == *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF))
     || v94 == *(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)))) )
  {
LABEL_21:
    LODWORD(a5) = 1;
    return (unsigned int)a5;
  }
LABEL_39:
  v32 = *(_QWORD *)a1;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 16 )
    goto LABEL_40;
LABEL_81:
  v32 = **(_QWORD **)(v32 + 16);
LABEL_40:
  v71 = v6;
  v73 = v32;
  sub_14AA4E0((__int64)&v99, *(_DWORD *)(v32 + 8) >> 8);
  sub_14AA4E0((__int64)&v103, *(_DWORD *)(v73 + 8) >> 8);
  sub_14BB090(a1, (__int64)&v99, a3, 0, a4, a5, a6, 0);
  sub_14BB090(v71, (__int64)&v103, a3, 0, a4, a5, a6, 0);
  LOBYTE(a4) = v100;
  v96 = v100;
  if ( v100 <= 0x40 )
  {
    v33 = v99;
LABEL_42:
    v34 = v103 | v33;
    goto LABEL_43;
  }
  sub_16A4FD0(&v95, &v99);
  LOBYTE(a4) = v96;
  if ( v96 <= 0x40 )
  {
    v33 = v95;
    goto LABEL_42;
  }
  sub_16A89F0(&v95, &v103);
  LODWORD(a4) = v96;
  v34 = v95;
  v96 = 0;
  v98 = a4;
  v97 = v95;
  if ( (unsigned int)a4 > 0x40 )
  {
    LOBYTE(a5) = (_DWORD)a4 == (unsigned int)sub_16A58F0(&v97);
    if ( v34 )
    {
      j_j___libc_free_0_0(v34);
      if ( v96 > 0x40 )
      {
        if ( v95 )
          j_j___libc_free_0_0(v95);
      }
    }
    goto LABEL_44;
  }
LABEL_43:
  LOBYTE(a5) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)a4) == v34;
LABEL_44:
  if ( v106 > 0x40 && v105 )
    j_j___libc_free_0_0(v105);
  if ( v104 > 0x40 && v103 )
    j_j___libc_free_0_0(v103);
  if ( v102 > 0x40 && v101 )
    j_j___libc_free_0_0(v101);
  if ( v100 > 0x40 && v99 )
    j_j___libc_free_0_0(v99);
  return (unsigned int)a5;
}
