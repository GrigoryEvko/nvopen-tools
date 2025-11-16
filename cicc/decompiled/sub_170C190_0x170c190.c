// Function: sub_170C190
// Address: 0x170c190
//
__int64 __fastcall sub_170C190(__int64 *a1, __int64 a2, unsigned __int64 *a3, bool *a4)
{
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r15
  unsigned int v11; // r15d
  bool v12; // r9
  int v13; // eax
  int v14; // eax
  int v15; // r10d
  unsigned __int8 v16; // al
  __int64 v17; // r14
  __int64 v18; // rcx
  _QWORD **v19; // r11
  bool v20; // al
  _QWORD *v21; // rdx
  int v22; // r10d
  __int64 v23; // rcx
  __int64 v24; // r15
  unsigned __int64 v25; // r13
  __int64 v26; // rax
  int v27; // eax
  unsigned __int64 v28; // rax
  _QWORD *v29; // rax
  unsigned int v30; // r13d
  __int64 v31; // rax
  bool v32; // cc
  __int64 v33; // rdx
  __int64 v34; // rcx
  unsigned int v35; // r14d
  bool v36; // al
  __int64 v37; // rdx
  unsigned int v38; // eax
  __int64 v39; // r11
  int v40; // r10d
  int v41; // eax
  __int64 v42; // rdx
  __int64 v43; // rax
  unsigned int v44; // r13d
  bool v45; // al
  __int64 v46; // rcx
  __int64 *v47; // rax
  __int64 v48; // rsi
  unsigned __int64 v49; // rcx
  __int64 v50; // rcx
  char v51; // al
  unsigned __int8 v52; // si
  int v53; // ecx
  unsigned int v54; // eax
  bool v55; // al
  char v56; // al
  __int64 v57; // rax
  unsigned int v58; // r14d
  __int64 v59; // rax
  unsigned int v60; // r13d
  int v61; // r14d
  unsigned int v62; // r15d
  __int64 v63; // rax
  char v64; // dl
  bool v65; // al
  bool v66; // al
  __int64 v67; // rax
  bool v68; // al
  __int64 v69; // rax
  __int64 v70; // rax
  unsigned int v71; // r13d
  __int64 v72; // rax
  char v73; // si
  bool v74; // al
  int v75; // eax
  _QWORD **v76; // [rsp+0h] [rbp-80h]
  __int64 v77; // [rsp+8h] [rbp-78h]
  unsigned int v78; // [rsp+8h] [rbp-78h]
  __int64 v79; // [rsp+8h] [rbp-78h]
  int v80; // [rsp+10h] [rbp-70h]
  __int64 v81; // [rsp+10h] [rbp-70h]
  _QWORD **v82; // [rsp+10h] [rbp-70h]
  _QWORD **v83; // [rsp+10h] [rbp-70h]
  _QWORD **v84; // [rsp+10h] [rbp-70h]
  int v85; // [rsp+10h] [rbp-70h]
  int v86; // [rsp+10h] [rbp-70h]
  __int64 v87; // [rsp+10h] [rbp-70h]
  unsigned int v88; // [rsp+1Ch] [rbp-64h]
  int v89; // [rsp+1Ch] [rbp-64h]
  int v90; // [rsp+1Ch] [rbp-64h]
  int v91; // [rsp+1Ch] [rbp-64h]
  int v92; // [rsp+1Ch] [rbp-64h]
  bool v93; // [rsp+20h] [rbp-60h]
  __int64 v94; // [rsp+20h] [rbp-60h]
  __int64 v95; // [rsp+20h] [rbp-60h]
  bool v96; // [rsp+20h] [rbp-60h]
  bool v97; // [rsp+20h] [rbp-60h]
  int v98; // [rsp+20h] [rbp-60h]
  bool v99; // [rsp+20h] [rbp-60h]
  int v100; // [rsp+20h] [rbp-60h]
  int v101; // [rsp+20h] [rbp-60h]
  unsigned __int64 v102; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v103; // [rsp+38h] [rbp-48h]
  _QWORD *v104; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v105; // [rsp+48h] [rbp-38h]

  v6 = a2;
  if ( *(_BYTE *)(a2 + 16) > 0x10u )
    goto LABEL_7;
  if ( sub_1593BB0(a2, a2, (__int64)a3, (__int64)a4) )
  {
LABEL_3:
    *a4 = 1;
    return v6;
  }
  if ( *(_BYTE *)(a2 + 16) == 13 )
  {
    v35 = *(_DWORD *)(a2 + 32);
    if ( v35 <= 0x40 )
      v36 = *(_QWORD *)(a2 + 24) == 0;
    else
      v36 = v35 == (unsigned int)sub_16A57B0(a2 + 24);
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 16 )
      goto LABEL_7;
    v57 = sub_15A1020((_BYTE *)a2, a2, v7, v8);
    if ( !v57 || *(_BYTE *)(v57 + 16) != 13 )
    {
      v61 = *(_QWORD *)(*(_QWORD *)a2 + 32LL);
      if ( !v61 )
        goto LABEL_3;
      v62 = 0;
      while ( 1 )
      {
        a2 = v62;
        v63 = sub_15A0A60(v6, v62);
        if ( !v63 )
          goto LABEL_7;
        v64 = *(_BYTE *)(v63 + 16);
        if ( v64 != 9 )
        {
          if ( v64 != 13 )
            goto LABEL_7;
          if ( *(_DWORD *)(v63 + 32) <= 0x40u )
          {
            v65 = *(_QWORD *)(v63 + 24) == 0;
          }
          else
          {
            v98 = *(_DWORD *)(v63 + 32);
            v65 = v98 == (unsigned int)sub_16A57B0(v63 + 24);
          }
          if ( !v65 )
            goto LABEL_7;
        }
        if ( v61 == ++v62 )
          goto LABEL_3;
      }
    }
    v58 = *(_DWORD *)(v57 + 32);
    if ( v58 <= 0x40 )
      v36 = *(_QWORD *)(v57 + 24) == 0;
    else
      v36 = v58 == (unsigned int)sub_16A57B0(v57 + 24);
  }
  if ( v36 )
    goto LABEL_3;
LABEL_7:
  v11 = *((_DWORD *)a3 + 2);
  if ( v11 > 0x40 )
  {
    v27 = sub_16A57B0((__int64)a3);
    if ( v11 - v27 <= 0x40 && *(_QWORD *)*a3 == 1 )
      goto LABEL_3;
    v12 = v11 == v27;
  }
  else
  {
    v12 = *a3 == 0;
    if ( *a3 == 1 )
      goto LABEL_3;
  }
  if ( v12 )
    return 0;
  if ( v11 <= 0x40 )
  {
    v28 = *a3;
    if ( !*a3 || (v28 & (v28 - 1)) != 0 )
    {
      v16 = *(_BYTE *)(v6 + 16);
      if ( v16 != 13 )
        goto LABEL_61;
    }
    else
    {
      _BitScanReverse64(&v28, v28);
      v15 = (v28 ^ 0xFFFFFFC0) + 64;
      v16 = *(_BYTE *)(v6 + 16);
      if ( v16 != 13 )
        goto LABEL_13;
    }
    v103 = v11;
    v17 = 0;
    v94 = v6;
    v88 = 0;
LABEL_43:
    v29 = (_QWORD *)*a3;
    v105 = v11;
    v102 = (unsigned __int64)v29;
LABEL_44:
    v104 = (_QWORD *)*a3;
    goto LABEL_45;
  }
  v13 = sub_16A5940((__int64)a3);
  v12 = 0;
  if ( v13 != 1 )
  {
    v16 = *(_BYTE *)(v6 + 16);
    if ( v16 == 13 )
      goto LABEL_98;
LABEL_61:
    v15 = -1;
    goto LABEL_13;
  }
  v14 = sub_16A57B0((__int64)a3);
  v12 = 0;
  v15 = v11 - 1 - v14;
  v16 = *(_BYTE *)(v6 + 16);
  if ( v16 != 13 )
  {
LABEL_13:
    v88 = 0;
    v17 = v6;
    v18 = 0;
    v19 = &v104;
    while ( 1 )
    {
      if ( v16 <= 0x17u )
        return 0;
      if ( (unsigned int)v16 - 35 > 0x11 )
        goto LABEL_62;
      if ( v16 != 39 )
        break;
      v76 = v19;
      v79 = v18;
      v85 = v15;
      v99 = v12;
      v66 = sub_15F2380(v17);
      v12 = v99;
      v15 = v85;
      v23 = v79;
      v19 = v76;
      *a4 = v66;
      if ( v99 && !v66 )
        return 0;
      v67 = *(_QWORD *)(v17 - 24);
      if ( *(_BYTE *)(v67 + 16) != 13 )
      {
        v70 = *(_QWORD *)(v17 + 8);
        if ( !v70 || *(_QWORD *)(v70 + 8) )
          return 0;
LABEL_111:
        v88 = 0;
        v42 = 0;
        goto LABEL_72;
      }
      v9 = *(_QWORD *)(v17 - 48);
      if ( *(_DWORD *)(v67 + 32) <= 0x40u )
      {
        v21 = (_QWORD *)*a3;
        if ( *(_QWORD *)(v67 + 24) == *a3 )
        {
LABEL_163:
          v17 = v23;
          goto LABEL_53;
        }
      }
      else
      {
        a2 = (__int64)a3;
        v68 = sub_16A5220(v67 + 24, (const void **)a3);
        v12 = v99;
        v15 = v85;
        v23 = v79;
        v19 = v76;
        if ( v68 )
          goto LABEL_163;
      }
      v69 = *(_QWORD *)(v17 + 8);
      if ( !v69 || *(_QWORD *)(v69 + 8) )
        return 0;
      v88 = 1;
      v42 = 24;
LABEL_72:
      if ( (*(_BYTE *)(v17 + 23) & 0x40) != 0 )
        v43 = *(_QWORD *)(v17 - 8);
      else
        v43 = v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF);
      v94 = *(_QWORD *)(v43 + v42);
      v16 = *(_BYTE *)(v94 + 16);
      if ( v16 == 13 )
      {
        v11 = *((_DWORD *)a3 + 2);
        v103 = v11;
        if ( v11 <= 0x40 )
          goto LABEL_43;
        goto LABEL_99;
      }
      v18 = v17;
      v17 = v94;
    }
    if ( v16 == 47 && v15 > 0 )
    {
      if ( *(_BYTE *)(*(_QWORD *)(v17 - 24) + 16LL) == 13 )
      {
        v77 = v18;
        v80 = v15;
        v93 = v12;
        v20 = sub_15F2380(v17);
        v22 = v80;
        *a4 = v20;
        v23 = v77;
        if ( !v93 || v20 )
        {
          v24 = *(_QWORD *)(v17 - 24);
          v25 = *((unsigned int *)a3 + 2);
          v78 = *(_DWORD *)(v24 + 32);
          if ( v78 > 0x40 )
          {
            v87 = v23;
            v101 = v22;
            v75 = sub_16A57B0(v24 + 24);
            v22 = v101;
            v23 = v87;
            if ( v78 - v75 <= 0x40 && v25 > **(_QWORD **)(v24 + 24) )
              v25 = **(_QWORD **)(v24 + 24);
          }
          else if ( v25 > *(_QWORD *)(v24 + 24) )
          {
            v25 = *(_QWORD *)(v24 + 24);
          }
          a2 = (unsigned int)v25;
          if ( (_DWORD)v25 == v22 )
          {
            v9 = *(_QWORD *)(v17 - 48);
            v17 = v23;
            goto LABEL_53;
          }
          if ( (int)v25 >= v22 )
          {
            v26 = *(_QWORD *)(v17 + 8);
            if ( v26 )
            {
              if ( !*(_QWORD *)(v26 + 8) )
              {
                a2 = (int)v25 - v22;
                v88 = 1;
                v9 = sub_15A0680(*(_QWORD *)v17, a2, 0);
                if ( *(_BYTE *)(v9 + 16) <= 0x10u )
                  goto LABEL_54;
                goto LABEL_82;
              }
            }
          }
        }
      }
      return 0;
    }
LABEL_62:
    v37 = *(_QWORD *)(v17 + 8);
    if ( !v37 || *(_QWORD *)(v37 + 8) || (unsigned __int8)(v16 - 60) > 0xCu )
      return 0;
    if ( v16 == 62 )
    {
      v91 = v15;
      v95 = (__int64)v19;
      v54 = sub_1643030(**(_QWORD **)(v17 - 24));
      sub_16A5A50((__int64)&v102, (__int64 *)a3, v54);
      a2 = (__int64)&v102;
      sub_16A5B10(v95, &v102, *((_DWORD *)a3 + 2));
      v19 = (_QWORD **)v95;
      v15 = v91;
      if ( v105 <= 0x40 )
      {
        v12 = v104 == (_QWORD *)*a3;
      }
      else
      {
        a2 = (__int64)a3;
        v55 = sub_16A5220(v95, (const void **)a3);
        v19 = (_QWORD **)v95;
        v15 = v91;
        v12 = v55;
        if ( v104 )
        {
          v83 = (_QWORD **)v95;
          v96 = v55;
          j_j___libc_free_0_0(v104);
          v19 = v83;
          v15 = v91;
          v12 = v96;
        }
      }
      if ( !v12 )
        goto LABEL_32;
      if ( *((_DWORD *)a3 + 2) <= 0x40u && (v56 = v103, v103 <= 0x40) )
      {
        *((_DWORD *)a3 + 2) = v103;
        *a3 = (0xFFFFFFFFFFFFFFFFLL >> -v56) & v102;
      }
      else
      {
        a2 = (__int64)&v102;
        v84 = v19;
        v92 = v15;
        v97 = v12;
        sub_16A51C0((__int64)a3, (__int64)&v102);
        v12 = v97;
        v15 = v92;
        v19 = v84;
        if ( v103 > 0x40 && v102 )
        {
          j_j___libc_free_0_0(v102);
          v19 = v84;
          v15 = v92;
          v12 = v97;
        }
      }
      goto LABEL_111;
    }
    v81 = (__int64)v19;
    v89 = v15;
    if ( v16 != 60 || v12 )
      return 0;
    v38 = sub_1643030(**(_QWORD **)(v17 - 24));
    a2 = (__int64)a3;
    sub_16A5B10(v81, a3, v38);
    v39 = v81;
    v40 = v89;
    if ( *((_DWORD *)a3 + 2) > 0x40u && *a3 )
    {
      j_j___libc_free_0_0(*a3);
      v39 = v81;
      v40 = v89;
    }
    v82 = (_QWORD **)v39;
    v90 = v40;
    *a3 = (unsigned __int64)v104;
    *((_DWORD *)a3 + 2) = v105;
    v41 = sub_1643030(*(_QWORD *)v17);
    v15 = v90;
    v42 = 0;
    v12 = 0;
    v88 = 0;
    v19 = v82;
    if ( v15 + 1 == v41 )
      v15 = -1;
    goto LABEL_72;
  }
LABEL_98:
  v103 = v11;
  v17 = 0;
  v94 = v6;
  v88 = 0;
LABEL_99:
  sub_16A4FD0((__int64)&v102, (const void **)a3);
  v105 = *((_DWORD *)a3 + 2);
  if ( v105 <= 0x40 )
    goto LABEL_44;
  sub_16A4FD0((__int64)&v104, (const void **)a3);
LABEL_45:
  sub_16AE5C0(v94 + 24, (__int64)a3, (__int64)&v102, (__int64)&v104);
  v30 = v105;
  if ( v105 <= 0x40 )
  {
    if ( !v104 )
      goto LABEL_47;
    goto LABEL_32;
  }
  if ( v30 != (unsigned int)sub_16A57B0((__int64)&v104) )
  {
    if ( v104 )
      j_j___libc_free_0_0(v104);
LABEL_32:
    if ( v103 > 0x40 && v102 )
      j_j___libc_free_0_0(v102);
    return 0;
  }
LABEL_47:
  a2 = (__int64)&v102;
  v31 = sub_15A1070(*(_QWORD *)v94, (__int64)&v102);
  v32 = v105 <= 0x40;
  *a4 = 1;
  v9 = v31;
  if ( !v32 && v104 )
    j_j___libc_free_0_0(v104);
  if ( v103 > 0x40 && v102 )
    j_j___libc_free_0_0(v102);
LABEL_53:
  if ( *(_BYTE *)(v9 + 16) > 0x10u )
    goto LABEL_81;
LABEL_54:
  if ( sub_1593BB0(v9, a2, (__int64)v21, v23) )
  {
LABEL_55:
    *a4 = 1;
    return v9;
  }
  if ( *(_BYTE *)(v9 + 16) == 13 )
  {
    v44 = *(_DWORD *)(v9 + 32);
    if ( v44 <= 0x40 )
      v45 = *(_QWORD *)(v9 + 24) == 0;
    else
      v45 = v44 == (unsigned int)sub_16A57B0(v9 + 24);
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)v9 + 8LL) != 16 )
      goto LABEL_81;
    v59 = sub_15A1020((_BYTE *)v9, a2, v33, v34);
    if ( !v59 || *(_BYTE *)(v59 + 16) != 13 )
    {
      v100 = *(_QWORD *)(*(_QWORD *)v9 + 32LL);
      if ( !v100 )
        goto LABEL_55;
      v71 = 0;
      while ( 1 )
      {
        v72 = sub_15A0A60(v9, v71);
        if ( !v72 )
          goto LABEL_81;
        v73 = *(_BYTE *)(v72 + 16);
        if ( v73 != 9 )
        {
          if ( v73 != 13 )
            goto LABEL_81;
          if ( *(_DWORD *)(v72 + 32) <= 0x40u )
          {
            v74 = *(_QWORD *)(v72 + 24) == 0;
          }
          else
          {
            v86 = *(_DWORD *)(v72 + 32);
            v74 = v86 == (unsigned int)sub_16A57B0(v72 + 24);
          }
          if ( !v74 )
            goto LABEL_81;
        }
        if ( v100 == ++v71 )
          goto LABEL_55;
      }
    }
    v60 = *(_DWORD *)(v59 + 32);
    if ( v60 <= 0x40 )
      v45 = *(_QWORD *)(v59 + 24) == 0;
    else
      v45 = v60 == (unsigned int)sub_16A57B0(v59 + 24);
  }
  if ( v45 )
    goto LABEL_55;
LABEL_81:
  if ( v17 )
  {
LABEL_82:
    if ( (*(_BYTE *)(v17 + 23) & 0x40) != 0 )
      v46 = *(_QWORD *)(v17 - 8);
    else
      v46 = v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF);
    v47 = (__int64 *)(v46 + 24LL * v88);
    if ( *v47 )
    {
      v48 = v47[1];
      v49 = v47[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v49 = v48;
      if ( v48 )
        *(_QWORD *)(v48 + 16) = *(_QWORD *)(v48 + 16) & 3LL | v49;
    }
    *v47 = v9;
    v50 = *(_QWORD *)(v9 + 8);
    v47[1] = v50;
    if ( v50 )
      *(_QWORD *)(v50 + 16) = (unsigned __int64)(v47 + 1) | *(_QWORD *)(v50 + 16) & 3LL;
    v47[2] = (v9 + 8) | v47[2] & 3;
    *(_QWORD *)(v9 + 8) = v47;
    v9 = v17;
    sub_170B990(*a1, v17);
    while ( 1 )
    {
      v53 = *(unsigned __int8 *)(v9 + 16);
      if ( (unsigned int)(v53 - 35) <= 0x11 )
      {
        v51 = sub_15F2380(v9);
        v52 = v51 & *a4;
        *a4 = v52;
        if ( v51 != v52 )
        {
          sub_15F2330(v9, v52);
          sub_170B990(*a1, v9);
        }
      }
      else if ( (_BYTE)v53 == 60 )
      {
        *a4 = 0;
      }
      if ( v6 == v9 )
        break;
      v9 = (__int64)sub_1648700(*(_QWORD *)(v9 + 8));
    }
  }
  return v9;
}
