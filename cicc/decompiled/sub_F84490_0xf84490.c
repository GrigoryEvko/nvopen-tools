// Function: sub_F84490
// Address: 0xf84490
//
char __fastcall sub_F84490(__int64 a1, unsigned __int8 **a2, unsigned __int8 **a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v9; // rax
  unsigned __int8 *v10; // rcx
  _QWORD *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // rax
  unsigned __int8 *v15; // rsi
  __int64 v16; // r13
  __int64 v17; // r9
  __int64 v18; // rax
  int v19; // edi
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // r14
  __int64 v23; // r12
  __int64 *v24; // rax
  _QWORD *v25; // r12
  unsigned __int8 v26; // cl
  __int64 v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // rax
  int v30; // esi
  __int64 v31; // r8
  int v32; // esi
  unsigned int v33; // r9d
  __int64 *v34; // rax
  __int64 v35; // r14
  _QWORD *v36; // rdx
  unsigned int v37; // r9d
  __int64 *v38; // rax
  __int64 v39; // r14
  unsigned __int8 v40; // al
  char v41; // dl
  char v42; // dh
  __int64 v43; // r12
  char v44; // al
  __int64 v45; // r14
  __int16 v46; // cx
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // r9
  __int64 v50; // r8
  __int64 v51; // rsi
  __int64 v52; // r14
  __int64 v53; // r12
  bool v54; // zf
  unsigned int v55; // eax
  __int64 v56; // r14
  __int64 v57; // r9
  __int64 v58; // rdx
  _QWORD *v59; // rdi
  int v60; // r9d
  __int64 v61; // rsi
  int v62; // r9d
  __int64 v63; // rax
  unsigned __int8 *v64; // rcx
  int v65; // eax
  int v66; // r10d
  int v67; // r10d
  bool v68; // r12
  unsigned __int8 *v69; // rax
  int v70; // eax
  __int64 v71; // r9
  int v72; // r10d
  __int64 v73; // rdi
  __int64 v74; // rcx
  unsigned __int8 *v75; // rax
  __int64 v76; // rax
  char v77; // r15
  __int64 v78; // rax
  int v79; // r12d
  unsigned int *v80; // r12
  __int64 v81; // rax
  unsigned int *v82; // r13
  __int64 v83; // rdx
  int v84; // r10d
  unsigned __int64 *v85; // r14
  unsigned __int64 *v86; // rdi
  int v87; // r13d
  int v88; // eax
  __int16 v90; // [rsp+0h] [rbp-150h]
  char v91; // [rsp+18h] [rbp-138h]
  int v94; // [rsp+28h] [rbp-128h]
  int v95; // [rsp+28h] [rbp-128h]
  __int64 v96[4]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v97; // [rsp+50h] [rbp-100h]
  __int64 v98[2]; // [rsp+60h] [rbp-F0h] BYREF
  unsigned __int8 *v99; // [rsp+70h] [rbp-E0h]
  __int16 v100; // [rsp+80h] [rbp-D0h]
  unsigned int *v101; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v102; // [rsp+98h] [rbp-B8h]
  _QWORD v103[5]; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v104; // [rsp+C8h] [rbp-88h]
  __int64 v105; // [rsp+D0h] [rbp-80h]
  __int64 v106; // [rsp+D8h] [rbp-78h]
  void **v107; // [rsp+E0h] [rbp-70h]
  void **v108; // [rsp+E8h] [rbp-68h]
  __int64 v109; // [rsp+F0h] [rbp-60h]
  int v110; // [rsp+F8h] [rbp-58h]
  __int16 v111; // [rsp+FCh] [rbp-54h]
  char v112; // [rsp+FEh] [rbp-52h]
  __int64 v113; // [rsp+100h] [rbp-50h]
  __int64 v114; // [rsp+108h] [rbp-48h]
  void *v115; // [rsp+110h] [rbp-40h] BYREF
  void *v116; // [rsp+118h] [rbp-38h] BYREF

  v9 = (_QWORD *)sub_D47930(a4);
  if ( !v9 )
    return (char)v9;
  v10 = *a3;
  v11 = v9;
  v12 = 0x1FFFFFFFE0LL;
  v13 = *((_QWORD *)*a3 - 1);
  if ( (*((_DWORD *)*a3 + 1) & 0x7FFFFFF) != 0 )
  {
    v14 = 0;
    do
    {
      if ( v11 == *(_QWORD **)(v13 + 32LL * *((unsigned int *)v10 + 18) + 8 * v14) )
      {
        v12 = 32 * v14;
        goto LABEL_7;
      }
      ++v14;
    }
    while ( (*((_DWORD *)*a3 + 1) & 0x7FFFFFF) != (_DWORD)v14 );
    v12 = 0x1FFFFFFFE0LL;
  }
LABEL_7:
  v15 = *a2;
  v16 = *(_QWORD *)(v13 + v12);
  if ( *(_BYTE *)v16 <= 0x1Cu )
    v16 = 0;
  v17 = *((_QWORD *)v15 - 1);
  v18 = 0x1FFFFFFFE0LL;
  v19 = *((_DWORD *)*a2 + 1) & 0x7FFFFFF;
  if ( v19 )
  {
    v20 = 0;
    do
    {
      if ( v11 == *(_QWORD **)(v17 + 32LL * *((unsigned int *)v15 + 18) + 8 * v20) )
      {
        v18 = 32 * v20;
        goto LABEL_14;
      }
      ++v20;
    }
    while ( v19 != (_DWORD)v20 );
    v21 = *(_QWORD *)(v17 + 0x1FFFFFFFE0LL);
    if ( !v21 )
LABEL_22:
      BUG();
  }
  else
  {
LABEL_14:
    v21 = *(_QWORD *)(v17 + v18);
    if ( !v21 )
      goto LABEL_22;
  }
  LOBYTE(v9) = v16 == 0 || *(_BYTE *)v21 <= 0x1Cu;
  v91 = (char)v9;
  if ( (_BYTE)v9 )
    return (char)v9;
  if ( *((_QWORD *)v10 + 1) == *((_QWORD *)v15 + 1) )
  {
    v99 = *a2;
    v98[0] = 0;
    v98[1] = 0;
    if ( v15 != (unsigned __int8 *)-8192LL && v15 != (unsigned __int8 *)-4096LL )
      sub_BD73F0((__int64)v98);
    v60 = *(_DWORD *)(a1 + 504);
    if ( v60 )
    {
      v61 = *(_QWORD *)(a1 + 488);
      v62 = v60 - 1;
      LODWORD(v63) = v62 & (((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4));
      v64 = *(unsigned __int8 **)(v61 + 24LL * (unsigned int)v63 + 16);
      if ( v64 == v99 )
      {
LABEL_65:
        sub_D68D70(v98);
        goto LABEL_17;
      }
      v84 = 1;
      while ( v64 != (unsigned __int8 *)-4096LL )
      {
        v63 = v62 & (unsigned int)(v63 + v84);
        v64 = *(unsigned __int8 **)(v61 + 24 * v63 + 16);
        if ( v99 == v64 )
          goto LABEL_65;
        ++v84;
      }
    }
    if ( (unsigned __int8)sub_F7DC80(a1, *a3, (unsigned __int8 *)v16, a4) )
      goto LABEL_65;
    v69 = *a2;
    v101 = 0;
    v102 = 0;
    v103[0] = v69;
    if ( v69 + 4096 != 0 && v69 != 0 && v69 != (unsigned __int8 *)-8192LL )
      sub_BD73F0((__int64)&v101);
    v70 = *(_DWORD *)(a1 + 504);
    if ( v70 )
    {
      v71 = *(_QWORD *)(a1 + 488);
      v72 = v70 - 1;
      LODWORD(v73) = (v70 - 1) & ((LODWORD(v103[0]) >> 9) ^ (LODWORD(v103[0]) >> 4));
      v74 = *(_QWORD *)(v71 + 24LL * (unsigned int)v73 + 16);
      if ( v103[0] == v74 )
      {
LABEL_89:
        sub_D68D70(&v101);
        sub_D68D70(v98);
        goto LABEL_90;
      }
      v88 = 1;
      while ( v74 != -4096 )
      {
        v73 = v72 & (unsigned int)(v73 + v88);
        v74 = *(_QWORD *)(v71 + 24 * v73 + 16);
        if ( v103[0] == v74 )
          goto LABEL_89;
        ++v88;
      }
    }
    v77 = sub_F7DC80(a1, *a2, (unsigned __int8 *)v21, a4);
    sub_D68D70(&v101);
    sub_D68D70(v98);
    if ( !v77 )
      goto LABEL_17;
LABEL_90:
    v75 = *a3;
    *a3 = *a2;
    *a2 = v75;
    v76 = v16;
    v16 = v21;
    v21 = v76;
  }
LABEL_17:
  v22 = *(_QWORD *)(v21 + 8);
  v23 = *(_QWORD *)a1;
  v24 = sub_DD8400(*(_QWORD *)a1, v16);
  v9 = sub_DC5820(v23, (__int64)v24, v22);
  v25 = v9;
  if ( v16 == v21 )
    return (char)v9;
  v9 = sub_DD8400(*(_QWORD *)a1, v21);
  if ( v25 != v9 )
    return (char)v9;
  v26 = *(_BYTE *)v16;
  if ( *(_BYTE *)v16 <= 0x1Cu )
  {
LABEL_74:
    if ( (v26 & 0xFD) != 0x2C && v26 != 54 )
    {
LABEL_36:
      LOBYTE(v9) = sub_F841B0((__int64 **)a1, (unsigned __int8 *)v16, v21, 1u);
      if ( !(_BYTE)v9 )
        return (char)v9;
      goto LABEL_37;
    }
LABEL_33:
    v40 = *(_BYTE *)v21;
    if ( *(_BYTE *)v21 != 42 && (v40 & 0xFD) != 0x2C && v40 != 54 )
      goto LABEL_36;
    v68 = (*(_BYTE *)(v16 + 1) & 2) != 0;
    if ( (*(_BYTE *)(v16 + 1) & 2) != 0 )
      v68 = (*(_BYTE *)(v21 + 1) & 2) != 0;
    if ( ((*(_BYTE *)(v16 + 1) >> 1) & 2) != 0 && (*(_BYTE *)(v21 + 1) & 4) != 0 )
    {
      LOBYTE(v9) = sub_F841B0((__int64 **)a1, (unsigned __int8 *)v16, v21, 1u);
      v91 = (char)v9;
      if ( !(_BYTE)v9 )
        return (char)v9;
    }
    else
    {
      LOBYTE(v9) = sub_F841B0((__int64 **)a1, (unsigned __int8 *)v16, v21, 1u);
      if ( !(_BYTE)v9 )
        return (char)v9;
      if ( !v68 )
        goto LABEL_37;
    }
    sub_B447F0((unsigned __int8 *)v16, v68 || (*(_BYTE *)(v16 + 1) & 2) != 0);
    sub_B44850((unsigned __int8 *)v16, ((*(_BYTE *)(v16 + 1) & 4) != 0) | v91);
LABEL_37:
    if ( *(_QWORD *)(v21 + 8) == *(_QWORD *)(v16 + 8) )
    {
      v56 = v16;
    }
    else
    {
      if ( *(_BYTE *)v16 == 84 )
      {
        v43 = sub_AA5190(*(_QWORD *)(v16 + 40));
        if ( !v43 )
          BUG();
        v44 = v42;
      }
      else
      {
        v78 = sub_B46B10(v16, 0);
        v41 = 0;
        v43 = v78 + 24;
        v44 = 0;
      }
      v45 = *(_QWORD *)(v43 + 16);
      LOBYTE(v46) = v41;
      HIBYTE(v46) = v44;
      v90 = v46;
      v47 = sub_AA48A0(v45);
      v108 = &v116;
      v106 = v47;
      v107 = &v115;
      v101 = (unsigned int *)v103;
      v102 = 0x200000000LL;
      v115 = &unk_49DA100;
      v111 = 512;
      LOWORD(v105) = 0;
      v109 = 0;
      v110 = 0;
      v112 = 7;
      v113 = 0;
      v114 = 0;
      v103[4] = 0;
      v104 = 0;
      v116 = &unk_49DA0B0;
      sub_A88F30((__int64)&v101, v45, v43, v90);
      v50 = *(_QWORD *)(v21 + 48);
      v98[0] = v50;
      if ( v50 )
      {
        sub_B96E90((__int64)v98, v50, 1);
        v50 = v98[0];
      }
      sub_F80810((__int64)&v101, 0, v50, v48, v50, v49);
      v51 = v98[0];
      if ( v98[0] )
        sub_B91220((__int64)v98, v98[0]);
      v97 = 257;
      if ( **(_BYTE **)(a1 + 16) )
      {
        v96[0] = *(_QWORD *)(a1 + 16);
        LOBYTE(v97) = 3;
      }
      v52 = *(_QWORD *)(v16 + 8);
      v53 = *(_QWORD *)(v21 + 8);
      v94 = sub_BCB060(v52);
      v54 = v94 == (unsigned int)sub_BCB060(v53);
      v55 = 38;
      if ( v54 )
        v55 = 49;
      if ( v53 == v52 )
      {
        v56 = v16;
      }
      else
      {
        v51 = v55;
        v95 = v55;
        v56 = (*((__int64 (__fastcall **)(void **, _QWORD, __int64, __int64))*v107 + 15))(v107, v55, v16, v53);
        if ( !v56 )
        {
          v100 = 257;
          v56 = sub_B51D30(v95, v16, v53, (__int64)v98, 0, 0);
          if ( (unsigned __int8)sub_920620(v56) )
          {
            v79 = v110;
            if ( v109 )
              sub_B99FD0(v56, 3u, v109);
            sub_B45150(v56, v79);
          }
          v51 = v56;
          (*((void (__fastcall **)(void **, __int64, __int64 *, __int64, __int64))*v108 + 2))(
            v108,
            v56,
            v96,
            v104,
            v105);
          v80 = v101;
          v81 = 4LL * (unsigned int)v102;
          v82 = &v101[v81];
          if ( v101 != &v101[v81] )
          {
            do
            {
              v83 = *((_QWORD *)v80 + 1);
              v51 = *v80;
              v80 += 4;
              sub_B99FD0(v56, v51, v83);
            }
            while ( v82 != v80 );
          }
        }
      }
      nullsub_61();
      v115 = &unk_49DA100;
      nullsub_63();
      if ( v101 != (unsigned int *)v103 )
        _libc_free(v101, v51);
    }
    sub_BD84D0(v21, v56);
    v58 = *(unsigned int *)(a6 + 8);
    LODWORD(v9) = v58;
    if ( *(_DWORD *)(a6 + 12) <= (unsigned int)v58 )
    {
      v85 = (unsigned __int64 *)sub_C8D7D0(a6, a6 + 16, 0, 0x18u, (unsigned __int64 *)&v101, v57);
      v86 = &v85[3 * *(unsigned int *)(a6 + 8)];
      if ( v86 )
      {
        *v86 = 6;
        v86[1] = 0;
        v86[2] = v21;
        if ( v21 != -8192 && v21 != -4096 )
          sub_BD73F0((__int64)v86);
      }
      sub_F17F80(a6, v85);
      v87 = (int)v101;
      if ( a6 + 16 != *(_QWORD *)a6 )
        _libc_free(*(_QWORD *)a6, v85);
      ++*(_DWORD *)(a6 + 8);
      *(_QWORD *)a6 = v85;
      *(_DWORD *)(a6 + 12) = v87;
      LOBYTE(v9) = a6;
    }
    else
    {
      v59 = (_QWORD *)(*(_QWORD *)a6 + 24 * v58);
      if ( v59 )
      {
        *v59 = 6;
        v59[1] = 0;
        v59[2] = v21;
        if ( v21 != -8192 && v21 != -4096 )
          sub_BD73F0((__int64)v59);
        LODWORD(v9) = *(_DWORD *)(a6 + 8);
      }
      LODWORD(v9) = (_DWORD)v9 + 1;
      *(_DWORD *)(a6 + 8) = (_DWORD)v9;
    }
    return (char)v9;
  }
  v27 = *(_QWORD *)(v16 + 40);
  v28 = *(_QWORD *)(v21 + 40);
  if ( v27 == v28
    || (v29 = *(_QWORD *)(*(_QWORD *)a1 + 48LL), v30 = *(_DWORD *)(v29 + 24), v31 = *(_QWORD *)(v29 + 8), !v30) )
  {
LABEL_32:
    if ( v26 == 42 )
      goto LABEL_33;
    goto LABEL_74;
  }
  v32 = v30 - 1;
  v33 = v32 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
  v34 = (__int64 *)(v31 + 16LL * v33);
  v35 = *v34;
  if ( v27 != *v34 )
  {
    v65 = 1;
    while ( v35 != -4096 )
    {
      v66 = v65 + 1;
      v33 = v32 & (v65 + v33);
      v34 = (__int64 *)(v31 + 16LL * v33);
      v35 = *v34;
      if ( v27 == *v34 )
        goto LABEL_27;
      v65 = v66;
    }
    goto LABEL_32;
  }
LABEL_27:
  v36 = (_QWORD *)v34[1];
  if ( !v36 )
    goto LABEL_32;
  v37 = v32 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
  v38 = (__int64 *)(v31 + 16LL * v37);
  v39 = *v38;
  if ( *v38 == v28 )
  {
LABEL_29:
    v9 = (_QWORD *)v38[1];
    if ( v36 != v9 )
    {
      while ( v9 )
      {
        v9 = (_QWORD *)*v9;
        if ( v36 == v9 )
          goto LABEL_32;
      }
      return (char)v9;
    }
    goto LABEL_32;
  }
  LODWORD(v9) = 1;
  while ( v39 != -4096 )
  {
    v67 = (_DWORD)v9 + 1;
    v37 = v32 & ((_DWORD)v9 + v37);
    v38 = (__int64 *)(v31 + 16LL * v37);
    v39 = *v38;
    if ( v28 == *v38 )
      goto LABEL_29;
    LODWORD(v9) = v67;
  }
  return (char)v9;
}
