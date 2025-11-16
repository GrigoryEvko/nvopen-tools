// Function: sub_2E17AE0
// Address: 0x2e17ae0
//
void __fastcall sub_2E17AE0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        const void *a5,
        __int64 a6)
{
  unsigned __int64 v6; // r15
  unsigned __int64 v9; // rbx
  __int64 v10; // r12
  __int64 v11; // r9
  unsigned __int64 v12; // rcx
  __int64 v13; // r11
  int v14; // edi
  unsigned __int64 v15; // rax
  int v16; // edi
  unsigned int v17; // esi
  __int64 v18; // r10
  unsigned int v19; // ecx
  __int64 v20; // rsi
  unsigned int v21; // r8d
  __int64 v22; // rax
  __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  unsigned __int64 i; // rax
  __int64 j; // r10
  __int16 v27; // dx
  unsigned int v28; // r11d
  __int64 *v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // r8
  __int64 v32; // r9
  size_t v33; // r14
  unsigned __int64 v34; // r12
  __int64 v35; // rbx
  _QWORD *v36; // rax
  _QWORD *v37; // rdx
  __int64 v38; // rax
  __int64 k; // r12
  __int16 v40; // ax
  __int64 v41; // r13
  __int64 v42; // r12
  int v43; // r15d
  unsigned int v44; // r14d
  __int64 v45; // rax
  unsigned __int16 v46; // dx
  unsigned __int64 v47; // rcx
  __int64 v48; // rsi
  __int64 v49; // rcx
  unsigned __int64 v50; // rcx
  _QWORD *v51; // rcx
  __int64 *v52; // rdx
  __int64 v53; // rsi
  __int64 v54; // rdx
  unsigned __int64 *v55; // r9
  unsigned __int64 v56; // r8
  __int64 v57; // r14
  unsigned __int64 *v58; // r13
  unsigned __int64 v59; // rbx
  unsigned __int64 v60; // rdi
  unsigned __int64 v61; // rdi
  unsigned int v62; // eax
  __int64 v63; // rdx
  __int64 *v64; // r14
  __int64 v65; // rax
  int *v66; // rdi
  int *v67; // r12
  __int64 v68; // r14
  int v69; // ebx
  unsigned __int64 v70; // rcx
  unsigned int v71; // eax
  unsigned int v72; // r13d
  __int64 v73; // rbx
  int v74; // r10d
  __int64 v75; // rcx
  unsigned __int64 v76; // rax
  int v77; // esi
  __int64 v78; // rax
  __int64 v79; // r15
  unsigned int v80; // eax
  __int64 v81; // rsi
  __int64 v82; // rax
  unsigned __int64 v83; // rdx
  __int64 v84; // r8
  unsigned __int64 v85; // r14
  __int64 v86; // rcx
  __int64 *v87; // rax
  __int64 *v88; // rsi
  void *v89; // rax
  __int64 v90; // r8
  _QWORD *v91; // rdx
  _QWORD *v92; // rdi
  int *v93; // rdi
  unsigned __int64 v94; // rbx
  unsigned __int64 v95; // rdi
  int v96; // edx
  int v97; // r10d
  __int128 v98; // [rsp-10h] [rbp-100h]
  unsigned __int64 v99; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v100; // [rsp+8h] [rbp-E8h]
  __int64 v101; // [rsp+10h] [rbp-E0h]
  unsigned __int64 *v102; // [rsp+18h] [rbp-D8h]
  int v103; // [rsp+18h] [rbp-D8h]
  unsigned __int64 *v104; // [rsp+18h] [rbp-D8h]
  unsigned __int64 *v105; // [rsp+18h] [rbp-D8h]
  unsigned __int64 *v106; // [rsp+18h] [rbp-D8h]
  unsigned __int64 *v107; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v108; // [rsp+20h] [rbp-D0h]
  const void *v109; // [rsp+28h] [rbp-C8h]
  _QWORD *v110; // [rsp+38h] [rbp-B8h]
  __int64 v111; // [rsp+50h] [rbp-A0h]
  unsigned __int64 v113; // [rsp+58h] [rbp-98h]
  __int64 v114; // [rsp+58h] [rbp-98h]
  const void *v115; // [rsp+60h] [rbp-90h]
  void *srca; // [rsp+68h] [rbp-88h]
  void *srcb; // [rsp+68h] [rbp-88h]
  void *srcc; // [rsp+68h] [rbp-88h]
  __int64 v120; // [rsp+70h] [rbp-80h]
  int *v121; // [rsp+70h] [rbp-80h]
  int v122; // [rsp+70h] [rbp-80h]
  __int64 v124; // [rsp+78h] [rbp-78h]
  int *v125; // [rsp+80h] [rbp-70h] BYREF
  __int64 v126; // [rsp+88h] [rbp-68h]
  _BYTE v127[96]; // [rsp+90h] [rbp-60h] BYREF

  v6 = a3;
  v9 = a4;
  v10 = *(_QWORD *)(a2 + 56);
  v11 = *(_QWORD *)(a1 + 32);
  if ( a3 != v10 )
  {
    do
    {
      while ( 1 )
      {
        v12 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v12 )
          BUG();
        v13 = *(_QWORD *)(v11 + 128);
        v14 = *(_DWORD *)(v11 + 144);
        if ( ((*(__int64 *)v12 >> 2) & 1) != 0 )
          break;
        v77 = *(_DWORD *)(v12 + 44);
        v78 = *(_QWORD *)v12;
        if ( (v77 & 4) != 0 )
        {
          while ( 1 )
          {
            v15 = v78 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v15 + 44) & 4) == 0 )
              break;
            v78 = *(_QWORD *)v15;
          }
          if ( !v14 )
            goto LABEL_101;
        }
        else
        {
          v15 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v14 )
            goto LABEL_101;
        }
LABEL_6:
        v16 = v14 - 1;
        v17 = v16 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v18 = *(_QWORD *)(v13 + 16LL * v17);
        if ( v15 == v18 )
          goto LABEL_7;
        v122 = 1;
        while ( v18 != -4096 )
        {
          v17 = v16 & (v122 + v17);
          v18 = *(_QWORD *)(v13 + 16LL * v17);
          if ( v18 == v15 )
            goto LABEL_7;
          ++v122;
        }
        v6 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
        if ( ((*(__int64 *)v12 >> 2) & 1) != 0 )
          goto LABEL_105;
        v77 = *(_DWORD *)(v12 + 44);
LABEL_101:
        v79 = *(_QWORD *)v12;
        if ( (v77 & 4) == 0 )
          goto LABEL_113;
        while ( 1 )
        {
          v6 = v79 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v6 + 44) & 4) == 0 )
            break;
          v79 = *(_QWORD *)v6;
        }
LABEL_105:
        if ( v6 == v10 )
          goto LABEL_7;
      }
      if ( v14 )
      {
        v15 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
        goto LABEL_6;
      }
LABEL_113:
      v6 = v12;
    }
    while ( v12 != v10 );
  }
LABEL_7:
  if ( a4 == a2 + 48 )
  {
LABEL_90:
    v75 = *(_QWORD *)(*(_QWORD *)(v11 + 152) + 16LL * *(unsigned int *)(a2 + 24) + 8);
    if ( ((v75 >> 1) & 3) != 0 )
      v76 = v75 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (int)(((v75 >> 1) & 3) - 1));
    else
      v76 = *(_QWORD *)(v75 & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL | 6;
    v111 = v76;
    v9 = a2 + 48;
    goto LABEL_21;
  }
  v19 = *(_DWORD *)(v11 + 144);
  v20 = *(_QWORD *)(v11 + 128);
  v21 = v19 - 1;
  while ( !v19 )
  {
LABEL_87:
    if ( !v9 )
      BUG();
    if ( (*(_BYTE *)v9 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v9 + 44) & 8) != 0 )
        v9 = *(_QWORD *)(v9 + 8);
    }
    v9 = *(_QWORD *)(v9 + 8);
    if ( v9 == a2 + 48 )
      goto LABEL_90;
  }
  LODWORD(v22) = v21 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v23 = *(_QWORD *)(v20 + 16LL * (unsigned int)v22);
  if ( v23 != v9 )
  {
    v74 = 1;
    while ( v23 != -4096 )
    {
      v22 = v21 & ((_DWORD)v22 + v74);
      v23 = *(_QWORD *)(v20 + 16 * v22);
      if ( v23 == v9 )
        goto LABEL_11;
      ++v74;
    }
    goto LABEL_87;
  }
LABEL_11:
  v24 = v9;
  for ( i = v9; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  if ( (*(_DWORD *)(v9 + 44) & 8) != 0 )
  {
    do
      v24 = *(_QWORD *)(v24 + 8);
    while ( (*(_BYTE *)(v24 + 44) & 8) != 0 );
  }
  for ( j = *(_QWORD *)(v24 + 8); j != i; i = *(_QWORD *)(i + 8) )
  {
    v27 = *(_WORD *)(i + 68);
    if ( (unsigned __int16)(v27 - 14) > 4u && v27 != 24 )
      break;
  }
  v28 = v21 & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
  v29 = (__int64 *)(v20 + 16LL * v28);
  v30 = *v29;
  if ( *v29 != i )
  {
    v96 = 1;
    while ( v30 != -4096 )
    {
      v97 = v96 + 1;
      v28 = v21 & (v96 + v28);
      v29 = (__int64 *)(v20 + 16LL * v28);
      v30 = *v29;
      if ( *v29 == i )
        goto LABEL_20;
      v96 = v97;
    }
    v29 = (__int64 *)(v20 + 16LL * v19);
  }
LABEL_20:
  v111 = v29[1];
LABEL_21:
  sub_2FADE80(v11, a2, v6, v9);
  v33 = 4 * a6;
  v125 = (int *)v127;
  v126 = 0xC00000000LL;
  if ( (unsigned __int64)(4 * a6) > 0x30 )
  {
    sub_C8D5F0((__int64)&v125, v127, (4 * a6) >> 2, 4u, v31, v32);
    v93 = &v125[(unsigned int)v126];
    goto LABEL_135;
  }
  if ( v33 )
  {
    v93 = (int *)v127;
LABEL_135:
    memcpy(v93, a5, v33);
    LODWORD(v33) = v126;
  }
  v124 = v6;
  v109 = (const void *)(a1 + 168);
  LODWORD(v126) = v33 + ((4 * a6) >> 2);
  v34 = v9;
  v108 = v9;
  v35 = a1;
  while ( 2 )
  {
    while ( 2 )
    {
      while ( 2 )
      {
        if ( v34 == v124 )
          goto LABEL_69;
LABEL_25:
        v36 = (_QWORD *)(*(_QWORD *)v34 & 0xFFFFFFFFFFFFFFF8LL);
        v37 = v36;
        if ( !v36 )
          BUG();
        v34 = *(_QWORD *)v34 & 0xFFFFFFFFFFFFFFF8LL;
        v38 = *v36;
        if ( (v38 & 4) == 0 && (*((_BYTE *)v37 + 44) & 4) != 0 )
        {
          for ( k = v38; ; k = *(_QWORD *)v34 )
          {
            v34 = k & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v34 + 44) & 4) == 0 )
              break;
          }
        }
        v40 = *(_WORD *)(v34 + 68);
        if ( (unsigned __int16)(v40 - 14) <= 4u )
          continue;
        break;
      }
      if ( v40 == 24 )
        continue;
      break;
    }
    v41 = *(_QWORD *)(v34 + 32);
    if ( v41 == v41 + 40LL * (*(_DWORD *)(v34 + 40) & 0xFFFFFF) )
      continue;
    break;
  }
  v113 = v34;
  v42 = v41 + 40LL * (*(_DWORD *)(v34 + 40) & 0xFFFFFF);
  while ( 2 )
  {
    while ( 2 )
    {
      if ( *(_BYTE *)v41 || (v43 = *(_DWORD *)(v41 + 8), v43 >= 0) )
      {
LABEL_35:
        v41 += 40;
        if ( v42 == v41 )
          goto LABEL_68;
        continue;
      }
      break;
    }
    v44 = v43 & 0x7FFFFFFF;
    v45 = v43 & 0x7FFFFFFF;
    v120 = 8 * v45;
    v46 = (*(_DWORD *)v41 >> 8) & 0xFFF;
    if ( !v46 )
    {
LABEL_63:
      v47 = *(unsigned int *)(v35 + 160);
      if ( (unsigned int)v47 <= v44 || !*(_QWORD *)(*(_QWORD *)(v35 + 152) + 8LL * (v43 & 0x7FFFFFFF)) )
        goto LABEL_65;
      goto LABEL_35;
    }
    v47 = *(unsigned int *)(v35 + 160);
    if ( v44 < (unsigned int)v47 )
    {
      v32 = *(_QWORD *)(*(_QWORD *)(v35 + 152) + 8 * v45);
      v110 = (_QWORD *)(*(_QWORD *)(v35 + 152) + 8 * v45);
      if ( v32 )
      {
        v48 = *(_QWORD *)(v35 + 8);
        v49 = *(_QWORD *)(16 * v45 + *(_QWORD *)(v48 + 56));
        if ( !v49 )
          goto LABEL_35;
        if ( (v49 & 4) != 0 )
          goto LABEL_35;
        v50 = v49 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v50 || !*(_BYTE *)(v48 + 48) || !*(_BYTE *)(v50 + 43) )
          goto LABEL_35;
        v51 = *(_QWORD **)(v32 + 104);
        if ( v51 )
        {
          if ( (*(_BYTE *)(v41 + 3) & 0x10) != 0 )
          {
            v52 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(v35 + 16) + 272LL) + 16LL * v46);
            v53 = *v52;
            v54 = v52[1];
            while ( v53 != v51[14] || v54 != v51[15] )
            {
              v51 = (_QWORD *)v51[13];
              if ( !v51 )
              {
                v102 = *(unsigned __int64 **)(*(_QWORD *)(v35 + 152) + 8 * v45);
                sub_2E0AFD0(v32);
                v55 = v102;
                v56 = v102[12];
                if ( !v56 )
                  goto LABEL_58;
                if ( *(_QWORD *)(v56 + 16) )
                {
                  v103 = v43 & 0x7FFFFFFF;
                  v57 = v41;
                  v58 = v55;
                  v101 = v35;
                  v59 = *(_QWORD *)(v56 + 16);
                  do
                  {
                    v99 = v56;
                    sub_2E10270(*(_QWORD *)(v59 + 24));
                    v60 = v59;
                    v59 = *(_QWORD *)(v59 + 16);
                    j_j___libc_free_0(v60);
                    v56 = v99;
                  }
                  while ( v59 );
                  goto LABEL_56;
                }
                goto LABEL_57;
              }
            }
          }
          goto LABEL_35;
        }
        v107 = *(unsigned __int64 **)(*(_QWORD *)(v35 + 152) + 8 * v45);
        sub_2E0AFD0(v32);
        v55 = v107;
        v56 = v107[12];
        if ( v56 )
        {
          if ( *(_QWORD *)(v56 + 16) )
          {
            v103 = v43 & 0x7FFFFFFF;
            v57 = v41;
            v58 = v55;
            v101 = v35;
            v94 = *(_QWORD *)(v56 + 16);
            do
            {
              v100 = v56;
              sub_2E10270(*(_QWORD *)(v94 + 24));
              v95 = v94;
              v94 = *(_QWORD *)(v94 + 16);
              j_j___libc_free_0(v95);
              v56 = v100;
            }
            while ( v94 );
LABEL_56:
            v55 = v58;
            v35 = v101;
            v41 = v57;
            v44 = v103;
          }
LABEL_57:
          v104 = v55;
          j_j___libc_free_0(v56);
          v55 = v104;
        }
LABEL_58:
        v61 = v55[8];
        if ( (unsigned __int64 *)v61 != v55 + 10 )
        {
          v105 = v55;
          _libc_free(v61);
          v55 = v105;
        }
        if ( (unsigned __int64 *)*v55 != v55 + 2 )
        {
          v106 = v55;
          _libc_free(*v55);
          v55 = v106;
        }
        j_j___libc_free_0((unsigned __int64)v55);
        *v110 = 0;
        goto LABEL_63;
      }
    }
LABEL_65:
    v62 = v44 + 1;
    if ( (unsigned int)v47 >= v44 + 1 || v62 == v47 )
    {
LABEL_66:
      v63 = *(_QWORD *)(v35 + 152);
    }
    else
    {
      if ( v62 < v47 )
      {
        *(_DWORD *)(v35 + 160) = v62;
        goto LABEL_66;
      }
      v84 = *(_QWORD *)(v35 + 168);
      v85 = v62 - v47;
      if ( v62 > (unsigned __int64)*(unsigned int *)(v35 + 164) )
      {
        srcb = *(void **)(v35 + 168);
        sub_C8D5F0(v35 + 152, v109, v62, 8u, v84, v32);
        v84 = (__int64)srcb;
      }
      v63 = *(_QWORD *)(v35 + 152);
      v86 = *(unsigned int *)(v35 + 160);
      v87 = (__int64 *)(v63 + 8 * v86);
      v88 = &v87[v85];
      if ( v87 != v88 )
      {
        do
          *v87++ = v84;
        while ( v88 != v87 );
        LODWORD(v86) = *(_DWORD *)(v35 + 160);
        v63 = *(_QWORD *)(v35 + 152);
      }
      *(_DWORD *)(v35 + 160) = v85 + v86;
    }
    v41 += 40;
    v64 = (__int64 *)(v63 + v120);
    v65 = sub_2E10F30(v43);
    *v64 = v65;
    sub_2E11E80((_QWORD *)v35, v65);
    sub_2E16DB0((__int64)&v125, v43);
    if ( v42 != v41 )
      continue;
    break;
  }
LABEL_68:
  v34 = v113;
  if ( v113 != v124 )
    goto LABEL_25;
LABEL_69:
  v66 = v125;
  v121 = &v125[(unsigned int)v126];
  if ( v121 == v125 )
    goto LABEL_82;
  v67 = v125;
  v68 = v35;
  v115 = (const void *)(v35 + 168);
  while ( 2 )
  {
    while ( 2 )
    {
      v69 = *v67;
      if ( *v67 >= 0 )
      {
LABEL_71:
        if ( v121 == ++v67 )
          goto LABEL_81;
        continue;
      }
      break;
    }
    v70 = *(unsigned int *)(v68 + 160);
    v71 = v69 & 0x7FFFFFFF;
    if ( (v69 & 0x7FFFFFFFu) >= (unsigned int)v70 || (srca = *(void **)(*(_QWORD *)(v68 + 152) + 8LL * v71)) == 0 )
    {
      v80 = v71 + 1;
      if ( (unsigned int)v70 < v80 )
      {
        v83 = v80;
        if ( v80 != v70 )
        {
          if ( v80 >= v70 )
          {
            v89 = *(void **)(v68 + 168);
            v90 = v83 - v70;
            if ( v83 > *(unsigned int *)(v68 + 164) )
            {
              v114 = v83 - v70;
              srcc = *(void **)(v68 + 168);
              sub_C8D5F0(v68 + 152, v115, v83, 8u, v90, v32);
              v70 = *(unsigned int *)(v68 + 160);
              v90 = v114;
              v89 = srcc;
            }
            v81 = *(_QWORD *)(v68 + 152);
            v91 = (_QWORD *)(v81 + 8 * v70);
            v92 = &v91[v90];
            if ( v91 != v92 )
            {
              do
                *v91++ = v89;
              while ( v92 != v91 );
              LODWORD(v70) = *(_DWORD *)(v68 + 160);
              v81 = *(_QWORD *)(v68 + 152);
            }
            *(_DWORD *)(v68 + 160) = v90 + v70;
            goto LABEL_112;
          }
          *(_DWORD *)(v68 + 160) = v80;
        }
      }
      v81 = *(_QWORD *)(v68 + 152);
LABEL_112:
      v82 = sub_2E10F30(v69);
      *(_QWORD *)(v81 + 8LL * (v69 & 0x7FFFFFFF)) = v82;
      srca = (void *)v82;
      sub_2E11E80((_QWORD *)v68, v82);
    }
    if ( !*((_DWORD *)srca + 18) )
      goto LABEL_71;
    if ( *((_QWORD *)srca + 13) )
    {
      v72 = v69;
      v73 = *((_QWORD *)srca + 13);
      do
      {
        sub_2E17360((_QWORD *)v68, v124, v108, v111, v73, v72, *(_OWORD *)(v73 + 112));
        v73 = *(_QWORD *)(v73 + 104);
      }
      while ( v73 );
      v69 = v72;
    }
    ++v67;
    sub_2E0AF60((__int64)srca);
    *((_QWORD *)&v98 + 1) = -1;
    *(_QWORD *)&v98 = -1;
    sub_2E17360((_QWORD *)v68, v124, v108, v111, (__int64)srca, v69, v98);
    if ( v121 != v67 )
      continue;
    break;
  }
LABEL_81:
  v66 = v125;
LABEL_82:
  if ( v66 != (int *)v127 )
    _libc_free((unsigned __int64)v66);
}
