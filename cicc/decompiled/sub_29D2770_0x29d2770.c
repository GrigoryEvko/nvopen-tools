// Function: sub_29D2770
// Address: 0x29d2770
//
__int64 __fastcall sub_29D2770(__int64 a1, __int64 a2, _QWORD *a3, _QWORD *a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rcx
  _QWORD *v10; // rax
  __int64 *v11; // rsi
  __int64 v12; // rdi
  _QWORD *v13; // rdx
  unsigned int v14; // r12d
  unsigned __int64 v16; // rdx
  __int64 v17; // r13
  __int64 v18; // rbx
  __int64 v19; // r8
  unsigned int v20; // edi
  _QWORD *v21; // rax
  __int64 v22; // rcx
  _QWORD *v23; // rax
  __int64 v24; // r12
  __int64 v25; // r10
  unsigned int v26; // esi
  int v27; // eax
  int v28; // r8d
  unsigned int v29; // eax
  int v30; // ecx
  _QWORD *v31; // rdx
  __int64 v32; // rsi
  int v33; // r14d
  _QWORD *v34; // rdi
  __int64 v35; // rax
  __int64 v36; // r12
  __int64 v37; // rbx
  __int64 v38; // r14
  unsigned int v39; // eax
  char *v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  char *v43; // rax
  unsigned int v44; // edx
  unsigned int v45; // eax
  __int64 v46; // rsi
  __int64 v47; // rax
  __int64 v48; // rax
  _BYTE *v49; // r11
  __int64 v50; // r13
  __int64 v51; // rax
  __int64 v52; // rdi
  __int64 v53; // rcx
  unsigned int v54; // esi
  __int64 v55; // rdx
  __int64 *v56; // rax
  __int64 v57; // r8
  _QWORD *v58; // rax
  __int64 v59; // r8
  unsigned int v60; // edx
  unsigned int v61; // esi
  unsigned int v62; // ecx
  _QWORD *v63; // rdx
  _BYTE *v64; // r10
  __int64 v65; // r13
  __int64 v66; // rcx
  int v67; // esi
  int v68; // esi
  unsigned int v69; // ecx
  int v70; // eax
  __int64 *v71; // rdi
  __int64 v72; // r8
  __int64 *v73; // r10
  int v74; // eax
  int v75; // eax
  int v76; // esi
  int v77; // esi
  unsigned int v78; // ecx
  __int64 v79; // r8
  int v80; // r9d
  __int64 v81; // r8
  _QWORD *v82; // rsi
  unsigned int v83; // r14d
  int v84; // edi
  __int64 v85; // rax
  char v86; // dl
  unsigned __int64 v87; // rcx
  __int64 v88; // rcx
  __int64 v89; // rdx
  __int64 v90; // rcx
  int v91; // edx
  __int64 v92; // rax
  int v93; // r8d
  __int64 v94; // rdi
  int v95; // r8d
  unsigned int v96; // esi
  __int64 *v97; // rax
  __int64 v98; // r10
  int v99; // r9d
  int v100; // eax
  int v101; // r9d
  _BYTE *v102; // [rsp+8h] [rbp-198h]
  _QWORD *v103; // [rsp+10h] [rbp-190h]
  _BYTE *v104; // [rsp+10h] [rbp-190h]
  int v105; // [rsp+10h] [rbp-190h]
  __int64 v106; // [rsp+10h] [rbp-190h]
  _QWORD *v107; // [rsp+10h] [rbp-190h]
  __int64 v109; // [rsp+20h] [rbp-180h]
  unsigned int v110; // [rsp+20h] [rbp-180h]
  __int64 v111; // [rsp+20h] [rbp-180h]
  int v112; // [rsp+20h] [rbp-180h]
  __int64 v113; // [rsp+20h] [rbp-180h]
  __int64 v114; // [rsp+28h] [rbp-178h]
  __int64 v115; // [rsp+30h] [rbp-170h]
  _QWORD *v116; // [rsp+30h] [rbp-170h]
  __int64 v117; // [rsp+38h] [rbp-168h]
  __int64 v118; // [rsp+38h] [rbp-168h]
  int v119; // [rsp+38h] [rbp-168h]
  int v120; // [rsp+38h] [rbp-168h]
  _QWORD *v121; // [rsp+38h] [rbp-168h]
  char v122; // [rsp+47h] [rbp-159h] BYREF
  __int64 v123; // [rsp+48h] [rbp-158h] BYREF
  __int64 v124; // [rsp+50h] [rbp-150h] BYREF
  char *v125; // [rsp+58h] [rbp-148h]
  __int64 v126; // [rsp+60h] [rbp-140h]
  int v127; // [rsp+68h] [rbp-138h]
  char v128; // [rsp+6Ch] [rbp-134h]
  char v129; // [rsp+70h] [rbp-130h] BYREF

  v9 = *(unsigned int *)(a1 + 88);
  v10 = *(_QWORD **)(a1 + 80);
  v11 = &v10[v9];
  v12 = (8 * v9) >> 3;
  if ( (8 * v9) >> 5 )
  {
    v13 = &v10[4 * ((8 * v9) >> 5)];
    while ( *v10 != a2 )
    {
      if ( v10[1] == a2 )
      {
        ++v10;
        goto LABEL_8;
      }
      if ( v10[2] == a2 )
      {
        v10 += 2;
        goto LABEL_8;
      }
      if ( v10[3] == a2 )
      {
        v10 += 3;
        goto LABEL_8;
      }
      v10 += 4;
      if ( v13 == v10 )
      {
        v12 = v11 - v10;
        goto LABEL_11;
      }
    }
    goto LABEL_8;
  }
LABEL_11:
  switch ( v12 )
  {
    case 2LL:
LABEL_109:
      if ( *v10 == a2 )
        goto LABEL_8;
      ++v10;
LABEL_111:
      if ( *v10 != a2 )
        break;
LABEL_8:
      v14 = 0;
      if ( v11 != v10 )
        return v14;
      break;
    case 3LL:
      if ( *v10 == a2 )
        goto LABEL_8;
      ++v10;
      goto LABEL_109;
    case 1LL:
      goto LABEL_111;
  }
  v16 = v9 + 1;
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 92) )
  {
    v121 = a4;
    sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), v16, 8u, a5, a6);
    v16 = *(unsigned int *)(a1 + 88);
    a4 = v121;
    v11 = (__int64 *)(*(_QWORD *)(a1 + 80) + 8 * v16);
  }
  *v11 = a2;
  ++*(_DWORD *)(a1 + 88);
  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
  {
    v116 = a4;
    sub_B2C6D0(a2, (__int64)v11, v16, v9);
    v17 = *(_QWORD *)(a2 + 96);
    a4 = v116;
    v117 = v17 + 40LL * *(_QWORD *)(a2 + 104);
    if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    {
      sub_B2C6D0(a2, (__int64)v11, v89, v90);
      v17 = *(_QWORD *)(a2 + 96);
      a4 = v116;
    }
  }
  else
  {
    v17 = *(_QWORD *)(a2 + 96);
    v117 = v17 + 40LL * *(_QWORD *)(a2 + 104);
  }
  if ( v117 == v17 )
    goto LABEL_35;
  v114 = a2;
  v18 = 0;
  do
  {
    v24 = *(_QWORD *)(a1 + 48);
    v25 = *(_QWORD *)(*a4 + v18);
    if ( v24 == *(_QWORD *)(a1 + 56) )
      v24 = *(_QWORD *)(*(_QWORD *)(a1 + 72) - 8LL) + 512LL;
    v26 = *(_DWORD *)(v24 - 8);
    v115 = v24 - 32;
    if ( v26 )
    {
      a6 = v26 - 1;
      v19 = *(_QWORD *)(v24 - 24);
      v20 = a6 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v21 = (_QWORD *)(v19 + 16LL * v20);
      v22 = *v21;
      if ( *v21 == v17 )
      {
LABEL_21:
        v23 = v21 + 1;
        goto LABEL_22;
      }
      v112 = 1;
      v31 = 0;
      while ( v22 != -4096 )
      {
        if ( !v31 && v22 == -8192 )
          v31 = v21;
        v20 = a6 & (v112 + v20);
        v21 = (_QWORD *)(v19 + 16LL * v20);
        v22 = *v21;
        if ( *v21 == v17 )
          goto LABEL_21;
        ++v112;
      }
      if ( !v31 )
        v31 = v21;
      v75 = *(_DWORD *)(v24 - 16);
      ++*(_QWORD *)(v24 - 32);
      v30 = v75 + 1;
      if ( 4 * (v75 + 1) < 3 * v26 )
      {
        if ( v26 - *(_DWORD *)(v24 - 12) - v30 <= v26 >> 3 )
        {
          v107 = a4;
          v113 = v25;
          sub_29D0890(v115, v26);
          v80 = *(_DWORD *)(v24 - 8);
          if ( !v80 )
          {
LABEL_180:
            ++*(_DWORD *)(v24 - 16);
            BUG();
          }
          a6 = (unsigned int)(v80 - 1);
          v81 = *(_QWORD *)(v24 - 24);
          v82 = 0;
          v83 = a6 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
          v25 = v113;
          a4 = v107;
          v30 = *(_DWORD *)(v24 - 16) + 1;
          v84 = 1;
          v31 = (_QWORD *)(v81 + 16LL * v83);
          v85 = *v31;
          if ( v17 != *v31 )
          {
            while ( v85 != -4096 )
            {
              if ( v85 == -8192 && !v82 )
                v82 = v31;
              v83 = a6 & (v84 + v83);
              v31 = (_QWORD *)(v81 + 16LL * v83);
              v85 = *v31;
              if ( *v31 == v17 )
                goto LABEL_95;
              ++v84;
            }
            if ( v82 )
              v31 = v82;
          }
        }
        goto LABEL_95;
      }
    }
    else
    {
      ++*(_QWORD *)(v24 - 32);
    }
    v103 = a4;
    v109 = v25;
    sub_29D0890(v115, 2 * v26);
    v27 = *(_DWORD *)(v24 - 8);
    if ( !v27 )
      goto LABEL_180;
    v28 = v27 - 1;
    a6 = *(_QWORD *)(v24 - 24);
    v25 = v109;
    a4 = v103;
    v29 = (v27 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
    v30 = *(_DWORD *)(v24 - 16) + 1;
    v31 = (_QWORD *)(a6 + 16LL * v29);
    v32 = *v31;
    if ( v17 != *v31 )
    {
      v33 = 1;
      v34 = 0;
      while ( v32 != -4096 )
      {
        if ( v32 == -8192 && !v34 )
          v34 = v31;
        v29 = v28 & (v33 + v29);
        v31 = (_QWORD *)(a6 + 16LL * v29);
        v32 = *v31;
        if ( *v31 == v17 )
          goto LABEL_95;
        ++v33;
      }
      if ( v34 )
        v31 = v34;
    }
LABEL_95:
    *(_DWORD *)(v24 - 16) = v30;
    if ( *v31 != -4096 )
      --*(_DWORD *)(v24 - 12);
    *v31 = v17;
    v23 = v31 + 1;
    v31[1] = 0;
LABEL_22:
    *v23 = v25;
    v17 += 40;
    v18 += 8;
  }
  while ( v17 != v117 );
  a2 = v114;
LABEL_35:
  v124 = 0;
  v125 = &v129;
  v35 = *(_QWORD *)(a2 + 80);
  v126 = 32;
  v127 = 0;
  v128 = 1;
  if ( !v35 )
    BUG();
  v36 = *(_QWORD *)(v35 + 32);
  v37 = v35 - 24;
  v38 = 1;
  while ( 2 )
  {
    v122 = 0;
    v123 = 0;
    v39 = sub_29D0C70(a1, v36, v38, &v123, (__int64)&v122, a6);
    if ( !(_BYTE)v39 )
    {
      v86 = v128;
      v14 = v39;
      goto LABEL_124;
    }
    if ( !v123 )
    {
      v14 = v39;
      v87 = *(_QWORD *)(v37 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v87 == v37 + 48 )
        goto LABEL_181;
      if ( !v87 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v87 - 24) - 30 > 0xA )
LABEL_181:
        BUG();
      v86 = v128;
      if ( (*(_DWORD *)(v87 - 20) & 0x7FFFFFF) == 0 )
        goto LABEL_135;
      v88 = *(_QWORD *)(v87 - 32LL * (*(_DWORD *)(v87 - 20) & 0x7FFFFFF) - 24);
      if ( v122 && *(_BYTE *)(*(_QWORD *)(v88 + 8) + 8LL) != 7 )
      {
        v14 = 0;
        goto LABEL_124;
      }
      if ( *(_BYTE *)v88 <= 0x15u )
      {
LABEL_134:
        *a3 = v88;
LABEL_135:
        --*(_DWORD *)(a1 + 88);
        goto LABEL_124;
      }
      v92 = *(_QWORD *)(a1 + 48);
      if ( v92 == *(_QWORD *)(a1 + 56) )
        v92 = *(_QWORD *)(*(_QWORD *)(a1 + 72) - 8LL) + 512LL;
      v93 = *(_DWORD *)(v92 - 8);
      v94 = *(_QWORD *)(v92 - 24);
      if ( v93 )
      {
        v95 = v93 - 1;
        v96 = v95 & (((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4));
        v97 = (__int64 *)(v94 + 16LL * v96);
        v98 = *v97;
        if ( *v97 == v88 )
        {
LABEL_145:
          v88 = v97[1];
          goto LABEL_134;
        }
        v100 = 1;
        while ( v98 != -4096 )
        {
          v101 = v100 + 1;
          v96 = v95 & (v100 + v96);
          v97 = (__int64 *)(v94 + 16LL * v96);
          v98 = *v97;
          if ( *v97 == v88 )
            goto LABEL_145;
          v100 = v101;
        }
      }
      v88 = 0;
      goto LABEL_134;
    }
    if ( !v128 )
      goto LABEL_45;
    v43 = v125;
    v41 = HIDWORD(v126);
    v40 = &v125[8 * HIDWORD(v126)];
    if ( v125 == v40 )
    {
LABEL_68:
      if ( HIDWORD(v126) < (unsigned int)v126 )
      {
        ++HIDWORD(v126);
        *(_QWORD *)v40 = v123;
        ++v124;
LABEL_46:
        LOWORD(v38) = 1;
        v36 = *(_QWORD *)(v123 + 56);
LABEL_47:
        if ( !v36 )
          BUG();
        if ( *(_BYTE *)(v36 - 24) != 84 )
        {
          v37 = v123;
          continue;
        }
        v46 = *(_QWORD *)(v36 - 32);
        v47 = 0x1FFFFFFFE0LL;
        if ( (*(_DWORD *)(v36 - 20) & 0x7FFFFFF) != 0 )
        {
          v48 = 0;
          do
          {
            if ( v37 == *(_QWORD *)(v46 + 32LL * *(unsigned int *)(v36 + 48) + 8 * v48) )
            {
              v47 = 32 * v48;
              goto LABEL_54;
            }
            ++v48;
          }
          while ( (*(_DWORD *)(v36 - 20) & 0x7FFFFFF) != (_DWORD)v48 );
          v47 = 0x1FFFFFFFE0LL;
        }
LABEL_54:
        v49 = *(_BYTE **)(v46 + v47);
        v50 = *(_QWORD *)(a1 + 48);
        v51 = *(_QWORD *)(a1 + 56);
        v52 = *(_QWORD *)(a1 + 72);
        if ( *v49 <= 0x15u )
        {
LABEL_55:
          if ( v51 != v50 )
            goto LABEL_56;
LABEL_67:
          v65 = *(_QWORD *)(v52 - 8);
          v54 = *(_DWORD *)(v65 + 504);
          v53 = *(_QWORD *)(v65 + 488);
          v50 = v65 + 512;
          goto LABEL_57;
        }
        if ( v51 != v50 )
        {
          v53 = *(_QWORD *)(v50 - 24);
          v54 = *(_DWORD *)(v50 - 8);
          v59 = v53;
          v60 = v54;
          if ( !v54 )
          {
            v49 = 0;
            goto LABEL_57;
          }
          goto LABEL_65;
        }
        v66 = *(_QWORD *)(v52 - 8);
        v60 = *(_DWORD *)(v66 + 504);
        v59 = *(_QWORD *)(v66 + 488);
        if ( !v60 )
        {
          v50 = v66 + 512;
          v49 = 0;
          v118 = v66 + 480;
          v55 = v36 - 24;
          goto LABEL_72;
        }
LABEL_65:
        v61 = v60 - 1;
        v62 = (v60 - 1) & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
        v63 = (_QWORD *)(v59 + 16LL * v62);
        v64 = (_BYTE *)*v63;
        if ( v49 != (_BYTE *)*v63 )
        {
          v91 = 1;
          while ( v64 != (_BYTE *)-4096LL )
          {
            v99 = v91 + 1;
            v62 = v61 & (v91 + v62);
            v63 = (_QWORD *)(v59 + 16LL * v62);
            v64 = (_BYTE *)*v63;
            if ( v49 == (_BYTE *)*v63 )
              goto LABEL_66;
            v91 = v99;
          }
          v49 = 0;
          goto LABEL_55;
        }
LABEL_66:
        v49 = (_BYTE *)v63[1];
        if ( v51 == v50 )
          goto LABEL_67;
LABEL_56:
        v53 = *(_QWORD *)(v50 - 24);
        v54 = *(_DWORD *)(v50 - 8);
LABEL_57:
        v55 = v36 - 24;
        v118 = v50 - 32;
        if ( v54 )
        {
          v110 = ((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4);
          a6 = (v54 - 1) & v110;
          v56 = (__int64 *)(v53 + 16 * a6);
          v57 = *v56;
          if ( v55 == *v56 )
          {
LABEL_59:
            v58 = v56 + 1;
LABEL_60:
            *v58 = v49;
            LOWORD(v38) = 0;
            v36 = *(_QWORD *)(v36 + 8);
            goto LABEL_47;
          }
          v105 = 1;
          v71 = 0;
          while ( v57 != -4096 )
          {
            if ( !v71 && v57 == -8192 )
              v71 = v56;
            a6 = (v54 - 1) & (v105 + (_DWORD)a6);
            v56 = (__int64 *)(v53 + 16LL * (unsigned int)a6);
            v57 = *v56;
            if ( v55 == *v56 )
              goto LABEL_59;
            ++v105;
          }
          if ( !v71 )
            v71 = v56;
          v74 = *(_DWORD *)(v50 - 16);
          ++*(_QWORD *)(v50 - 32);
          v70 = v74 + 1;
          if ( 4 * v70 < 3 * v54 )
          {
            if ( v54 - *(_DWORD *)(v50 - 12) - v70 <= v54 >> 3 )
            {
              v102 = v49;
              v106 = v36 - 24;
              sub_29D0890(v118, v54);
              v76 = *(_DWORD *)(v50 - 8);
              if ( !v76 )
              {
LABEL_182:
                ++*(_DWORD *)(v50 - 16);
                BUG();
              }
              v77 = v76 - 1;
              a6 = *(_QWORD *)(v50 - 24);
              v55 = v36 - 24;
              v78 = v77 & v110;
              v49 = v102;
              v70 = *(_DWORD *)(v50 - 16) + 1;
              v71 = (__int64 *)(a6 + 16LL * (v77 & v110));
              v79 = *v71;
              if ( v106 != *v71 )
              {
                v120 = 1;
                v73 = 0;
                while ( v79 != -4096 )
                {
                  if ( !v73 && v79 == -8192 )
                    v73 = v71;
                  v78 = v77 & (v120 + v78);
                  v71 = (__int64 *)(a6 + 16LL * v78);
                  v79 = *v71;
                  if ( v106 == *v71 )
                    goto LABEL_86;
                  ++v120;
                }
                goto LABEL_77;
              }
            }
            goto LABEL_86;
          }
        }
        else
        {
LABEL_72:
          ++*(_QWORD *)(v50 - 32);
          v54 = 0;
        }
        v104 = v49;
        v111 = v55;
        sub_29D0890(v118, 2 * v54);
        v67 = *(_DWORD *)(v50 - 8);
        if ( !v67 )
          goto LABEL_182;
        v55 = v111;
        v68 = v67 - 1;
        a6 = *(_QWORD *)(v50 - 24);
        v49 = v104;
        v69 = v68 & (((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4));
        v70 = *(_DWORD *)(v50 - 16) + 1;
        v71 = (__int64 *)(a6 + 16LL * v69);
        v72 = *v71;
        if ( *v71 != v111 )
        {
          v119 = 1;
          v73 = 0;
          while ( v72 != -4096 )
          {
            if ( !v73 && v72 == -8192 )
              v73 = v71;
            v69 = v68 & (v119 + v69);
            v71 = (__int64 *)(a6 + 16LL * v69);
            v72 = *v71;
            if ( v111 == *v71 )
              goto LABEL_86;
            ++v119;
          }
LABEL_77:
          if ( v73 )
            v71 = v73;
        }
LABEL_86:
        *(_DWORD *)(v50 - 16) = v70;
        if ( *v71 != -4096 )
          --*(_DWORD *)(v50 - 12);
        *v71 = v55;
        v58 = v71 + 1;
        v71[1] = 0;
        goto LABEL_60;
      }
LABEL_45:
      sub_C8CC70((__int64)&v124, v123, (__int64)v40, v41, v42, a6);
      v45 = v44;
      if ( (_BYTE)v44 )
        goto LABEL_46;
      v86 = v128;
      v14 = v45;
LABEL_124:
      if ( !v86 )
        _libc_free((unsigned __int64)v125);
    }
    else
    {
      while ( v123 != *(_QWORD *)v43 )
      {
        v43 += 8;
        if ( v40 == v43 )
          goto LABEL_68;
      }
      return 0;
    }
    return v14;
  }
}
