// Function: sub_1AA9AF0
// Address: 0x1aa9af0
//
unsigned __int64 __fastcall sub_1AA9AF0(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        char a7,
        __int64 *a8)
{
  __int64 v9; // rbx
  unsigned __int64 result; // rax
  int v11; // ecx
  __int64 *v12; // r12
  int v13; // ecx
  __int64 v14; // rsi
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // r8
  bool v18; // r14
  __int64 *v19; // r15
  __int64 v20; // rbx
  _BOOL4 v21; // eax
  char v22; // cl
  bool v23; // zf
  __int64 v24; // r13
  __int64 v25; // rcx
  unsigned int v26; // r9d
  __int64 *v27; // rdx
  __int64 v28; // rdi
  int v29; // edx
  int v30; // edx
  __int64 v31; // rcx
  unsigned int v32; // esi
  __int64 *v33; // rax
  __int64 v34; // r9
  _QWORD *v35; // rcx
  _QWORD *v36; // rdx
  __int64 v37; // rax
  __int64 v38; // r8
  __int64 v39; // r12
  unsigned int v40; // esi
  __int64 v41; // r9
  unsigned int v42; // edi
  __int64 *v43; // r15
  __int64 v44; // rcx
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 *v47; // r13
  __int64 v48; // r15
  int v49; // eax
  int v50; // r8d
  __int64 v51; // rdi
  unsigned int v52; // edx
  __int64 *v53; // rax
  __int64 v54; // r9
  __int64 *v55; // rbx
  _QWORD *v56; // rdx
  _QWORD *v57; // rax
  _QWORD *v58; // r12
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rdi
  int v62; // edx
  int v63; // r8d
  _QWORD *v64; // rdx
  int v65; // eax
  int v66; // r10d
  _QWORD *v67; // rax
  unsigned int v68; // ecx
  __int64 *v69; // rax
  unsigned int v70; // edx
  unsigned int v71; // esi
  __int64 v72; // rdx
  unsigned int v73; // r11d
  __int64 v74; // r9
  __int64 v75; // rcx
  __int64 *v76; // r15
  __int64 v77; // rdi
  __int64 *v78; // rax
  __int64 v79; // r14
  __int64 v80; // rsi
  __int64 v81; // r8
  __int64 v82; // rdi
  __int64 v83; // rdi
  unsigned int v84; // edi
  int v85; // r10d
  int v86; // r10d
  __int64 v87; // rdx
  int v88; // eax
  __int64 v89; // rdi
  __int64 v90; // rax
  __int64 *v91; // r8
  __int64 *v92; // r10
  int v93; // eax
  int v94; // eax
  int v95; // r14d
  int v96; // r14d
  __int64 v97; // r11
  __int64 v98; // rdi
  int v99; // r8d
  __int64 *v100; // r10
  int v101; // ecx
  int v102; // r14d
  int v103; // r14d
  __int64 v104; // r11
  __int64 *v105; // rsi
  __int64 v106; // rdi
  int v107; // r10d
  int v108; // edx
  __int64 *v109; // rax
  int v110; // eax
  int v111; // r10d
  int v112; // r10d
  __int64 *v113; // rcx
  int v114; // esi
  __int64 v115; // rdx
  __int64 v116; // rdi
  unsigned int v117; // edi
  int v118; // r10d
  __int64 *v119; // rsi
  int v120; // r11d
  int v121; // r11d
  __int64 v122; // r10
  __int64 *v123; // rsi
  int v124; // r9d
  unsigned int v125; // ecx
  __int64 v126; // rdi
  int v127; // r11d
  int v128; // r11d
  __int64 v129; // r10
  unsigned int v130; // ecx
  __int64 v131; // rdi
  int v132; // r9d
  int v133; // esi
  __int64 v134; // [rsp+10h] [rbp-80h]
  int v135; // [rsp+10h] [rbp-80h]
  unsigned int v138; // [rsp+28h] [rbp-68h]
  __int64 v139; // [rsp+28h] [rbp-68h]
  __int64 v140; // [rsp+28h] [rbp-68h]
  __int64 v142; // [rsp+40h] [rbp-50h]
  __int64 v143; // [rsp+40h] [rbp-50h]
  __int64 *v144; // [rsp+40h] [rbp-50h]
  unsigned int v145; // [rsp+40h] [rbp-50h]
  __int64 v146; // [rsp+40h] [rbp-50h]
  __int64 v147; // [rsp+40h] [rbp-50h]
  unsigned int v148; // [rsp+40h] [rbp-50h]
  __int64 v149; // [rsp+40h] [rbp-50h]
  char v151; // [rsp+48h] [rbp-48h]
  __int64 *v152; // [rsp+48h] [rbp-48h]
  __int64 v153[7]; // [rsp+58h] [rbp-38h] BYREF

  v9 = a1;
  if ( a5 )
  {
    if ( **(_QWORD **)(a5 + 56) != a1 )
    {
      sub_15D0340(a5, a2);
      goto LABEL_4;
    }
    *(_BYTE *)(a5 + 72) = 0;
    v37 = sub_22077B0(56);
    v39 = v37;
    if ( v37 )
    {
      *(_QWORD *)(v37 + 8) = 0;
      *(_DWORD *)(v37 + 16) = 0;
      *(_QWORD *)v37 = a2;
      *(_QWORD *)(v37 + 24) = 0;
      *(_QWORD *)(v37 + 32) = 0;
      *(_QWORD *)(v37 + 40) = 0;
      *(_QWORD *)(v37 + 48) = -1;
    }
    v40 = *(_DWORD *)(a5 + 48);
    if ( v40 )
    {
      v41 = *(_QWORD *)(a5 + 32);
      v42 = (v40 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v43 = (__int64 *)(v41 + 16LL * v42);
      v44 = *v43;
      if ( a2 == *v43 )
      {
LABEL_37:
        v38 = v43[1];
        v43[1] = v39;
        if ( v38 )
        {
          v45 = *(_QWORD *)(v38 + 24);
          if ( v45 )
          {
            v143 = v38;
            j_j___libc_free_0(v45, *(_QWORD *)(v38 + 40) - v45);
            v38 = v143;
          }
          j_j___libc_free_0(v38, 56);
          v39 = v43[1];
        }
        goto LABEL_41;
      }
      v108 = 1;
      v109 = 0;
      while ( v44 != -8 )
      {
        if ( v44 == -16 && !v109 )
          v109 = v43;
        LODWORD(v38) = v108 + 1;
        v42 = (v40 - 1) & (v108 + v42);
        v43 = (__int64 *)(v41 + 16LL * v42);
        v44 = *v43;
        if ( a2 == *v43 )
          goto LABEL_37;
        ++v108;
      }
      if ( v109 )
        v43 = v109;
      v110 = *(_DWORD *)(a5 + 40);
      ++*(_QWORD *)(a5 + 24);
      v88 = v110 + 1;
      if ( 4 * v88 < 3 * v40 )
      {
        if ( v40 - *(_DWORD *)(a5 + 44) - v88 > v40 >> 3 )
          goto LABEL_117;
        v148 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
        sub_15CFCF0(a5 + 24, v40);
        v111 = *(_DWORD *)(a5 + 48);
        if ( !v111 )
          goto LABEL_234;
        v112 = v111 - 1;
        v41 = *(_QWORD *)(a5 + 32);
        v113 = 0;
        v114 = 1;
        LODWORD(v115) = v112 & v148;
        v88 = *(_DWORD *)(a5 + 40) + 1;
        v43 = (__int64 *)(v41 + 16LL * (v112 & v148));
        v116 = *v43;
        if ( a2 == *v43 )
          goto LABEL_117;
        while ( v116 != -8 )
        {
          if ( v116 == -16 && !v113 )
            v113 = v43;
          LODWORD(v38) = v114 + 1;
          v115 = v112 & (unsigned int)(v115 + v114);
          v43 = (__int64 *)(v41 + 16 * v115);
          v116 = *v43;
          if ( a2 == *v43 )
            goto LABEL_117;
          ++v114;
        }
        goto LABEL_163;
      }
    }
    else
    {
      ++*(_QWORD *)(a5 + 24);
    }
    sub_15CFCF0(a5 + 24, 2 * v40);
    v85 = *(_DWORD *)(a5 + 48);
    if ( !v85 )
      goto LABEL_234;
    v86 = v85 - 1;
    v41 = *(_QWORD *)(a5 + 32);
    LODWORD(v87) = v86 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v88 = *(_DWORD *)(a5 + 40) + 1;
    v43 = (__int64 *)(v41 + 16LL * (unsigned int)v87);
    v89 = *v43;
    if ( a2 == *v43 )
      goto LABEL_117;
    v133 = 1;
    v113 = 0;
    while ( v89 != -8 )
    {
      if ( !v113 && v89 == -16 )
        v113 = v43;
      LODWORD(v38) = v133 + 1;
      v87 = v86 & (unsigned int)(v87 + v133);
      v43 = (__int64 *)(v41 + 16 * v87);
      v89 = *v43;
      if ( a2 == *v43 )
        goto LABEL_117;
      ++v133;
    }
LABEL_163:
    if ( v113 )
      v43 = v113;
LABEL_117:
    *(_DWORD *)(a5 + 40) = v88;
    if ( *v43 != -8 )
      --*(_DWORD *)(a5 + 44);
    v43[1] = v39;
    *v43 = a2;
LABEL_41:
    v46 = *(unsigned int *)(a5 + 8);
    if ( !(_DWORD)v46 )
    {
      if ( !*(_DWORD *)(a5 + 12) )
      {
        sub_16CD150(a5, (const void *)(a5 + 16), 0, 8, v38, v41);
        v46 = *(unsigned int *)(a5 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a5 + 8 * v46) = a2;
      ++*(_DWORD *)(a5 + 8);
LABEL_45:
      *(_QWORD *)(a5 + 56) = v39;
      goto LABEL_4;
    }
    v71 = *(_DWORD *)(a5 + 48);
    v72 = **(_QWORD **)a5;
    if ( v71 )
    {
      v73 = v71 - 1;
      v74 = *(_QWORD *)(a5 + 32);
      v145 = ((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4);
      v75 = (v71 - 1) & v145;
      v76 = (__int64 *)(v74 + 16 * v75);
      v77 = *v76;
      v78 = v76;
      if ( v72 == *v76 )
      {
LABEL_97:
        v79 = v78[1];
LABEL_98:
        v78[1] = 0;
        v80 = *(_QWORD *)(v39 + 32);
        v153[0] = v79;
        if ( v80 == *(_QWORD *)(v39 + 40) )
        {
          sub_15CE310(v39 + 24, (_BYTE *)v80, v153);
        }
        else
        {
          if ( v80 )
          {
            *(_QWORD *)v80 = v79;
            v80 = *(_QWORD *)(v39 + 32);
          }
          v80 += 8;
          *(_QWORD *)(v39 + 32) = v80;
        }
        v81 = v76[1];
        v76[1] = v79;
        if ( v81 )
        {
          v82 = *(_QWORD *)(v81 + 24);
          if ( v82 )
          {
            v146 = v81;
            j_j___libc_free_0(v82, *(_QWORD *)(v81 + 40) - v82);
            v81 = v146;
          }
          v80 = 56;
          j_j___libc_free_0(v81, 56);
          v79 = v76[1];
        }
        *(_QWORD *)(v79 + 8) = v39;
        v83 = v76[1];
        if ( *(_DWORD *)(v83 + 16) != *(_DWORD *)(*(_QWORD *)(v83 + 8) + 16LL) + 1 )
          sub_1AA5500(v83, v80, v72, v75, v81, v74);
        **(_QWORD **)a5 = a2;
        goto LABEL_45;
      }
      v138 = v75;
      v90 = *v76;
      v91 = (__int64 *)(v74 + 16LL * (v73 & v145));
      v92 = 0;
      v135 = 1;
      while ( v90 != -8 )
      {
        if ( !v92 && v90 == -16 )
          v92 = v91;
        v138 = v73 & (v138 + v135);
        v91 = (__int64 *)(v74 + 16LL * v138);
        v90 = *v91;
        if ( v72 == *v91 )
        {
          v78 = (__int64 *)(v74 + 16LL * (v73 & v145));
          v76 = (__int64 *)(v74 + 16LL * v138);
          goto LABEL_141;
        }
        ++v135;
      }
      v93 = *(_DWORD *)(a5 + 40);
      if ( !v92 )
        v92 = v91;
      ++*(_QWORD *)(a5 + 24);
      v94 = v93 + 1;
      v76 = v92;
      if ( 4 * v94 < 3 * v71 )
      {
        if ( v71 - *(_DWORD *)(a5 + 44) - v94 <= v71 >> 3 )
        {
          v140 = v72;
          sub_15CFCF0(a5 + 24, v71);
          v120 = *(_DWORD *)(a5 + 48);
          if ( !v120 )
            goto LABEL_234;
          v121 = v120 - 1;
          v122 = *(_QWORD *)(a5 + 32);
          v123 = 0;
          v72 = v140;
          v124 = 1;
          v125 = v121 & v145;
          v94 = *(_DWORD *)(a5 + 40) + 1;
          v76 = (__int64 *)(v122 + 16LL * (v121 & v145));
          v126 = *v76;
          if ( v140 != *v76 )
          {
            while ( v126 != -8 )
            {
              if ( !v123 && v126 == -16 )
                v123 = v76;
              v125 = v121 & (v124 + v125);
              v76 = (__int64 *)(v122 + 16LL * v125);
              v126 = *v76;
              if ( v140 == *v76 )
                goto LABEL_131;
              ++v124;
            }
LABEL_186:
            if ( v123 )
              v76 = v123;
            goto LABEL_131;
          }
        }
        goto LABEL_131;
      }
    }
    else
    {
      ++*(_QWORD *)(a5 + 24);
    }
    v149 = v72;
    sub_15CFCF0(a5 + 24, 2 * v71);
    v127 = *(_DWORD *)(a5 + 48);
    if ( !v127 )
      goto LABEL_234;
    v72 = v149;
    v128 = v127 - 1;
    v129 = *(_QWORD *)(a5 + 32);
    v130 = v128 & (((unsigned int)v149 >> 9) ^ ((unsigned int)v149 >> 4));
    v94 = *(_DWORD *)(a5 + 40) + 1;
    v76 = (__int64 *)(v129 + 16LL * v130);
    v131 = *v76;
    if ( v149 != *v76 )
    {
      v132 = 1;
      v123 = 0;
      while ( v131 != -8 )
      {
        if ( !v123 && v131 == -16 )
          v123 = v76;
        v130 = v128 & (v132 + v130);
        v76 = (__int64 *)(v129 + 16LL * v130);
        v131 = *v76;
        if ( v149 == *v76 )
          goto LABEL_131;
        ++v132;
      }
      goto LABEL_186;
    }
LABEL_131:
    *(_DWORD *)(a5 + 40) = v94;
    if ( *v76 != -8 )
      --*(_DWORD *)(a5 + 44);
    *v76 = v72;
    v76[1] = 0;
    v71 = *(_DWORD *)(a5 + 48);
    if ( v71 )
    {
      v73 = v71 - 1;
      v74 = *(_QWORD *)(a5 + 32);
      v145 = ((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4);
      v75 = (v71 - 1) & v145;
      v78 = (__int64 *)(v74 + 16 * v75);
      v77 = *v78;
      if ( v72 == *v78 )
        goto LABEL_97;
LABEL_141:
      v99 = 1;
      v100 = 0;
      while ( v77 != -8 )
      {
        if ( !v100 && v77 == -16 )
          v100 = v78;
        v75 = v73 & (v99 + (_DWORD)v75);
        v78 = (__int64 *)(v74 + 16LL * (unsigned int)v75);
        v77 = *v78;
        if ( v72 == *v78 )
          goto LABEL_97;
        ++v99;
      }
      v101 = *(_DWORD *)(a5 + 40);
      if ( v100 )
        v78 = v100;
      ++*(_QWORD *)(a5 + 24);
      v75 = (unsigned int)(v101 + 1);
      if ( 4 * (int)v75 < 3 * v71 )
      {
        if ( v71 - ((_DWORD)v75 + *(_DWORD *)(a5 + 44)) > v71 >> 3 )
        {
LABEL_137:
          *(_DWORD *)(a5 + 40) = v75;
          if ( *v78 != -8 )
            --*(_DWORD *)(a5 + 44);
          *v78 = v72;
          v79 = 0;
          v78[1] = 0;
          goto LABEL_98;
        }
        v139 = v72;
        sub_15CFCF0(a5 + 24, v71);
        v102 = *(_DWORD *)(a5 + 48);
        if ( v102 )
        {
          v103 = v102 - 1;
          v104 = *(_QWORD *)(a5 + 32);
          v72 = v139;
          LODWORD(v74) = v103 & v145;
          v75 = (unsigned int)(*(_DWORD *)(a5 + 40) + 1);
          v105 = (__int64 *)(v104 + 16LL * (v103 & v145));
          v106 = *v105;
          v78 = v105;
          if ( v139 != *v105 )
          {
            v107 = 1;
            v78 = 0;
            while ( v106 != -8 )
            {
              if ( !v78 && v106 == -16 )
                v78 = v105;
              LODWORD(v74) = v103 & (v107 + v74);
              v105 = (__int64 *)(v104 + 16LL * (unsigned int)v74);
              v106 = *v105;
              if ( v139 == *v105 )
              {
                v78 = (__int64 *)(v104 + 16LL * (unsigned int)v74);
                goto LABEL_137;
              }
              ++v107;
            }
            if ( !v78 )
              v78 = v105;
          }
          goto LABEL_137;
        }
LABEL_234:
        ++*(_DWORD *)(a5 + 40);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a5 + 24);
    }
    v147 = v72;
    sub_15CFCF0(a5 + 24, 2 * v71);
    v95 = *(_DWORD *)(a5 + 48);
    if ( v95 )
    {
      v72 = v147;
      v96 = v95 - 1;
      v97 = *(_QWORD *)(a5 + 32);
      v75 = (unsigned int)(*(_DWORD *)(a5 + 40) + 1);
      LODWORD(v74) = v96 & (((unsigned int)v147 >> 9) ^ ((unsigned int)v147 >> 4));
      v78 = (__int64 *)(v97 + 16LL * (unsigned int)v74);
      v98 = *v78;
      if ( v147 != *v78 )
      {
        v118 = 1;
        v119 = 0;
        while ( v98 != -8 )
        {
          if ( !v119 && v98 == -16 )
            v119 = v78;
          LODWORD(v74) = v96 & (v118 + v74);
          v78 = (__int64 *)(v97 + 16LL * (unsigned int)v74);
          v98 = *v78;
          if ( v147 == *v78 )
            goto LABEL_137;
          ++v118;
        }
        if ( v119 )
          v78 = v119;
      }
      goto LABEL_137;
    }
    goto LABEL_234;
  }
LABEL_4:
  result = a6;
  if ( !a6 )
    return result;
  v11 = *(_DWORD *)(a6 + 24);
  result = (unsigned __int64)a3;
  v12 = &a3[a4];
  if ( !v11 )
  {
    if ( a3 == v12 )
      return result;
    goto LABEL_32;
  }
  v13 = v11 - 1;
  v14 = *(_QWORD *)(a6 + 8);
  v15 = v13 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v16 = (__int64 *)(v14 + 16LL * v15);
  v17 = *v16;
  if ( v9 == *v16 )
  {
LABEL_7:
    result = v16[1];
    v142 = result;
    v18 = result != 0;
    if ( a3 == v12 )
    {
      v151 = 0;
LABEL_47:
      if ( v142 )
      {
        if ( v18 )
        {
          result = (unsigned __int64)a3;
          if ( a3 != v12 )
          {
            v152 = v12;
            v47 = a3;
            v48 = v9;
            v144 = 0;
            do
            {
              v49 = *(_DWORD *)(a6 + 24);
              if ( v49 )
              {
                v50 = v49 - 1;
                v51 = *(_QWORD *)(a6 + 8);
                v52 = (v49 - 1) & (((unsigned int)*v47 >> 9) ^ ((unsigned int)*v47 >> 4));
                v53 = (__int64 *)(v51 + 16LL * v52);
                v54 = *v53;
                if ( *v47 == *v53 )
                {
LABEL_57:
                  v55 = (__int64 *)v53[1];
                  if ( v55 )
                  {
                    v56 = (_QWORD *)v55[9];
                    v57 = (_QWORD *)v55[8];
                    if ( v56 != v57 )
                    {
LABEL_59:
                      v58 = &v56[*((unsigned int *)v55 + 20)];
                      v57 = sub_16CC9F0((__int64)(v55 + 7), v48);
                      if ( v48 == *v57 )
                      {
                        v60 = v55[9];
                        if ( v60 == v55[8] )
                          v61 = *((unsigned int *)v55 + 21);
                        else
                          v61 = *((unsigned int *)v55 + 20);
                        v64 = (_QWORD *)(v60 + 8 * v61);
                      }
                      else
                      {
                        v59 = v55[9];
                        if ( v59 != v55[8] )
                        {
                          v57 = (_QWORD *)(v59 + 8LL * *((unsigned int *)v55 + 20));
                          goto LABEL_62;
                        }
                        v57 = (_QWORD *)(v59 + 8LL * *((unsigned int *)v55 + 21));
                        v64 = v57;
                      }
                      goto LABEL_70;
                    }
                    while ( 1 )
                    {
                      v58 = &v57[*((unsigned int *)v55 + 21)];
                      if ( v57 == v58 )
                      {
                        v64 = v57;
                      }
                      else
                      {
                        do
                        {
                          if ( v48 == *v57 )
                            break;
                          ++v57;
                        }
                        while ( v58 != v57 );
                        v64 = v58;
                      }
LABEL_70:
                      while ( v64 != v57 )
                      {
                        if ( *v57 < 0xFFFFFFFFFFFFFFFELL )
                          break;
                        ++v57;
                      }
LABEL_62:
                      if ( v57 != v58 )
                        break;
                      v55 = (__int64 *)*v55;
                      if ( !v55 )
                        goto LABEL_54;
                      v56 = (_QWORD *)v55[9];
                      v57 = (_QWORD *)v55[8];
                      if ( v56 != v57 )
                        goto LABEL_59;
                    }
                    if ( sub_1377F70((__int64)(v55 + 7), v48) )
                    {
                      if ( v144 )
                      {
                        v67 = (_QWORD *)*v144;
                        v68 = 1;
                        if ( *v144 )
                        {
                          do
                          {
                            v67 = (_QWORD *)*v67;
                            ++v68;
                          }
                          while ( v67 );
                        }
                        v69 = (__int64 *)*v55;
                        v70 = 1;
                        if ( *v55 )
                        {
                          do
                          {
                            v69 = (__int64 *)*v69;
                            ++v70;
                          }
                          while ( v69 );
                        }
                        if ( v68 >= v70 )
                          v55 = v144;
                        v144 = v55;
                      }
                      else
                      {
                        v144 = v55;
                      }
                    }
                  }
                }
                else
                {
                  v65 = 1;
                  while ( v54 != -8 )
                  {
                    v66 = v65 + 1;
                    v52 = v50 & (v65 + v52);
                    v53 = (__int64 *)(v51 + 16LL * v52);
                    v54 = *v53;
                    if ( *v47 == *v53 )
                      goto LABEL_57;
                    v65 = v66;
                  }
                }
              }
LABEL_54:
              ++v47;
            }
            while ( v152 != v47 );
            result = (unsigned __int64)v144;
            if ( v144 )
              return (unsigned __int64)sub_1400330((__int64)v144, a2, a6);
          }
        }
        else
        {
          result = (unsigned __int64)sub_1400330(v142, a2, a6);
          if ( v151 )
          {
            v35 = *(_QWORD **)(v142 + 32);
            result = 0;
            if ( a2 != *v35 )
            {
              do
              {
                result = (unsigned int)(result + 1);
                v36 = &v35[result];
              }
              while ( a2 != *v36 );
              *v36 = *v35;
              result = *(_QWORD *)(v142 + 32);
              *(_QWORD *)result = a2;
            }
          }
        }
      }
      return result;
    }
LABEL_8:
    v19 = a3;
    v134 = v9;
    v20 = a5;
    v151 = 0;
    do
    {
      result = *(unsigned int *)(v20 + 48);
      if ( (_DWORD)result )
      {
        v24 = *v19;
        v25 = *(_QWORD *)(v20 + 32);
        v26 = (result - 1) & (((unsigned int)*v19 >> 9) ^ ((unsigned int)*v19 >> 4));
        v27 = (__int64 *)(v25 + 16LL * v26);
        v28 = *v27;
        if ( *v19 == *v27 )
        {
LABEL_18:
          result = v25 + 16 * result;
          if ( v27 != (__int64 *)result && v27[1] )
          {
            if ( a7 )
            {
              result = a6;
              v29 = *(_DWORD *)(a6 + 24);
              if ( v29 )
              {
                v30 = v29 - 1;
                v31 = *(_QWORD *)(a6 + 8);
                v32 = v30 & (((unsigned int)*v19 >> 9) ^ ((unsigned int)*v19 >> 4));
                v33 = (__int64 *)(v31 + 16LL * v32);
                v34 = *v33;
                if ( v24 == *v33 )
                {
LABEL_23:
                  result = v33[1];
                  if ( result )
                  {
                    result = sub_1377F70(result + 56, v134);
                    if ( !(_DWORD)result )
                    {
                      result = (unsigned __int64)a8;
                      *(_BYTE *)a8 = 1;
                    }
                  }
                }
                else
                {
                  result = 1;
                  while ( v34 != -8 )
                  {
                    v84 = result + 1;
                    v32 = v30 & (result + v32);
                    v33 = (__int64 *)(v31 + 16LL * v32);
                    v34 = *v33;
                    if ( v24 == *v33 )
                      goto LABEL_23;
                    result = v84;
                  }
                }
              }
            }
            if ( v142 )
            {
              v21 = sub_1377F70(v142 + 56, v24);
              v22 = v151;
              v23 = !v21;
              if ( !v21 )
                v22 = 1;
              result = 0;
              if ( !v23 )
                v18 = 0;
              v151 = v22;
            }
          }
        }
        else
        {
          v62 = 1;
          while ( v28 != -8 )
          {
            v63 = v62 + 1;
            v26 = (result - 1) & (v62 + v26);
            v27 = (__int64 *)(v25 + 16LL * v26);
            v28 = *v27;
            if ( v24 == *v27 )
              goto LABEL_18;
            v62 = v63;
          }
        }
      }
      ++v19;
    }
    while ( v19 != v12 );
    v9 = v134;
    goto LABEL_47;
  }
  result = 1;
  while ( v17 != -8 )
  {
    v117 = result + 1;
    v15 = v13 & (result + v15);
    v16 = (__int64 *)(v14 + 16LL * v15);
    v17 = *v16;
    if ( v9 == *v16 )
      goto LABEL_7;
    result = v117;
  }
  if ( a3 != v12 )
  {
LABEL_32:
    v142 = 0;
    v18 = 0;
    goto LABEL_8;
  }
  return result;
}
