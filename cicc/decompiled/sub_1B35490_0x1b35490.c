// Function: sub_1B35490
// Address: 0x1b35490
//
__int64 __fastcall sub_1B35490(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        unsigned __int8 a15)
{
  _QWORD *v16; // r13
  __int64 v17; // r12
  _QWORD *v18; // rax
  _QWORD *v19; // rbx
  int v20; // r8d
  int v21; // r9d
  __int64 v22; // r15
  __int64 v23; // rax
  _QWORD *v24; // rax
  _DWORD *v25; // r15
  unsigned __int64 v26; // rbx
  unsigned int *v27; // r12
  unsigned __int64 v28; // rax
  __int64 v29; // r11
  unsigned int v30; // ecx
  unsigned int v31; // edx
  unsigned int *v32; // rax
  __int64 v33; // rdi
  __int64 v34; // rdx
  unsigned int *v35; // rsi
  __int64 v36; // r13
  _QWORD *v37; // rax
  __int64 v38; // r12
  unsigned __int8 v39; // al
  int v40; // eax
  _DWORD *v41; // rax
  int v42; // r9d
  double v43; // xmm4_8
  double v44; // xmm5_8
  _DWORD *v45; // r11
  __int64 v46; // r15
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rcx
  unsigned __int64 v50; // rdx
  __int64 v51; // rdx
  int v52; // r15d
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rbx
  int v56; // ebx
  __int64 v57; // rax
  __int64 v58; // rdx
  int v59; // eax
  __int64 v60; // rbx
  unsigned int v61; // r15d
  __int64 v62; // rax
  __int64 v63; // r14
  __int64 v64; // rcx
  _QWORD *v65; // rax
  bool v66; // r15
  __int64 v67; // rdi
  _QWORD *v68; // r12
  __int64 v69; // rax
  __int64 *v70; // r8
  __int64 *v71; // r14
  __int64 v72; // r15
  __int64 *v73; // rax
  int v74; // eax
  __int64 v75; // rsi
  int v76; // ecx
  unsigned int v77; // edx
  _QWORD *v78; // rax
  _QWORD *v79; // rdi
  int v80; // eax
  int v81; // esi
  __int64 v82; // rcx
  unsigned int v83; // edx
  _QWORD *v84; // rax
  _QWORD *v85; // rdi
  unsigned __int64 v86; // rdx
  _QWORD **v87; // r12
  __int64 v88; // rbx
  __int64 v89; // r13
  int v90; // eax
  int v91; // ecx
  __int64 v92; // rsi
  unsigned int v93; // edx
  __int64 *v94; // rax
  __int64 v95; // rdi
  int v97; // eax
  _DWORD *v98; // r9
  __int64 v99; // rax
  __int64 v100; // rcx
  unsigned __int64 v101; // rdx
  __int64 v102; // rdx
  int v103; // eax
  _DWORD *v104; // r9
  __int64 v105; // rdx
  _QWORD *v106; // rax
  __int64 v107; // rsi
  unsigned __int64 v108; // rcx
  __int64 v109; // rcx
  int v110; // eax
  _DWORD *v111; // r9
  __int64 v112; // rdx
  _QWORD *v113; // rax
  __int64 v114; // rdi
  unsigned __int64 v115; // rsi
  __int64 v116; // rsi
  int v117; // eax
  int v118; // esi
  __int64 v119; // rcx
  unsigned int v120; // edx
  __int64 *v121; // rax
  __int64 v122; // rdi
  __int64 v123; // rax
  double v124; // xmm4_8
  double v125; // xmm5_8
  __int64 v126; // rsi
  unsigned __int64 v127; // rax
  int v128; // eax
  __int64 v129; // rsi
  int v130; // ecx
  unsigned int v131; // edx
  __int64 *v132; // rax
  __int64 v133; // r8
  int v134; // eax
  int v135; // r8d
  int v136; // eax
  int v137; // r8d
  int v138; // eax
  int v139; // r8d
  __int64 v140; // rdx
  unsigned __int64 v141; // rcx
  __int64 v142; // rdx
  unsigned __int64 v143; // rcx
  int v144; // eax
  int v145; // r8d
  int v146; // eax
  int v147; // edi
  __int64 v150; // [rsp+10h] [rbp-650h]
  __int64 v152; // [rsp+28h] [rbp-638h]
  _QWORD *v154; // [rsp+38h] [rbp-628h]
  __int64 v155; // [rsp+38h] [rbp-628h]
  unsigned __int8 v156; // [rsp+47h] [rbp-619h]
  __int64 v157; // [rsp+48h] [rbp-618h]
  __int64 v158; // [rsp+50h] [rbp-610h] BYREF
  __int64 v159; // [rsp+58h] [rbp-608h]
  _DWORD *v160; // [rsp+220h] [rbp-440h] BYREF
  __int64 v161; // [rsp+228h] [rbp-438h]
  _BYTE v162[1072]; // [rsp+230h] [rbp-430h] BYREF

  v16 = a1;
  v17 = a1[1];
  v156 = a15;
  v160 = v162;
  v161 = 0x4000000000LL;
  if ( !v17 )
  {
LABEL_61:
    sub_15F20C0(v16);
    v80 = *(_DWORD *)(a3 + 24);
    if ( v80 )
    {
      v81 = v80 - 1;
      v82 = *(_QWORD *)(a3 + 8);
      v83 = (v80 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v84 = (_QWORD *)(v82 + 16LL * v83);
      v85 = (_QWORD *)*v84;
      if ( v16 == (_QWORD *)*v84 )
      {
LABEL_63:
        *v84 = -16;
        --*(_DWORD *)(a3 + 16);
        ++*(_DWORD *)(a3 + 20);
      }
      else
      {
        v144 = 1;
        while ( v85 != (_QWORD *)-8LL )
        {
          v145 = v144 + 1;
          v83 = v81 & (v144 + v83);
          v84 = (_QWORD *)(v82 + 16LL * v83);
          v85 = (_QWORD *)*v84;
          if ( v16 == (_QWORD *)*v84 )
            goto LABEL_63;
          v144 = v145;
        }
      }
    }
    v86 = *(_QWORD *)(a2 + 576) & 0xFFFFFFFFFFFFFFF8LL;
    if ( (*(_QWORD *)(a2 + 576) & 4) != 0 )
    {
      v87 = *(_QWORD ***)v86;
      v88 = *(_QWORD *)v86 + 8LL * *(unsigned int *)(v86 + 8);
    }
    else
    {
      v87 = (_QWORD **)(a2 + 576);
      if ( !v86 )
      {
LABEL_71:
        v156 = 1;
        v41 = v160;
        goto LABEL_72;
      }
      v88 = a2 + 584;
    }
    while ( (_QWORD **)v88 != v87 )
    {
      v89 = (__int64)*v87;
      sub_15F20C0(*v87);
      v90 = *(_DWORD *)(a3 + 24);
      if ( v90 )
      {
        v91 = v90 - 1;
        v92 = *(_QWORD *)(a3 + 8);
        v93 = (v90 - 1) & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
        v94 = (__int64 *)(v92 + 16LL * v93);
        v95 = *v94;
        if ( v89 == *v94 )
        {
LABEL_68:
          *v94 = -16;
          --*(_DWORD *)(a3 + 16);
          ++*(_DWORD *)(a3 + 20);
        }
        else
        {
          v136 = 1;
          while ( v95 != -8 )
          {
            v137 = v136 + 1;
            v93 = v91 & (v136 + v93);
            v94 = (__int64 *)(v92 + 16LL * v93);
            v95 = *v94;
            if ( v89 == *v94 )
              goto LABEL_68;
            v136 = v137;
          }
        }
      }
      ++v87;
    }
    goto LABEL_71;
  }
  do
  {
    while ( 1 )
    {
      v18 = sub_1648700(v17);
      v19 = v18;
      if ( *((_BYTE *)v18 + 16) == 55 )
        break;
      v17 = *(_QWORD *)(v17 + 8);
      if ( !v17 )
        goto LABEL_8;
    }
    v22 = (unsigned int)sub_1B34670(a3, (__int64)v18);
    v23 = (unsigned int)v161;
    if ( (unsigned int)v161 >= HIDWORD(v161) )
    {
      sub_16CD150((__int64)&v160, v162, 0, 16, v20, v21);
      v23 = (unsigned int)v161;
    }
    v24 = &v160[4 * v23];
    *v24 = v22;
    v24[1] = v19;
    LODWORD(v161) = v161 + 1;
    v17 = *(_QWORD *)(v17 + 8);
  }
  while ( v17 );
LABEL_8:
  v25 = v160;
  v26 = 4LL * (unsigned int)v161;
  v27 = &v160[v26];
  if ( v160 != &v160[v26] )
  {
    _BitScanReverse64(&v28, (__int64)(v26 * 4) >> 4);
    sub_1B33B90((__int64)v160, (unsigned __int64)&v160[v26], 2LL * (int)(63 - (v28 ^ 0x3F)));
    if ( v26 <= 64 )
    {
      sub_1B316F0((__int64)v25, v27);
    }
    else
    {
      sub_1B316F0((__int64)v25, v25 + 64);
      for ( ; v27 != (unsigned int *)v29; *((_QWORD *)v35 + 1) = v33 )
      {
        while ( 1 )
        {
          v30 = *(_DWORD *)v29;
          v31 = *(_DWORD *)(v29 - 16);
          v32 = (unsigned int *)(v29 - 16);
          v33 = *(_QWORD *)(v29 + 8);
          if ( *(_DWORD *)v29 < v31 )
            break;
          v126 = v29;
          v29 += 16;
          *(_DWORD *)v126 = v30;
          *(_QWORD *)(v126 + 8) = v33;
          if ( v27 == (unsigned int *)v29 )
            goto LABEL_14;
        }
        do
        {
          v32[4] = v31;
          v34 = *((_QWORD *)v32 + 1);
          v35 = v32;
          v32 -= 4;
          *((_QWORD *)v32 + 5) = v34;
          v31 = *v32;
        }
        while ( v30 < *v32 );
        v29 += 16;
        *v35 = v30;
      }
    }
  }
LABEL_14:
  if ( !v16[1] )
    goto LABEL_61;
  v154 = v16;
  v36 = v16[1];
  while ( 1 )
  {
    while ( 1 )
    {
      v37 = sub_1648700(v36);
      v38 = (__int64)v37;
      if ( !a15 )
        break;
      v39 = *((_BYTE *)v37 + 16);
      if ( v39 <= 0x17u )
        break;
      if ( v39 != 78 )
      {
        if ( v39 == 71 )
        {
          v97 = sub_1B34670(a3, v38);
          v159 = 0;
          LODWORD(v158) = v97;
          v41 = sub_1B31870(v160, (__int64)&v160[4 * (unsigned int)v161], &v158);
          if ( v98 == v41 )
            goto LABEL_72;
          v99 = *(_QWORD *)(*(_QWORD *)(*((_QWORD *)v41 - 1) - 48LL) - 24LL);
          if ( v99 )
          {
            if ( *(_QWORD *)(v38 - 24) )
            {
              v100 = *(_QWORD *)(v38 - 16);
              v101 = *(_QWORD *)(v38 - 8) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v101 = v100;
              if ( v100 )
                *(_QWORD *)(v100 + 16) = *(_QWORD *)(v100 + 16) & 3LL | v101;
            }
            *(_QWORD *)(v38 - 24) = v99;
            v102 = *(_QWORD *)(v99 + 8);
            *(_QWORD *)(v38 - 16) = v102;
            if ( v102 )
              *(_QWORD *)(v102 + 16) = (v38 - 16) | *(_QWORD *)(v102 + 16) & 3LL;
            *(_QWORD *)(v38 - 8) = (v99 + 8) | *(_QWORD *)(v38 - 8) & 3LL;
            *(_QWORD *)(v99 + 8) = v38 - 24;
          }
          else if ( *(_QWORD *)(v38 - 24) )
          {
            v142 = *(_QWORD *)(v38 - 16);
            v143 = *(_QWORD *)(v38 - 8) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v143 = v142;
            if ( v142 )
              *(_QWORD *)(v142 + 16) = v143 | *(_QWORD *)(v142 + 16) & 3LL;
            *(_QWORD *)(v38 - 24) = 0;
          }
        }
        else
        {
          if ( v39 != 56 )
            break;
          v103 = sub_1B34670(a3, v38);
          v159 = 0;
          LODWORD(v158) = v103;
          v41 = sub_1B31870(v160, (__int64)&v160[4 * (unsigned int)v161], &v158);
          if ( v104 == v41 )
            goto LABEL_72;
          v105 = *(_QWORD *)(*(_QWORD *)(*((_QWORD *)v41 - 1) - 48LL) - 24LL);
          v106 = (_QWORD *)(v38 - 24LL * (*(_DWORD *)(v38 + 20) & 0xFFFFFFF));
          if ( *v106 )
          {
            v107 = v106[1];
            v108 = v106[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v108 = v107;
            if ( v107 )
              *(_QWORD *)(v107 + 16) = *(_QWORD *)(v107 + 16) & 3LL | v108;
          }
          *v106 = v105;
          if ( v105 )
          {
            v109 = *(_QWORD *)(v105 + 8);
            v106[1] = v109;
            if ( v109 )
              *(_QWORD *)(v109 + 16) = (unsigned __int64)(v106 + 1) | *(_QWORD *)(v109 + 16) & 3LL;
            v106[2] = (v105 + 8) | v106[2] & 3LL;
            *(_QWORD *)(v105 + 8) = v106;
          }
        }
        goto LABEL_45;
      }
      v52 = *(_DWORD *)(v38 + 20) & 0xFFFFFFF;
      if ( *(char *)(v38 + 23) < 0 )
      {
        v53 = sub_1648A40(v38);
        v55 = v53 + v54;
        if ( *(char *)(v38 + 23) >= 0 )
        {
          if ( (unsigned int)(v55 >> 4) )
LABEL_157:
            BUG();
        }
        else if ( (unsigned int)((v55 - sub_1648A40(v38)) >> 4) )
        {
          if ( *(char *)(v38 + 23) >= 0 )
            goto LABEL_157;
          v56 = *(_DWORD *)(sub_1648A40(v38) + 8);
          if ( *(char *)(v38 + 23) >= 0 )
            BUG();
          v57 = sub_1648A40(v38);
          v59 = *(_DWORD *)(v57 + v58 - 4) - v56;
          goto LABEL_40;
        }
      }
      v59 = 0;
LABEL_40:
      v60 = 0;
      v61 = v52 - 1 - v59;
      if ( v61 )
      {
        v62 = a3;
        v63 = v61;
        v64 = v62;
        do
        {
          v65 = *(_QWORD **)(v38 + 24 * (v60 - (*(_DWORD *)(v38 + 20) & 0xFFFFFFF)));
          v66 = v154 == v65 && v65 != 0;
          if ( v66 )
          {
            v150 = v64;
            v110 = sub_1B34670(v64, v38);
            v159 = 0;
            LODWORD(v158) = v110;
            v41 = sub_1B31870(v160, (__int64)&v160[4 * (unsigned int)v161], &v158);
            if ( v111 == v41 )
            {
              v156 = v66;
              goto LABEL_72;
            }
            v64 = v150;
            v112 = *(_QWORD *)(*(_QWORD *)(*((_QWORD *)v41 - 1) - 48LL) - 24LL);
            v113 = (_QWORD *)(v38 + 24 * (v60 - (*(_DWORD *)(v38 + 20) & 0xFFFFFFF)));
            if ( *v113 )
            {
              v114 = v113[1];
              v115 = v113[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v115 = v114;
              if ( v114 )
                *(_QWORD *)(v114 + 16) = *(_QWORD *)(v114 + 16) & 3LL | v115;
            }
            *v113 = v112;
            if ( v112 )
            {
              v116 = *(_QWORD *)(v112 + 8);
              v113[1] = v116;
              if ( v116 )
                *(_QWORD *)(v116 + 16) = (unsigned __int64)(v113 + 1) | *(_QWORD *)(v116 + 16) & 3LL;
              v113[2] = (v112 + 8) | v113[2] & 3LL;
              *(_QWORD *)(v112 + 8) = v113;
            }
          }
          ++v60;
        }
        while ( v63 != v60 );
        a3 = v64;
      }
LABEL_45:
      v36 = *(_QWORD *)(v36 + 8);
      if ( !v36 )
      {
LABEL_46:
        v16 = v154;
        v67 = v154[1];
        if ( !v67 )
          goto LABEL_61;
        v155 = a3;
        while ( 1 )
        {
          v68 = sub_1648700(v67);
          v152 = *(v68 - 6);
          v69 = *(_QWORD *)(a2 + 576);
          if ( (v69 & 4) != 0 )
            break;
          v70 = (__int64 *)(a2 + 576);
          if ( (v69 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            v157 = a2 + 584;
            goto LABEL_51;
          }
LABEL_54:
          sub_15F20C0(v68);
          v74 = *(_DWORD *)(v155 + 24);
          if ( v74 )
          {
            v75 = *(_QWORD *)(v155 + 8);
            v76 = v74 - 1;
            v77 = (v74 - 1) & (((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4));
            v78 = (_QWORD *)(v75 + 16LL * v77);
            v79 = (_QWORD *)*v78;
            if ( v68 == (_QWORD *)*v78 )
            {
LABEL_56:
              *v78 = -16;
              --*(_DWORD *)(v155 + 16);
              ++*(_DWORD *)(v155 + 20);
            }
            else
            {
              v134 = 1;
              while ( v79 != (_QWORD *)-8LL )
              {
                v135 = v134 + 1;
                v77 = v76 & (v134 + v77);
                v78 = (_QWORD *)(v75 + 16LL * v77);
                v79 = (_QWORD *)*v78;
                if ( v68 == (_QWORD *)*v78 )
                  goto LABEL_56;
                v134 = v135;
              }
            }
          }
          if ( a15 )
          {
            if ( *(_BYTE *)(v152 + 16) == 54 && sub_1648CD0(v152, 0) )
            {
              sub_15F20C0((_QWORD *)v152);
              v128 = *(_DWORD *)(v155 + 24);
              if ( v128 )
              {
                v129 = *(_QWORD *)(v155 + 8);
                v130 = v128 - 1;
                v131 = (v128 - 1) & (((unsigned int)v152 >> 9) ^ ((unsigned int)v152 >> 4));
                v132 = (__int64 *)(v129 + 16LL * v131);
                v133 = *v132;
                if ( *v132 == v152 )
                {
LABEL_124:
                  *v132 = -16;
                  --*(_DWORD *)(v155 + 16);
                  ++*(_DWORD *)(v155 + 20);
                }
                else
                {
                  v146 = 1;
                  while ( v133 != -8 )
                  {
                    v147 = v146 + 1;
                    v131 = v130 & (v146 + v131);
                    v132 = (__int64 *)(v129 + 16LL * v131);
                    v133 = *v132;
                    if ( v152 == *v132 )
                      goto LABEL_124;
                    v146 = v147;
                  }
                }
              }
            }
          }
          v67 = v16[1];
          if ( !v67 )
          {
            a3 = v155;
            goto LABEL_61;
          }
        }
        v127 = v69 & 0xFFFFFFFFFFFFFFF8LL;
        v70 = *(__int64 **)v127;
        v157 = *(_QWORD *)v127 + 8LL * *(unsigned int *)(v127 + 8);
LABEL_51:
        if ( (__int64 *)v157 != v70 )
        {
          v71 = v70;
          do
          {
            v72 = *v71++;
            v73 = (__int64 *)sub_15F2050((__int64)v16);
            sub_15A5590((__int64)&v158, v73, 0, 0);
            sub_1AE9B50(v72, (__int64)v68, &v158);
            sub_129E320((__int64)&v158, (__int64)v68);
          }
          while ( (__int64 *)v157 != v71 );
        }
        goto LABEL_54;
      }
    }
    v36 = *(_QWORD *)(v36 + 8);
    if ( *(_BYTE *)(v38 + 16) == 54 )
      break;
LABEL_16:
    if ( !v36 )
      goto LABEL_46;
  }
  v40 = sub_1B34670(a3, v38);
  v159 = 0;
  LODWORD(v158) = v40;
  v41 = sub_1B31870(v160, (__int64)&v160[4 * (unsigned int)v161], &v158);
  if ( v45 != v41 )
  {
    v46 = *(_QWORD *)(*((_QWORD *)v41 - 1) - 48LL);
    if ( a15 )
    {
      v47 = *(_QWORD *)(v46 - 24);
      v48 = *(_QWORD *)(v38 - 24);
      if ( v47 )
      {
        if ( v48 )
        {
          v49 = *(_QWORD *)(v38 - 16);
          v50 = *(_QWORD *)(v38 - 8) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v50 = v49;
          if ( v49 )
            *(_QWORD *)(v49 + 16) = *(_QWORD *)(v49 + 16) & 3LL | v50;
        }
        *(_QWORD *)(v38 - 24) = v47;
        v51 = *(_QWORD *)(v47 + 8);
        *(_QWORD *)(v38 - 16) = v51;
        if ( v51 )
          *(_QWORD *)(v51 + 16) = (v38 - 16) | *(_QWORD *)(v51 + 16) & 3LL;
        *(_QWORD *)(v38 - 8) = (v47 + 8) | *(_QWORD *)(v38 - 8) & 3LL;
        *(_QWORD *)(v47 + 8) = v38 - 24;
      }
      else if ( v48 )
      {
        v140 = *(_QWORD *)(v38 - 16);
        v141 = *(_QWORD *)(v38 - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v141 = v140;
        if ( v140 )
          *(_QWORD *)(v140 + 16) = v141 | *(_QWORD *)(v140 + 16) & 3LL;
        *(_QWORD *)(v38 - 24) = 0;
      }
LABEL_108:
      sub_15F20C0((_QWORD *)v38);
      v117 = *(_DWORD *)(a3 + 24);
      if ( v117 )
      {
        v118 = v117 - 1;
        v119 = *(_QWORD *)(a3 + 8);
        v120 = (v117 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
        v121 = (__int64 *)(v119 + 16LL * v120);
        v122 = *v121;
        if ( v38 == *v121 )
        {
LABEL_110:
          *v121 = -16;
          --*(_DWORD *)(a3 + 16);
          ++*(_DWORD *)(a3 + 20);
        }
        else
        {
          v138 = 1;
          while ( v122 != -8 )
          {
            v139 = v138 + 1;
            v120 = v118 & (v138 + v120);
            v121 = (__int64 *)(v119 + 16LL * v120);
            v122 = *v121;
            if ( v38 == *v121 )
              goto LABEL_110;
            v138 = v139;
          }
        }
      }
      goto LABEL_16;
    }
    if ( v46 )
    {
      if ( !a6 )
      {
LABEL_105:
        if ( v38 == v46 )
          v46 = sub_1599EF0(*(__int64 ***)v38);
        goto LABEL_107;
      }
    }
    else if ( !a6 )
    {
LABEL_107:
      sub_164D160(v38, v46, a7, a8, a9, a10, v43, v44, a13, a14);
      goto LABEL_108;
    }
    if ( (*(_QWORD *)(v38 + 48) || *(__int16 *)(v38 + 18) < 0)
      && sub_1625790(v38, 11)
      && !(unsigned __int8)sub_14BFF20(v46, a4, 0, a6, v38, a5) )
    {
      sub_1B31B30(a6, (__int64 ***)v38);
    }
    goto LABEL_105;
  }
  if ( !v42 )
  {
    v123 = sub_1599EF0(*(__int64 ***)v38);
    sub_164D160(v38, v123, a7, a8, a9, a10, v124, v125, a13, a14);
    goto LABEL_108;
  }
  v156 = 0;
LABEL_72:
  if ( v41 != (_DWORD *)v162 )
    _libc_free((unsigned __int64)v41);
  return v156;
}
