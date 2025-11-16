// Function: sub_2CA2C90
// Address: 0x2ca2c90
//
void __fastcall sub_2CA2C90(__int64 a1, __int64 *a2, __int64 a3, unsigned int a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v9; // rbx
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 i; // rbx
  __int64 v13; // r12
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // rdx
  __int64 v18; // r11
  _QWORD *v19; // r13
  _QWORD *v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // rcx
  unsigned int v23; // esi
  __int64 v24; // r9
  unsigned int v25; // ecx
  __int64 v26; // rdx
  _QWORD *v27; // r8
  int v28; // esi
  __int64 v29; // rax
  int v30; // ecx
  __int64 v31; // r8
  __int64 v32; // rdi
  int v33; // ecx
  unsigned int v34; // edx
  __int64 *v35; // rax
  __int64 v36; // r9
  __int64 v37; // rdx
  _DWORD *v38; // r15
  unsigned int v39; // r11d
  unsigned __int64 v40; // r12
  unsigned __int64 v41; // r10
  _DWORD *v42; // rbx
  unsigned __int64 v43; // r15
  unsigned __int64 *v44; // r14
  unsigned __int64 v45; // r13
  unsigned int v46; // edx
  unsigned int v47; // esi
  unsigned int v48; // ecx
  bool v49; // cf
  _QWORD **v50; // rdx
  unsigned int *v51; // rsi
  char v52; // al
  __int64 *v53; // rax
  __int64 *v54; // rdx
  __int64 v55; // rax
  unsigned int v56; // r11d
  unsigned __int64 v57; // r10
  _QWORD *v58; // rax
  _QWORD *v59; // rsi
  unsigned int v60; // eax
  __int64 *v61; // r9
  unsigned int v62; // r8d
  __int64 *v63; // rax
  __int64 v64; // rax
  __int64 *v65; // rax
  __int64 *v66; // rdx
  char v67; // di
  _QWORD *v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rsi
  unsigned int v71; // ecx
  _QWORD *v72; // rax
  _QWORD *v73; // r8
  __int64 *v74; // rax
  __int64 v75; // rdi
  _BYTE *v76; // rsi
  int v77; // eax
  signed __int64 v78; // r8
  __int64 v79; // rcx
  __int64 v80; // rax
  unsigned __int64 v81; // rax
  __int64 v82; // r12
  unsigned __int64 v83; // r12
  __int64 v84; // rax
  char *v85; // rcx
  unsigned __int64 v86; // r12
  char *v87; // rax
  _QWORD *v88; // rdi
  int v89; // eax
  int v90; // edx
  bool v91; // al
  int v92; // r9d
  int v93; // r9d
  __int64 v94; // r8
  unsigned int v95; // esi
  _QWORD *v96; // rax
  int v97; // eax
  int v98; // eax
  _QWORD *v99; // rax
  __int64 *v100; // rax
  unsigned int v101; // r11d
  unsigned __int64 v102; // r10
  __int64 v103; // rdi
  _BYTE *v104; // rsi
  int v105; // r10d
  __int64 v106; // rax
  __int64 *v107; // rax
  int v108; // r9d
  int v109; // r10d
  _QWORD *v110; // rcx
  unsigned __int64 v111; // [rsp+8h] [rbp-148h]
  unsigned int v112; // [rsp+10h] [rbp-140h]
  unsigned __int64 v113; // [rsp+18h] [rbp-138h]
  __int64 *v114; // [rsp+18h] [rbp-138h]
  __int64 *v115; // [rsp+18h] [rbp-138h]
  unsigned int v117; // [rsp+30h] [rbp-120h]
  __int64 v121; // [rsp+58h] [rbp-F8h]
  __int64 v122; // [rsp+60h] [rbp-F0h]
  unsigned __int64 v123; // [rsp+70h] [rbp-E0h]
  unsigned __int64 v124; // [rsp+70h] [rbp-E0h]
  unsigned int v125; // [rsp+70h] [rbp-E0h]
  unsigned __int64 v126; // [rsp+70h] [rbp-E0h]
  unsigned int v127; // [rsp+78h] [rbp-D8h]
  unsigned int v128; // [rsp+78h] [rbp-D8h]
  _QWORD *v129; // [rsp+78h] [rbp-D8h]
  unsigned __int64 v130; // [rsp+78h] [rbp-D8h]
  unsigned __int64 v131; // [rsp+78h] [rbp-D8h]
  unsigned __int64 v132; // [rsp+78h] [rbp-D8h]
  unsigned int v133; // [rsp+80h] [rbp-D0h]
  unsigned int v134; // [rsp+84h] [rbp-CCh]
  unsigned int v135; // [rsp+84h] [rbp-CCh]
  unsigned int v136; // [rsp+84h] [rbp-CCh]
  unsigned int v137; // [rsp+84h] [rbp-CCh]
  unsigned __int64 *n; // [rsp+88h] [rbp-C8h]
  _DWORD *v139; // [rsp+90h] [rbp-C0h]
  __int64 v140; // [rsp+90h] [rbp-C0h]
  int v141; // [rsp+90h] [rbp-C0h]
  __int64 v142; // [rsp+90h] [rbp-C0h]
  void *src; // [rsp+98h] [rbp-B8h]
  __int64 v144; // [rsp+A0h] [rbp-B0h]
  unsigned __int64 v145; // [rsp+A0h] [rbp-B0h]
  __int64 v146; // [rsp+A8h] [rbp-A8h]
  __int64 v147; // [rsp+A8h] [rbp-A8h]
  _QWORD *v149; // [rsp+B8h] [rbp-98h]
  __int64 v150; // [rsp+B8h] [rbp-98h]
  __int64 v151; // [rsp+B8h] [rbp-98h]
  char *v152; // [rsp+B8h] [rbp-98h]
  __int64 v153; // [rsp+B8h] [rbp-98h]
  __int64 v154; // [rsp+C0h] [rbp-90h]
  __int64 v155; // [rsp+C0h] [rbp-90h]
  _QWORD *v156; // [rsp+C8h] [rbp-88h]
  unsigned __int64 v157; // [rsp+C8h] [rbp-88h]
  __int64 v158; // [rsp+D0h] [rbp-80h] BYREF
  _BYTE *v159; // [rsp+D8h] [rbp-78h] BYREF
  _QWORD *v160; // [rsp+E0h] [rbp-70h] BYREF
  _QWORD *v161; // [rsp+E8h] [rbp-68h] BYREF
  __int64 v162; // [rsp+F0h] [rbp-60h] BYREF
  __int64 v163; // [rsp+F8h] [rbp-58h] BYREF
  __int64 *v164; // [rsp+100h] [rbp-50h]
  __int64 *v165; // [rsp+108h] [rbp-48h]
  __int64 *v166; // [rsp+110h] [rbp-40h]
  __int64 v167; // [rsp+118h] [rbp-38h]

  v6 = a2[1] - *a2;
  v133 = qword_50124A8;
  v165 = &v163;
  v166 = &v163;
  v7 = v6 >> 3;
  LODWORD(v163) = 0;
  v164 = 0;
  v167 = 0;
  if ( (unsigned __int64)v6 > 0x1FFFFFFFFFFFFFF8LL )
    sub_4262D8((__int64)"vector::reserve");
  if ( !v7 )
    goto LABEL_71;
  v9 = 32 * v7;
  v10 = sub_22077B0(32 * v7);
  v154 = v9 + v10;
  v11 = *a2;
  v146 = (a2[1] - *a2) >> 3;
  if ( !v146 )
  {
    src = (void *)v10;
    goto LABEL_69;
  }
  src = (void *)v10;
  v144 = a1 + 32;
  v156 = (_QWORD *)(a1 + 72);
  for ( i = 0; i != v146; ++i )
  {
    v13 = *(_QWORD *)(v11 + 8 * i);
    v14 = *(_QWORD *)(a1 + 184);
    v160 = *(_QWORD **)v13;
    v15 = sub_D95540((__int64)v160);
    v16 = sub_D97050(v14, v15);
    v17 = *(_QWORD **)(a1 + 80);
    v18 = v16;
    v19 = v160;
    if ( v17 )
    {
      v20 = v156;
      do
      {
        while ( 1 )
        {
          v21 = v17[2];
          v22 = v17[3];
          if ( v17[4] >= (unsigned __int64)v160 )
            break;
          v17 = (_QWORD *)v17[3];
          if ( !v22 )
            goto LABEL_10;
        }
        v20 = v17;
        v17 = (_QWORD *)v17[2];
      }
      while ( v21 );
LABEL_10:
      if ( v156 != v20 && v20[4] <= (unsigned __int64)v160 )
        v19 = (_QWORD *)v20[5];
    }
    v23 = *(_DWORD *)(a1 + 56);
    if ( !v23 )
    {
      ++*(_QWORD *)(a1 + 32);
      v161 = 0;
LABEL_130:
      v153 = v18;
      sub_2C96F50(v144, 2 * v23);
      v92 = *(_DWORD *)(a1 + 56);
      v18 = v153;
      if ( v92 )
      {
        v93 = v92 - 1;
        v94 = *(_QWORD *)(a1 + 40);
        v149 = v160;
        v95 = v93 & (((unsigned int)v160 >> 9) ^ ((unsigned int)v160 >> 4));
        v88 = (_QWORD *)(v94 + 16LL * v95);
        v96 = (_QWORD *)*v88;
        if ( v160 == (_QWORD *)*v88 )
        {
LABEL_132:
          v97 = *(_DWORD *)(a1 + 48);
          v161 = v88;
          v90 = v97 + 1;
        }
        else
        {
          v109 = 1;
          v110 = 0;
          while ( v96 != (_QWORD *)-4096LL )
          {
            if ( v96 == (_QWORD *)-8192LL && !v110 )
              v110 = v88;
            v95 = v93 & (v109 + v95);
            v88 = (_QWORD *)(v94 + 16LL * v95);
            v96 = (_QWORD *)*v88;
            if ( v160 == (_QWORD *)*v88 )
              goto LABEL_132;
            ++v109;
          }
          if ( !v110 )
            v110 = v88;
          v90 = *(_DWORD *)(a1 + 48) + 1;
          v161 = v110;
          v88 = v110;
        }
      }
      else
      {
        v161 = 0;
        v88 = 0;
        v149 = v160;
        v90 = *(_DWORD *)(a1 + 48) + 1;
      }
      goto LABEL_120;
    }
    v24 = *(_QWORD *)(a1 + 40);
    v149 = v160;
    v25 = (v23 - 1) & (((unsigned int)v160 >> 9) ^ ((unsigned int)v160 >> 4));
    v26 = v24 + 16LL * v25;
    v27 = *(_QWORD **)v26;
    if ( v160 == *(_QWORD **)v26 )
    {
LABEL_15:
      v28 = *(_DWORD *)(v26 + 8);
      goto LABEL_16;
    }
    v141 = 1;
    v88 = 0;
    while ( v27 != (_QWORD *)-4096LL )
    {
      if ( v27 == (_QWORD *)-8192LL && !v88 )
        v88 = (_QWORD *)v26;
      v25 = (v23 - 1) & (v141 + v25);
      v26 = v24 + 16LL * v25;
      v27 = *(_QWORD **)v26;
      if ( v160 == *(_QWORD **)v26 )
        goto LABEL_15;
      ++v141;
    }
    v89 = *(_DWORD *)(a1 + 48);
    if ( !v88 )
      v88 = (_QWORD *)v26;
    ++*(_QWORD *)(a1 + 32);
    v90 = v89 + 1;
    v161 = v88;
    if ( 4 * (v89 + 1) >= 3 * v23 )
      goto LABEL_130;
    if ( v23 - *(_DWORD *)(a1 + 52) - v90 <= v23 >> 3 )
    {
      v142 = v18;
      sub_2C96F50(v144, v23);
      sub_2C95990(v144, (__int64 *)&v160, &v161);
      v88 = v161;
      v18 = v142;
      v149 = v160;
      v90 = *(_DWORD *)(a1 + 48) + 1;
    }
LABEL_120:
    *(_DWORD *)(a1 + 48) = v90;
    if ( *v88 != -4096 )
      --*(_DWORD *)(a1 + 52);
    *((_DWORD *)v88 + 2) = 0;
    v28 = 0;
    *v88 = v149;
LABEL_16:
    v29 = *(_QWORD *)(a1 + 192);
    v30 = *(_DWORD *)(v29 + 24);
    v31 = *(_QWORD *)(v29 + 8);
    v32 = *(_QWORD *)(*(_QWORD *)(***(_QWORD ***)(v13 + 8) + 16LL) + 40LL);
    if ( !v30 )
      goto LABEL_96;
    v33 = v30 - 1;
    v34 = v33 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
    v35 = (__int64 *)(v31 + 16LL * v34);
    v36 = *v35;
    if ( v32 != *v35 )
    {
      v77 = 1;
      while ( v36 != -4096 )
      {
        v105 = v77 + 1;
        v34 = v33 & (v77 + v34);
        v35 = (__int64 *)(v31 + 16LL * v34);
        v36 = *v35;
        if ( v32 == *v35 )
          goto LABEL_18;
        v77 = v105;
      }
LABEL_96:
      v37 = 0;
      if ( v154 != v10 )
        goto LABEL_19;
LABEL_97:
      v78 = v154 - (_QWORD)src;
      v79 = (v154 - (__int64)src) >> 5;
      if ( v79 == 0x3FFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"vector::_M_realloc_insert");
      v80 = 1;
      if ( v79 )
        v80 = (v154 - (__int64)src) >> 5;
      v49 = __CFADD__(v79, v80);
      v81 = v79 + v80;
      if ( v49 )
      {
        v83 = 0x7FFFFFFFFFFFFFE0LL;
        goto LABEL_105;
      }
      if ( v81 )
      {
        v82 = 0x3FFFFFFFFFFFFFFLL;
        if ( v81 <= 0x3FFFFFFFFFFFFFFLL )
          v82 = v81;
        v83 = 32 * v82;
LABEL_105:
        v140 = v37;
        v151 = v18;
        v84 = sub_22077B0(v83);
        v18 = v151;
        v37 = v140;
        v78 = v154 - (_QWORD)src;
        v85 = (char *)v84;
        v86 = v84 + v83;
      }
      else
      {
        v86 = 0;
        v85 = 0;
      }
      v87 = &v85[v78];
      if ( &v85[v78] )
      {
        *(_QWORD *)v87 = v18;
        *((_QWORD *)v87 + 1) = v19;
        *((_QWORD *)v87 + 2) = v37;
        *((_DWORD *)v87 + 6) = v28;
      }
      v10 = (__int64)&v85[v78 + 32];
      if ( v78 > 0 )
      {
        v85 = (char *)memmove(v85, src, v78);
      }
      else if ( !src )
      {
LABEL_110:
        v154 = v86;
        src = v85;
        goto LABEL_22;
      }
      v152 = v85;
      j_j___libc_free_0((unsigned __int64)src);
      v85 = v152;
      goto LABEL_110;
    }
LABEL_18:
    v37 = v35[1];
    if ( v154 == v10 )
      goto LABEL_97;
LABEL_19:
    if ( v10 )
    {
      *(_QWORD *)v10 = v18;
      *(_QWORD *)(v10 + 8) = v19;
      *(_QWORD *)(v10 + 16) = v37;
      *(_DWORD *)(v10 + 24) = v28;
    }
    v10 += 32;
LABEL_22:
    v11 = *a2;
  }
  v145 = (a2[1] - v11) >> 3;
  if ( !v145 )
    goto LABEL_69;
  v122 = a1;
  v38 = src;
  v157 = 0;
LABEL_27:
  v39 = v38[6];
  v40 = ++v157;
  if ( v39 <= 0x3E8 )
  {
    v41 = v145;
    if ( v145 <= v157 )
      goto LABEL_69;
    v139 = v38;
    v121 = 0;
    v150 = *(_QWORD *)v38;
    v155 = *((_QWORD *)v38 + 1);
    v147 = *((_QWORD *)v38 + 2);
    n = *(unsigned __int64 **)(v11 + 8 * v157 - 8);
    v42 = v38;
    v43 = *n;
    while ( 1 )
    {
LABEL_30:
      v44 = *(unsigned __int64 **)(v11 + 8 * v40);
      v45 = *v44;
      if ( v43 == *v44 )
        goto LABEL_88;
      if ( v155 != *((_QWORD *)v42 + 5) )
        break;
      if ( *((_QWORD *)v42 + 6) != v147 )
        goto LABEL_88;
      if ( v147 )
        break;
      ++v40;
      v42 += 8;
      if ( v40 >= v41 )
      {
LABEL_89:
        v38 = v139;
LABEL_26:
        v38 += 8;
        goto LABEL_27;
      }
    }
    if ( v150 == *((_QWORD *)v42 + 4) )
    {
      v46 = v42[14];
      if ( (v46 > 3 || v39 > 3) && v46 <= 0x3E8 )
      {
        v47 = v39 - v46;
        v48 = v46 - v39;
        v49 = v46 < v39;
        if ( v46 > v39 )
          v46 = v39;
        if ( v49 )
          v48 = v47;
        v134 = v46;
        if ( v48 <= v46 )
        {
          v50 = (_QWORD **)v44[1];
          v123 = v41;
          v127 = v39;
          v51 = (unsigned int *)n[1];
          v158 = 0;
          v159 = 0;
          v52 = sub_2C9EEF0(v122, v51, v50, &v158, (__int64 *)&v159, a3, 0);
          v39 = v127;
          v41 = v123;
          if ( !v52 )
            goto LABEL_87;
          v53 = v164;
          if ( v164 )
          {
            v54 = &v163;
            do
            {
              while ( v45 <= v53[4] && (v45 != v53[4] || v43 <= v53[5]) )
              {
                v54 = v53;
                v53 = (__int64 *)v53[2];
                if ( !v53 )
                  goto LABEL_48;
              }
              v53 = (__int64 *)v53[3];
            }
            while ( v53 );
LABEL_48:
            if ( v54 != &v163 && v45 >= v54[4] && (v45 != v54[4] || v43 >= v54[5]) )
            {
              v160 = (_QWORD *)v54[6];
              v62 = *((_DWORD *)v54 + 14);
              goto LABEL_74;
            }
          }
          v55 = sub_D95540(v43);
          v56 = v127;
          v57 = v123;
          if ( *(_BYTE *)(v55 + 8) == 14 )
          {
            v160 = sub_DCC810(*(__int64 **)(v122 + 184), v45, v43, 0, 0);
            v106 = sub_D970F0(*(_QWORD *)(v122 + 184));
            v59 = v160;
            v39 = v127;
            v41 = v123;
            if ( v160 != (_QWORD *)v106 )
              goto LABEL_54;
            goto LABEL_87;
          }
          if ( !v121 )
          {
            v107 = sub_DCAF50(*(__int64 **)(v122 + 184), v43, 0);
            v57 = v123;
            v56 = v127;
            v121 = (__int64)v107;
          }
          v124 = v57;
          v128 = v56;
          v58 = sub_DC7ED0(*(__int64 **)(v122 + 184), v45, v121, 0, 0);
          v41 = v124;
          v39 = v128;
          v160 = v58;
          v59 = v58;
LABEL_54:
          v113 = v41;
          v125 = v39;
          v60 = sub_CEFE70(*(_QWORD *)(v122 + 184), (__int64)v59);
          v61 = &v163;
          v39 = v125;
          v62 = v60;
          v41 = v113;
          v129 = v160;
          v63 = v164;
          if ( !v164 )
            goto LABEL_63;
          do
          {
            while ( v45 <= v63[4] && (v45 != v63[4] || v43 <= v63[5]) )
            {
              v61 = v63;
              v63 = (__int64 *)v63[2];
              if ( !v63 )
                goto LABEL_61;
            }
            v63 = (__int64 *)v63[3];
          }
          while ( v63 );
LABEL_61:
          if ( v61 == &v163 || v45 < v61[4] || v45 == v61[4] && v43 < v61[5] )
          {
LABEL_63:
            v111 = v113;
            v112 = v125;
            v117 = v62;
            v114 = v61;
            v64 = sub_22077B0(0x40u);
            *(_QWORD *)(v64 + 32) = v45;
            *(_QWORD *)(v64 + 40) = v43;
            *(_QWORD *)(v64 + 48) = 0;
            *(_DWORD *)(v64 + 56) = 0;
            v126 = v64;
            v65 = sub_2C96C60(&v162, v114, (unsigned __int64 *)(v64 + 32));
            if ( v66 )
            {
              v67 = 1;
              if ( &v163 != v66 && !v65 && v45 >= v66[4] )
              {
                v67 = 0;
                if ( v45 == v66[4] )
                  v67 = v43 < v66[5];
              }
              sub_220F040(v67, v126, v66, &v163);
              ++v167;
              v61 = (__int64 *)v126;
              v62 = v117;
              v39 = v112;
              v41 = v111;
            }
            else
            {
              v115 = v65;
              j_j___libc_free_0(v126);
              v41 = v111;
              v39 = v112;
              v62 = v117;
              v61 = v115;
            }
          }
          *((_DWORD *)v61 + 14) = v62;
          v61[6] = (__int64)v129;
LABEL_74:
          if ( (v133 > a4 || v62 <= 8)
            && v134 > v62
            && (*v159 != 23
             || v159 == *(_BYTE **)(*(_QWORD *)(**(_QWORD **)v44[1] + 16LL) + 40LL)
             || (v131 = v41,
                 v136 = v39,
                 v91 = sub_DAEB40(*(_QWORD *)(v122 + 184), (__int64)v160, (__int64)v159),
                 v39 = v136,
                 v41 = v131,
                 v91)) )
          {
            v130 = v41;
            v135 = v39;
            v68 = (_QWORD *)sub_22077B0(0x10u);
            if ( v68 )
            {
              *v68 = n[1];
              v68[1] = v44[1];
            }
            v161 = v68;
            v69 = *(unsigned int *)(a5 + 24);
            v70 = *(_QWORD *)(a5 + 8);
            if ( (_DWORD)v69 )
            {
              v71 = (v69 - 1) & (((unsigned int)v160 >> 9) ^ ((unsigned int)v160 >> 4));
              v72 = (_QWORD *)(v70 + 16LL * v71);
              v73 = (_QWORD *)*v72;
              if ( v160 == (_QWORD *)*v72 )
              {
LABEL_82:
                if ( v72 != (_QWORD *)(v70 + 16 * v69) )
                {
                  v74 = sub_2C93820(a5, (__int64 *)&v160);
                  v39 = v135;
                  v41 = v130;
                  v75 = *v74;
                  v76 = *(_BYTE **)(*v74 + 8);
                  if ( v76 == *(_BYTE **)(*v74 + 16) )
                  {
                    sub_2C908A0(v75, v76, &v161);
                    v41 = v130;
                    v39 = v135;
                  }
                  else
                  {
                    if ( v76 )
                    {
                      *(_QWORD *)v76 = v161;
                      v76 = *(_BYTE **)(v75 + 8);
                    }
                    *(_QWORD *)(v75 + 8) = v76 + 8;
                  }
                  goto LABEL_87;
                }
              }
              else
              {
                v98 = 1;
                while ( v73 != (_QWORD *)-4096LL )
                {
                  v108 = v98 + 1;
                  v71 = (v69 - 1) & (v98 + v71);
                  v72 = (_QWORD *)(v70 + 16LL * v71);
                  v73 = (_QWORD *)*v72;
                  if ( v160 == (_QWORD *)*v72 )
                    goto LABEL_82;
                  v98 = v108;
                }
              }
            }
            v99 = (_QWORD *)sub_22077B0(0x18u);
            if ( v99 )
            {
              *v99 = 0;
              v99[1] = 0;
              v99[2] = 0;
            }
            *sub_2C93820(a5, (__int64 *)&v160) = (__int64)v99;
            v100 = sub_2C93820(a5, (__int64 *)&v160);
            v101 = v135;
            v102 = v130;
            v103 = *v100;
            v104 = *(_BYTE **)(*v100 + 8);
            if ( v104 == *(_BYTE **)(*v100 + 16) )
            {
              sub_2C908A0(v103, v104, &v161);
              v102 = v130;
              v101 = v135;
            }
            else
            {
              if ( v104 )
              {
                *(_QWORD *)v104 = v161;
                v104 = *(_BYTE **)(v103 + 8);
              }
              *(_QWORD *)(v103 + 8) = v104 + 8;
            }
            v132 = v102;
            v137 = v101;
            sub_2C95CA0(a6, &v160);
            v39 = v137;
            v41 = v132;
            v11 = *a2;
          }
          else
          {
LABEL_87:
            v11 = *a2;
          }
        }
      }
    }
LABEL_88:
    ++v40;
    v42 += 8;
    if ( v40 >= v41 )
      goto LABEL_89;
    goto LABEL_30;
  }
  if ( v145 > v157 )
    goto LABEL_26;
LABEL_69:
  if ( src )
    j_j___libc_free_0((unsigned __int64)src);
LABEL_71:
  sub_2C91580((unsigned __int64)v164);
}
