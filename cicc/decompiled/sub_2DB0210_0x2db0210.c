// Function: sub_2DB0210
// Address: 0x2db0210
//
__int64 __fastcall sub_2DB0210(
        int a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 *a4,
        __int64 **a5,
        __int64 a6,
        _DWORD *a7)
{
  unsigned __int64 *v8; // r12
  __int64 v10; // rbx
  __int64 v11; // rsi
  __int64 j; // rdx
  __int64 v13; // rcx
  unsigned __int64 v14; // r8
  __int64 v15; // r9
  __int64 i; // r15
  __int64 v17; // r8
  unsigned __int64 v18; // rax
  __int64 v19; // r9
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // rax
  unsigned int v23; // r14d
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  _QWORD *v28; // rbx
  _QWORD *v29; // r15
  void (__fastcall *v30)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v31; // rax
  __int64 v33; // rax
  int v34; // eax
  unsigned int v35; // ebx
  int v36; // eax
  unsigned int v37; // ecx
  const char *v38; // r13
  __int64 *v39; // rax
  unsigned __int64 v40; // r15
  size_t v41; // rdx
  __int64 v42; // r14
  _BYTE *v43; // rdx
  __int64 v44; // rbx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // r14
  unsigned __int16 v48; // bx
  __int64 v49; // r13
  __int64 v50; // rax
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // r15
  unsigned __int64 v54; // rbx
  int v55; // eax
  unsigned int v56; // edx
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rdx
  _QWORD *v60; // r13
  void *v61; // r12
  _QWORD *v62; // rdi
  const __m128i *v63; // rsi
  _QWORD *v64; // r13
  int v65; // eax
  unsigned int v66; // r13d
  __int64 *v67; // rax
  __int64 v68; // r13
  __int64 *v69; // r15
  __int64 *v70; // rbx
  __int64 v71; // r14
  __int64 v72; // rax
  _BYTE *v73; // rdi
  _QWORD *v74; // rax
  __int64 v75; // rax
  __int64 v76; // rbx
  unsigned __int64 v77; // r13
  unsigned __int64 v78; // r14
  __int64 v79; // rax
  _QWORD *v80; // r12
  __int64 v81; // r15
  _QWORD *v82; // rdi
  __int64 v83; // r9
  __int64 v84; // rbx
  unsigned __int16 v85; // r15
  int v86; // r14d
  _QWORD *v87; // rax
  __int64 v88; // r13
  int v89; // ecx
  __int64 v90; // r14
  __int64 v91; // rax
  __int64 v92; // rcx
  __int64 v93; // rax
  __int64 *v94; // rdi
  _QWORD *v95; // rax
  __int64 v96; // rsi
  unsigned __int8 *v97; // rsi
  __int64 *v98; // rax
  _QWORD *v99; // rdi
  __int64 *v100; // rax
  unsigned int v101; // ebx
  __int64 v102; // rax
  __int64 v103; // rax
  unsigned __int64 v104; // rdx
  __int64 v105; // r12
  _QWORD *v106; // rax
  __int64 v107; // rbx
  __int64 *v108; // r14
  int v109; // r13d
  _QWORD *v110; // rax
  __int64 v111; // r15
  unsigned int v112; // ecx
  __int64 v113; // r13
  __int64 v114; // rax
  __int64 v115; // rcx
  __int64 v116; // rax
  __int64 *v117; // rdi
  _QWORD *v118; // rax
  __int64 v119; // rsi
  unsigned __int8 *v120; // rsi
  __int64 *v121; // rax
  _QWORD *v122; // rdi
  __int64 v123; // rax
  __int64 *v124; // rax
  unsigned __int64 *v125; // [rsp+8h] [rbp-558h]
  int v126; // [rsp+24h] [rbp-53Ch]
  _BYTE *v127; // [rsp+30h] [rbp-530h]
  bool s; // [rsp+40h] [rbp-520h]
  __int64 v129; // [rsp+48h] [rbp-518h]
  __int64 *v130; // [rsp+48h] [rbp-518h]
  __int64 v131; // [rsp+48h] [rbp-518h]
  __int64 *v132; // [rsp+50h] [rbp-510h]
  unsigned int v133; // [rsp+58h] [rbp-508h]
  __int64 v134; // [rsp+58h] [rbp-508h]
  __int64 *v137; // [rsp+68h] [rbp-4F8h]
  int v138; // [rsp+70h] [rbp-4F0h]
  __int64 v139; // [rsp+70h] [rbp-4F0h]
  __int64 v140; // [rsp+70h] [rbp-4F0h]
  void *v142; // [rsp+88h] [rbp-4D8h]
  __int64 v143; // [rsp+88h] [rbp-4D8h]
  const __m128i *v144; // [rsp+88h] [rbp-4D8h]
  unsigned __int64 v146; // [rsp+A8h] [rbp-4B8h]
  unsigned __int16 v147; // [rsp+A8h] [rbp-4B8h]
  __int64 *v148; // [rsp+A8h] [rbp-4B8h]
  __int64 v149; // [rsp+A8h] [rbp-4B8h]
  __int64 *v150; // [rsp+A8h] [rbp-4B8h]
  unsigned __int16 v151; // [rsp+A8h] [rbp-4B8h]
  __int64 v152; // [rsp+B0h] [rbp-4B0h] BYREF
  unsigned __int16 v153; // [rsp+B8h] [rbp-4A8h]
  __m128i *v154; // [rsp+C0h] [rbp-4A0h] BYREF
  __m128i *v155; // [rsp+C8h] [rbp-498h]
  const __m128i *v156; // [rsp+D0h] [rbp-490h]
  unsigned __int64 v157; // [rsp+E0h] [rbp-480h] BYREF
  __int64 v158; // [rsp+E8h] [rbp-478h]
  _QWORD v159[2]; // [rsp+F0h] [rbp-470h] BYREF
  void *v160[2]; // [rsp+100h] [rbp-460h] BYREF
  _BYTE v161[16]; // [rsp+110h] [rbp-450h] BYREF
  __int16 v162; // [rsp+120h] [rbp-440h]
  unsigned int v163; // [rsp+140h] [rbp-420h]
  __int64 *v164; // [rsp+150h] [rbp-410h] BYREF
  __int64 v165; // [rsp+158h] [rbp-408h]
  _BYTE v166[128]; // [rsp+160h] [rbp-400h] BYREF
  __int64 *v167; // [rsp+1E0h] [rbp-380h] BYREF
  __int64 v168; // [rsp+1E8h] [rbp-378h]
  _BYTE v169[128]; // [rsp+1F0h] [rbp-370h] BYREF
  unsigned __int64 v170[2]; // [rsp+270h] [rbp-2F0h] BYREF
  _BYTE v171[512]; // [rsp+280h] [rbp-2E0h] BYREF
  __int64 v172; // [rsp+480h] [rbp-E0h]
  __int64 v173; // [rsp+488h] [rbp-D8h]
  unsigned __int64 *v174; // [rsp+490h] [rbp-D0h]
  __int64 v175; // [rsp+498h] [rbp-C8h]
  char v176; // [rsp+4A0h] [rbp-C0h]
  __int64 v177; // [rsp+4A8h] [rbp-B8h]
  char *v178; // [rsp+4B0h] [rbp-B0h]
  __int64 v179; // [rsp+4B8h] [rbp-A8h]
  int v180; // [rsp+4C0h] [rbp-A0h]
  char v181; // [rsp+4C4h] [rbp-9Ch]
  char v182; // [rsp+4C8h] [rbp-98h] BYREF
  __int16 v183; // [rsp+508h] [rbp-58h]
  _QWORD *v184; // [rsp+510h] [rbp-50h]
  _QWORD *v185; // [rsp+518h] [rbp-48h]
  __int64 v186; // [rsp+520h] [rbp-40h]

  v8 = a4;
  v178 = &v182;
  if ( a4 )
    v8 = v170;
  v10 = a2 + 72;
  v170[0] = (unsigned __int64)v171;
  v164 = (__int64 *)v166;
  v170[1] = 0x1000000000LL;
  v174 = a4;
  v183 = 0;
  v165 = 0x1000000000LL;
  v172 = 0;
  v173 = 0;
  v175 = 0;
  v176 = 1;
  v177 = 0;
  v179 = 8;
  v180 = 0;
  v181 = 1;
  v184 = 0;
  v185 = 0;
  v186 = 0;
  v167 = (__int64 *)v169;
  v11 = 41;
  v168 = 0x1000000000LL;
  sub_B2D610(a2, 41);
  for ( i = *(_QWORD *)(v10 + 8); v10 != i; i = *(_QWORD *)(i + 8) )
  {
    while ( 1 )
    {
      if ( !i )
        BUG();
      v17 = i - 24;
      v18 = *(_QWORD *)(i + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v18 == i + 24 )
        goto LABEL_175;
      if ( !v18 )
        BUG();
      v19 = v18 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v18 - 24) - 30 > 0xA )
LABEL_175:
        BUG();
      if ( *(_BYTE *)(v18 - 24) == 35 )
      {
        v20 = (unsigned int)v165;
        v21 = (unsigned int)v165 + 1LL;
        if ( v21 > HIDWORD(v165) )
        {
          v11 = (__int64)v166;
          v140 = v19;
          sub_C8D5F0((__int64)&v164, v166, v21, 8u, v17, v19);
          v20 = (unsigned int)v165;
          v19 = v140;
          v17 = i - 24;
        }
        v164[v20] = v19;
        LODWORD(v165) = v165 + 1;
      }
      v22 = sub_AA5EB0(v17);
      if ( v22 )
      {
        if ( (*(_BYTE *)(v22 + 2) & 1) != 0 )
          break;
      }
      i = *(_QWORD *)(i + 8);
      if ( v10 == i )
        goto LABEL_19;
    }
    j = (unsigned int)v168;
    v14 = (unsigned int)v168 + 1LL;
    if ( v14 > HIDWORD(v168) )
    {
      v11 = (__int64)v169;
      v149 = v22;
      sub_C8D5F0((__int64)&v167, v169, (unsigned int)v168 + 1LL, 8u, v14, v15);
      j = (unsigned int)v168;
      v22 = v149;
    }
    v13 = (__int64)v167;
    v167[j] = v22;
    LODWORD(v168) = v168 + 1;
  }
LABEL_19:
  v23 = 0;
  if ( (_DWORD)v165 )
  {
    v33 = sub_B2E500(a2);
    v34 = sub_B2A630(v33);
    v138 = v34;
    if ( v34 > 10 )
    {
      if ( v34 == 12 )
        goto LABEL_20;
    }
    else if ( v34 > 6 )
    {
      goto LABEL_20;
    }
    v132 = (__int64 *)sub_B2BE50(a2);
    v146 = (unsigned int)v165;
    v35 = v165;
    if ( !a1 )
      goto LABEL_42;
    v66 = (unsigned int)(v165 + 63) >> 6;
    v160[0] = v161;
    v160[1] = (void *)0x600000000LL;
    if ( v66 > 6 )
    {
      sub_C8D5F0((__int64)v160, v161, v66, 8u, v14, v15);
      v11 = 0;
      memset(v160[0], 0, 8LL * v66);
      LODWORD(v160[1]) = v66;
      v146 = (unsigned int)v165;
    }
    else
    {
      if ( v66 )
      {
        j = 8LL * v66;
        if ( j )
        {
          v11 = 0;
          memset(v161, 0, j);
        }
      }
      LODWORD(v160[1]) = v66;
    }
    v163 = v35;
    v133 = 0;
    v67 = &v164[v146];
    v148 = v164;
    v130 = v67;
    if ( v164 != v67 )
    {
      do
      {
        v15 = (__int64)v167;
        v68 = *v148;
        v69 = v167;
        v70 = &v167[(unsigned int)v168];
        if ( v167 != v70 )
        {
          while ( 1 )
          {
            v71 = *v69;
            v72 = sub_FFD350((__int64)v8, v11, j, v13, v14, v15);
            v11 = v68;
            if ( (unsigned __int8)sub_D0EBA0(v71, v68, 0, v72, 0) )
              break;
            if ( v70 == ++v69 )
              goto LABEL_86;
          }
          j = (__int64)v160[0];
          v13 = v133;
          v11 = 1LL << v133;
          *((_QWORD *)v160[0] + (v133 >> 6)) |= 1LL << v133;
        }
LABEL_86:
        ++v148;
        ++v133;
      }
      while ( v130 != v148 );
      v35 = v163;
    }
    v73 = v160[0];
    v13 = v35 >> 6;
    if ( v35 >> 6 )
    {
      v74 = v160[0];
      j = (__int64)v160[0] + 8 * (unsigned int)(v13 - 1) + 8;
      while ( *v74 == -1 )
      {
        if ( (_QWORD *)j == ++v74 )
          goto LABEL_129;
      }
    }
    else
    {
LABEL_129:
      v101 = v35 & 0x3F;
      if ( !v101 || (v102 = (unsigned int)v13, v13 = v101, j = (1LL << v101) - 1, *((_QWORD *)v160[0] + v102) == j) )
      {
        v146 = (unsigned int)v165;
        goto LABEL_132;
      }
    }
    v75 = sub_B2BE50(a2);
    v76 = (unsigned int)v165;
    v131 = v75;
    if ( (_DWORD)v165 )
    {
      v134 = (__int64)v8;
      v77 = 0;
      v78 = 0;
      do
      {
        v13 = (__int64)v160[0];
        j = (__int64)v164;
        v79 = *((_QWORD *)v160[0] + ((unsigned int)v78 >> 6));
        v80 = (_QWORD *)v164[v78];
        if ( _bittest64(&v79, v78) )
        {
          v164[v77++] = (__int64)v80;
        }
        else
        {
          v81 = v80[5];
          v82 = sub_BD2C40(72, unk_3F148B8);
          if ( v82 )
            sub_B4C8A0((__int64)v82, v131, (__int64)(v80 + 3), 0);
          sub_B43D60(v80);
          v11 = (__int64)a5;
          v157 = 0x100000000000001LL;
          v158 = 0x1000101000000LL;
          v159[0] = 0;
          sub_FC3C00(v81, a5, v134, a6, (__int64)&v157, v83, 0, 0);
        }
        ++v78;
      }
      while ( v76 != v78 );
      v123 = (unsigned int)v165;
      v146 = v77;
      v8 = (unsigned __int64 *)v134;
      if ( v77 != (unsigned int)v165 )
      {
        if ( v77 >= (unsigned int)v165 )
        {
          if ( v77 > HIDWORD(v165) )
          {
            sub_C8D5F0((__int64)&v164, v166, v77, 8u, v14, v15);
            v123 = (unsigned int)v165;
          }
          v11 = v77;
          v124 = &v164[v123];
          for ( j = (__int64)&v164[v77]; (__int64 *)j != v124; ++v124 )
          {
            if ( v124 )
              *v124 = 0;
          }
          LODWORD(v165) = v77;
          v73 = v160[0];
          goto LABEL_132;
        }
        LODWORD(v165) = v77;
      }
      v73 = v160[0];
    }
    else
    {
      v146 = 0;
      v73 = v160[0];
    }
LABEL_132:
    if ( v73 != v161 )
      _libc_free((unsigned __int64)v73);
LABEL_42:
    v23 = 1;
    if ( v146 )
    {
      if ( (unsigned int)(v138 - 4) <= 1
        && ((v36 = a7[8], (unsigned int)(v36 - 36) <= 1) || (unsigned int)(v36 - 1) <= 1)
        && (v37 = a7[12], v37 <= 0x31)
        && (s = ((0x20000006381E0uLL >> v37) & 1) == 0, ((0x20000006381E0uLL >> v37) & 1) != 0)
        && a7[13] == 3 )
      {
        v38 = *(const char **)(a3 + 529200);
        v100 = (__int64 *)sub_BCB120(v132);
        v40 = sub_BCF640(v100, 0);
        v126 = *(_DWORD *)(a3 + 533084);
      }
      else
      {
        v38 = *(const char **)(a3 + 529192);
        v160[0] = (void *)sub_BCE3C0(v132, 0);
        v39 = (__int64 *)sub_BCB120(v132);
        s = 1;
        v40 = sub_BCF480(v39, v160, 1, 0);
        v126 = *(_DWORD *)(a3 + 533080);
      }
      v41 = 0;
      v42 = *(_QWORD *)(a2 + 40);
      if ( v38 )
        v41 = strlen(v38);
      v129 = sub_BA8CA0(v42, (__int64)v38, v41, v40);
      v127 = v43;
      if ( v146 == 1 )
      {
        v105 = *(_QWORD *)(*v164 + 40);
        v106 = sub_2DAFBB0((_QWORD *)*v164);
        v157 = (unsigned __int64)v159;
        v158 = 0x100000000LL;
        if ( s )
        {
          v159[0] = v106;
          LODWORD(v158) = 1;
        }
        sub_B43C20((__int64)&v154, v105);
        v162 = 257;
        v107 = (unsigned int)v158;
        v108 = (__int64 *)v157;
        v109 = v158 + 1;
        v144 = v154;
        v151 = (unsigned __int16)v155;
        v110 = sub_BD2C40(88, (int)v158 + 1);
        v111 = (__int64)v110;
        if ( v110 )
        {
          v112 = v109 & 0x7FFFFFF;
          v113 = (__int64)v110;
          sub_B44260((__int64)v110, **(_QWORD **)(v129 + 16), 56, v112, (__int64)v144, v151);
          *(_QWORD *)(v111 + 72) = 0;
          sub_B4A290(v111, v129, (__int64)v127, v108, v107, (__int64)v160, 0, 0);
        }
        else
        {
          v113 = 0;
        }
        if ( !*v127 )
        {
          if ( sub_B92180((__int64)v127) )
          {
            v114 = sub_B92180(a2);
            v115 = v114;
            if ( v114 )
            {
              v116 = *(_QWORD *)(v114 + 8);
              v117 = (__int64 *)(v116 & 0xFFFFFFFFFFFFFFF8LL);
              if ( (v116 & 4) != 0 )
                v117 = (__int64 *)*v117;
              v118 = sub_B01860(v117, 0, 0, v115, 0, 0, 0, 1);
              sub_B10CB0(v160, (__int64)v118);
              if ( (void **)(v111 + 48) == v160 )
              {
                if ( v160[0] )
                  sub_B91220((__int64)v160, (__int64)v160[0]);
              }
              else
              {
                v119 = *(_QWORD *)(v111 + 48);
                if ( v119 )
                  sub_B91220(v111 + 48, v119);
                v120 = (unsigned __int8 *)v160[0];
                *(void **)(v111 + 48) = v160[0];
                if ( v120 )
                  sub_B976B0((__int64)v160, v120, v111 + 48);
              }
            }
          }
        }
        *(_WORD *)(v111 + 2) = (4 * v126) | *(_WORD *)(v111 + 2) & 0xF003;
        v121 = (__int64 *)sub_BD5C60(v113);
        *(_QWORD *)(v111 + 72) = sub_A7A090((__int64 *)(v111 + 72), v121, -1, 36);
        sub_B43C20((__int64)v160, v105);
        v11 = unk_3F148B8;
        v122 = sub_BD2C40(72, unk_3F148B8);
        if ( v122 )
        {
          v11 = (__int64)v132;
          sub_B4C8A0((__int64)v122, (__int64)v132, (__int64)v160[0], (unsigned __int16)v160[1]);
        }
        if ( (_QWORD *)v157 != v159 )
          _libc_free(v157);
      }
      else
      {
        v154 = 0;
        v155 = 0;
        v156 = 0;
        if ( (_DWORD)v165 )
        {
          v44 = (unsigned int)v165;
          v154 = (__m128i *)sub_22077B0(v44 * 16);
          v155 = v154;
          v156 = &v154[v44];
        }
        v157 = (unsigned __int64)v159;
        v158 = 0x100000000LL;
        v160[0] = "unwind_resume";
        v162 = 259;
        v45 = sub_22077B0(0x50u);
        v139 = v45;
        if ( v45 )
          sub_AA4D50(v45, (__int64)v132, (__int64)v160, a2, 0);
        sub_B43C20((__int64)&v152, v139);
        v160[0] = "exn.obj";
        v162 = 259;
        v46 = sub_BCE3C0(v132, 0);
        v47 = v152;
        v48 = v153;
        v49 = v46;
        v50 = sub_BD2DA0(80);
        v53 = v50;
        if ( v50 )
        {
          sub_B44260(v50, v49, 55, 0x8000000u, v47, v48);
          *(_DWORD *)(v53 + 72) = v146;
          sub_BD6B50((unsigned __int8 *)v53, (const char **)v160);
          sub_BD2A10(v53, *(_DWORD *)(v53 + 72), 1);
        }
        v54 = (unsigned __int64)v164;
        v137 = &v164[(unsigned int)v165];
        if ( v164 != v137 )
        {
          v125 = v8;
          do
          {
            v60 = *(_QWORD **)v54;
            v61 = *(void **)(*(_QWORD *)v54 + 40LL);
            sub_B43C20((__int64)v160, (__int64)v61);
            v142 = v160[0];
            v147 = (unsigned __int16)v160[1];
            v62 = sub_BD2C40(72, 1u);
            if ( v62 )
              sub_B4C8F0((__int64)v62, v139, 1u, (__int64)v142, v147);
            v160[0] = v61;
            v63 = v155;
            v160[1] = (void *)(v139 & 0xFFFFFFFFFFFFFFFBLL);
            if ( v155 == v156 )
            {
              sub_F38BA0((const __m128i **)&v154, v155, (const __m128i *)v160);
            }
            else
            {
              if ( v155 )
              {
                *v155 = _mm_loadu_si128((const __m128i *)v160);
                v63 = v155;
              }
              v155 = (__m128i *)&v63[1];
            }
            v64 = sub_2DAFBB0(v60);
            v65 = *(_DWORD *)(v53 + 4) & 0x7FFFFFF;
            if ( v65 == *(_DWORD *)(v53 + 72) )
            {
              sub_B48D90(v53);
              v65 = *(_DWORD *)(v53 + 4) & 0x7FFFFFF;
            }
            v55 = (v65 + 1) & 0x7FFFFFF;
            v56 = v55 | *(_DWORD *)(v53 + 4) & 0xF8000000;
            v57 = *(_QWORD *)(v53 - 8) + 32LL * (unsigned int)(v55 - 1);
            *(_DWORD *)(v53 + 4) = v56;
            if ( *(_QWORD *)v57 )
            {
              v58 = *(_QWORD *)(v57 + 8);
              **(_QWORD **)(v57 + 16) = v58;
              if ( v58 )
                *(_QWORD *)(v58 + 16) = *(_QWORD *)(v57 + 16);
            }
            *(_QWORD *)v57 = v64;
            if ( v64 )
            {
              v59 = v64[2];
              *(_QWORD *)(v57 + 8) = v59;
              if ( v59 )
                *(_QWORD *)(v59 + 16) = v57 + 8;
              *(_QWORD *)(v57 + 16) = v64 + 2;
              v64[2] = v57;
            }
            v54 += 8LL;
            *(_QWORD *)(*(_QWORD *)(v53 - 8)
                      + 32LL * *(unsigned int *)(v53 + 72)
                      + 8LL * ((*(_DWORD *)(v53 + 4) & 0x7FFFFFFu) - 1)) = v61;
          }
          while ( v137 != (__int64 *)v54 );
          v8 = v125;
        }
        if ( s )
        {
          v103 = (unsigned int)v158;
          v104 = (unsigned int)v158 + 1LL;
          if ( v104 > HIDWORD(v158) )
          {
            sub_C8D5F0((__int64)&v157, v159, v104, 8u, v51, v52);
            v103 = (unsigned int)v158;
          }
          *(_QWORD *)(v157 + 8 * v103) = v53;
          LODWORD(v158) = v158 + 1;
        }
        sub_B43C20((__int64)&v152, v139);
        v84 = (unsigned int)v158;
        v162 = 257;
        v143 = v152;
        v85 = v153;
        v86 = v158 + 1;
        v150 = (__int64 *)v157;
        v87 = sub_BD2C40(88, (int)v158 + 1);
        v88 = (__int64)v87;
        if ( v87 )
        {
          v89 = v86;
          v90 = (__int64)v87;
          sub_B44260((__int64)v87, **(_QWORD **)(v129 + 16), 56, v89 & 0x7FFFFFF, v143, v85);
          *(_QWORD *)(v88 + 72) = 0;
          sub_B4A290(v88, v129, (__int64)v127, v150, v84, (__int64)v160, 0, 0);
        }
        else
        {
          v90 = 0;
        }
        if ( !*v127 )
        {
          if ( sub_B92180((__int64)v127) )
          {
            v91 = sub_B92180(a2);
            v92 = v91;
            if ( v91 )
            {
              v93 = *(_QWORD *)(v91 + 8);
              v94 = (__int64 *)(v93 & 0xFFFFFFFFFFFFFFF8LL);
              if ( (v93 & 4) != 0 )
                v94 = (__int64 *)*v94;
              v95 = sub_B01860(v94, 0, 0, v92, 0, 0, 0, 1);
              sub_B10CB0(v160, (__int64)v95);
              if ( (void **)(v88 + 48) == v160 )
              {
                if ( v160[0] )
                  sub_B91220((__int64)v160, (__int64)v160[0]);
              }
              else
              {
                v96 = *(_QWORD *)(v88 + 48);
                if ( v96 )
                  sub_B91220(v88 + 48, v96);
                v97 = (unsigned __int8 *)v160[0];
                *(void **)(v88 + 48) = v160[0];
                if ( v97 )
                  sub_B976B0((__int64)v160, v97, v88 + 48);
              }
            }
          }
        }
        *(_WORD *)(v88 + 2) = (4 * v126) | *(_WORD *)(v88 + 2) & 0xF003;
        v98 = (__int64 *)sub_BD5C60(v90);
        *(_QWORD *)(v88 + 72) = sub_A7A090((__int64 *)(v88 + 72), v98, -1, 36);
        sub_B43C20((__int64)v160, v139);
        v11 = unk_3F148B8;
        v99 = sub_BD2C40(72, unk_3F148B8);
        if ( v99 )
        {
          v11 = (__int64)v132;
          sub_B4C8A0((__int64)v99, (__int64)v132, (__int64)v160[0], (unsigned __int16)v160[1]);
        }
        if ( v8 )
        {
          v11 = (__int64)v154;
          sub_FFB3D0((__int64)v8, (unsigned __int64 *)v154, v155 - v154, v13, v14, v15);
        }
        if ( (_QWORD *)v157 != v159 )
          _libc_free(v157);
        if ( v154 )
        {
          v11 = (char *)v156 - (char *)v154;
          j_j___libc_free_0((unsigned __int64)v154);
        }
      }
      v23 = 1;
    }
  }
LABEL_20:
  if ( v167 != (__int64 *)v169 )
    _libc_free((unsigned __int64)v167);
  if ( v164 != (__int64 *)v166 )
    _libc_free((unsigned __int64)v164);
  sub_FFCE90((__int64)v170, v11, j, v13, v14, v15);
  sub_FFD870((__int64)v170, v11, v24, v25, v26, v27);
  sub_FFBC40((__int64)v170, v11);
  v28 = v185;
  v29 = v184;
  if ( v185 != v184 )
  {
    do
    {
      v30 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v29[7];
      *v29 = &unk_49E5048;
      if ( v30 )
        v30(v29 + 5, v29 + 5, 3);
      *v29 = &unk_49DB368;
      v31 = v29[3];
      if ( v31 != 0 && v31 != -4096 && v31 != -8192 )
        sub_BD60C0(v29 + 1);
      v29 += 9;
    }
    while ( v28 != v29 );
    v29 = v184;
  }
  if ( v29 )
    j_j___libc_free_0((unsigned __int64)v29);
  if ( !v181 )
    _libc_free((unsigned __int64)v178);
  if ( (_BYTE *)v170[0] != v171 )
    _libc_free(v170[0]);
  return v23;
}
