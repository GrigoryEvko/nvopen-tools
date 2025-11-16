// Function: sub_301D730
// Address: 0x301d730
//
__int64 __fastcall sub_301D730(__int64 a1, __int64 a2)
{
  __int64 v4; // r14
  unsigned int v5; // eax
  unsigned int v6; // r15d
  __int64 v7; // rsi
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // r9
  __int64 v11; // r8
  __int64 *v12; // rdi
  _QWORD *v13; // r12
  __int64 v14; // rax
  __int64 (*v15)(); // rdx
  __int64 (*v16)(); // rax
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  _BOOL4 v24; // eax
  unsigned __int8 **v25; // rbx
  _QWORD *v26; // rdi
  unsigned __int64 v27; // rdx
  _QWORD *v28; // rax
  __int64 v29; // r15
  unsigned __int8 v30; // al
  __int64 v31; // rax
  __int64 v32; // r8
  __int64 v33; // rcx
  int v34; // eax
  __int64 v35; // rax
  _QWORD *v36; // r14
  unsigned __int64 v37; // rbx
  __int64 v38; // rax
  bool v39; // cf
  char (__fastcall *v40)(__int64, __int64); // rax
  __int64 v41; // r13
  int v42; // eax
  __int64 v43; // rax
  unsigned __int8 *v44; // rsi
  bool v45; // zf
  _QWORD *v46; // r13
  _QWORD *v47; // rax
  __int64 v48; // r9
  __int64 v49; // rdx
  __int64 v50; // r9
  int v51; // eax
  int v52; // eax
  char v53; // al
  _QWORD *v54; // rax
  __int64 v55; // r9
  __int64 v56; // rax
  unsigned __int64 v57; // rdx
  __int64 v58; // rax
  unsigned __int64 v59; // rcx
  unsigned __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // rsi
  __int64 v63; // rax
  __int64 v64; // r8
  unsigned __int64 v65; // rcx
  unsigned __int64 v66; // rdx
  __int64 v67; // rax
  unsigned __int8 **v68; // r9
  __int64 v69; // rbx
  __int64 *v70; // r12
  __int64 *v71; // r15
  __int64 v72; // rdi
  __int64 v73; // rax
  __int64 v74; // rax
  unsigned int v75; // eax
  __int64 v76; // rdx
  char v77; // al
  unsigned __int64 v78; // rdi
  unsigned __int64 v79; // rdi
  __int64 *v80; // r15
  __int64 *v81; // r12
  __int64 v82; // rsi
  __int64 v83; // rdi
  __int64 *v84; // rax
  __int64 *v85; // r12
  __int64 *v86; // r15
  __int64 v87; // rdi
  unsigned int v88; // ecx
  __int64 v89; // rax
  __int64 *v90; // r15
  __int64 *v91; // r12
  __int64 v92; // rsi
  __int64 v93; // rdi
  _BYTE *v94; // rax
  _BYTE *v95; // r12
  unsigned __int64 v96; // r15
  unsigned __int64 v97; // rdi
  int v98; // eax
  unsigned __int64 v99; // r12
  __int64 v100; // r14
  __int64 *v101; // rbx
  __int64 *v102; // r13
  __int64 v103; // rdi
  __int64 v104; // rax
  __int64 v105; // rax
  unsigned int v106; // eax
  __int64 v107; // rdx
  char v108; // al
  unsigned __int64 v109; // rdi
  unsigned __int64 v110; // rdi
  __int64 *v111; // rbx
  __int64 *v112; // r12
  __int64 v113; // rsi
  __int64 v114; // rdi
  __int64 *v115; // rax
  __int64 *v116; // r13
  __int64 *v117; // r12
  __int64 v118; // rdi
  unsigned int v119; // ecx
  __int64 v120; // rax
  __int64 *v121; // rdi
  __int64 *v122; // r12
  __int64 *v123; // rbx
  __int64 v124; // rsi
  __int64 v125; // rdi
  _BYTE *v126; // r12
  _BYTE *v127; // rbx
  unsigned __int64 v128; // r13
  unsigned __int64 v129; // rdi
  unsigned __int64 v130; // [rsp+0h] [rbp-1A0h]
  unsigned __int64 v131; // [rsp+10h] [rbp-190h]
  unsigned __int64 v132; // [rsp+10h] [rbp-190h]
  unsigned __int8 v133; // [rsp+18h] [rbp-188h]
  __int64 v134; // [rsp+20h] [rbp-180h]
  unsigned int v135; // [rsp+20h] [rbp-180h]
  _QWORD *v136; // [rsp+28h] [rbp-178h]
  unsigned __int64 v137; // [rsp+28h] [rbp-178h]
  unsigned __int64 v138; // [rsp+28h] [rbp-178h]
  const char *v139; // [rsp+28h] [rbp-178h]
  unsigned __int64 v140; // [rsp+28h] [rbp-178h]
  unsigned __int64 v141; // [rsp+28h] [rbp-178h]
  __int64 v142; // [rsp+30h] [rbp-170h]
  __int64 v143; // [rsp+30h] [rbp-170h]
  __int64 v144; // [rsp+30h] [rbp-170h]
  __int64 v145; // [rsp+30h] [rbp-170h]
  _QWORD *v146; // [rsp+30h] [rbp-170h]
  unsigned __int64 v147; // [rsp+30h] [rbp-170h]
  _QWORD *v148; // [rsp+30h] [rbp-170h]
  unsigned __int64 v149; // [rsp+30h] [rbp-170h]
  unsigned __int8 **v150; // [rsp+30h] [rbp-170h]
  unsigned __int64 v151; // [rsp+30h] [rbp-170h]
  _BYTE *v152; // [rsp+30h] [rbp-170h]
  _BYTE *v153; // [rsp+30h] [rbp-170h]
  _QWORD *v154; // [rsp+38h] [rbp-168h]
  const char *v155; // [rsp+38h] [rbp-168h]
  __int64 v156; // [rsp+40h] [rbp-160h] BYREF
  unsigned __int8 *v157; // [rsp+48h] [rbp-158h] BYREF
  unsigned __int8 *v158; // [rsp+50h] [rbp-150h] BYREF
  __int64 v159; // [rsp+58h] [rbp-148h]
  char v160; // [rsp+60h] [rbp-140h] BYREF
  _BYTE *v161; // [rsp+68h] [rbp-138h]
  __int64 v162; // [rsp+70h] [rbp-130h]
  _BYTE v163[56]; // [rsp+78h] [rbp-128h] BYREF
  __int64 v164; // [rsp+B0h] [rbp-F0h]
  __int64 v165; // [rsp+B8h] [rbp-E8h]
  char v166; // [rsp+C0h] [rbp-E0h]
  __int64 v167; // [rsp+C4h] [rbp-DCh]
  unsigned __int8 *v168; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v169; // [rsp+D8h] [rbp-C8h]
  __int64 v170; // [rsp+E0h] [rbp-C0h]
  __int64 v171; // [rsp+E8h] [rbp-B8h]
  const char *v172; // [rsp+F0h] [rbp-B0h]
  const char *v173; // [rsp+F8h] [rbp-A8h]
  __int64 v174; // [rsp+100h] [rbp-A0h]
  __int64 v175; // [rsp+108h] [rbp-98h]
  __int64 i; // [rsp+110h] [rbp-90h]
  __int64 *v177; // [rsp+118h] [rbp-88h]
  __int64 v178; // [rsp+120h] [rbp-80h]
  _BYTE v179[32]; // [rsp+128h] [rbp-78h] BYREF
  __int64 *v180; // [rsp+148h] [rbp-58h]
  __int64 v181; // [rsp+150h] [rbp-50h]
  _QWORD v182[9]; // [rsp+158h] [rbp-48h] BYREF

  v4 = *(_QWORD *)a2;
  v156 = sub_B2D7E0(*(_QWORD *)a2, "function-instrument", 0x13u);
  LOBYTE(v5) = sub_A71840((__int64)&v156);
  v6 = v5;
  if ( (_BYTE)v5 )
  {
    v6 = 0;
    v22 = sub_A72240(&v156);
    if ( v23 == 11 )
    {
      v24 = *(_QWORD *)v22 != 0x776C612D79617278LL || *(_WORD *)(v22 + 8) != 31073 || *(_BYTE *)(v22 + 10) != 115;
      LOBYTE(v6) = !v24;
    }
  }
  if ( !sub_A71840((__int64)&v156)
    || (v20 = sub_A72240(&v156), v21 != 10)
    || *(_QWORD *)v20 != 0x76656E2D79617278LL
    || *(_WORD *)(v20 + 8) != 29285 )
  {
    v7 = (__int64)"xray-ignore-loops";
    v8 = sub_B2D7E0(v4, "xray-ignore-loops", 0x11u);
    v10 = v8;
    if ( (_BYTE)v6 )
    {
      v154 = (_QWORD *)(a2 + 320);
LABEL_5:
      v11 = *(_QWORD *)(a2 + 328);
      goto LABEL_6;
    }
    v7 = (__int64)"xray-instruction-threshold";
    v25 = (unsigned __int8 **)v8;
    v142 = v8;
    v9 = sub_B2D810(v4, "xray-instruction-threshold", 0x1Au, -1);
    if ( v9 == -1 )
      return 0;
    v11 = *(_QWORD *)(a2 + 328);
    v10 = v142;
    v154 = (_QWORD *)(a2 + 320);
    if ( v11 == a2 + 320 )
    {
      v27 = 0;
    }
    else
    {
      v26 = *(_QWORD **)(a2 + 328);
      v27 = 0;
      do
      {
        v28 = (_QWORD *)v26[7];
        if ( v28 != v26 + 6 )
        {
          LODWORD(v7) = 0;
          do
          {
            v28 = (_QWORD *)v28[1];
            LODWORD(v7) = v7 + 1;
          }
          while ( v28 != v26 + 6 );
          v7 = (unsigned int)v7;
          v27 += (unsigned int)v7;
        }
        v26 = (_QWORD *)v26[1];
      }
      while ( v26 != v154 );
    }
    if ( v142 )
    {
      if ( v9 > v27 )
        return 0;
      goto LABEL_6;
    }
    v137 = v27;
    v149 = v9;
    v58 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_501FE44);
    v59 = v149;
    v60 = v137;
    if ( v58
      && (v61 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v58 + 104LL))(v58, &unk_501FE44),
          v59 = v149,
          v60 = v137,
          v61) )
    {
      v166 = 0;
      v150 = (unsigned __int8 **)(v61 + 200);
      v158 = (unsigned __int8 *)&v160;
      v159 = 0x100000000LL;
      v161 = v163;
      v162 = 0x600000000LL;
      v164 = 0;
      v165 = 0;
      v167 = 0;
    }
    else
    {
      v158 = (unsigned __int8 *)&v160;
      v159 = 0x100000000LL;
      v161 = v163;
      v162 = 0x600000000LL;
      v98 = *(_DWORD *)(a2 + 120);
      v132 = v60;
      v140 = v59;
      v164 = 0;
      v166 = 0;
      LODWORD(v167) = 0;
      v165 = a2;
      HIDWORD(v167) = v98;
      v150 = &v158;
      sub_2E708A0((__int64)&v158);
      v59 = v140;
      v60 = v132;
    }
    v131 = v60;
    v138 = v59;
    v62 = (__int64)&unk_50208AC;
    v63 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_50208AC);
    v65 = v138;
    v66 = v131;
    if ( v63 )
    {
      v67 = (*(__int64 (__fastcall **)(__int64, void *, unsigned __int64, unsigned __int64))(*(_QWORD *)v63 + 104LL))(
              v63,
              &unk_50208AC,
              v131,
              v138);
      v68 = 0;
      v66 = v131;
      v62 = v67 + 200;
      v65 = v138;
      if ( v67 )
        v68 = (unsigned __int8 **)(v67 + 200);
      v25 = v68;
    }
    v168 = 0;
    v177 = (__int64 *)v179;
    v178 = 0x400000000LL;
    v169 = 0;
    v170 = 0;
    LODWORD(v171) = 0;
    v172 = 0;
    v173 = 0;
    v174 = 0;
    v175 = 0;
    i = 0;
    v180 = v182;
    v181 = 0;
    v182[0] = 0;
    v182[1] = 1;
    if ( !v25 )
    {
      v62 = (__int64)v150;
      v130 = v66;
      v141 = v65;
      sub_2EA84A0((__int64)&v168, (__int64)v150, v66, v65, v64, (__int64)&v168);
      v66 = v130;
      v65 = v141;
      v25 = &v168;
    }
    if ( v25[5] != v25[4] || v65 <= v66 )
    {
      sub_301D560((__int64)&v168);
      v139 = v173;
      v151 = (unsigned __int64)v172;
      if ( v172 != v173 )
      {
        do
        {
          v69 = *(_QWORD *)v151;
          v70 = *(__int64 **)(*(_QWORD *)v151 + 16LL);
          if ( *(__int64 **)(*(_QWORD *)v151 + 8LL) == v70 )
          {
            *(_BYTE *)(v69 + 152) = 1;
          }
          else
          {
            v71 = *(__int64 **)(*(_QWORD *)v151 + 8LL);
            do
            {
              v72 = *v71++;
              sub_2EA4EF0(v72, v62);
            }
            while ( v70 != v71 );
            *(_BYTE *)(v69 + 152) = 1;
            v73 = *(_QWORD *)(v69 + 8);
            if ( v73 != *(_QWORD *)(v69 + 16) )
              *(_QWORD *)(v69 + 16) = v73;
          }
          v74 = *(_QWORD *)(v69 + 32);
          if ( v74 != *(_QWORD *)(v69 + 40) )
            *(_QWORD *)(v69 + 40) = v74;
          ++*(_QWORD *)(v69 + 56);
          if ( *(_BYTE *)(v69 + 84) )
          {
            *(_QWORD *)v69 = 0;
          }
          else
          {
            v75 = 4 * (*(_DWORD *)(v69 + 76) - *(_DWORD *)(v69 + 80));
            v76 = *(unsigned int *)(v69 + 72);
            if ( v75 < 0x20 )
              v75 = 32;
            if ( (unsigned int)v76 > v75 )
            {
              sub_C8C990(v69 + 56, v62);
            }
            else
            {
              v62 = 0xFFFFFFFFLL;
              memset(*(void **)(v69 + 64), -1, 8 * v76);
            }
            v77 = *(_BYTE *)(v69 + 84);
            *(_QWORD *)v69 = 0;
            if ( !v77 )
              _libc_free(*(_QWORD *)(v69 + 64));
          }
          v78 = *(_QWORD *)(v69 + 32);
          if ( v78 )
          {
            v62 = *(_QWORD *)(v69 + 48) - v78;
            j_j___libc_free_0(v78);
          }
          v79 = *(_QWORD *)(v69 + 8);
          if ( v79 )
          {
            v62 = *(_QWORD *)(v69 + 24) - v79;
            j_j___libc_free_0(v79);
          }
          v151 += 8LL;
        }
        while ( v139 != (const char *)v151 );
        if ( v172 != v173 )
          v173 = v172;
      }
      v80 = v180;
      v81 = &v180[2 * (unsigned int)v181];
      if ( v180 != v81 )
      {
        do
        {
          v82 = v80[1];
          v83 = *v80;
          v80 += 2;
          sub_C7D6A0(v83, v82, 16);
        }
        while ( v81 != v80 );
      }
      LODWORD(v181) = 0;
      if ( (_DWORD)v178 )
      {
        v84 = v177;
        v182[0] = 0;
        v85 = &v177[(unsigned int)v178];
        v86 = v177 + 1;
        v175 = *v177;
        for ( i = v175 + 4096; v85 != v86; v84 = v177 )
        {
          v87 = *v86;
          v88 = (unsigned int)(v86 - v84) >> 7;
          v89 = 4096LL << v88;
          if ( v88 >= 0x1E )
            v89 = 0x40000000000LL;
          ++v86;
          sub_C7D6A0(v87, v89, 16);
        }
        LODWORD(v178) = 1;
        sub_C7D6A0(*v84, 4096, 16);
        v90 = v180;
        v91 = &v180[2 * (unsigned int)v181];
        if ( v180 == v91 )
          goto LABEL_153;
        do
        {
          v92 = v90[1];
          v93 = *v90;
          v90 += 2;
          sub_C7D6A0(v93, v92, 16);
        }
        while ( v91 != v90 );
      }
      v91 = v180;
LABEL_153:
      if ( v91 != v182 )
        _libc_free((unsigned __int64)v91);
      if ( v177 != (__int64 *)v179 )
        _libc_free((unsigned __int64)v177);
      if ( v172 )
        j_j___libc_free_0((unsigned __int64)v172);
      v7 = 16LL * (unsigned int)v171;
      sub_C7D6A0(v169, v7, 8);
      v94 = v161;
      v95 = &v161[8 * (unsigned int)v162];
      if ( v161 != v95 )
      {
        do
        {
          v96 = *((_QWORD *)v95 - 1);
          v95 -= 8;
          if ( v96 )
          {
            v97 = *(_QWORD *)(v96 + 24);
            if ( v97 != v96 + 40 )
            {
              v152 = v94;
              _libc_free(v97);
              v94 = v152;
            }
            v7 = 80;
            v153 = v94;
            j_j___libc_free_0(v96);
            v94 = v153;
          }
        }
        while ( v94 != v95 );
        v95 = v161;
      }
      if ( v95 != v163 )
        _libc_free((unsigned __int64)v95);
      if ( v158 != (unsigned __int8 *)&v160 )
        _libc_free((unsigned __int64)v158);
      goto LABEL_5;
    }
    sub_301D560((__int64)&v168);
    v99 = (unsigned __int64)v172;
    v155 = v173;
    if ( v172 != v173 )
    {
      do
      {
        v100 = *(_QWORD *)v99;
        v101 = *(__int64 **)(*(_QWORD *)v99 + 16LL);
        if ( *(__int64 **)(*(_QWORD *)v99 + 8LL) == v101 )
        {
          *(_BYTE *)(v100 + 152) = 1;
        }
        else
        {
          v102 = *(__int64 **)(*(_QWORD *)v99 + 8LL);
          do
          {
            v103 = *v102++;
            sub_2EA4EF0(v103, v62);
          }
          while ( v101 != v102 );
          *(_BYTE *)(v100 + 152) = 1;
          v104 = *(_QWORD *)(v100 + 8);
          if ( *(_QWORD *)(v100 + 16) != v104 )
            *(_QWORD *)(v100 + 16) = v104;
        }
        v105 = *(_QWORD *)(v100 + 32);
        if ( v105 != *(_QWORD *)(v100 + 40) )
          *(_QWORD *)(v100 + 40) = v105;
        ++*(_QWORD *)(v100 + 56);
        if ( *(_BYTE *)(v100 + 84) )
        {
          *(_QWORD *)v100 = 0;
        }
        else
        {
          v106 = 4 * (*(_DWORD *)(v100 + 76) - *(_DWORD *)(v100 + 80));
          v107 = *(unsigned int *)(v100 + 72);
          if ( v106 < 0x20 )
            v106 = 32;
          if ( (unsigned int)v107 > v106 )
          {
            sub_C8C990(v100 + 56, v62);
          }
          else
          {
            v62 = 0xFFFFFFFFLL;
            memset(*(void **)(v100 + 64), -1, 8 * v107);
          }
          v108 = *(_BYTE *)(v100 + 84);
          *(_QWORD *)v100 = 0;
          if ( !v108 )
            _libc_free(*(_QWORD *)(v100 + 64));
        }
        v109 = *(_QWORD *)(v100 + 32);
        if ( v109 )
        {
          v62 = *(_QWORD *)(v100 + 48) - v109;
          j_j___libc_free_0(v109);
        }
        v110 = *(_QWORD *)(v100 + 8);
        if ( v110 )
        {
          v62 = *(_QWORD *)(v100 + 24) - v110;
          j_j___libc_free_0(v110);
        }
        v99 += 8LL;
      }
      while ( v155 != (const char *)v99 );
      if ( v172 != v173 )
        v173 = v172;
    }
    v111 = v180;
    v112 = &v180[2 * (unsigned int)v181];
    if ( v180 != v112 )
    {
      do
      {
        v113 = v111[1];
        v114 = *v111;
        v111 += 2;
        sub_C7D6A0(v114, v113, 16);
      }
      while ( v112 != v111 );
    }
    LODWORD(v181) = 0;
    if ( (_DWORD)v178 )
    {
      v115 = v177;
      v182[0] = 0;
      v116 = &v177[(unsigned int)v178];
      v117 = v177 + 1;
      v175 = *v177;
      i = v175 + 4096;
      if ( v116 != v177 + 1 )
      {
        do
        {
          v118 = *v117;
          v119 = (unsigned int)(v117 - v177) >> 7;
          v120 = 4096LL << v119;
          if ( v119 >= 0x1E )
            v120 = 0x40000000000LL;
          ++v117;
          sub_C7D6A0(v118, v120, 16);
        }
        while ( v116 != v117 );
        v115 = v177;
      }
      LODWORD(v178) = 1;
      sub_C7D6A0(*v115, 4096, 16);
      v121 = v180;
      v122 = &v180[2 * (unsigned int)v181];
      if ( v180 == v122 )
        goto LABEL_209;
      v123 = v180;
      do
      {
        v124 = v123[1];
        v125 = *v123;
        v123 += 2;
        sub_C7D6A0(v125, v124, 16);
      }
      while ( v122 != v123 );
    }
    v121 = v180;
LABEL_209:
    if ( v121 != v182 )
      _libc_free((unsigned __int64)v121);
    if ( v177 != (__int64 *)v179 )
      _libc_free((unsigned __int64)v177);
    if ( v172 )
      j_j___libc_free_0((unsigned __int64)v172);
    sub_C7D6A0(v169, 16LL * (unsigned int)v171, 8);
    v126 = v161;
    v127 = &v161[8 * (unsigned int)v162];
    if ( v161 != v127 )
    {
      do
      {
        v128 = *((_QWORD *)v127 - 1);
        v127 -= 8;
        if ( v128 )
        {
          v129 = *(_QWORD *)(v128 + 24);
          if ( v129 != v128 + 40 )
            _libc_free(v129);
          j_j___libc_free_0(v128);
        }
      }
      while ( v126 != v127 );
      v126 = v161;
    }
    if ( v126 != v163 )
      _libc_free((unsigned __int64)v126);
    if ( v158 != (unsigned __int8 *)&v160 )
      _libc_free((unsigned __int64)v158);
    return v6;
  }
  if ( !(_BYTE)v6 )
    return 0;
  v7 = (__int64)"xray-ignore-loops";
  sub_B2D7E0(v4, "xray-ignore-loops", 0x11u);
  v11 = *(_QWORD *)(a2 + 328);
  v154 = (_QWORD *)(a2 + 320);
LABEL_6:
  if ( v154 == (_QWORD *)v11 )
    return 0;
  while ( v11 + 48 == (*(_QWORD *)(v11 + 48) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v11 = *(_QWORD *)(v11 + 8);
    if ( (_QWORD *)v11 == v154 )
      return 0;
  }
  v12 = *(__int64 **)(a2 + 16);
  v13 = 0;
  v14 = *v12;
  v15 = *(__int64 (**)())(*v12 + 128);
  if ( v15 != sub_2DAC790 )
  {
    v144 = v11;
    v35 = ((__int64 (__fastcall *)(__int64 *, __int64, __int64 (*)(), unsigned __int64, __int64, __int64))v15)(
            v12,
            v7,
            v15,
            v9,
            v11,
            v10);
    v12 = *(__int64 **)(a2 + 16);
    v11 = v144;
    v13 = (_QWORD *)v35;
    v14 = *v12;
  }
  v16 = *(__int64 (**)())(v14 + 120);
  if ( v16 == sub_301C9A0
    || (v143 = v11,
        v29 = *(_QWORD *)(v11 + 56),
        v30 = ((__int64 (__fastcall *)(__int64 *, __int64, __int64 (*)(), unsigned __int64, __int64, __int64))v16)(
                v12,
                v7,
                v15,
                v9,
                v11,
                v10),
        v11 = v143,
        (v133 = v30) == 0) )
  {
    v17 = **(_QWORD **)(v11 + 32);
    v18 = sub_B2BE50(v17);
    v169 = 24;
    v170 = v17;
    v171 = 0;
    v172 = 0;
    v168 = (unsigned __int8 *)&unk_49D9E88;
    v173 = "An attempt to perform XRay instrumentation for an unsupported target.";
    LOWORD(v177) = 259;
    sub_B6EB20(v18, (__int64)&v168);
    return 0;
  }
  if ( !(unsigned __int8)sub_B2D620(v4, "xray-skip-entry", 0xFu) )
  {
    v31 = v13[1];
    v32 = v143;
    v158 = *(unsigned __int8 **)(v29 + 56);
    v33 = v31 - 1440;
    if ( v158 )
    {
      v134 = v31 - 1440;
      sub_B96E90((__int64)&v158, (__int64)v158, 1);
      v32 = v143;
      v33 = v134;
      v168 = v158;
      if ( v158 )
      {
        sub_B976B0((__int64)&v158, v158, (__int64)&v168);
        v33 = v134;
        v158 = 0;
        v32 = v143;
      }
    }
    else
    {
      v168 = 0;
    }
    v169 = 0;
    v170 = 0;
    sub_301D240(v32, v29, (__int64)&v168, v33);
    if ( v168 )
      sub_B91220((__int64)&v168, (__int64)v168);
    if ( v158 )
      sub_B91220((__int64)&v158, (__int64)v158);
  }
  v6 = sub_B2D620(v4, "xray-skip-exit", 0xEu);
  if ( !(_BYTE)v6 )
  {
    v34 = *(_DWORD *)(*(_QWORD *)(a2 + 8) + 544LL);
    switch ( v34 )
    {
      case 1:
      case 3:
      case 12:
      case 14:
      case 16:
      case 17:
      case 18:
      case 19:
      case 28:
      case 29:
      case 36:
        v36 = *(_QWORD **)(a2 + 328);
        v135 = v34 - 28;
        if ( v154 == v36 )
          return v133;
        break;
      case 25:
      case 33:
        sub_301CC90(a2, (__int64)v13, 0x100u);
        return v133;
      default:
        sub_301CC90(a2, (__int64)v13, 1u);
        return v133;
    }
    while ( 1 )
    {
      v136 = v36 + 6;
      v37 = sub_2E313E0((__int64)v36);
      if ( v36 + 6 != (_QWORD *)v37 )
        break;
LABEL_83:
      v36 = (_QWORD *)v36[1];
      if ( v154 == v36 )
        return v133;
    }
    while ( 1 )
    {
      v51 = *(_DWORD *)(v37 + 44);
      if ( (v51 & 4) != 0 || (v51 & 8) == 0 )
        v38 = (*(_QWORD *)(*(_QWORD *)(v37 + 16) + 24LL) >> 5) & 1LL;
      else
        LOBYTE(v38) = sub_2E88A90(v37, 32, 1);
      v39 = (_BYTE)v38 == 0;
      v40 = *(char (__fastcall **)(__int64, __int64))(*v13 + 1328LL);
      v41 = v39 ? 0 : 0x26;
      if ( v40 != sub_2FDE950 )
        break;
      v42 = *(_DWORD *)(v37 + 44);
      if ( (v42 & 4) != 0 || (v42 & 8) == 0 )
      {
        if ( (*(_QWORD *)(*(_QWORD *)(v37 + 16) + 24LL) & 0x20LL) == 0 )
          goto LABEL_59;
      }
      else if ( !sub_2E88A90(v37, 32, 1) )
      {
        goto LABEL_59;
      }
      v52 = *(_DWORD *)(v37 + 44);
      if ( (v52 & 4) != 0 || (v52 & 8) == 0 )
        v53 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v37 + 16) + 24LL) >> 7;
      else
        v53 = sub_2E88A90(v37, 128, 1);
      if ( v53 )
        goto LABEL_92;
LABEL_59:
      if ( (_DWORD)v41 )
      {
        v43 = -40 * v41;
        goto LABEL_61;
      }
LABEL_76:
      if ( (*(_BYTE *)v37 & 4) != 0 )
      {
        v37 = *(_QWORD *)(v37 + 8);
        if ( v136 == (_QWORD *)v37 )
          goto LABEL_83;
      }
      else
      {
        while ( (*(_BYTE *)(v37 + 44) & 8) != 0 )
          v37 = *(_QWORD *)(v37 + 8);
        v37 = *(_QWORD *)(v37 + 8);
        if ( v136 == (_QWORD *)v37 )
          goto LABEL_83;
      }
    }
    if ( !v40((__int64)v13, v37) )
      goto LABEL_59;
LABEL_92:
    if ( v135 <= 1 )
    {
      v43 = -1560;
LABEL_61:
      v44 = *(unsigned __int8 **)(v37 + 56);
      v145 = v13[1] + v43;
      v157 = v44;
      if ( v44 )
      {
        sub_B96E90((__int64)&v157, (__int64)v44, 1);
        v168 = v157;
        if ( v157 )
        {
          sub_B976B0((__int64)&v157, v157, (__int64)&v168);
          v169 = 0;
          v170 = 0;
          v45 = (*(_BYTE *)(v37 + 44) & 4) == 0;
          v157 = 0;
          v46 = (_QWORD *)v36[4];
          v158 = v168;
          if ( !v45 )
          {
            if ( v168 )
              sub_B96E90((__int64)&v158, (__int64)v168, 1);
LABEL_66:
            v47 = sub_2E7B380(v46, v145, &v158, 0);
            v48 = (__int64)v47;
            if ( v158 )
            {
              v146 = v47;
              sub_B91220((__int64)&v158, (__int64)v158);
              v48 = (__int64)v146;
            }
            v147 = v48;
            sub_2E326B0((__int64)v36, (__int64 *)v37, v48);
            v49 = v169;
            v50 = v147;
            if ( v169 )
            {
LABEL_69:
              sub_2E882B0(v50, (__int64)v46, v49);
              v50 = v147;
            }
LABEL_70:
            if ( v170 )
              sub_2E88680(v50, (__int64)v46, v170);
            if ( v168 )
              sub_B91220((__int64)&v168, (__int64)v168);
            if ( v157 )
              sub_B91220((__int64)&v157, (__int64)v157);
            goto LABEL_76;
          }
          if ( v168 )
            sub_B96E90((__int64)&v158, (__int64)v168, 1);
LABEL_102:
          v54 = sub_2E7B380(v46, v145, &v158, 0);
          v55 = (__int64)v54;
          if ( v158 )
          {
            v148 = v54;
            sub_B91220((__int64)&v158, (__int64)v158);
            v55 = (__int64)v148;
          }
          v147 = v55;
          sub_2E31040(v36 + 5, v55);
          v50 = v147;
          v56 = *(_QWORD *)v147;
          v57 = *(_QWORD *)v37 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v147 + 8) = v37;
          *(_QWORD *)v147 = v57 | v56 & 7;
          *(_QWORD *)(v57 + 8) = v147;
          *(_QWORD *)v37 = v147 | *(_QWORD *)v37 & 7LL;
          v49 = v169;
          if ( v169 )
            goto LABEL_69;
          goto LABEL_70;
        }
      }
      else
      {
        v168 = 0;
      }
      v169 = 0;
      v170 = 0;
      v46 = (_QWORD *)v36[4];
      if ( (*(_BYTE *)(v37 + 44) & 4) != 0 )
      {
        v158 = 0;
        goto LABEL_66;
      }
      v158 = 0;
      goto LABEL_102;
    }
    goto LABEL_59;
  }
  return v6;
}
