// Function: sub_34C80D0
// Address: 0x34c80d0
//
float __fastcall sub_34C80D0(
        _QWORD *a1,
        __int64 a2,
        __int64 *a3,
        __int64 *a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12)
{
  __int64 v12; // r13
  _QWORD *v13; // r12
  __int64 v14; // rax
  _QWORD *v15; // r14
  __int64 v16; // r9
  __int64 (*v17)(void); // rax
  __int64 v18; // r8
  unsigned int v19; // eax
  unsigned int *v20; // rdx
  double v21; // xmm4_8
  double v22; // xmm5_8
  void **v23; // rax
  __int64 p_base; // rdx
  __int64 v25; // rax
  _QWORD *v26; // rbx
  __int64 v27; // r14
  __int64 v28; // r15
  __int64 v29; // rcx
  __int64 v30; // rsi
  unsigned __int64 v31; // rax
  __int64 i; // rdi
  int v33; // edx
  __int64 v34; // rdi
  __int64 *v35; // rdx
  __int64 v36; // rdx
  __int16 v37; // ax
  char *v38; // rax
  __int64 v39; // rcx
  __int64 v40; // rdi
  __int64 (*v41)(); // rax
  double v42; // xmm4_8
  double v43; // xmm5_8
  __int64 (*v44)(); // rax
  float (__fastcall *v45)(float, __int64, int); // rbx
  __int64 v46; // rsi
  float result; // xmm0_4
  __int64 v48; // rax
  __int64 *v49; // rcx
  __int64 v50; // rax
  unsigned int *v51; // r15
  __int64 *v52; // rdx
  void **v53; // rbx
  __int64 v54; // rax
  unsigned int *v55; // r9
  size_t v56; // rax
  __int64 v57; // rsi
  unsigned int *v58; // r13
  __int64 v59; // r9
  __int64 v60; // rdx
  __int64 v61; // rcx
  _DWORD *v62; // rcx
  int *v63; // rdi
  __int64 v64; // rdx
  int *v65; // r15
  _QWORD *v66; // r12
  __int64 v67; // rdx
  __int64 v68; // rbx
  __int64 v69; // rdx
  unsigned __int64 v70; // rcx
  int v71; // ebx
  unsigned int v72; // esi
  int v73; // edx
  _DWORD *v74; // rax
  int v75; // eax
  __int64 v76; // rax
  __int64 (*v77)(); // rax
  __int64 v78; // rdi
  __int64 v79; // rax
  int v80; // ecx
  __int64 v81; // rsi
  int v82; // ecx
  unsigned int v83; // edx
  __int64 *v84; // rax
  __int64 v85; // r8
  __int64 v86; // rcx
  __int64 v87; // rdx
  __int64 v88; // rax
  __int64 v89; // r14
  __int64 *v90; // rbx
  __int64 v91; // r12
  __int64 *v92; // r13
  __int64 v93; // rsi
  _QWORD *v94; // rax
  _QWORD *v95; // r9
  __int16 v96; // ax
  __int16 v97; // ax
  __int64 v98; // rcx
  unsigned __int64 v99; // rdx
  __int64 *v100; // rax
  __int64 v101; // rcx
  __int64 (__fastcall *v102)(__int64); // rax
  int v103; // eax
  unsigned int v104; // r15d
  unsigned int v105; // esi
  int v106; // edx
  unsigned int *v107; // rax
  __int64 v108; // rcx
  unsigned int v109; // edx
  __int64 v110; // r15
  __int64 v111; // rbx
  unsigned __int64 v112; // rdx
  unsigned int v113; // eax
  __int64 v114; // rcx
  __int64 *v115; // rbx
  __int64 v116; // rax
  bool v117; // zf
  __int64 v118; // rcx
  __int16 v119; // ax
  unsigned __int64 v120; // rdi
  __int64 v121; // rax
  __int64 v122; // r15
  _QWORD *v123; // rcx
  __int64 v124; // rdi
  unsigned __int64 v125; // rax
  __int64 v126; // rax
  __int64 v127; // r14
  __int64 v128; // rdi
  unsigned __int64 v129; // r10
  int v130; // eax
  int v131; // edi
  unsigned int v132; // edi
  unsigned int v133; // edx
  unsigned int v134; // r8d
  __int64 v135; // rcx
  _QWORD *v136; // r14
  int v137; // r13d
  __int64 v138; // r12
  __int64 v139; // rbx
  unsigned __int64 v140; // rdi
  __int64 v141; // rax
  _QWORD *v142; // rsi
  __int64 v143; // rax
  unsigned int v144; // ecx
  __int64 *v145; // rdi
  int v146; // r11d
  unsigned int *v147; // rdi
  int v148; // esi
  unsigned int v149; // edx
  int v150; // r11d
  unsigned int *v151; // rdi
  int v152; // esi
  unsigned int v153; // edx
  int v154; // r11d
  __int64 v155; // r9
  unsigned __int64 v156; // rax
  __int64 *v157; // rsi
  __int64 *v158; // rdi
  int v159; // [rsp+Ch] [rbp-1D4h]
  _QWORD *v160; // [rsp+18h] [rbp-1C8h]
  __int64 v161; // [rsp+20h] [rbp-1C0h]
  __int64 v162; // [rsp+28h] [rbp-1B8h]
  _QWORD *v163; // [rsp+28h] [rbp-1B8h]
  __int64 v164; // [rsp+28h] [rbp-1B8h]
  char v165; // [rsp+28h] [rbp-1B8h]
  _QWORD *v166; // [rsp+30h] [rbp-1B0h]
  __int64 v167; // [rsp+30h] [rbp-1B0h]
  _QWORD *v168; // [rsp+38h] [rbp-1A8h]
  __int64 v169; // [rsp+38h] [rbp-1A8h]
  char v170; // [rsp+40h] [rbp-1A0h]
  __int64 v171; // [rsp+40h] [rbp-1A0h]
  _QWORD *v172; // [rsp+40h] [rbp-1A0h]
  char v173; // [rsp+40h] [rbp-1A0h]
  __int64 v174; // [rsp+40h] [rbp-1A0h]
  int v175; // [rsp+48h] [rbp-198h]
  unsigned __int64 v176; // [rsp+48h] [rbp-198h]
  unsigned int v177; // [rsp+50h] [rbp-190h]
  int *v178; // [rsp+50h] [rbp-190h]
  __int64 v179; // [rsp+58h] [rbp-188h]
  __int64 v180; // [rsp+58h] [rbp-188h]
  float v181; // [rsp+58h] [rbp-188h]
  float v183; // [rsp+68h] [rbp-178h]
  unsigned int v185; // [rsp+78h] [rbp-168h]
  unsigned __int64 v186; // [rsp+78h] [rbp-168h]
  bool v187; // [rsp+80h] [rbp-160h]
  __int64 v188; // [rsp+80h] [rbp-160h]
  float v189; // [rsp+88h] [rbp-158h]
  __int64 v190; // [rsp+88h] [rbp-158h]
  unsigned int v191; // [rsp+88h] [rbp-158h]
  int v192; // [rsp+88h] [rbp-158h]
  _QWORD v193[2]; // [rsp+90h] [rbp-150h] BYREF
  char v194; // [rsp+A0h] [rbp-140h]
  __int64 v195; // [rsp+B0h] [rbp-130h] BYREF
  __int64 v196; // [rsp+B8h] [rbp-128h]
  __int64 *v197; // [rsp+C0h] [rbp-120h] BYREF
  unsigned int v198; // [rsp+C8h] [rbp-118h]
  void *base; // [rsp+100h] [rbp-E0h] BYREF
  __int64 v200; // [rsp+108h] [rbp-D8h]
  _BYTE v201[64]; // [rsp+110h] [rbp-D0h] BYREF
  __int64 v202; // [rsp+150h] [rbp-90h] BYREF
  char *v203; // [rsp+158h] [rbp-88h]
  __int64 v204; // [rsp+160h] [rbp-80h]
  int v205; // [rsp+168h] [rbp-78h]
  char v206; // [rsp+16Ch] [rbp-74h]
  char v207; // [rsp+170h] [rbp-70h] BYREF

  v12 = a2;
  v13 = a1;
  v14 = a1[1];
  v15 = *(_QWORD **)(v14 + 32);
  v179 = 0;
  v160 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v14 + 16) + 200LL))(*(_QWORD *)(v14 + 16));
  v17 = *(__int64 (**)(void))(**(_QWORD **)(a1[1] + 16LL) + 128LL);
  if ( v17 != sub_2DAC790 )
    v179 = v17();
  v18 = *(unsigned int *)(a2 + 112);
  v206 = 1;
  v203 = &v207;
  v204 = 8;
  v202 = 0;
  v19 = v18 & 0x7FFFFFFF;
  v205 = 0;
  if ( ((unsigned int)v18 & 0x7FFFFFFF) >= *((_DWORD *)v15 + 62) )
  {
    v177 = 0;
    v159 = 0;
  }
  else
  {
    v20 = (unsigned int *)(v15[30] + 40LL * v19);
    v177 = v20[4];
    if ( v177 )
      v177 = **((_DWORD **)v20 + 1);
    a2 = *v20;
    v159 = *v20;
  }
  *(_QWORD *)&v21 = *(unsigned int *)(v12 + 116);
  *(_QWORD *)&v22 = dword_44D0BE0;
  v183 = *(float *)(v12 + 116);
  if ( INFINITY != *(float *)&v21 )
  {
    v108 = v19;
    v109 = *(_DWORD *)(*(_QWORD *)(a1[3] + 80LL) + 4LL * v19);
    if ( v109 )
    {
      v18 = v109;
      v19 = v109 & 0x7FFFFFFF;
      v108 = v109 & 0x7FFFFFFF;
    }
    v110 = a1[2];
    v111 = 8 * v108;
    v112 = *(unsigned int *)(v110 + 160);
    if ( v19 < (unsigned int)v112 )
    {
      a2 = *(_QWORD *)(v110 + 152);
      v118 = *(_QWORD *)(a2 + 8 * v108);
      if ( v118 )
      {
        *(_QWORD *)&a6 = dword_44D0BE0;
        v117 = INFINITY == *(float *)(v118 + 116);
LABEL_162:
        if ( v117 )
        {
          *(_QWORD *)&a11 = dword_44D0BE0;
          *(_DWORD *)(v12 + 116) = dword_44D0BE0;
          v183 = *(float *)&dword_44D0BE0;
        }
        else
        {
          *(_QWORD *)&a11 = *(unsigned int *)(v12 + 116);
          v183 = *(float *)(v12 + 116);
        }
        goto LABEL_8;
      }
    }
    v113 = v19 + 1;
    if ( (unsigned int)v112 < v113 )
    {
      v129 = v113;
      if ( v113 != v112 )
      {
        if ( v113 >= v112 )
        {
          v155 = *(_QWORD *)(v110 + 168);
          v156 = v113 - v112;
          if ( v129 > *(unsigned int *)(v110 + 164) )
          {
            v186 = v156;
            v188 = *(_QWORD *)(v110 + 168);
            v192 = v18;
            sub_C8D5F0(v110 + 152, (const void *)(v110 + 168), v129, 8u, v18, v155);
            v112 = *(unsigned int *)(v110 + 160);
            v156 = v186;
            v155 = v188;
            LODWORD(v18) = v192;
          }
          v114 = *(_QWORD *)(v110 + 152);
          v157 = (__int64 *)(v114 + 8 * v112);
          v158 = &v157[v156];
          if ( v157 != v158 )
          {
            do
              *v157++ = v155;
            while ( v158 != v157 );
            LODWORD(v112) = *(_DWORD *)(v110 + 160);
            v114 = *(_QWORD *)(v110 + 152);
          }
          *(_DWORD *)(v110 + 160) = v112 + v156;
          goto LABEL_161;
        }
        *(_DWORD *)(v110 + 160) = v113;
      }
    }
    v114 = *(_QWORD *)(v110 + 152);
LABEL_161:
    v115 = (__int64 *)(v114 + v111);
    v116 = sub_2E10F30(v18);
    *v115 = v116;
    a2 = v116;
    v190 = v116;
    sub_2E11E80((_QWORD *)v110, v116);
    *(_QWORD *)&a6 = dword_44D0BE0;
    v117 = INFINITY == *(float *)(v190 + 116);
    goto LABEL_162;
  }
LABEL_8:
  v187 = a4 != 0 && a3 != 0;
  if ( v187 )
  {
    v120 = *a4 & 0xFFFFFFFFFFFFFFF8LL;
    v121 = *(_QWORD *)(v120 + 16);
    if ( v121 )
    {
      v122 = *(_QWORD *)(v121 + 24);
    }
    else
    {
      v141 = *(_QWORD *)(v13[2] + 32LL);
      v142 = *(_QWORD **)(v141 + 296);
      v143 = *(unsigned int *)(v141 + 304);
      if ( v143 )
      {
        v144 = *(_DWORD *)(v120 + 24) | (*a4 >> 1) & 3;
        do
        {
          v145 = &v142[2 * (v143 >> 1)];
          if ( v144 >= (*(_DWORD *)((*v145 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v145 >> 1) & 3) )
          {
            v142 = v145 + 2;
            v143 = v143 - (v143 >> 1) - 1;
          }
          else
          {
            v143 >>= 1;
          }
        }
        while ( v143 > 0 );
      }
      v122 = *(v142 - 1);
    }
    a12 = 0.0;
    a2 = 1;
    *(float *)&v191 = sub_2E13860(1u, 0, v13[6], v122, v13[5]) + 0.0;
    *(float *)a5.m128i_i32 = sub_2E13860(0, 1u, v13[6], v122, v13[5]);
    *(_QWORD *)&a11 = v191;
    v185 = 2;
    *(float *)&a11 = *(float *)&v191 + *(float *)a5.m128i_i32;
    v189 = *(float *)&v191 + *(float *)a5.m128i_i32;
  }
  else
  {
    v185 = 0;
    a6 = 0.0;
    v189 = 0.0;
  }
  v23 = (void **)&v197;
  p_base = (__int64)&base;
  v195 = 0;
  v196 = 1;
  do
    *(_DWORD *)v23++ = -1;
  while ( v23 != &base );
  v25 = *(unsigned int *)(v12 + 112);
  if ( (int)v25 < 0 )
  {
    v26 = *(_QWORD **)(v15[7] + 16 * (v25 & 0x7FFFFFFF) + 8);
  }
  else
  {
    p_base = v15[38];
    v26 = *(_QWORD **)(p_base + 8 * v25);
  }
  if ( !v26 )
  {
LABEL_43:
    if ( !v187 )
      goto LABEL_61;
LABEL_44:
    *(_QWORD *)&a8 = dword_44D0BE0;
    if ( INFINITY == v183 )
      goto LABEL_176;
    goto LABEL_45;
  }
  if ( (*((_BYTE *)v26 + 4) & 8) == 0 )
  {
LABEL_16:
    v170 = 0;
    v162 = 0;
    v166 = v15;
    v27 = v179;
    while ( 1 )
    {
      v28 = v26[2];
      do
        v26 = (_QWORD *)v26[4];
      while ( v26 && ((*((_BYTE *)v26 + 4) & 8) != 0 || v28 == v26[2]) );
      v29 = v28;
      v30 = *(_QWORD *)(v13[2] + 32LL);
      v31 = v28;
      if ( (*(_DWORD *)(v28 + 44) & 4) != 0 )
      {
        do
          v31 = *(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL;
        while ( (*(_BYTE *)(v31 + 44) & 4) != 0 );
      }
      if ( (*(_DWORD *)(v28 + 44) & 8) != 0 )
      {
        do
          v29 = *(_QWORD *)(v29 + 8);
        while ( (*(_BYTE *)(v29 + 44) & 8) != 0 );
      }
      for ( i = *(_QWORD *)(v29 + 8); i != v31; v31 = *(_QWORD *)(v31 + 8) )
      {
        v33 = *(unsigned __int16 *)(v31 + 68);
        v29 = (unsigned int)(v33 - 14);
        if ( (unsigned __int16)(v33 - 14) > 4u && (_WORD)v33 != 24 )
          break;
      }
      v34 = *(_QWORD *)(v30 + 128);
      a2 = *(unsigned int *)(v30 + 144);
      if ( !(_DWORD)a2 )
        goto LABEL_100;
      v29 = ((_DWORD)a2 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v35 = (__int64 *)(v34 + 16 * v29);
      v18 = *v35;
      if ( v31 != *v35 )
        break;
LABEL_30:
      v36 = v35[1];
      if ( v187 )
      {
        p_base = *(_DWORD *)((v36 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v36 >> 1) & 3;
        if ( (unsigned int)p_base < (*(_DWORD *)((*a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*a3 >> 1) & 3) )
          goto LABEL_41;
        v29 = *a4 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (unsigned int)p_base > (*(_DWORD *)(v29 + 24) | (unsigned int)(*a4 >> 1) & 3) )
          goto LABEL_41;
      }
      v37 = *(_WORD *)(v28 + 68);
      ++v185;
      if ( v37 == 20 )
      {
        v74 = *(_DWORD **)(v28 + 32);
        p_base = (__int64)(v74 + 10);
        goto LABEL_102;
      }
      p_base = *(_QWORD *)v27;
      v29 = *(_QWORD *)(*(_QWORD *)v27 + 520LL);
      if ( (__int64 (__fastcall *)(__int64))v29 == sub_2DCA430 )
        goto LABEL_35;
      a2 = v27;
      ((void (__fastcall *)(_QWORD *, __int64, __int64))v29)(v193, v27, v28);
      v74 = (_DWORD *)v193[0];
      p_base = v193[1];
      if ( v194 )
      {
LABEL_102:
        a2 = *(unsigned int *)(p_base + 8);
        if ( v74[2] != (_DWORD)a2 )
          goto LABEL_103;
        p_base = *(_DWORD *)p_base >> 8;
        LOWORD(p_base) = p_base & 0xFFF;
        if ( (_WORD)p_base != ((*v74 >> 8) & 0xFFF) )
          goto LABEL_103;
        if ( !v26 )
        {
LABEL_42:
          v15 = v166;
          goto LABEL_43;
        }
      }
      else
      {
LABEL_103:
        v37 = *(_WORD *)(v28 + 68);
LABEL_35:
        if ( v37 == 10 )
          goto LABEL_41;
        if ( !v206 )
          goto LABEL_104;
        v38 = v203;
        v29 = HIDWORD(v204);
        p_base = (__int64)&v203[8 * HIDWORD(v204)];
        if ( v203 != (char *)p_base )
        {
          while ( v28 != *(_QWORD *)v38 )
          {
            v38 += 8;
            if ( (char *)p_base == v38 )
              goto LABEL_134;
          }
          goto LABEL_41;
        }
LABEL_134:
        if ( HIDWORD(v204) < (unsigned int)v204 )
        {
          ++HIDWORD(v204);
          *(_QWORD *)p_base = v28;
          ++v202;
        }
        else
        {
LABEL_104:
          a2 = v28;
          sub_C8CC70((__int64)&v202, v28, p_base, v29, v18, v16);
          if ( !(_BYTE)p_base )
            goto LABEL_41;
        }
        v75 = *(_DWORD *)(v28 + 44);
        if ( (v75 & 4) != 0 || (v75 & 8) == 0 )
        {
          v76 = (*(_QWORD *)(*(_QWORD *)(v28 + 16) + 24LL) >> 9) & 1LL;
        }
        else
        {
          a2 = 512;
          LOBYTE(v76) = sub_2E88A90(v28, 512, 1);
        }
        if ( (_BYTE)v76 )
        {
          v77 = *(__int64 (**)())(*(_QWORD *)v27 + 536LL);
          if ( v77 != sub_2EEE480 )
          {
            a2 = v28;
            if ( ((unsigned __int8 (__fastcall *)(__int64, __int64))v77)(v27, v28) )
            {
              a2 = *(unsigned int *)(v12 + 112);
              if ( (unsigned int)sub_2E8E710(v28, a2, 0, 0, 0) != -1 )
              {
                result = -1.0;
                *(_DWORD *)(v12 + 116) = dword_44D0BE0;
                goto LABEL_54;
              }
            }
          }
        }
        *(_QWORD *)&a7 = dword_44D0BE0;
        *(_QWORD *)&a8 = LODWORD(v183);
        if ( INFINITY == v183 )
        {
          *(_QWORD *)&a12 = 1065353216;
          v181 = 1.0;
        }
        else
        {
          v78 = *(_QWORD *)(v28 + 24);
          v169 = v78;
          if ( v78 != v162 )
          {
            v79 = v13[4];
            v80 = *(_DWORD *)(v79 + 24);
            v81 = *(_QWORD *)(v79 + 8);
            if ( v80 )
            {
              v82 = v80 - 1;
              v83 = v82 & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
              v84 = (__int64 *)(v81 + 16LL * v83);
              v85 = *v84;
              if ( v78 == *v84 )
              {
LABEL_114:
                v86 = v84[1];
                if ( v86 )
                {
                  v87 = *(_QWORD *)(v169 + 112);
                  v88 = *(unsigned int *)(v169 + 120);
                  if ( v87 != v87 + 8 * v88 )
                  {
                    v180 = v27;
                    v172 = v26;
                    v89 = v86 + 56;
                    v90 = *(__int64 **)(v169 + 112);
                    v163 = v13;
                    v91 = v86;
                    v161 = v12;
                    v92 = (__int64 *)(v87 + 8 * v88);
                    while ( 1 )
                    {
                      v93 = *v90;
                      if ( *(_BYTE *)(v91 + 84) )
                      {
                        v94 = *(_QWORD **)(v91 + 64);
                        v95 = &v94[*(unsigned int *)(v91 + 76)];
                        if ( v94 == v95 )
                          goto LABEL_127;
                        while ( v93 != *v94 )
                        {
                          if ( v95 == ++v94 )
                            goto LABEL_127;
                        }
                      }
                      else if ( !sub_C8CA60(v89, v93) )
                      {
LABEL_127:
                        v12 = v161;
                        v13 = v163;
                        v27 = v180;
                        v26 = v172;
                        v97 = sub_2E89D80(v28, *(_DWORD *)(v161 + 112), 0);
                        v173 = HIBYTE(v97);
                        a2 = (unsigned __int8)v97;
                        *(float *)a5.m128i_i32 = sub_2E13950(HIBYTE(v97), v97, v13[6], v28, v13[5]);
                        v181 = *(float *)a5.m128i_i32;
                        if ( v173 )
                          goto LABEL_128;
                        v170 = 1;
                        goto LABEL_125;
                      }
                      if ( v92 == ++v90 )
                      {
                        v27 = v180;
                        v26 = v172;
                        v13 = v163;
                        v12 = v161;
                        break;
                      }
                    }
                  }
                }
              }
              else
              {
                v130 = 1;
                while ( v85 != -4096 )
                {
                  v131 = v130 + 1;
                  v83 = v82 & (v130 + v83);
                  v84 = (__int64 *)(v81 + 16LL * v83);
                  v85 = *v84;
                  if ( v169 == *v84 )
                    goto LABEL_114;
                  v130 = v131;
                }
              }
            }
            v96 = sub_2E89D80(v28, *(_DWORD *)(v12 + 112), 0);
            a2 = (unsigned __int8)v96;
            *(float *)a5.m128i_i32 = sub_2E13950(HIBYTE(v96), v96, v13[6], v28, v13[5]);
            v170 = 0;
            v181 = *(float *)a5.m128i_i32;
            goto LABEL_125;
          }
          v119 = sub_2E89D80(v28, *(_DWORD *)(v12 + 112), 0);
          v165 = HIBYTE(v119);
          a2 = (unsigned __int8)v119;
          *(float *)a5.m128i_i32 = sub_2E13950(HIBYTE(v119), v119, v13[6], v28, v13[5]);
          v181 = *(float *)a5.m128i_i32;
          if ( v165 && v170 )
          {
LABEL_128:
            v98 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v13[2] + 32LL) + 152LL) + 16LL * *(unsigned int *)(v169 + 24) + 8);
            if ( ((v98 >> 1) & 3) != 0 )
              v99 = v98 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (int)(((v98 >> 1) & 3) - 1));
            else
              v99 = *(_QWORD *)(v98 & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL | 6;
            v164 = v99;
            v100 = (__int64 *)sub_2E09D00((__int64 *)v12, v99);
            v101 = *(unsigned int *)(v12 + 8);
            v170 = 1;
            a2 = 3 * v101;
            if ( v100 != (__int64 *)(*(_QWORD *)v12 + 24 * v101) )
            {
              p_base = v164;
              a2 = v164 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_DWORD *)((*v100 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v100 >> 1) & 3)) <= (*(_DWORD *)((v164 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v164 >> 1) & 3) )
              {
                HIDWORD(a12) = 0;
                *(float *)&a12 = 3.0 * v181;
                v181 = 3.0 * v181;
              }
            }
          }
LABEL_125:
          *(_QWORD *)&a11 = LODWORD(v189);
          *(float *)&a11 = v189 + v181;
          v189 = v189 + v181;
          v162 = v169;
        }
        if ( *(_WORD *)(v28 + 68) != 20 )
        {
          v102 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v27 + 520LL);
          if ( v102 == sub_2DCA430 )
            goto LABEL_41;
          a2 = v27;
          ((void (__fastcall *)(void **, __int64, __int64))v102)(&base, v27, v28);
          if ( !v201[0] )
            goto LABEL_41;
        }
        a2 = *(unsigned int *)(v12 + 112);
        v103 = sub_34C7490(v28, a2, v160, (__int64)v166);
        v104 = v103;
        if ( v103
          && (v103 < 0
           || (p_base = *(_QWORD *)(*(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v166 + 16LL) + 200LL))(*(_QWORD *)(*v166 + 16LL))
                                              + 248)
                                  + 16LL),
               *(_BYTE *)(p_base + v104))
           && (p_base = v104 >> 6, a2 = v166[48], (*(_QWORD *)(a2 + 8 * p_base) & (1LL << v104)) == 0)) )
        {
          if ( (v196 & 1) != 0 )
          {
            v16 = (__int64)&v197;
            v106 = 7;
          }
          else
          {
            v105 = v198;
            v16 = (__int64)v197;
            v106 = v198 - 1;
            if ( !v198 )
            {
              v132 = v196;
              ++v195;
              v107 = 0;
              v133 = ((unsigned int)v196 >> 1) + 1;
              goto LABEL_221;
            }
          }
          a2 = v106 & (37 * v104);
          v107 = (unsigned int *)(v16 + 8 * a2);
          v18 = *v107;
          if ( v104 != (_DWORD)v18 )
          {
            v146 = 1;
            v147 = 0;
            while ( (_DWORD)v18 != -1 )
            {
              if ( (_DWORD)v18 == -2 && !v147 )
                v147 = v107;
              a2 = v106 & (unsigned int)(a2 + v146);
              v107 = (unsigned int *)(v16 + 8 * a2);
              v18 = *v107;
              if ( v104 == (_DWORD)v18 )
                goto LABEL_147;
              ++v146;
            }
            v134 = 24;
            v105 = 8;
            if ( v147 )
              v107 = v147;
            v132 = v196;
            ++v195;
            v133 = ((unsigned int)v196 >> 1) + 1;
            if ( (v196 & 1) == 0 )
            {
              v105 = v198;
LABEL_221:
              v134 = 3 * v105;
            }
            v16 = 4 * v133;
            if ( (unsigned int)v16 >= v134 )
            {
              sub_34C7990((__int64)&v195, 2 * v105);
              if ( (v196 & 1) != 0 )
              {
                v16 = (__int64)&v197;
                v152 = 7;
              }
              else
              {
                v16 = (__int64)v197;
                if ( !v198 )
                {
LABEL_290:
                  LODWORD(v196) = (2 * ((unsigned int)v196 >> 1) + 2) | v196 & 1;
                  BUG();
                }
                v152 = v198 - 1;
              }
              v132 = v196;
              v153 = v152 & (37 * v104);
              v107 = (unsigned int *)(v16 + 8LL * v153);
              v18 = *v107;
              if ( v104 != (_DWORD)v18 )
              {
                v154 = 1;
                v151 = 0;
                while ( (_DWORD)v18 != -1 )
                {
                  if ( (_DWORD)v18 == -2 && !v151 )
                    v151 = v107;
                  v153 = v152 & (v154 + v153);
                  v107 = (unsigned int *)(v16 + 8LL * v153);
                  v18 = *v107;
                  if ( v104 == (_DWORD)v18 )
                    goto LABEL_257;
                  ++v154;
                }
LABEL_255:
                if ( v151 )
                  v107 = v151;
LABEL_257:
                v132 = v196;
              }
            }
            else
            {
              v18 = v105 - HIDWORD(v196) - v133;
              if ( (unsigned int)v18 <= v105 >> 3 )
              {
                sub_34C7990((__int64)&v195, v105);
                if ( (v196 & 1) != 0 )
                {
                  v16 = (__int64)&v197;
                  v148 = 7;
                }
                else
                {
                  v16 = (__int64)v197;
                  if ( !v198 )
                    goto LABEL_290;
                  v148 = v198 - 1;
                }
                v132 = v196;
                v149 = v148 & (37 * v104);
                v107 = (unsigned int *)(v16 + 8LL * v149);
                v18 = *v107;
                if ( v104 != (_DWORD)v18 )
                {
                  v150 = 1;
                  v151 = 0;
                  while ( (_DWORD)v18 != -1 )
                  {
                    if ( (_DWORD)v18 == -2 && !v151 )
                      v151 = v107;
                    v149 = v148 & (v150 + v149);
                    v107 = (unsigned int *)(v16 + 8LL * v149);
                    v18 = *v107;
                    if ( v104 == (_DWORD)v18 )
                      goto LABEL_257;
                    ++v150;
                  }
                  goto LABEL_255;
                }
              }
            }
            a2 = 2 * (v132 >> 1) + 2;
            LODWORD(v196) = a2 | v132 & 1;
            if ( *v107 != -1 )
              --HIDWORD(v196);
            *v107 = v104;
            p_base = (__int64)(v107 + 1);
            a5.m128i_i64[0] = 0;
            v107[1] = 0;
            goto LABEL_148;
          }
LABEL_147:
          a5.m128i_i64[0] = v107[1];
          p_base = (__int64)(v107 + 1);
LABEL_148:
          *(float *)a5.m128i_i32 = *(float *)a5.m128i_i32 + v181;
          *(_DWORD *)p_base = a5.m128i_i32[0];
          if ( !v26 )
            goto LABEL_42;
        }
        else
        {
LABEL_41:
          if ( !v26 )
            goto LABEL_42;
        }
      }
    }
    v73 = 1;
    while ( v18 != -4096 )
    {
      v16 = (unsigned int)(v73 + 1);
      v29 = ((_DWORD)a2 - 1) & (unsigned int)(v73 + v29);
      v35 = (__int64 *)(v34 + 16LL * (unsigned int)v29);
      v18 = *v35;
      if ( *v35 == v31 )
        goto LABEL_30;
      v73 = v16;
    }
LABEL_100:
    v35 = (__int64 *)(v34 + 16LL * (unsigned int)a2);
    goto LABEL_30;
  }
  while ( 1 )
  {
    v26 = (_QWORD *)v26[4];
    if ( !v26 )
      break;
    if ( (*((_BYTE *)v26 + 4) & 8) == 0 )
      goto LABEL_16;
  }
  if ( v187 )
    goto LABEL_44;
LABEL_61:
  if ( !((unsigned int)v196 >> 1) )
    goto LABEL_175;
  if ( !v159 )
  {
    if ( v177 )
    {
      v177 = 0;
      v48 = *(_DWORD *)(v12 + 112) & 0x7FFFFFFF;
      if ( (unsigned int)v48 < *((_DWORD *)v15 + 62) )
        *(_DWORD *)(v15[30] + 40 * v48 + 16) = 0;
    }
  }
  base = v201;
  v200 = 0x800000000LL;
  if ( (v196 & 1) != 0 )
  {
    v53 = &base;
    v49 = (__int64 *)&v197;
  }
  else
  {
    v49 = v197;
    v50 = v198;
    v51 = (unsigned int *)&v197[v198];
    v52 = v197;
    if ( v197 == (__int64 *)v51 )
    {
      v53 = (void **)v197;
      goto LABEL_215;
    }
    v53 = (void **)&v197[v198];
  }
  v51 = (unsigned int *)v49;
  do
  {
    if ( *v51 <= 0xFFFFFFFD )
      break;
    v51 += 2;
  }
  while ( v51 != (unsigned int *)v53 );
  v52 = (__int64 *)&v197;
  v54 = 8;
  if ( (v196 & 1) == 0 )
  {
    v52 = v197;
    v50 = v198;
LABEL_215:
    v54 = v50;
  }
  v55 = (unsigned int *)&v52[v54];
  v56 = 0;
  if ( v55 == v51 )
  {
    v189 = v189 * 1.01;
  }
  else
  {
    v57 = v12;
    v58 = v55;
    v59 = v57;
    do
    {
      v60 = *v51;
      if ( v177 != (_DWORD)v60 )
      {
        v61 = (unsigned int)v56;
        a5 = (__m128i)v51[1];
        if ( (unsigned int)v56 >= (unsigned __int64)HIDWORD(v200) )
        {
          v176 = ((unsigned __int64)(unsigned int)_mm_cvtsi128_si32(a5) << 32) | v60;
          if ( HIDWORD(v200) < (unsigned __int64)(unsigned int)v56 + 1 )
          {
            v174 = v59;
            sub_C8D5F0((__int64)&base, v201, (unsigned int)v56 + 1LL, 8u, v18, v59);
            v61 = (unsigned int)v200;
            v59 = v174;
          }
          *((_QWORD *)base + v61) = v176;
          v56 = (unsigned int)(v200 + 1);
          LODWORD(v200) = v200 + 1;
        }
        else
        {
          v62 = (char *)base + 8 * (unsigned int)v56;
          if ( v62 )
          {
            *v62 = v60;
            v62[1] = a5.m128i_i32[0];
            LODWORD(v56) = v200;
          }
          v56 = (unsigned int)(v56 + 1);
          LODWORD(v200) = v56;
        }
      }
      for ( v51 += 2; v53 != (void **)v51; v51 += 2 )
      {
        if ( *v51 <= 0xFFFFFFFD )
          break;
      }
    }
    while ( v51 != v58 );
    v63 = (int *)base;
    v12 = v59;
    v64 = 2 * v56;
    if ( v56 > 1 )
    {
      qsort(base, v56, 8u, (__compar_fn_t)sub_34C73F0);
      v63 = (int *)base;
      v64 = 2LL * (unsigned int)v200;
    }
    v178 = &v63[v64];
    if ( &v63[v64] != v63 )
    {
      v168 = v13;
      v65 = v63;
      v66 = v15;
      v171 = (__int64)(v15 + 30);
      while ( 1 )
      {
        v70 = *((unsigned int *)v66 + 62);
        v71 = *(_DWORD *)(v12 + 112);
        v175 = *v65;
        v72 = (v66[8] & 0x7FFFFFFF) + 1;
        if ( v72 > (unsigned int)v70 && v72 != v70 )
        {
          if ( v72 < v70 )
          {
            v67 = v66[30];
            v135 = v67 + 40 * v70;
            if ( v135 != v67 + 40LL * v72 )
            {
              v167 = v12;
              v136 = v66;
              v137 = *(_DWORD *)(v12 + 112);
              v138 = v67 + 40LL * v72;
              v139 = v135;
              do
              {
                v139 -= 40;
                v140 = *(_QWORD *)(v139 + 8);
                if ( v140 != v139 + 24 )
                  _libc_free(v140);
              }
              while ( v138 != v139 );
              v71 = v137;
              v67 = v136[30];
              v12 = v167;
              v66 = v136;
            }
            *((_DWORD *)v66 + 62) = v72;
            goto LABEL_89;
          }
          sub_34C7F50(v171, v72 - v70, (const void **)v66 + 32, v70);
        }
        v67 = v66[30];
LABEL_89:
        v68 = v67 + 40LL * (v71 & 0x7FFFFFFF);
        v69 = *(unsigned int *)(v68 + 16);
        if ( v69 + 1 > (unsigned __int64)*(unsigned int *)(v68 + 20) )
        {
          sub_C8D5F0(v68 + 8, (const void *)(v68 + 24), v69 + 1, 4u, v69 + 1, v59);
          v69 = *(unsigned int *)(v68 + 16);
        }
        v65 += 2;
        *(_DWORD *)(*(_QWORD *)(v68 + 8) + 4 * v69) = v175;
        ++*(_DWORD *)(v68 + 16);
        if ( v178 == v65 )
        {
          v15 = v66;
          v63 = (int *)base;
          v13 = v168;
          break;
        }
      }
    }
    *(_QWORD *)&v21 = LODWORD(v189);
    *(float *)&v21 = v189 * 1.01;
    v189 = v189 * 1.01;
    if ( v63 != (int *)v201 )
      _libc_free((unsigned __int64)v63);
  }
LABEL_175:
  *(_QWORD *)&a12 = dword_44D0BE0;
  if ( INFINITY == v183 )
  {
LABEL_176:
    result = -1.0;
    goto LABEL_54;
  }
  v123 = *(_QWORD **)v12;
  v124 = v13[2];
  v16 = *(_QWORD *)v12 + 24LL * *(unsigned int *)(v12 + 8);
  v18 = *(_QWORD *)(v124 + 32);
  if ( *(_QWORD *)v12 == v16 )
  {
LABEL_195:
    a2 = *(_QWORD *)(v124 + 184);
    if ( !(unsigned __int8)sub_2E0ADA0(v12, (__int64 *)a2, *(unsigned int *)(v124 + 192)) )
    {
      a2 = v12;
      if ( !(unsigned __int8)sub_34C78E0((__int64)v13, v12) )
      {
        v126 = *(unsigned int *)(v12 + 112);
        if ( (int)v126 < 0 )
          v127 = *(_QWORD *)(v15[7] + 16 * (v126 & 0x7FFFFFFF) + 8);
        else
          v127 = *(_QWORD *)(v15[38] + 8 * v126);
        if ( !v127 )
        {
LABEL_203:
          result = -1.0;
          *(_DWORD *)(v12 + 116) = dword_44D0BE0;
          goto LABEL_54;
        }
        while ( 1 )
        {
          v128 = *(_QWORD *)(v127 + 16);
          if ( (unsigned int)*(unsigned __int16 *)(v128 + 68) - 1 <= 1 )
          {
            a2 = 0xCCCCCCCCCCCCCCCDLL * ((v127 - *(_QWORD *)(v128 + 32)) >> 3);
            if ( sub_2E8E6C0(v128, a2) )
              break;
          }
          v127 = *(_QWORD *)(v127 + 32);
          if ( !v127 )
            goto LABEL_203;
        }
      }
    }
  }
  else
  {
    p_base = v18 + 96;
    while ( 1 )
    {
      v125 = *v123 & 0xFFFFFFFFFFFFFFF8LL;
      while ( 1 )
      {
        v125 = *(_QWORD *)(v125 + 8);
        if ( p_base == v125 )
          break;
        if ( *(_QWORD *)(v125 + 16) )
          goto LABEL_193;
      }
      v125 = *(_QWORD *)(v18 + 96);
LABEL_193:
      a2 = *(unsigned int *)((v123[1] & 0xFFFFFFFFFFFFFFF8LL) + 24);
      if ( *(_DWORD *)((v125 & 0xFFFFFFFFFFFFFFF8LL) + 24) < (unsigned int)a2 )
        break;
      v123 += 3;
      if ( (_QWORD *)v16 == v123 )
        goto LABEL_195;
    }
  }
LABEL_45:
  v39 = 0;
  v40 = *(_QWORD *)(v13[1] + 16LL);
  v41 = *(__int64 (**)())(*(_QWORD *)v40 + 128LL);
  if ( v41 != sub_2DAC790 )
    v39 = ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD, double, double, double, double, double, double, double, double))v41)(
            v40,
            a2,
            p_base,
            0,
            *(double *)a5.m128i_i64,
            a6,
            a7,
            a8,
            v21,
            v22,
            a11,
            a12);
  if ( (unsigned __int8)sub_34C7590(v12, v13[2], v13[3], v39, v18, v16) )
  {
    *(_QWORD *)&v43 = LODWORD(v189);
    *(float *)&v43 = v189 * 0.5;
    v189 = v189 * 0.5;
  }
  v44 = *(__int64 (**)())(*v160 + 32LL);
  if ( v44 != sub_2FF5190 )
    v189 = (float)((int (__fastcall *)(_QWORD *, __int64, _QWORD, double, double, double, double, double, double, double, double))v44)(
                    v160,
                    v12,
                    v13[1],
                    *(double *)a5.m128i_i64,
                    a6,
                    a7,
                    a8,
                    v42,
                    v43,
                    a11,
                    a12)
         * v189;
  v45 = *(float (__fastcall **)(float, __int64, int))(*v13 + 16LL);
  if ( v187 )
  {
    v46 = (*(_DWORD *)((*a4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*a4 >> 1) & 3)
        - (*(_DWORD *)((*a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*a3 >> 1) & 3);
    if ( v45 == sub_2F4C0C0 )
    {
LABEL_53:
      result = v189 / (float)(v46 + 400);
      goto LABEL_54;
    }
  }
  else
  {
    v46 = (unsigned int)sub_2E0B010(v12);
    if ( v45 == sub_2F4C0C0 )
      goto LABEL_53;
  }
  result = v189;
  ((void (__fastcall *)(_QWORD *, __int64, _QWORD, float))v45)(v13, v46, v185, v189);
LABEL_54:
  if ( (v196 & 1) == 0 )
    sub_C7D6A0((__int64)v197, 8LL * v198, 4);
  if ( !v206 )
    _libc_free((unsigned __int64)v203);
  return result;
}
