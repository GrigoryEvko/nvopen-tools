// Function: sub_23F7470
// Address: 0x23f7470
//
__int64 __fastcall sub_23F7470(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  __int64 v4; // r13
  __int64 *v6; // r15
  __int64 v7; // rax
  __int64 *i; // r12
  __int64 *v9; // rdi
  __int64 *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 *v13; // rax
  __int64 v14; // rsi
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r12
  unsigned int *v18; // rax
  int v19; // edx
  unsigned int *v20; // rcx
  __int64 v21; // r9
  __int64 v22; // r12
  unsigned int *v23; // rax
  int v24; // ecx
  unsigned int *v25; // rdx
  __int64 v26; // rcx
  _QWORD *v27; // rax
  _QWORD *v28; // r12
  unsigned __int64 v29; // rdi
  int v30; // eax
  _QWORD *v31; // rdi
  __int64 v32; // r15
  __int64 v33; // r13
  __int64 v34; // rdi
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // r15
  unsigned __int64 v38; // rsi
  unsigned __int64 *v39; // rax
  int v40; // edx
  unsigned __int64 *v41; // rcx
  unsigned __int16 v42; // r12
  _QWORD *v43; // rdi
  int v44; // eax
  __int64 v45; // rax
  _QWORD *v46; // rbx
  _QWORD *v47; // r12
  __int64 v48; // rax
  __int64 v49; // rax
  unsigned __int64 v50; // rdi
  __int64 v51; // r12
  _QWORD *v52; // rdi
  __int64 v53; // r9
  __int64 *v54; // r15
  __int64 v55; // rdx
  unsigned __int64 v56; // rax
  int v57; // edx
  unsigned __int64 v58; // r15
  __int64 v59; // r15
  __int64 *v60; // rax
  __int64 *v61; // r13
  __int64 *v62; // rax
  unsigned __int64 v63; // rax
  __int64 *v64; // rsi
  __int64 **v65; // r8
  __int64 *v66; // rax
  _QWORD *v67; // rax
  __int64 v68; // r15
  __int64 v69; // r13
  unsigned int *v70; // r13
  unsigned int *v71; // rbx
  __int64 v72; // rdx
  unsigned int v73; // esi
  __int64 v74; // rax
  __int64 v75; // r15
  __int64 v76; // rax
  __int64 v77; // rdx
  __int64 v78; // rax
  __int64 v79; // rsi
  __int64 v80; // rax
  __int64 *v81; // rax
  __int64 v82; // rsi
  unsigned __int8 *v83; // rsi
  __int64 v84; // rbx
  __int64 *v85; // rax
  __int64 v86; // rsi
  __int64 v87; // r9
  __int64 v88; // r14
  unsigned int *v89; // rax
  int v90; // esi
  unsigned int *v91; // rdx
  __int64 v92; // rbx
  __int64 v93; // r8
  __int64 v94; // r9
  unsigned int *v95; // rax
  int v96; // ecx
  unsigned int *v97; // rdx
  char v98; // al
  __int64 v99; // rsi
  __int64 v100; // rdi
  __int64 v101; // rax
  __int64 v102; // r8
  __int64 v103; // r9
  __int64 v104; // r10
  __int64 v105; // rax
  unsigned __int64 v106; // rdx
  __int64 **v107; // rax
  __int64 *v108; // rax
  char v109; // bl
  __int64 v110; // rax
  __int64 v111; // rax
  __int64 v112; // rax
  __int64 v113; // rbx
  __int64 v114; // rax
  _QWORD *v115; // rax
  __int64 v116; // r15
  unsigned int *v117; // rbx
  unsigned int *v118; // r13
  __int64 v119; // rdx
  unsigned int v120; // esi
  unsigned __int64 v121; // r8
  __int64 v122; // rax
  __m128i si128; // xmm0
  unsigned int *v124; // rdi
  size_t v125; // rdx
  __int64 v126; // rsi
  __int64 v127; // rdx
  unsigned __int64 v128; // rax
  unsigned __int64 v129; // rsi
  unsigned __int64 v130; // r14
  unsigned __int64 v131; // rbx
  unsigned int *v132; // r12
  unsigned int *v133; // r14
  __int64 v134; // rbx
  __int64 v135; // rdx
  unsigned int v136; // esi
  __int64 v137; // [rsp-10h] [rbp-520h]
  __int64 v138; // [rsp-8h] [rbp-518h]
  __int64 v139; // [rsp+0h] [rbp-510h]
  __int64 v140; // [rsp+8h] [rbp-508h]
  unsigned __int64 v141; // [rsp+10h] [rbp-500h]
  __int64 v142; // [rsp+18h] [rbp-4F8h]
  __int64 *v143; // [rsp+20h] [rbp-4F0h]
  char v144; // [rsp+28h] [rbp-4E8h]
  unsigned __int64 v145; // [rsp+28h] [rbp-4E8h]
  __int64 v146; // [rsp+30h] [rbp-4E0h]
  unsigned __int64 v147; // [rsp+40h] [rbp-4D0h]
  void *v148; // [rsp+48h] [rbp-4C8h]
  int v149; // [rsp+48h] [rbp-4C8h]
  char v150; // [rsp+5Fh] [rbp-4B1h]
  __int64 v151; // [rsp+68h] [rbp-4A8h]
  char v152; // [rsp+68h] [rbp-4A8h]
  __int64 *v153; // [rsp+68h] [rbp-4A8h]
  __int64 **v154; // [rsp+68h] [rbp-4A8h]
  __int64 *v155; // [rsp+68h] [rbp-4A8h]
  unsigned __int64 v156; // [rsp+68h] [rbp-4A8h]
  __int64 v157; // [rsp+78h] [rbp-498h]
  __int64 v158; // [rsp+90h] [rbp-480h]
  __int64 v159; // [rsp+90h] [rbp-480h]
  unsigned __int64 v160; // [rsp+98h] [rbp-478h]
  __int64 *v161; // [rsp+98h] [rbp-478h]
  __int64 v162; // [rsp+A0h] [rbp-470h]
  __int64 v163; // [rsp+A8h] [rbp-468h]
  __int64 v166; // [rsp+C0h] [rbp-450h]
  __int64 v167; // [rsp+C8h] [rbp-448h]
  __int64 v168; // [rsp+C8h] [rbp-448h]
  __int64 v169; // [rsp+C8h] [rbp-448h]
  __int64 *v170; // [rsp+C8h] [rbp-448h]
  unsigned __int64 v171; // [rsp+D0h] [rbp-440h]
  __int64 *src; // [rsp+E8h] [rbp-428h]
  __int64 v173; // [rsp+F0h] [rbp-420h]
  __int64 *v174; // [rsp+F8h] [rbp-418h]
  __int64 *v175; // [rsp+120h] [rbp-3F0h] BYREF
  __int64 v176; // [rsp+128h] [rbp-3E8h] BYREF
  void *dest; // [rsp+130h] [rbp-3E0h]
  size_t v178; // [rsp+138h] [rbp-3D8h]
  _QWORD v179[2]; // [rsp+140h] [rbp-3D0h] BYREF
  __int64 v180[2]; // [rsp+150h] [rbp-3C0h] BYREF
  _BYTE v181[16]; // [rsp+160h] [rbp-3B0h] BYREF
  __int16 v182; // [rsp+170h] [rbp-3A0h]
  unsigned int **v183; // [rsp+180h] [rbp-390h] BYREF
  _QWORD v184[2]; // [rsp+188h] [rbp-388h] BYREF
  __int64 v185; // [rsp+198h] [rbp-378h]
  __int64 *v186; // [rsp+1A0h] [rbp-370h]
  __int16 v187; // [rsp+1A8h] [rbp-368h]
  __int64 v188[2]; // [rsp+1B0h] [rbp-360h] BYREF
  __int64 *v189; // [rsp+1C0h] [rbp-350h] BYREF
  __int64 v190; // [rsp+1C8h] [rbp-348h]
  _BYTE v191[64]; // [rsp+1D0h] [rbp-340h] BYREF
  signed __int64 v192; // [rsp+210h] [rbp-300h] BYREF
  char *v193; // [rsp+218h] [rbp-2F8h]
  __int64 v194; // [rsp+220h] [rbp-2F0h]
  char v195; // [rsp+228h] [rbp-2E8h] BYREF
  __int16 v196; // [rsp+230h] [rbp-2E0h]
  unsigned int *v197; // [rsp+270h] [rbp-2A0h] BYREF
  size_t n; // [rsp+278h] [rbp-298h]
  _QWORD v199[4]; // [rsp+280h] [rbp-290h] BYREF
  __int64 v200; // [rsp+2A0h] [rbp-270h]
  __int64 *v201; // [rsp+2A8h] [rbp-268h]
  __int64 v202; // [rsp+2B0h] [rbp-260h]
  __int64 *v203; // [rsp+2B8h] [rbp-258h]
  void **v204; // [rsp+2C0h] [rbp-250h]
  _QWORD *v205; // [rsp+2C8h] [rbp-248h]
  __int64 v206; // [rsp+2D0h] [rbp-240h]
  int v207; // [rsp+2D8h] [rbp-238h]
  __int16 v208; // [rsp+2DCh] [rbp-234h]
  char v209; // [rsp+2DEh] [rbp-232h]
  __int64 v210; // [rsp+2E0h] [rbp-230h]
  __int64 v211; // [rsp+2E8h] [rbp-228h]
  __int64 *v212; // [rsp+2F0h] [rbp-220h] BYREF
  unsigned __int64 v213; // [rsp+2F8h] [rbp-218h]
  _QWORD v214[2]; // [rsp+300h] [rbp-210h] BYREF
  _BYTE v215[24]; // [rsp+310h] [rbp-200h] BYREF
  char *v216; // [rsp+328h] [rbp-1E8h]
  char v217; // [rsp+338h] [rbp-1D8h] BYREF
  void *v218; // [rsp+3A8h] [rbp-168h]
  __int64 v219[8]; // [rsp+3B8h] [rbp-158h] BYREF
  _QWORD *v220; // [rsp+3F8h] [rbp-118h]
  unsigned int v221; // [rsp+408h] [rbp-108h]
  unsigned __int64 v222; // [rsp+418h] [rbp-F8h]
  char v223; // [rsp+42Ch] [rbp-E4h]
  unsigned __int64 v224; // [rsp+488h] [rbp-88h]
  char v225; // [rsp+49Ch] [rbp-74h]

  LODWORD(v4) = 0;
  if ( (unsigned __int8)sub_B2D610(a1, 37) )
    return (unsigned int)v4;
  v6 = (__int64 *)(a1 + 72);
  v171 = sub_B2BEC0(a1);
  v7 = sub_B2BE50(a1);
  sub_D5EB90((__int64)v215, v171, a2, v7, 257, 0);
  v4 = *(_QWORD *)(a1 + 80);
  v189 = (__int64 *)v191;
  v190 = 0x400000000LL;
  if ( a1 + 72 == v4 )
  {
    i = 0;
  }
  else
  {
    if ( !v4 )
      BUG();
    while ( 1 )
    {
      i = *(__int64 **)(v4 + 32);
      if ( i != (__int64 *)(v4 + 24) )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( v6 == (__int64 *)v4 )
        goto LABEL_9;
      if ( !v4 )
        BUG();
    }
  }
  while ( (__int64 *)v4 != v6 )
  {
    if ( !i )
    {
      v183 = (unsigned int **)&unk_49D94D0;
      v184[0] = v171;
      BUG();
    }
    src = i - 3;
    v183 = (unsigned int **)&unk_49D94D0;
    v184[0] = v171;
    v84 = i[2];
    v192 = (signed __int64)&unk_49D94D0;
    v193 = (char *)v171;
    v85 = (__int64 *)sub_AA48A0(v84);
    v212 = (__int64 *)&unk_49D94D0;
    v203 = v85;
    v204 = (void **)&v212;
    v205 = v214;
    v197 = (unsigned int *)v199;
    v213 = (unsigned __int64)v193;
    v200 = v84;
    n = 0x200000000LL;
    v206 = 0;
    v207 = 0;
    v208 = 512;
    v209 = 7;
    v210 = 0;
    v211 = 0;
    v214[0] = &unk_49DA0B0;
    v201 = i;
    LOWORD(v202) = 0;
    if ( i != (__int64 *)(v84 + 48) )
    {
      v86 = *(_QWORD *)sub_B46C60((__int64)src);
      v180[0] = v86;
      if ( v86 && (sub_B96E90((__int64)v180, v86, 1), (v88 = v180[0]) != 0) )
      {
        v89 = v197;
        v90 = n;
        v91 = &v197[4 * (unsigned int)n];
        if ( v197 != v91 )
        {
          while ( *v89 )
          {
            v89 += 4;
            if ( v91 == v89 )
              goto LABEL_202;
          }
          *((_QWORD *)v89 + 1) = v180[0];
LABEL_166:
          sub_B91220((__int64)v180, v88);
          goto LABEL_167;
        }
LABEL_202:
        if ( (unsigned int)n >= (unsigned __int64)HIDWORD(n) )
        {
          v131 = v158 & 0xFFFFFFFF00000000LL;
          v158 &= 0xFFFFFFFF00000000LL;
          if ( HIDWORD(n) < (unsigned __int64)(unsigned int)n + 1 )
          {
            sub_C8D5F0((__int64)&v197, v199, (unsigned int)n + 1LL, 0x10u, (unsigned int)n + 1LL, v87);
            v91 = &v197[4 * (unsigned int)n];
          }
          *(_QWORD *)v91 = v131;
          *((_QWORD *)v91 + 1) = v88;
          v88 = v180[0];
          LODWORD(n) = n + 1;
        }
        else
        {
          if ( v91 )
          {
            *v91 = 0;
            *((_QWORD *)v91 + 1) = v88;
            v90 = n;
            v88 = v180[0];
          }
          LODWORD(n) = v90 + 1;
        }
      }
      else
      {
        sub_93FB40((__int64)&v197, 0);
        v88 = v180[0];
      }
      if ( v88 )
        goto LABEL_166;
    }
LABEL_167:
    v192 = (signed __int64)&unk_49D94D0;
    nullsub_63();
    v92 = sub_B9C770(v203, 0, 0, 0, 1);
    if ( v92 )
    {
      v95 = v197;
      v96 = n;
      v97 = &v197[4 * (unsigned int)n];
      if ( v197 == v97 )
      {
LABEL_192:
        if ( (unsigned int)n >= (unsigned __int64)HIDWORD(n) )
        {
          v129 = (unsigned int)n + 1LL;
          v130 = v160 & 0xFFFFFFFF00000000LL | 0x1F;
          v160 = v130;
          if ( HIDWORD(n) < v129 )
          {
            sub_C8D5F0((__int64)&v197, v199, v129, 0x10u, v93, v94);
            v97 = &v197[4 * (unsigned int)n];
          }
          *(_QWORD *)v97 = v130;
          *((_QWORD *)v97 + 1) = v92;
          LODWORD(n) = n + 1;
        }
        else
        {
          if ( v97 )
          {
            *v97 = 31;
            *((_QWORD *)v97 + 1) = v92;
            v96 = n;
          }
          LODWORD(n) = v96 + 1;
        }
      }
      else
      {
        while ( *v95 != 31 )
        {
          v95 += 4;
          if ( v97 == v95 )
            goto LABEL_192;
        }
        *((_QWORD *)v95 + 1) = v92;
      }
    }
    else
    {
      sub_93FB40((__int64)&v197, 31);
    }
    v183 = (unsigned int **)&unk_49D94D0;
    nullsub_63();
    v98 = *((_BYTE *)i - 24);
    switch ( v98 )
    {
      case '=':
        if ( (*((_BYTE *)i - 22) & 1) != 0 )
          goto LABEL_181;
        v99 = *(i - 2);
        v100 = *(i - 7);
        break;
      case '>':
        if ( (*((_BYTE *)i - 22) & 1) != 0 )
          goto LABEL_181;
        v100 = *(i - 7);
        v99 = *(_QWORD *)(*(i - 11) + 8);
        break;
      case 'A':
        if ( (*((_BYTE *)i - 22) & 1) != 0 )
          goto LABEL_181;
        v100 = *(i - 15);
        v99 = *(_QWORD *)(*(i - 11) + 8);
        break;
      default:
        if ( v98 != 66 || (*((_BYTE *)i - 22) & 1) != 0 )
          goto LABEL_181;
        v100 = *(i - 11);
        v99 = *(_QWORD *)(*(i - 7) + 8);
        break;
    }
    v101 = sub_23F6380(v100, v99, v171, (__int64)v215, (__int64)&v197, a3);
    v104 = v101;
    if ( v101 )
    {
      if ( a4[5] )
      {
        v109 = a4[4];
        v163 = v101;
        v196 = 257;
        HIDWORD(v183) = 0;
        v110 = sub_BCB2B0(v203);
        v180[0] = sub_ACD640(v110, v109, 1u);
        v111 = sub_BCB2A0(v203);
        v112 = sub_B35180((__int64)&v197, v111, 6u, (__int64)v180, 1u, (__int64)v183, (__int64)&v192);
        LOWORD(v186) = 257;
        v113 = v112;
        v114 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v204 + 2))(v204, 28, v163, v112);
        v102 = v137;
        v103 = v138;
        if ( !v114 )
        {
          v196 = 257;
          v169 = sub_B504D0(28, v163, v113, (__int64)&v192, 0, 0);
          (*(void (__fastcall **)(_QWORD *, __int64, unsigned int ***, __int64 *, __int64))(*v205 + 16LL))(
            v205,
            v169,
            &v183,
            v201,
            v202);
          v114 = v169;
          if ( v197 != &v197[4 * (unsigned int)n] )
          {
            v170 = i;
            v132 = v197;
            v133 = &v197[4 * (unsigned int)n];
            v134 = v114;
            do
            {
              v135 = *((_QWORD *)v132 + 1);
              v136 = *v132;
              v132 += 4;
              sub_B99FD0(v134, v136, v135);
            }
            while ( v133 != v132 );
            i = v170;
            v114 = v134;
          }
        }
        v104 = v114;
      }
      v105 = (unsigned int)v190;
      v106 = (unsigned int)v190 + 1LL;
      if ( v106 > HIDWORD(v190) )
      {
        v168 = v104;
        sub_C8D5F0((__int64)&v189, v191, v106, 0x10u, v102, v103);
        v105 = (unsigned int)v190;
        v104 = v168;
      }
      v107 = (__int64 **)&v189[2 * v105];
      v107[1] = (__int64 *)v104;
      *v107 = src;
      LODWORD(v190) = v190 + 1;
    }
LABEL_181:
    nullsub_61();
    v212 = (__int64 *)&unk_49D94D0;
    nullsub_63();
    if ( v197 != (unsigned int *)v199 )
      _libc_free((unsigned __int64)v197);
    for ( i = (__int64 *)i[1]; ; i = *(__int64 **)(v4 + 32) )
    {
      v108 = (__int64 *)(v4 - 24);
      if ( !v4 )
        v108 = 0;
      if ( i != v108 + 6 )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( v6 == (__int64 *)v4 )
        goto LABEL_9;
      if ( !v4 )
        BUG();
    }
  }
LABEL_9:
  LOBYTE(v179[0]) = 0;
  dest = v179;
  v178 = 0;
  if ( a4[2] )
  {
    v192 = 34;
    v197 = (unsigned int *)v199;
    v122 = sub_22409D0((__int64)&v197, (unsigned __int64 *)&v192, 0);
    v197 = (unsigned int *)v122;
    v199[0] = v192;
    *(__m128i *)v122 = _mm_load_si128((const __m128i *)&xmmword_4380220);
    si128 = _mm_load_si128((const __m128i *)&xmmword_4380230);
    *(_WORD *)(v122 + 32) = 29540;
    *(__m128i *)(v122 + 16) = si128;
    n = v192;
    *((_BYTE *)v197 + v192) = 0;
    if ( *a4 )
    {
      if ( 0x3FFFFFFFFFFFFFFFLL - n <= 7 )
        goto LABEL_268;
      sub_2241490((unsigned __int64 *)&v197, "_minimal", 8u);
    }
    if ( a4[1] )
      goto LABEL_230;
    if ( 0x3FFFFFFFFFFFFFFFLL - n > 5 )
    {
      sub_2241490((unsigned __int64 *)&v197, "_abort", 6u);
LABEL_230:
      v124 = (unsigned int *)dest;
      v125 = n;
      if ( v197 == (unsigned int *)v199 )
      {
        if ( n )
        {
          if ( n == 1 )
            *(_BYTE *)dest = v199[0];
          else
            memcpy(dest, v199, n);
          v125 = n;
          v124 = (unsigned int *)dest;
        }
        v178 = v125;
        *((_BYTE *)v124 + v125) = 0;
        v124 = v197;
        goto LABEL_234;
      }
      if ( dest == v179 )
      {
        dest = v197;
        v178 = n;
        v179[0] = v199[0];
      }
      else
      {
        v126 = v179[0];
        dest = v197;
        v178 = n;
        v179[0] = v199[0];
        if ( v124 )
        {
          v197 = v124;
          v199[0] = v126;
LABEL_234:
          n = 0;
          *(_BYTE *)v124 = 0;
          if ( v197 != (unsigned int *)v199 )
            j_j___libc_free_0((unsigned __int64)v197);
          v9 = v189;
          v44 = v190;
          v127 = 2LL * (unsigned int)v190;
          v161 = &v189[v127];
          if ( &v189[v127] != v189 )
            goto LABEL_11;
LABEL_70:
          LOBYTE(v4) = v44 != 0;
          if ( dest != v179 )
            j_j___libc_free_0((unsigned __int64)dest);
          v9 = v189;
          goto LABEL_73;
        }
      }
      v197 = (unsigned int *)v199;
      v124 = (unsigned int *)v199;
      goto LABEL_234;
    }
LABEL_268:
    sub_4262D8((__int64)"basic_string::append");
  }
  v9 = v189;
  v161 = &v189[2 * (unsigned int)v190];
  if ( v189 != v161 )
  {
LABEL_11:
    v174 = v9;
    v162 = 0;
    v10 = (__int64 *)&unk_49D94D0;
    while ( 1 )
    {
      v4 = 0;
      v11 = *v174;
      v184[0] = v171;
      v183 = (unsigned int **)v10;
      v12 = *(_QWORD *)(v11 + 40);
      v193 = (char *)v171;
      if ( v11 )
        v4 = v11 + 24;
      v192 = (signed __int64)v10;
      v13 = (__int64 *)sub_AA48A0(v12);
      v200 = v12;
      v203 = v13;
      v197 = (unsigned int *)v199;
      v204 = (void **)&v212;
      n = 0x200000000LL;
      v205 = v214;
      v206 = 0;
      v213 = (unsigned __int64)v193;
      v207 = 0;
      v208 = 512;
      v214[0] = &unk_49DA0B0;
      v209 = 7;
      v210 = 0;
      v211 = 0;
      v212 = v10;
      v201 = (__int64 *)v4;
      LOWORD(v202) = 0;
      if ( v4 != v12 + 48 )
      {
        if ( v4 )
          v4 -= 24;
        v14 = *(_QWORD *)sub_B46C60(v4);
        v180[0] = v14;
        if ( v14 && (v4 = (__int64)v180, sub_B96E90((__int64)v180, v14, 1), (v17 = v180[0]) != 0) )
        {
          v18 = v197;
          v19 = n;
          v20 = &v197[4 * (unsigned int)n];
          if ( v197 != v20 )
          {
            while ( *v18 )
            {
              v18 += 4;
              if ( v20 == v18 )
                goto LABEL_111;
            }
            *((_QWORD *)v18 + 1) = v180[0];
            goto LABEL_24;
          }
LABEL_111:
          if ( (unsigned int)n >= (unsigned __int64)HIDWORD(n) )
          {
            v4 = v139 & 0xFFFFFFFF00000000LL;
            v139 &= 0xFFFFFFFF00000000LL;
            if ( HIDWORD(n) < (unsigned __int64)(unsigned int)n + 1 )
            {
              sub_C8D5F0((__int64)&v197, v199, (unsigned int)n + 1LL, 0x10u, v15, v16);
              v20 = &v197[4 * (unsigned int)n];
            }
            *(_QWORD *)v20 = v4;
            *((_QWORD *)v20 + 1) = v17;
            v17 = v180[0];
            LODWORD(n) = n + 1;
          }
          else
          {
            if ( v20 )
            {
              *v20 = 0;
              *((_QWORD *)v20 + 1) = v17;
              v19 = n;
              v17 = v180[0];
            }
            LODWORD(n) = v19 + 1;
          }
        }
        else
        {
          sub_93FB40((__int64)&v197, 0);
          v17 = v180[0];
        }
        if ( v17 )
        {
          v4 = (__int64)v180;
LABEL_24:
          sub_B91220((__int64)v180, v17);
        }
      }
      v192 = (signed __int64)v10;
      nullsub_63();
      v22 = sub_B9C770(v203, 0, 0, 0, 1);
      if ( v22 )
      {
        v23 = v197;
        v24 = n;
        v25 = &v197[4 * (unsigned int)n];
        if ( v197 == v25 )
        {
LABEL_94:
          if ( (unsigned int)n >= (unsigned __int64)HIDWORD(n) )
          {
            v121 = (unsigned int)n + 1LL;
            v4 = v141 & 0xFFFFFFFF00000000LL | 0x1F;
            v141 = v4;
            if ( HIDWORD(n) < v121 )
            {
              sub_C8D5F0((__int64)&v197, v199, v121, 0x10u, v121, v21);
              v25 = &v197[4 * (unsigned int)n];
            }
            *(_QWORD *)v25 = v4;
            *((_QWORD *)v25 + 1) = v22;
            LODWORD(n) = n + 1;
          }
          else
          {
            if ( v25 )
            {
              *v25 = 31;
              *((_QWORD *)v25 + 1) = v22;
              v24 = n;
            }
            LODWORD(n) = v24 + 1;
          }
        }
        else
        {
          while ( *v23 != 31 )
          {
            v23 += 4;
            if ( v25 == v23 )
              goto LABEL_94;
          }
          *((_QWORD *)v23 + 1) = v22;
        }
      }
      else
      {
        sub_93FB40((__int64)&v197, 31);
      }
      v183 = (unsigned int **)v10;
      nullsub_63();
      v26 = v174[1];
      v167 = v26;
      if ( v26 && *(_BYTE *)v26 == 17 )
      {
        v27 = *(_QWORD **)(v26 + 24);
        if ( *(_DWORD *)(v26 + 32) > 0x40u )
          v27 = (_QWORD *)*v27;
        if ( !v27 )
          goto LABEL_66;
        v166 = v174[1];
      }
      else
      {
        v166 = 0;
      }
      if ( !v201 )
        BUG();
      v28 = (_QWORD *)v201[2];
      v196 = 257;
      v159 = sub_AA8550(v28, v201, (unsigned __int16)v202, (__int64)&v192, 0);
      v29 = v28[6] & 0xFFFFFFFFFFFFFFF8LL;
      if ( (_QWORD *)v29 == v28 + 6 )
      {
        v31 = 0;
      }
      else
      {
        if ( !v29 )
          BUG();
        v30 = *(unsigned __int8 *)(v29 - 24);
        v31 = (_QWORD *)(v29 - 24);
        if ( (unsigned int)(v30 - 30) >= 0xB )
          v31 = 0;
      }
      sub_B43D60(v31);
      v32 = *(_QWORD *)(v200 + 72);
      sub_B33910(&v175, (__int64 *)&v197);
      v183 = &v197;
      v184[0] = 0;
      v185 = v200;
      v184[1] = 0;
      if ( v200 != 0 && v200 != -4096 && v200 != -8192 )
        sub_BD73F0((__int64)v184);
      v186 = v201;
      v187 = v202;
      sub_B33910(v188, (__int64 *)&v197);
      v173 = v162;
      if ( !v162 )
      {
        v192 = (signed __int64)"trap";
        v196 = 259;
        v151 = sub_B2BE50(v32);
        v173 = sub_22077B0(0x50u);
        if ( v173 )
          sub_AA4D50(v173, v151, (__int64)&v192, v32, 0);
        LOWORD(v202) = 0;
        v200 = v173;
        v201 = (__int64 *)(v173 + 48);
        v150 = a4[3];
        v152 = v150 ^ 1;
        if ( a4[2] )
        {
          v148 = dest;
          v144 = a4[1];
          v146 = *(_QWORD *)(v173 + 72);
          v147 = v178;
          v192 = sub_B2BE50(v146);
          v54 = (__int64 *)v192;
          v193 = &v195;
          v194 = 0x800000000LL;
          sub_A77B20((__int64 **)&v192, 41);
          v55 = v146;
          if ( !v144 )
          {
            sub_A77B20((__int64 **)&v192, 36);
            v55 = v146;
          }
          v142 = *(_QWORD *)(v55 + 40);
          v143 = (__int64 *)sub_BCB120(v54);
          v145 = sub_A7B020(v54, -1, (__int64)&v192);
          v180[0] = (__int64)v181;
          v180[1] = 0;
          v56 = sub_BCF480(v143, v181, 0, 0);
          v58 = sub_BA8C10(v142, (__int64)v148, v147, v56, v145);
          if ( (_BYTE *)v180[0] != v181 )
          {
            v149 = v57;
            _libc_free(v180[0]);
            v57 = v149;
          }
          v182 = 257;
          v59 = sub_921880(&v197, v58, v57, 0, 0, (__int64)v180, 0);
          if ( v193 != &v195 )
            _libc_free((unsigned __int64)v193);
          v60 = (__int64 *)(v59 + 72);
          v61 = (__int64 *)(v59 + 72);
          if ( !v152 )
          {
LABEL_125:
            v62 = (__int64 *)sub_BD5C60(v59);
            v63 = sub_A7A090(v61, v62, -1, 41);
            v64 = v175;
            v65 = (__int64 **)(v59 + 48);
            *(_QWORD *)(v59 + 72) = v63;
            v192 = (signed __int64)v64;
            if ( v64 )
            {
              sub_B96E90((__int64)&v192, (__int64)v64, 1);
              v65 = (__int64 **)(v59 + 48);
              if ( (signed __int64 *)(v59 + 48) == &v192 )
              {
                if ( v192 )
                  sub_B91220((__int64)&v192, v192);
                goto LABEL_129;
              }
              v82 = *(_QWORD *)(v59 + 48);
              if ( !v82 )
              {
LABEL_152:
                v83 = (unsigned __int8 *)v192;
                *(_QWORD *)(v59 + 48) = v192;
                if ( v83 )
                  sub_B976B0((__int64)&v192, v83, (__int64)v65);
                goto LABEL_129;
              }
            }
            else if ( v65 == (__int64 **)&v192 || (v82 = *(_QWORD *)(v59 + 48)) == 0 )
            {
LABEL_129:
              if ( a4[2] && a4[1] )
              {
                v196 = 257;
                v115 = sub_BD2C40(72, 1u);
                v116 = (__int64)v115;
                if ( v115 )
                  sub_B4C8F0((__int64)v115, v159, 1u, 0, 0);
                (*(void (__fastcall **)(_QWORD *, __int64, signed __int64 *, __int64 *, __int64))(*v205 + 16LL))(
                  v205,
                  v116,
                  &v192,
                  v201,
                  v202);
                if ( v197 != &v197[4 * (unsigned int)n] )
                {
                  v155 = v10;
                  v117 = v197;
                  v118 = &v197[4 * (unsigned int)n];
                  do
                  {
                    v119 = *((_QWORD *)v117 + 1);
                    v120 = *v117;
                    v117 += 4;
                    sub_B99FD0(v116, v120, v119);
                  }
                  while ( v118 != v117 );
                  v10 = v155;
                }
              }
              else
              {
                v66 = (__int64 *)sub_BD5C60(v59);
                *(_QWORD *)(v59 + 72) = sub_A7A090(v61, v66, -1, 36);
                v196 = 257;
                v67 = sub_BD2C40(72, unk_3F148B8);
                v68 = (__int64)v67;
                if ( v67 )
                  sub_B4C8A0((__int64)v67, (__int64)v203, 0, 0);
                (*(void (__fastcall **)(_QWORD *, __int64, signed __int64 *, __int64 *, __int64))(*v205 + 16LL))(
                  v205,
                  v68,
                  &v192,
                  v201,
                  v202);
                v69 = 4LL * (unsigned int)n;
                if ( v197 != &v197[v69] )
                {
                  v153 = v10;
                  v70 = &v197[v69];
                  v71 = v197;
                  do
                  {
                    v72 = *((_QWORD *)v71 + 1);
                    v73 = *v71;
                    v71 += 4;
                    sub_B99FD0(v68, v73, v72);
                  }
                  while ( v70 != v71 );
                  v10 = v153;
                }
                if ( byte_4FE2748 )
                {
                  v74 = 0;
                  if ( v150 )
                    v74 = v173;
                  v162 = v74;
                }
              }
              goto LABEL_46;
            }
            v154 = v65;
            sub_B91220((__int64)v65, v82);
            v65 = v154;
            goto LABEL_152;
          }
        }
        else
        {
          if ( !v152 )
          {
            HIDWORD(v180[0]) = 0;
            v196 = 257;
            v59 = sub_B33D10((__int64)&v197, 0x162u, 0, 0, 0, 0, LODWORD(v180[0]), (__int64)&v192);
            v61 = (__int64 *)(v59 + 72);
            goto LABEL_125;
          }
          HIDWORD(v180[0]) = 0;
          v196 = 257;
          v75 = (char)a4[4];
          if ( !a4[5] )
          {
            v76 = *(_QWORD *)(v173 + 72);
            v77 = v76 + 72;
            v78 = *(_QWORD *)(v76 + 80);
            if ( v77 == v78 )
            {
              v75 = 0;
            }
            else
            {
              v79 = 0;
              do
              {
                v78 = *(_QWORD *)(v78 + 8);
                ++v79;
              }
              while ( v77 != v78 );
              v75 = v79;
            }
          }
          v80 = sub_BCB2B0(v203);
          v176 = sub_ACD640(v80, v75, 0);
          v59 = sub_B33D10((__int64)&v197, 0x169u, 0, 0, (int)&v176, 1, v180[0], (__int64)&v192);
          v60 = (__int64 *)(v59 + 72);
        }
        v61 = v60;
        v81 = (__int64 *)sub_BD5C60(v59);
        *(_QWORD *)(v59 + 72) = sub_A7A090(v61, v81, -1, 32);
        goto LABEL_125;
      }
LABEL_46:
      v33 = (__int64)v183;
      if ( v185 )
      {
        v34 = v157;
        LOWORD(v34) = v187;
        v157 = v34;
        sub_A88F30((__int64)v183, v185, (__int64)v186, v187);
        v33 = (__int64)v183;
      }
      else
      {
        v183[6] = 0;
        *(_QWORD *)(v33 + 56) = 0;
        *(_WORD *)(v33 + 64) = 0;
      }
      v192 = v188[0];
      if ( !v188[0] || (sub_B96E90((__int64)&v192, v188[0], 1), (v37 = v192) == 0) )
      {
        sub_93FB40(v33, 0);
        v37 = v192;
        goto LABEL_99;
      }
      v38 = *(unsigned int *)(v33 + 8);
      v39 = *(unsigned __int64 **)v33;
      v40 = *(_DWORD *)(v33 + 8);
      v41 = (unsigned __int64 *)(*(_QWORD *)v33 + 16 * v38);
      if ( *(unsigned __int64 **)v33 == v41 )
      {
LABEL_102:
        v50 = *(unsigned int *)(v33 + 12);
        if ( v38 >= v50 )
        {
          v128 = v140 & 0xFFFFFFFF00000000LL;
          v140 &= 0xFFFFFFFF00000000LL;
          if ( v50 < v38 + 1 )
          {
            v156 = v128;
            sub_C8D5F0(v33, (const void *)(v33 + 16), v38 + 1, 0x10u, v35, v36);
            v128 = v156;
            v41 = (unsigned __int64 *)(*(_QWORD *)v33 + 16LL * *(unsigned int *)(v33 + 8));
          }
          *v41 = v128;
          v41[1] = v37;
          ++*(_DWORD *)(v33 + 8);
          v37 = v192;
        }
        else
        {
          if ( v41 )
          {
            *(_DWORD *)v41 = 0;
            v41[1] = v37;
            v40 = *(_DWORD *)(v33 + 8);
            v37 = v192;
          }
          *(_DWORD *)(v33 + 8) = v40 + 1;
        }
LABEL_99:
        if ( !v37 )
          goto LABEL_56;
        goto LABEL_55;
      }
      while ( *(_DWORD *)v39 )
      {
        v39 += 2;
        if ( v41 == v39 )
          goto LABEL_102;
      }
      v39[1] = v192;
LABEL_55:
      sub_B91220((__int64)&v192, v37);
LABEL_56:
      if ( v188[0] )
        sub_B91220((__int64)v188, v188[0]);
      if ( v185 != 0 && v185 != -4096 && v185 != -8192 )
        sub_BD60C0(v184);
      if ( v175 )
        sub_B91220((__int64)&v175, (__int64)v175);
      if ( v166 )
      {
        sub_B43C20((__int64)&v192, (__int64)v28);
        v4 = v192;
        v42 = (unsigned __int16)v193;
        v43 = sub_BD2C40(72, 1u);
        if ( v43 )
          sub_B4C8F0((__int64)v43, v173, 1u, v4, v42);
      }
      else
      {
        sub_B43C20((__int64)&v192, (__int64)v28);
        v51 = v192;
        v4 = (unsigned __int16)v193;
        v52 = sub_BD2C40(72, 3u);
        if ( v52 )
          sub_B4C9A0((__int64)v52, v173, v159, v167, 3u, v53, v51, v4);
      }
LABEL_66:
      nullsub_61();
      v212 = v10;
      nullsub_63();
      if ( v197 != (unsigned int *)v199 )
        _libc_free((unsigned __int64)v197);
      v174 += 2;
      if ( v161 == v174 )
      {
        v44 = v190;
        goto LABEL_70;
      }
    }
  }
  LOBYTE(v4) = (_DWORD)v190 != 0;
LABEL_73:
  if ( v9 != (__int64 *)v191 )
    _libc_free((unsigned __int64)v9);
  if ( !v225 )
    _libc_free(v224);
  if ( !v223 )
    _libc_free(v222);
  v45 = v221;
  if ( v221 )
  {
    v46 = v220;
    v47 = &v220[7 * v221];
    do
    {
      if ( *v46 != -8192 && *v46 != -4096 )
      {
        v48 = v46[6];
        if ( v48 != -4096 && v48 != 0 && v48 != -8192 )
          sub_BD60C0(v46 + 4);
        v49 = v46[3];
        if ( v49 != -4096 && v49 != 0 && v49 != -8192 )
          sub_BD60C0(v46 + 1);
      }
      v46 += 7;
    }
    while ( v47 != v46 );
    v45 = v221;
  }
  sub_C7D6A0((__int64)v220, 56 * v45, 8);
  sub_B32BF0(v219);
  v218 = &unk_49D94D0;
  nullsub_63();
  if ( v216 != &v217 )
    _libc_free((unsigned __int64)v216);
  return (unsigned int)v4;
}
