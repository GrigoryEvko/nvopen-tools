// Function: sub_905EE0
// Address: 0x905ee0
//
__int64 __fastcall sub_905EE0(
        char *a1,
        __int64 a2,
        _QWORD *a3,
        char *a4,
        unsigned __int8 a5,
        char **a6,
        __int64 *a7,
        unsigned int *a8,
        char a9,
        char a10,
        char a11,
        char a12)
{
  __int64 v13; // r13
  char *v14; // r15
  __int64 v15; // r12
  __int64 v16; // rbx
  __int64 (__fastcall *v17)(__int64 *); // r13
  __int64 v18; // rcx
  unsigned int v19; // r13d
  __int64 (__fastcall *v20)(__int64, __int64, __int64, _QWORD); // r13
  __int64 v21; // rcx
  unsigned int v22; // r8d
  const char **v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // r12
  __int64 v30; // rbx
  __int64 v31; // rax
  int v32; // r10d
  const char **v33; // r11
  __int64 v34; // r12
  int v35; // ebx
  __int64 v36; // r12
  int v37; // ebx
  void (__fastcall *v38)(_QWORD *, __int64); // rax
  void (__fastcall *v39)(__int64, __int64); // rax
  __int64 (__fastcall *v40)(__int64, _QWORD, _QWORD); // rax
  __int64 (__fastcall *v41)(__int64, unsigned __int64 *); // rax
  __int64 v42; // rcx
  unsigned int v43; // ebx
  __int64 v44; // rax
  const char *v45; // rbx
  __int64 v46; // r12
  size_t v47; // rax
  _BYTE *v48; // rdi
  size_t v49; // r13
  _BYTE *v50; // rax
  char v51; // bl
  __int64 v52; // rax
  const char *v53; // r12
  size_t v54; // rdx
  char *v55; // r13
  size_t v56; // rax
  size_t v57; // r12
  __int64 result; // rax
  size_t v59; // rax
  size_t v60; // rax
  __int64 v61; // rbx
  __int64 v62; // r15
  __int64 v63; // rax
  const char *v64; // rdi
  size_t v65; // rax
  __int64 v66; // rax
  __int64 v67; // r13
  __int64 v68; // rax
  char *v69; // rdx
  char v70; // al
  size_t v71; // rax
  __int64 v72; // r12
  __int64 v73; // r12
  unsigned int v74; // ebx
  __int64 v75; // r13
  size_t v76; // rax
  __int64 v77; // rax
  __int64 v78; // rax
  const char *v79; // r12
  size_t v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // rcx
  __int64 v83; // rbx
  __int64 v84; // rsi
  int v85; // ecx
  int v86; // r8d
  int v87; // r9d
  char v88; // al
  size_t v89; // rax
  __int64 v90; // r12
  _QWORD *v91; // r12
  __int64 v92; // rbx
  unsigned int v93; // eax
  __int64 v94; // rcx
  const char *v95; // r12
  size_t v96; // rdx
  __int64 v97; // rcx
  __int64 v98; // rcx
  char *v99; // rsi
  unsigned int v100; // r13d
  size_t v101; // rax
  __int64 v102; // rax
  __int64 v103; // rax
  __int64 v104; // rdx
  const char *v105; // r12
  __int64 (__fastcall *v106)(__int64, const char *); // rax
  size_t v107; // rax
  void *v108; // rsi
  const char *v109; // r12
  size_t v110; // rdx
  __int64 v111; // rcx
  __int64 v112; // rcx
  __int64 (__fastcall *v113)(__int64, unsigned __int64 *); // rax
  __int64 v114; // rcx
  unsigned int v115; // ebx
  size_t v116; // rdx
  __int64 v117; // rcx
  __int64 v118; // rcx
  size_t v119; // rax
  __int64 v120; // rax
  size_t v121; // rdx
  __int64 v122; // rcx
  size_t v123; // rdx
  __int64 v124; // rcx
  __int64 v125; // rcx
  __int64 v126; // r12
  __int64 (__fastcall *v127)(__int64, __int64); // rax
  __int64 v128; // rcx
  size_t v129; // rax
  __int64 v130; // r12
  __int64 v131; // rax
  char *v132; // rsi
  size_t v133; // rax
  _DWORD *v134; // rdi
  size_t v135; // r13
  __int64 v136; // r12
  __m128i *v137; // rax
  __m128i si128; // xmm0
  size_t v139; // rax
  size_t v140; // r13
  unsigned __int64 v141; // rax
  __int64 v142; // r12
  _BYTE *v143; // rax
  size_t v144; // rax
  _WORD *v145; // rdi
  size_t v146; // r13
  __int64 v147; // rsi
  size_t v148; // rax
  _BYTE *v149; // rdi
  size_t v150; // r13
  __int64 v151; // rbx
  int v152; // r12d
  const void *v153; // r15
  size_t v154; // rax
  size_t v155; // r14
  __int64 v156; // r13
  _BYTE *v157; // rax
  __int64 v158; // rdi
  _BYTE *v159; // rax
  __m128i *v160; // rax
  __m128i v161; // xmm0
  __int64 v162; // rdi
  __int64 v163; // rax
  __int64 v164; // rax
  __int64 v165; // rax
  void (__fastcall *v166)(__int64 *); // [rsp+8h] [rbp-368h]
  int v167; // [rsp+10h] [rbp-360h]
  void (__fastcall *v168)(__int64, __int64 (__fastcall *)(__int64, __int64, __int64), char **, __int64); // [rsp+10h] [rbp-360h]
  __int64 (__fastcall *v169)(_QWORD); // [rsp+20h] [rbp-350h]
  char *src; // [rsp+40h] [rbp-330h]
  _QWORD *srca; // [rsp+40h] [rbp-330h]
  __int64 v172; // [rsp+48h] [rbp-328h]
  char *v176; // [rsp+68h] [rbp-308h]
  char *v177; // [rsp+68h] [rbp-308h]
  char *v178; // [rsp+68h] [rbp-308h]
  char *v179; // [rsp+68h] [rbp-308h]
  char *v180; // [rsp+68h] [rbp-308h]
  char *v181; // [rsp+68h] [rbp-308h]
  char *v182; // [rsp+68h] [rbp-308h]
  __int64 v184; // [rsp+70h] [rbp-300h]
  __int64 v185; // [rsp+98h] [rbp-2D8h] BYREF
  unsigned __int64 v186; // [rsp+A0h] [rbp-2D0h] BYREF
  unsigned __int64 v187; // [rsp+A8h] [rbp-2C8h] BYREF
  __int64 v188; // [rsp+B0h] [rbp-2C0h] BYREF
  __int64 v189; // [rsp+B8h] [rbp-2B8h] BYREF
  __int64 v190; // [rsp+C0h] [rbp-2B0h] BYREF
  __int64 v191; // [rsp+C8h] [rbp-2A8h] BYREF
  _QWORD *v192; // [rsp+D0h] [rbp-2A0h] BYREF
  __int64 v193; // [rsp+D8h] [rbp-298h]
  _QWORD v194[2]; // [rsp+E0h] [rbp-290h] BYREF
  _QWORD v195[2]; // [rsp+F0h] [rbp-280h] BYREF
  _QWORD v196[2]; // [rsp+100h] [rbp-270h] BYREF
  __int64 v197[4]; // [rsp+110h] [rbp-260h] BYREF
  _QWORD v198[2]; // [rsp+130h] [rbp-240h] BYREF
  __int64 v199; // [rsp+140h] [rbp-230h] BYREF
  __int64 v200; // [rsp+150h] [rbp-220h] BYREF
  __int64 v201; // [rsp+158h] [rbp-218h]
  __int64 v202; // [rsp+160h] [rbp-210h] BYREF
  void *v203; // [rsp+170h] [rbp-200h] BYREF
  __int64 v204; // [rsp+178h] [rbp-1F8h]
  __int64 v205; // [rsp+180h] [rbp-1F0h]
  __int64 v206; // [rsp+188h] [rbp-1E8h]
  __int64 v207; // [rsp+190h] [rbp-1E0h]
  __int64 v208; // [rsp+198h] [rbp-1D8h]
  __int64 *v209; // [rsp+1A0h] [rbp-1D0h]
  char *v210; // [rsp+1B0h] [rbp-1C0h] BYREF
  __int64 v211; // [rsp+1B8h] [rbp-1B8h]
  char *v212; // [rsp+1C0h] [rbp-1B0h] BYREF
  char *v213; // [rsp+1C8h] [rbp-1A8h]
  __int64 v214; // [rsp+1D0h] [rbp-1A0h]
  int v215; // [rsp+1D8h] [rbp-198h] BYREF
  int v216; // [rsp+1DCh] [rbp-194h] BYREF
  char v217[4]; // [rsp+1E0h] [rbp-190h] BYREF
  int v218; // [rsp+1E4h] [rbp-18Ch] BYREF
  __int64 v219; // [rsp+1E8h] [rbp-188h] BYREF
  __int64 v220; // [rsp+1F0h] [rbp-180h] BYREF
  char v221[8]; // [rsp+1F8h] [rbp-178h] BYREF
  __int64 v222; // [rsp+200h] [rbp-170h] BYREF
  char v223[4]; // [rsp+208h] [rbp-168h] BYREF
  char v224[4]; // [rsp+20Ch] [rbp-164h] BYREF
  _QWORD v225[44]; // [rsp+210h] [rbp-160h] BYREF

  v13 = a2;
  if ( a11 != 1 && a9 )
  {
    sub_223E0D0(qword_4FD4BE0, "\"", 1);
    if ( a1 )
    {
      v59 = strlen(a1);
      sub_223E0D0(qword_4FD4BE0, a1, v59);
    }
    else
    {
      sub_222DC80(
        (char *)qword_4FD4BE0 + *(_QWORD *)(qword_4FD4BE0[0] - 24LL),
        *(_DWORD *)((char *)&qword_4FD4BE0[4] + *(_QWORD *)(qword_4FD4BE0[0] - 24LL)) | 1u);
    }
    sub_223E0D0(qword_4FD4BE0, "\" -libnvvm ", 11);
    if ( a4 )
    {
      sub_223E0D0(qword_4FD4BE0, "-nvvmir-library \"", 17);
      v60 = strlen(a4);
      sub_223E0D0(qword_4FD4BE0, a4, v60);
      sub_223E0D0(qword_4FD4BE0, "\" ", 2);
    }
    v61 = 0;
    sub_223E0D0(qword_4FD4BE0, "\"", 1);
    v62 = sub_223E0D0(qword_4FD4BE0, *a3, a3[1]);
    sub_223E0D0(v62, "\" -o \"", 6);
    v63 = sub_223E0D0(v62, a3[4], a3[5]);
    sub_223E0D0(v63, "\"", 1);
    if ( *((int *)a3 + 17) > 0 )
    {
      do
      {
        while ( 1 )
        {
          sub_223E0D0(qword_4FD4BE0, " ", 1);
          v66 = a3[9];
          v67 = *(_QWORD *)(v66 + 8 * v61);
          if ( !v67 )
            break;
          v64 = *(const char **)(v66 + 8 * v61++);
          v65 = strlen(v64);
          sub_223E0D0(qword_4FD4BE0, v67, v65);
          if ( *((_DWORD *)a3 + 17) <= (int)v61 )
            goto LABEL_79;
        }
        ++v61;
        sub_222DC80(
          (char *)qword_4FD4BE0 + *(_QWORD *)(qword_4FD4BE0[0] - 24LL),
          *(_DWORD *)((char *)&qword_4FD4BE0[4] + *(_QWORD *)(qword_4FD4BE0[0] - 24LL)) | 1u);
      }
      while ( *((_DWORD *)a3 + 17) > (int)v61 );
LABEL_79:
      v13 = a2;
    }
    sub_223E0D0(qword_4FD4BE0, " -keep -v\n", 10);
  }
  if ( a11 || a12 )
  {
    v14 = 0;
    *a8 = 0;
    v184 = sub_C996C0("LibNVVM", 7, *a3, a3[1]);
    if ( a12 )
      goto LABEL_41;
LABEL_7:
    v195[0] = v196;
    v208 = 0x100000000LL;
    v192 = v194;
    v203 = &unk_49DD210;
    v209 = v195;
    v193 = 0;
    LOBYTE(v194[0]) = 0;
    v195[1] = 0;
    LOBYTE(v196[0]) = 0;
    v204 = 0;
    v205 = 0;
    v206 = 0;
    v207 = 0;
    sub_CB5980(&v203, 0, 0, 0);
    sub_A3ACE0(v13, &v203, 1, 0, 0, 0);
    v15 = *v209;
    v16 = v209[1];
    v167 = *(_DWORD *)(sub_AE2980(v13 + 312, 0) + 4);
    if ( v13 )
    {
      sub_BA9C10(v13);
      j_j___libc_free_0(v13, 880);
    }
    a6[1] = 0;
    **a6 = 0;
    v17 = (__int64 (__fastcall *)(__int64 *))sub_12BC0F0(2151);
    v169 = (__int64 (__fastcall *)(_QWORD))sub_12BC0F0(46967);
    v19 = v17(&v185);
    if ( v19 )
    {
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v193) <= 0xF )
        goto LABEL_304;
      sub_2241490(&v192, "libnvvm: error: ", 16, v18);
      v79 = (const char *)v169(v19);
      v80 = strlen(v79);
      if ( v80 > 0x3FFFFFFFFFFFFFFFLL - v193 )
        goto LABEL_304;
      sub_2241490(&v192, v79, v80, v81);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v193) <= 0x2F )
        goto LABEL_304;
      sub_2241490(&v192, ": failed to create the libnvvm compilation unit\n", 48, v82);
    }
    else
    {
      v20 = (__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD))sub_12BC0F0(4660);
      v166 = (void (__fastcall *)(__int64 *))sub_12BC0F0(21257);
      v19 = v20(v185, v15, v16, 0);
      if ( !v19 )
      {
        v22 = *((_DWORD *)a3 + 17);
        v23 = (const char **)a3[9];
        if ( v22 && (!strcmp(*v23, "-opt") || !strcmp(*v23, "-llc")) )
          goto LABEL_13;
        sub_8FD2C0((__int64)v225, v22, (__int64)v23, v167 == 64);
        v91 = (_QWORD *)v225[0];
        v92 = v225[0] + 16LL * LODWORD(v225[1]);
        if ( v225[0] == v92 )
        {
LABEL_149:
          if ( a4 )
          {
            LOWORD(v214) = 257;
            if ( *a4 )
            {
              v210 = a4;
              LOBYTE(v214) = 3;
            }
            sub_C7EAD0(&v200, &v210, 0, 1, 0);
            if ( (v202 & 1) != 0 )
            {
              v100 = v200;
              if ( (_DWORD)v200 )
              {
                v172 = v201;
                sub_223E0D0(qword_4FD4BE0, "error in open: ", 15);
                v101 = strlen(a4);
                sub_223E0D0(qword_4FD4BE0, a4, v101);
                sub_223E0D0(qword_4FD4BE0, "\n", 1);
                (*(void (__fastcall **)(char **, __int64, _QWORD))(*(_QWORD *)v172 + 32LL))(&v210, v172, v100);
                v102 = sub_223E0D0(qword_4FD4BE0, v210, v211);
                v99 = "\n";
                sub_223E0D0(v102, "\n", 1);
                if ( v210 != (char *)&v212 )
                {
                  v99 = v212 + 1;
                  j_j___libc_free_0(v210, v212 + 1);
                }
                v166(&v185);
                if ( (v202 & 1) == 0 && v200 )
                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v200 + 8LL))(v200);
                v19 = 4;
                goto LABEL_158;
              }
            }
            v108 = *(void **)(v200 + 8);
            v19 = sub_12BCB00(v185, v108, *(_QWORD *)(v200 + 16) - (_QWORD)v108, 0);
            if ( (v202 & 1) == 0 && v200 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v200 + 8LL))(v200);
          }
          else
          {
            v108 = &unk_3EA0080;
            v19 = sub_12BCB00(v185, &unk_3EA0080, 455876, 0);
          }
          if ( !v19 )
          {
            if ( (_QWORD *)v225[0] != &v225[2] )
              _libc_free(v225[0], v108);
LABEL_13:
            v24 = *a3;
            v188 = 0;
            v189 = 0;
            v25 = a3[1];
            v190 = 0;
            v191 = 0;
            if ( a10 )
            {
              v26 = sub_904090(v24, v25);
              v29 = sub_904090(v26, v27);
            }
            else
            {
              v103 = sub_C80C90(v24, v25, 0);
              v29 = sub_C80C90(v103, v104, 0);
            }
            v30 = v28;
            LOWORD(v225[4]) = 773;
            v225[1] = v28;
            v225[2] = ".lnk.bc";
            v225[0] = v29;
            sub_CA0F50(v198, v225);
            v225[1] = v30;
            v225[0] = v29;
            LOWORD(v225[4]) = 773;
            v225[2] = ".opt.bc";
            sub_CA0F50(&v200, v225);
            if ( a11 || !a9 && !a10 )
              goto LABEL_31;
            v31 = sub_12BC0F0(48879);
            v32 = *((_DWORD *)a3 + 17);
            v33 = (const char **)a3[9];
            v168 = (void (__fastcall *)(__int64, __int64 (__fastcall *)(__int64, __int64, __int64), char **, __int64))v31;
            v210 = (char *)*a3;
            v211 = a3[4];
            v212 = a1;
            if ( v32 > 0 )
            {
              if ( !strcmp(*v33, "-lnk") )
              {
                sub_2240AE0(v198, a3 + 4);
                v33 = (const char **)a3[9];
                v32 = *((_DWORD *)a3 + 17);
              }
              else if ( !strcmp(*v33, "-opt") )
              {
                sub_2240AE0(v198, a3);
                sub_2240AE0(&v200, a3 + 4);
                v33 = (const char **)a3[9];
                v32 = *((_DWORD *)a3 + 17);
              }
              else if ( !strcmp(*v33, "-llc") )
              {
                sub_2240AE0(&v200, a3);
                v33 = (const char **)a3[9];
                v32 = *((_DWORD *)a3 + 17);
              }
            }
            v213 = (char *)v198[0];
            v214 = v200;
            sub_9685E0(
              v32,
              (_DWORD)v33,
              57069,
              (unsigned int)&v215,
              (unsigned int)&v219,
              (unsigned int)&v216,
              (__int64)&v220,
              (__int64)v217,
              (__int64)v221,
              (__int64)&v218,
              (__int64)&v222,
              (__int64)v223,
              0,
              (__int64)v224);
            v34 = v220;
            v35 = v216;
            if ( v216 != (_DWORD)v188 || v220 != v189 )
            {
              sub_95D500(&v188, &v189);
              LODWORD(v188) = v35;
              v189 = v34;
            }
            v36 = v222;
            v37 = v218;
            if ( v218 != (_DWORD)v190 || v222 != v191 )
            {
              sub_95D500(&v190, &v191);
              LODWORD(v190) = v37;
              v191 = v36;
            }
            if ( !a9 )
              goto LABEL_30;
            v130 = ((__int64 (*)(void))sub_CB7210)();
            v131 = *(_QWORD *)(v130 + 32);
            if ( (unsigned __int64)(*(_QWORD *)(v130 + 24) - v131) <= 2 )
            {
              v130 = sub_CB6200(v130, "[ \"", 3);
            }
            else
            {
              *(_BYTE *)(v131 + 2) = 34;
              *(_WORD *)v131 = 8283;
              *(_QWORD *)(v130 + 32) += 3LL;
            }
            v132 = v212;
            if ( v212 )
            {
              src = v212;
              v133 = strlen(v212);
              v134 = *(_DWORD **)(v130 + 32);
              v132 = src;
              v135 = v133;
              if ( v133 <= *(_QWORD *)(v130 + 24) - (_QWORD)v134 )
              {
                if ( v133 )
                {
                  memcpy(v134, src, v133);
                  v134 = (_DWORD *)(v135 + *(_QWORD *)(v130 + 32));
                  *(_QWORD *)(v130 + 32) = v134;
                }
LABEL_231:
                if ( *(_QWORD *)(v130 + 24) - (_QWORD)v134 <= 6u )
                {
                  v132 = "\" -lnk ";
                  v134 = (_DWORD *)v130;
                  sub_CB6200(v130, "\" -lnk ", 7);
                }
                else
                {
                  *v134 = 1814896674;
                  *((_WORD *)v134 + 2) = 27502;
                  *((_BYTE *)v134 + 6) = 32;
                  *(_QWORD *)(v130 + 32) += 7LL;
                }
                if ( a4 )
                {
                  v136 = sub_CB7210(v134, v132);
                  v137 = *(__m128i **)(v136 + 32);
                  if ( *(_QWORD *)(v136 + 24) - (_QWORD)v137 <= 0x10u )
                  {
                    v132 = "-nvvmir-library \"";
                    v136 = sub_CB6200(v136, "-nvvmir-library \"", 17);
                  }
                  else
                  {
                    si128 = _mm_load_si128((const __m128i *)&xmmword_3F0F5B0);
                    v137[1].m128i_i8[0] = 34;
                    *v137 = si128;
                    *(_QWORD *)(v136 + 32) += 17LL;
                  }
                  v139 = strlen(a4);
                  v134 = *(_DWORD **)(v136 + 32);
                  v140 = v139;
                  v141 = *(_QWORD *)(v136 + 24) - (_QWORD)v134;
                  if ( v140 > v141 )
                  {
                    v132 = a4;
                    v164 = sub_CB6200(v136, a4, v140);
                    v134 = *(_DWORD **)(v164 + 32);
                    v136 = v164;
                    v141 = *(_QWORD *)(v164 + 24) - (_QWORD)v134;
                  }
                  else if ( v140 )
                  {
                    v132 = a4;
                    memcpy(v134, a4, v140);
                    v165 = *(_QWORD *)(v136 + 24);
                    v134 = (_DWORD *)(v140 + *(_QWORD *)(v136 + 32));
                    *(_QWORD *)(v136 + 32) = v134;
                    v141 = v165 - (_QWORD)v134;
                  }
                  if ( v141 <= 1 )
                  {
                    v132 = "\" ";
                    v134 = (_DWORD *)v136;
                    sub_CB6200(v136, "\" ", 2);
                  }
                  else
                  {
                    *(_WORD *)v134 = 8226;
                    *(_QWORD *)(v136 + 32) += 2LL;
                  }
                }
                v142 = sub_CB7210(v134, v132);
                v143 = *(_BYTE **)(v142 + 32);
                if ( *(_BYTE **)(v142 + 24) == v143 )
                {
                  v142 = sub_CB6200(v142, "\"", 1);
                }
                else
                {
                  *v143 = 34;
                  ++*(_QWORD *)(v142 + 32);
                }
                if ( v210 )
                {
                  v180 = v210;
                  v144 = strlen(v210);
                  v145 = *(_WORD **)(v142 + 32);
                  v146 = v144;
                  if ( v144 <= *(_QWORD *)(v142 + 24) - (_QWORD)v145 )
                  {
                    if ( v144 )
                    {
                      memcpy(v145, v180, v144);
                      v145 = (_WORD *)(v146 + *(_QWORD *)(v142 + 32));
                      *(_QWORD *)(v142 + 32) = v145;
                    }
LABEL_247:
                    if ( *(_QWORD *)(v142 + 24) - (_QWORD)v145 <= 5u )
                    {
                      v142 = sub_CB6200(v142, "\" -o \"", 6);
                    }
                    else
                    {
                      *(_DWORD *)v145 = 1865228322;
                      v145[2] = 8736;
                      *(_QWORD *)(v142 + 32) += 6LL;
                    }
                    v147 = (__int64)v213;
                    if ( v213 )
                    {
                      v181 = v213;
                      v148 = strlen(v213);
                      v149 = *(_BYTE **)(v142 + 32);
                      v147 = (__int64)v181;
                      v150 = v148;
                      if ( v148 <= *(_QWORD *)(v142 + 24) - (_QWORD)v149 )
                      {
                        if ( v148 )
                        {
                          memcpy(v149, v181, v148);
                          v149 = (_BYTE *)(v150 + *(_QWORD *)(v142 + 32));
                          *(_QWORD *)(v142 + 32) = v149;
                        }
LABEL_253:
                        if ( v149 == *(_BYTE **)(v142 + 24) )
                        {
                          v147 = (__int64)"\"";
                          v149 = (_BYTE *)v142;
                          sub_CB6200(v142, "\"", 1);
                        }
                        else
                        {
                          *v149 = 34;
                          ++*(_QWORD *)(v142 + 32);
                        }
                        v151 = 8;
                        v152 = 1;
                        if ( v215 > 1 )
                        {
                          v182 = v14;
                          srca = a3;
                          do
                          {
                            v156 = sub_CB7210(v149, v147);
                            v157 = *(_BYTE **)(v156 + 32);
                            if ( (unsigned __int64)v157 < *(_QWORD *)(v156 + 24) )
                            {
                              *(_QWORD *)(v156 + 32) = v157 + 1;
                              *v157 = 32;
                            }
                            else
                            {
                              v149 = (_BYTE *)v156;
                              v147 = 32;
                              v156 = sub_CB5D20(v156, 32);
                            }
                            v153 = *(const void **)(v219 + v151);
                            if ( v153 )
                            {
                              v154 = strlen(*(const char **)(v219 + v151));
                              v149 = *(_BYTE **)(v156 + 32);
                              v155 = v154;
                              if ( v154 > *(_QWORD *)(v156 + 24) - (_QWORD)v149 )
                              {
                                v147 = (__int64)v153;
                                v149 = (_BYTE *)v156;
                                sub_CB6200(v156, v153, v154);
                              }
                              else if ( v154 )
                              {
                                v147 = (__int64)v153;
                                memcpy(v149, v153, v154);
                                *(_QWORD *)(v156 + 32) += v155;
                              }
                            }
                            ++v152;
                            v151 += 8;
                          }
                          while ( v215 > v152 );
                          v14 = v182;
                          a3 = srca;
                        }
                        v158 = sub_CB7210(v149, v147);
                        v159 = *(_BYTE **)(v158 + 32);
                        if ( (unsigned __int64)v159 >= *(_QWORD *)(v158 + 24) )
                        {
                          v147 = 32;
                          v158 = sub_CB5D20(v158, 32);
                        }
                        else
                        {
                          *(_QWORD *)(v158 + 32) = v159 + 1;
                          *v159 = 32;
                        }
                        v160 = *(__m128i **)(v158 + 32);
                        if ( *(_QWORD *)(v158 + 24) - (_QWORD)v160 <= 0x18u )
                        {
                          v147 = (__int64)"-nvvm-version=nvvm-latest";
                          sub_CB6200(v158, "-nvvm-version=nvvm-latest", 25);
                        }
                        else
                        {
                          v161 = _mm_load_si128((const __m128i *)&xmmword_3C23BC0);
                          v160[1].m128i_i8[8] = 116;
                          v160[1].m128i_i64[0] = 0x736574616C2D6D76LL;
                          *v160 = v161;
                          *(_QWORD *)(v158 + 32) += 25LL;
                        }
                        v162 = sub_CB7210(v158, v147);
                        v163 = *(_QWORD *)(v162 + 32);
                        if ( (unsigned __int64)(*(_QWORD *)(v162 + 24) - v163) <= 2 )
                        {
                          sub_CB6200(v162, " ]\n", 3);
                        }
                        else
                        {
                          *(_BYTE *)(v163 + 2) = 10;
                          *(_WORD *)v163 = 23840;
                          *(_QWORD *)(v162 + 32) += 3LL;
                        }
                        v168(v185, sub_903BA0, &v210, 61453);
                        v168(v185, sub_903730, &v210, 47710);
                        if ( a10 )
                        {
LABEL_30:
                          v168(v185, (__int64 (__fastcall *)(__int64, __int64, __int64))sub_9085A0, &v210, 64222);
                          v168(v185, (__int64 (__fastcall *)(__int64, __int64, __int64))sub_908220, &v210, 56993);
                        }
LABEL_31:
                        v38 = (void (__fastcall *)(_QWORD *, __int64))sub_12BC0F0(65261);
                        qmemcpy(v225, off_4B90FE0, 0x128u);
                        v38(v225, 37);
                        v39 = (void (__fastcall *)(__int64, __int64))sub_12BC0F0(48813);
                        v39(v185, 57069);
                        v40 = (__int64 (__fastcall *)(__int64, _QWORD, _QWORD))sub_12BC0F0(17185);
                        v19 = v40(v185, *((unsigned int *)a3 + 17), a3[9]);
                        v41 = (__int64 (__fastcall *)(__int64, unsigned __int64 *))sub_12BC0F0(41856);
                        v43 = v41(v185, &v186);
                        if ( v43 )
                        {
                          if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v193) <= 0xF )
                            goto LABEL_304;
                          sub_2241490(&v192, "libnvvm: error: ", 16, v42);
                          v177 = (char *)v169(v43);
                          v116 = strlen(v177);
                          if ( v116 > 0x3FFFFFFFFFFFFFFFLL - v193 )
                            goto LABEL_304;
                          sub_2241490(&v192, v177, v116, v117);
                          if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v193) <= 0x2A )
                            goto LABEL_304;
                          sub_2241490(&v192, ": failed to get the error/warning messages\n", 43, v118);
                        }
                        else
                        {
                          if ( v186 <= 1 )
                          {
                            if ( v19 )
                            {
LABEL_34:
                              v166(&v185);
LABEL_35:
                              if ( (__int64 *)v200 != &v202 )
                                j_j___libc_free_0(v200, v202 + 1);
                              if ( (__int64 *)v198[0] != &v199 )
                                j_j___libc_free_0(v198[0], v199 + 1);
                              sub_95D500(&v190, &v191);
                              sub_95D500(&v188, &v189);
                              goto LABEL_109;
                            }
LABEL_184:
                            v113 = (__int64 (__fastcall *)(__int64, unsigned __int64 *))sub_12BC0F0(61451);
                            v115 = v113(v185, &v187);
                            if ( v115 )
                            {
                              if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v193) > 0xF )
                              {
                                sub_2241490(&v192, "libnvvm: error: ", 16, v114);
                                v178 = (char *)v169(v115);
                                v123 = strlen(v178);
                                if ( v123 <= 0x3FFFFFFFFFFFFFFFLL - v193 )
                                {
                                  sub_2241490(&v192, v178, v123, v124);
                                  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v193) > 0x1E )
                                  {
                                    sub_2241490(&v192, ": failed to get the PTX output\n", 31, v125);
LABEL_186:
                                    v166(&v185);
                                    if ( !v19 )
                                      v19 = v115;
                                    goto LABEL_35;
                                  }
                                }
                              }
                            }
                            else
                            {
                              if ( v187 <= 1 )
                                goto LABEL_186;
                              v126 = sub_2207820(v187);
                              v127 = (__int64 (__fastcall *)(__int64, __int64))sub_12BC0F0(4111);
                              v115 = v127(v185, v126);
                              if ( v115 )
                              {
                                if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v193) > 0xF )
                                {
                                  sub_2241490(&v192, "libnvvm: error: ", 16, 0x3FFFFFFFFFFFFFFFLL);
                                  v179 = (char *)v169(v115);
                                  v129 = strlen(v179);
                                  if ( v129 <= 0x3FFFFFFFFFFFFFFFLL - v193 )
                                  {
                                    sub_2241490(&v192, v179, v129, 0x3FFFFFFFFFFFFFFFLL);
                                    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v193) > 0x1E )
                                    {
                                      sub_2241490(&v192, ": failed to get the PTX output\n", 31, 0x3FFFFFFFFFFFFFFFLL);
                                      goto LABEL_221;
                                    }
                                  }
                                }
                              }
                              else if ( v187 <= 0x3FFFFFFFFFFFFFFFLL - (__int64)a6[1] )
                              {
                                sub_2241490(a6, v126, v187, v128);
LABEL_221:
                                if ( v126 )
                                  j_j___libc_free_0_0(v126);
                                goto LABEL_186;
                              }
                            }
LABEL_304:
                            sub_4262D8((__int64)"basic_string::append");
                          }
                          v105 = (const char *)sub_2207820(v186);
                          v106 = (__int64 (__fastcall *)(__int64, const char *))sub_12BC0F0(46903);
                          v43 = v106(v185, v105);
                          if ( !v43 )
                          {
                            v121 = strlen(v105);
                            if ( v121 > 0x3FFFFFFFFFFFFFFFLL - v193 )
                              goto LABEL_304;
                            sub_2241490(&v192, v105, v121, v122);
                            j_j___libc_free_0_0(v105);
                            if ( v19 )
                              goto LABEL_34;
                            goto LABEL_184;
                          }
                          if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v193) <= 0xF )
                            goto LABEL_304;
                          sub_2241490(&v192, "libnvvm: error: ", 16, 0x3FFFFFFFFFFFFFFFLL);
                          v176 = (char *)v169(v43);
                          v107 = strlen(v176);
                          if ( v107 > 0x3FFFFFFFFFFFFFFFLL - v193 )
                            goto LABEL_304;
                          sub_2241490(&v192, v176, v107, 0x3FFFFFFFFFFFFFFFLL);
                          if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v193) <= 0x2A )
                            goto LABEL_304;
                          sub_2241490(&v192, ": failed to get the error/warning messages\n", 43, 0x3FFFFFFFFFFFFFFFLL);
                          if ( v105 )
                            j_j___libc_free_0_0(v105);
                        }
                        if ( v19 )
                        {
                          v19 = v43;
                          v166(&v185);
                          goto LABEL_35;
                        }
                        v19 = v43;
                        goto LABEL_184;
                      }
                      v142 = sub_CB6200(v142, v181, v148);
                    }
                    v149 = *(_BYTE **)(v142 + 32);
                    goto LABEL_253;
                  }
                  v142 = sub_CB6200(v142, v180, v144);
                }
                v145 = *(_WORD **)(v142 + 32);
                goto LABEL_247;
              }
              v130 = sub_CB6200(v130, src, v133);
            }
            v134 = *(_DWORD **)(v130 + 32);
            goto LABEL_231;
          }
        }
        else
        {
          while ( 1 )
          {
            v93 = sub_12BCB00(v185, *v91, v91[1], 0);
            if ( v93 )
              break;
            v91 += 2;
            if ( (_QWORD *)v92 == v91 )
              goto LABEL_149;
          }
          v19 = v93;
        }
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v193) > 0xF )
        {
          sub_2241490(&v192, "libnvvm: error: ", 16, v94);
          v95 = (const char *)v169(v19);
          v96 = strlen(v95);
          if ( v96 <= 0x3FFFFFFFFFFFFFFFLL - v193 )
          {
            sub_2241490(&v192, v95, v96, v97);
            if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v193) > 0x2C )
            {
              v99 = ": failed to link the module with the builtin\n";
              sub_2241490(&v192, ": failed to link the module with the builtin\n", 45, v98);
              v166(&v185);
LABEL_158:
              if ( (_QWORD *)v225[0] != &v225[2] )
                _libc_free(v225[0], v99);
              goto LABEL_109;
            }
          }
        }
        goto LABEL_304;
      }
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v193) <= 0xF )
        goto LABEL_304;
      sub_2241490(&v192, "libnvvm: error: ", 16, v21);
      v109 = (const char *)v169(v19);
      v110 = strlen(v109);
      if ( v110 > 0x3FFFFFFFFFFFFFFFLL - v193 )
        goto LABEL_304;
      sub_2241490(&v192, v109, v110, v111);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v193) <= 0x3A )
        goto LABEL_304;
      sub_2241490(&v192, ": failed to add the module to the libnvvm compilation unit\n", 59, v112);
      v166(&v185);
    }
LABEL_109:
    v83 = v193;
    *a8 = v19;
    if ( v83 )
    {
      v84 = sub_2207820(v83 + 1);
      *a7 = v84;
      sub_2241570(&v192, v84, v83, 0);
      *(_BYTE *)(*a7 + v83) = 0;
    }
    v203 = &unk_49DD210;
    sub_CB5840(&v203);
    if ( (_QWORD *)v195[0] != v196 )
      j_j___libc_free_0(v195[0], v196[0] + 1LL);
    if ( v192 != v194 )
      j_j___libc_free_0(v192, v194[0] + 1LL);
LABEL_41:
    if ( a11 )
      goto LABEL_65;
    if ( !*a7 )
    {
LABEL_52:
      if ( !a12 && *((_BYTE *)a3 + 66) && a6[1] )
      {
        v51 = **a6;
        LODWORD(v203) = 0;
        v52 = sub_2241E40();
        v53 = (const char *)a3[4];
        v54 = 0;
        v204 = v52;
        if ( v51 != 127 && v51 != -19 )
        {
          if ( v53 )
            v54 = strlen(v53);
          sub_CB7060(v225, v53, v54, &v203, 1);
          if ( !(_DWORD)v203 )
          {
            v55 = *a6;
            if ( *a6 )
            {
              v56 = strlen(*a6);
              v57 = v56;
              if ( v56 > v225[3] - v225[4] )
              {
                sub_CB6200(v225, v55, v56);
              }
              else if ( v56 )
              {
                memcpy((void *)v225[4], v55, v56);
                v225[4] += v57;
              }
            }
LABEL_64:
            sub_CB5B00(v225);
            goto LABEL_65;
          }
LABEL_204:
          if ( a1 )
          {
            v119 = strlen(a1);
            sub_223E0D0(qword_4FD4BE0, a1, v119);
          }
          else
          {
            sub_222DC80(
              (char *)qword_4FD4BE0 + *(_QWORD *)(qword_4FD4BE0[0] - 24LL),
              *(_DWORD *)((char *)&qword_4FD4BE0[4] + *(_QWORD *)(qword_4FD4BE0[0] - 24LL)) | 1u);
          }
          sub_223E0D0(qword_4FD4BE0, ": IO error: ", 12);
          (*(void (__fastcall **)(char **, __int64, _QWORD))(*(_QWORD *)v204 + 32LL))(&v210, v204, (unsigned int)v203);
          v120 = sub_223E0D0(qword_4FD4BE0, v210, v211);
          sub_223E0D0(v120, "\n", 1);
          if ( v210 != (char *)&v212 )
            j_j___libc_free_0(v210, v212 + 1);
          goto LABEL_64;
        }
        if ( v53 )
          v54 = strlen(v53);
        sub_CB7060(v225, v53, v54, &v203, 0);
        if ( (_DWORD)v203 )
          goto LABEL_204;
        sub_CB6200(v225, *a6, a6[1]);
        sub_CB5B00(v225);
      }
LABEL_65:
      result = v184;
      if ( v184 )
        result = sub_C9AF60(v184);
      goto LABEL_67;
    }
    v44 = sub_CB72A0();
    v45 = (const char *)*a7;
    v46 = v44;
    if ( *a7 )
    {
      v47 = strlen(v45);
      v48 = *(_BYTE **)(v46 + 32);
      v49 = v47;
      v50 = *(_BYTE **)(v46 + 24);
      if ( v49 <= v50 - v48 )
      {
        if ( v49 )
        {
          memcpy(v48, v45, v49);
          v50 = *(_BYTE **)(v46 + 24);
          v48 = (_BYTE *)(v49 + *(_QWORD *)(v46 + 32));
          *(_QWORD *)(v46 + 32) = v48;
        }
        goto LABEL_47;
      }
      v46 = sub_CB6200(v46, v45, v49);
    }
    v50 = *(_BYTE **)(v46 + 24);
    v48 = *(_BYTE **)(v46 + 32);
LABEL_47:
    if ( v50 == v48 )
    {
      sub_CB6200(v46, "\n", 1);
    }
    else
    {
      *v48 = 10;
      ++*(_QWORD *)(v46 + 32);
    }
    if ( *a7 )
      j_j___libc_free_0_0(*a7);
    *a7 = 0;
    goto LABEL_52;
  }
  v14 = 0;
  if ( !*((_BYTE *)a3 + 64) )
  {
LABEL_6:
    *a8 = 0;
    v184 = sub_C996C0("LibNVVM", 7, *a3, a3[1]);
    goto LABEL_7;
  }
  v211 = 0;
  v210 = (char *)&v212;
  LOBYTE(v212) = 0;
  v68 = sub_22077B0(8);
  v14 = (char *)v68;
  if ( v68 )
    sub_B6EEA0(v68);
  sub_B6F950(v14, a5);
  v69 = (char *)*a3;
  LOWORD(v225[4]) = 257;
  if ( *v69 )
  {
    v225[0] = v69;
    LOBYTE(v225[4]) = 3;
  }
  sub_C7EAD0(&v203, v225, 0, 1, 0);
  if ( (v205 & 1) == 0 || (v74 = (unsigned int)v203) == 0 )
  {
    if ( *((_BYTE *)a3 + 65) )
    {
      v200 = (__int64)v203;
      v203 = 0;
      sub_9047E0((__int64)v225, &v200, (__int64)a3);
      if ( v200 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v200 + 8LL))(v200);
      v70 = v225[1];
      v13 = v225[0];
      LOBYTE(v225[1]) &= ~2u;
      if ( (v70 & 1) == 0 )
        goto LABEL_146;
      v225[0] = 0;
      v200 = v13 | 1;
      if ( (v13 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        if ( a1 )
        {
          v71 = strlen(a1);
          sub_223E0D0(qword_4FD4BE0, a1, v71);
        }
        else
        {
          sub_222DC80(
            (char *)qword_4FD4BE0 + *(_QWORD *)(qword_4FD4BE0[0] - 24LL),
            *(_DWORD *)((char *)&qword_4FD4BE0[4] + *(_QWORD *)(qword_4FD4BE0[0] - 24LL)) | 1u);
        }
        sub_223E0D0(qword_4FD4BE0, ": input file ", 13);
        v72 = sub_223E0D0(qword_4FD4BE0, *a3, a3[1]);
        sub_223E0D0(v72, " read error: \"", 14);
        sub_223E0D0(v72, "\"\n", 2);
        *a8 = -1;
        if ( (v200 & 1) != 0 || (v200 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_C63C30(&v200);
        result = LOBYTE(v225[1]);
        if ( (v225[1] & 2) != 0 )
          sub_904700(v225);
        v73 = v225[0];
        if ( (v225[1] & 1) != 0 )
        {
LABEL_97:
          if ( v73 )
            result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v73 + 8LL))(v73);
          goto LABEL_131;
        }
LABEL_129:
        if ( v73 )
        {
          sub_BA9C10(v73);
          result = j_j___libc_free_0(v73, 880);
        }
        goto LABEL_131;
      }
    }
    else
    {
      memset(&v225[4], 0, 0x58u);
      sub_C7E010(v197, v203);
      sub_A01950(
        (unsigned int)&v200,
        (_DWORD)v14,
        (unsigned int)v225,
        v85,
        v86,
        v87,
        v197[0],
        v197[1],
        v197[2],
        v197[3]);
      if ( LOBYTE(v225[14]) )
      {
        LOBYTE(v225[14]) = 0;
        if ( v225[12] )
          ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v225[12])(&v225[10], &v225[10], 3);
      }
      if ( LOBYTE(v225[9]) )
      {
        LOBYTE(v225[9]) = 0;
        if ( v225[7] )
          ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v225[7])(&v225[5], &v225[5], 3);
      }
      if ( LOBYTE(v225[4]) )
      {
        LOBYTE(v225[4]) = 0;
        if ( v225[2] )
          ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v225[2])(v225, v225, 3);
      }
      v88 = v201;
      v13 = v200;
      LOBYTE(v201) = v201 & 0xFD;
      if ( (v88 & 1) == 0 )
        goto LABEL_146;
      v200 = 0;
      v225[0] = v13 | 1;
      if ( (v13 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        if ( a1 )
        {
          v89 = strlen(a1);
          sub_223E0D0(qword_4FD4BE0, a1, v89);
        }
        else
        {
          sub_222DC80(
            (char *)qword_4FD4BE0 + *(_QWORD *)(qword_4FD4BE0[0] - 24LL),
            *(_DWORD *)((char *)&qword_4FD4BE0[4] + *(_QWORD *)(qword_4FD4BE0[0] - 24LL)) | 1u);
        }
        sub_223E0D0(qword_4FD4BE0, ": input file ", 13);
        v90 = sub_223E0D0(qword_4FD4BE0, *a3, a3[1]);
        sub_223E0D0(v90, " read error: \"", 14);
        sub_223E0D0(v90, "\"\n", 2);
        *a8 = -1;
        if ( (v225[0] & 1) != 0 || (v225[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_C63C30(v225);
        result = (unsigned __int8)v201;
        if ( (v201 & 2) != 0 )
          sub_904700(&v200);
        v73 = v200;
        if ( (v201 & 1) != 0 )
          goto LABEL_97;
        goto LABEL_129;
      }
    }
    v13 = 0;
LABEL_146:
    if ( (v205 & 1) == 0 && v203 )
      (*(void (__fastcall **)(void *))(*(_QWORD *)v203 + 8LL))(v203);
    if ( v210 != (char *)&v212 )
      j_j___libc_free_0(v210, v212 + 1);
    goto LABEL_6;
  }
  v75 = v204;
  if ( a1 )
  {
    v76 = strlen(a1);
    sub_223E0D0(qword_4FD4BE0, a1, v76);
  }
  else
  {
    sub_222DC80(
      (char *)qword_4FD4BE0 + *(_QWORD *)(qword_4FD4BE0[0] - 24LL),
      *(_DWORD *)((char *)&qword_4FD4BE0[4] + *(_QWORD *)(qword_4FD4BE0[0] - 24LL)) | 1u);
  }
  sub_223E0D0(qword_4FD4BE0, ": error in open ", 16);
  v77 = sub_223E0D0(qword_4FD4BE0, *a3, a3[1]);
  sub_223E0D0(v77, "\n", 1);
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*(_QWORD *)v75 + 32LL))(v225, v75, v74);
  v78 = sub_223E0D0(qword_4FD4BE0, v225[0], v225[1]);
  sub_223E0D0(v78, "\n", 1);
  if ( (_QWORD *)v225[0] != &v225[2] )
    j_j___libc_free_0(v225[0], v225[2] + 1LL);
  result = (__int64)a8;
  *a8 = -1;
LABEL_131:
  if ( (v205 & 1) == 0 && v203 )
    result = (*(__int64 (__fastcall **)(void *))(*(_QWORD *)v203 + 8LL))(v203);
  if ( v210 != (char *)&v212 )
    result = j_j___libc_free_0(v210, v212 + 1);
LABEL_67:
  if ( v14 )
  {
    sub_B6E710(v14);
    return j_j___libc_free_0(v14, 8);
  }
  return result;
}
