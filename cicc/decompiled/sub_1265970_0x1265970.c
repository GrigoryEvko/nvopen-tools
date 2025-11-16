// Function: sub_1265970
// Address: 0x1265970
//
__int64 __fastcall sub_1265970(
        char *a1,
        __int64 a2,
        _QWORD *a3,
        char *a4,
        __int64 a5,
        char **a6,
        const char **a7,
        unsigned int *a8,
        unsigned int a9,
        char a10,
        char a11,
        unsigned int a12)
{
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r15
  char *v17; // rbx
  __int64 v18; // r12
  const char *v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  const char *v25; // rbx
  __int64 v26; // r12
  size_t v27; // rax
  _BYTE *v28; // rdi
  size_t v29; // r13
  _BYTE *v30; // rax
  __int64 result; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  size_t v36; // rax
  size_t v37; // rax
  __int64 v38; // rbx
  __int64 v39; // r15
  __int64 v40; // rax
  const char *v41; // rdi
  size_t v42; // rax
  __int64 v43; // rax
  __int64 v44; // r13
  _QWORD *v45; // rax
  __int64 v46; // rbx
  __int64 v47; // r12
  __int64 v48; // rax
  __int64 (__fastcall *v49)(__int64 *); // r13
  __int64 v50; // rcx
  unsigned int v51; // r13d
  __int64 (__fastcall *v52)(__int64, __int64, __int64, _QWORD); // r13
  __int64 v53; // rcx
  unsigned int v54; // r8d
  const char **v55; // rdx
  __int64 v56; // rdi
  __int64 v57; // rsi
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rdx
  void (__fastcall *v62)(_QWORD *, __int64); // rax
  void (__fastcall *v63)(__int64, __int64); // rax
  __int64 (__fastcall *v64)(__int64, _QWORD, _QWORD); // rax
  __int64 (__fastcall *v65)(__int64, unsigned __int64 *); // rax
  __int64 v66; // rcx
  unsigned int v67; // ebx
  __int64 v68; // rax
  char *v69; // rax
  char v70; // al
  size_t v71; // rax
  __int64 v72; // r12
  __int64 v73; // r12
  char v74; // bl
  __int64 v75; // rax
  const char *v76; // r12
  size_t v77; // rdx
  char *v78; // r13
  size_t v79; // rax
  size_t v80; // r12
  unsigned int v81; // r13d
  size_t v82; // rax
  __int64 v83; // rax
  __int64 v84; // rax
  const char *v85; // r12
  size_t v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // rcx
  __int64 v89; // rbx
  int v90; // edx
  int v91; // ecx
  int v92; // r8d
  int v93; // r9d
  char v94; // al
  __int64 v95; // rax
  size_t v96; // rax
  __int64 v97; // r12
  _QWORD *v98; // r12
  __int64 v99; // rbx
  unsigned int v100; // eax
  __int64 v101; // rcx
  const char *v102; // r12
  size_t v103; // rdx
  __int64 v104; // rcx
  __int64 v105; // rcx
  unsigned int v106; // r13d
  size_t v107; // rax
  __int64 v108; // rax
  __int64 v109; // rax
  __int64 v110; // rdx
  __int64 v111; // rdx
  __int64 v112; // rax
  int v113; // r10d
  const char **v114; // r11
  const char **v115; // rsi
  unsigned __int64 v116; // rdi
  __int64 v117; // rdx
  __int64 v118; // rcx
  __int64 v119; // r12
  int v120; // ebx
  __int64 v121; // r12
  int v122; // ebx
  const char *v123; // r12
  __int64 (__fastcall *v124)(__int64, const char *); // rax
  size_t v125; // rax
  void *v126; // rsi
  const char *v127; // r12
  size_t v128; // rdx
  __int64 v129; // rcx
  __int64 v130; // rcx
  size_t v131; // rdx
  __int64 v132; // rcx
  __int64 v133; // rcx
  __int64 (__fastcall *v134)(__int64, unsigned __int64 *); // rax
  __int64 v135; // rcx
  unsigned int v136; // ebx
  size_t v137; // rax
  __int64 v138; // rax
  size_t v139; // rdx
  __int64 v140; // rcx
  size_t v141; // rdx
  __int64 v142; // rcx
  __int64 v143; // rcx
  __int64 v144; // r12
  __int64 (__fastcall *v145)(__int64, __int64); // rax
  __int64 v146; // rcx
  size_t v147; // rax
  __int64 v148; // rcx
  __int64 v149; // r12
  __int64 v150; // rax
  unsigned __int64 v151; // rdx
  char *v152; // rsi
  size_t v153; // rax
  _DWORD *v154; // rdi
  size_t v155; // r13
  __int64 v156; // r12
  __m128i *v157; // rax
  __m128i si128; // xmm0
  size_t v159; // rax
  size_t v160; // r13
  unsigned __int64 v161; // rax
  _BYTE *v162; // rdx
  __int64 v163; // rcx
  __int64 v164; // r12
  _BYTE *v165; // rax
  size_t v166; // rax
  _WORD *v167; // rdi
  size_t v168; // r13
  __int64 v169; // rsi
  size_t v170; // rax
  _BYTE *v171; // rdi
  size_t v172; // r13
  __int64 v173; // rbx
  int v174; // r12d
  const void *v175; // r15
  size_t v176; // rax
  size_t v177; // r14
  __int64 v178; // r13
  _BYTE *v179; // rax
  __int64 v180; // rcx
  __int64 v181; // rdi
  _BYTE *v182; // rax
  __m128i *v183; // rax
  unsigned __int64 v184; // rdx
  __m128i v185; // xmm0
  __int64 v186; // rdi
  __int64 v187; // rax
  __int64 v188; // rax
  __int64 v189; // rax
  void (__fastcall *v190)(__int64, __int64 (__fastcall *)(__int64, char *, __int64, __int64), char **, __int64); // [rsp+8h] [rbp-368h]
  void (__fastcall *v191)(__int64 *); // [rsp+10h] [rbp-360h]
  int v192; // [rsp+18h] [rbp-358h]
  __int64 (__fastcall *v193)(_QWORD); // [rsp+28h] [rbp-348h]
  char *src; // [rsp+48h] [rbp-328h]
  _QWORD *srca; // [rsp+48h] [rbp-328h]
  unsigned __int8 v196; // [rsp+50h] [rbp-320h]
  void *v197; // [rsp+50h] [rbp-320h]
  char *v201; // [rsp+68h] [rbp-308h]
  char *v202; // [rsp+68h] [rbp-308h]
  char *v203; // [rsp+68h] [rbp-308h]
  char *v204; // [rsp+68h] [rbp-308h]
  char *v205; // [rsp+68h] [rbp-308h]
  char *v206; // [rsp+68h] [rbp-308h]
  char *v207; // [rsp+68h] [rbp-308h]
  __int64 v208; // [rsp+88h] [rbp-2E8h]
  __int64 v209; // [rsp+98h] [rbp-2D8h] BYREF
  unsigned __int64 v210; // [rsp+A0h] [rbp-2D0h] BYREF
  unsigned __int64 v211; // [rsp+A8h] [rbp-2C8h] BYREF
  __int64 v212; // [rsp+B0h] [rbp-2C0h] BYREF
  __int64 v213; // [rsp+B8h] [rbp-2B8h] BYREF
  __int64 v214; // [rsp+C0h] [rbp-2B0h] BYREF
  __int64 v215; // [rsp+C8h] [rbp-2A8h] BYREF
  __int64 v216; // [rsp+D0h] [rbp-2A0h] BYREF
  __int64 v217; // [rsp+D8h] [rbp-298h]
  const char *v218; // [rsp+E0h] [rbp-290h] BYREF
  __int64 v219; // [rsp+E8h] [rbp-288h]
  _QWORD v220[2]; // [rsp+F0h] [rbp-280h] BYREF
  _QWORD v221[2]; // [rsp+100h] [rbp-270h] BYREF
  _QWORD v222[2]; // [rsp+110h] [rbp-260h] BYREF
  __int64 v223[4]; // [rsp+120h] [rbp-250h] BYREF
  _QWORD v224[2]; // [rsp+140h] [rbp-230h] BYREF
  __int64 v225; // [rsp+150h] [rbp-220h] BYREF
  __int64 v226[2]; // [rsp+160h] [rbp-210h] BYREF
  __int64 v227; // [rsp+170h] [rbp-200h] BYREF
  void *v228; // [rsp+180h] [rbp-1F0h] BYREF
  __int64 v229; // [rsp+188h] [rbp-1E8h]
  __int64 v230; // [rsp+190h] [rbp-1E0h]
  __int64 v231; // [rsp+198h] [rbp-1D8h]
  int v232; // [rsp+1A0h] [rbp-1D0h]
  _QWORD *v233; // [rsp+1A8h] [rbp-1C8h]
  char *v234; // [rsp+1B0h] [rbp-1C0h] BYREF
  __int64 v235; // [rsp+1B8h] [rbp-1B8h]
  char *v236; // [rsp+1C0h] [rbp-1B0h] BYREF
  char *v237; // [rsp+1C8h] [rbp-1A8h]
  __int64 v238; // [rsp+1D0h] [rbp-1A0h]
  int v239; // [rsp+1D8h] [rbp-198h] BYREF
  int v240; // [rsp+1DCh] [rbp-194h] BYREF
  char v241[4]; // [rsp+1E0h] [rbp-190h] BYREF
  int v242; // [rsp+1E4h] [rbp-18Ch] BYREF
  __int64 v243; // [rsp+1E8h] [rbp-188h] BYREF
  __int64 v244; // [rsp+1F0h] [rbp-180h] BYREF
  char v245[8]; // [rsp+1F8h] [rbp-178h] BYREF
  __int64 v246; // [rsp+200h] [rbp-170h] BYREF
  char v247[4]; // [rsp+208h] [rbp-168h] BYREF
  char v248[4]; // [rsp+20Ch] [rbp-164h] BYREF
  _QWORD v249[44]; // [rsp+210h] [rbp-160h] BYREF

  v13 = a2;
  v14 = a12;
  v15 = a9;
  v196 = a5;
  if ( a11 != 1 && (_BYTE)a9 )
  {
    sub_223E0D0(qword_4FD4BE0, "\"", 1);
    if ( a1 )
    {
      v36 = strlen(a1);
      sub_223E0D0(qword_4FD4BE0, a1, v36);
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
      v37 = strlen(a4);
      sub_223E0D0(qword_4FD4BE0, a4, v37);
      sub_223E0D0(qword_4FD4BE0, "\" ", 2);
    }
    v38 = 0;
    sub_223E0D0(qword_4FD4BE0, "\"", 1);
    v39 = sub_223E0D0(qword_4FD4BE0, *a3, a3[1]);
    sub_223E0D0(v39, "\" -o \"", 6);
    v40 = sub_223E0D0(v39, a3[4], a3[5]);
    sub_223E0D0(v40, "\"", 1);
    if ( *((int *)a3 + 17) > 0 )
    {
      do
      {
        while ( 1 )
        {
          sub_223E0D0(qword_4FD4BE0, " ", 1);
          v43 = a3[9];
          v44 = *(_QWORD *)(v43 + 8 * v38);
          if ( !v44 )
            break;
          v41 = *(const char **)(v43 + 8 * v38++);
          v42 = strlen(v41);
          sub_223E0D0(qword_4FD4BE0, v44, v42);
          if ( *((_DWORD *)a3 + 17) <= (int)v38 )
            goto LABEL_35;
        }
        ++v38;
        sub_222DC80(
          (char *)qword_4FD4BE0 + *(_QWORD *)(qword_4FD4BE0[0] - 24LL),
          *(_DWORD *)((char *)&qword_4FD4BE0[4] + *(_QWORD *)(qword_4FD4BE0[0] - 24LL)) | 1u);
      }
      while ( *((_DWORD *)a3 + 17) > (int)v38 );
LABEL_35:
      v13 = a2;
    }
    a2 = (__int64)" -keep -v\n";
    sub_223E0D0(qword_4FD4BE0, " -keep -v\n", 10);
    if ( a11 )
      goto LABEL_37;
  }
  else if ( a11 )
  {
LABEL_37:
    v19 = "LibNVVM";
    *a8 = 0;
    v17 = (char *)*a3;
    v18 = a3[1];
    v16 = sub_16DA870("LibNVVM", a2, v14, v15, a5, a6);
    if ( !v16 )
    {
LABEL_8:
      if ( (_BYTE)a12 )
        goto LABEL_9;
LABEL_39:
      v218 = (const char *)v220;
      v233 = v221;
      v221[0] = v222;
      v228 = &unk_49EFBE0;
      v219 = 0;
      LOBYTE(v220[0]) = 0;
      v221[1] = 0;
      LOBYTE(v222[0]) = 0;
      v232 = 1;
      v231 = 0;
      v230 = 0;
      v229 = 0;
      sub_153BF40(v13, &v228, 1, 0, 0, 0);
      if ( v231 == v229 )
      {
        v45 = v233;
        v46 = *v233;
      }
      else
      {
        sub_16E7BA0(&v228);
        v45 = v233;
        v46 = *v233;
        if ( v231 != v229 )
        {
          sub_16E7BA0(&v228);
          v45 = v233;
        }
      }
      v47 = v45[1];
      v48 = sub_1632FA0(v13);
      v192 = sub_15A9520(v48, 0);
      if ( v13 )
      {
        sub_1633490(v13);
        j_j___libc_free_0(v13, 736);
      }
      a6[1] = 0;
      **a6 = 0;
      v49 = (__int64 (__fastcall *)(__int64 *))sub_12BC0F0(2151);
      v193 = (__int64 (__fastcall *)(_QWORD))sub_12BC0F0(46967);
      v51 = v49(&v209);
      if ( v51 )
      {
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v219) <= 0xF )
          goto LABEL_299;
        sub_2241490(&v218, "libnvvm: error: ", 16, v50);
        v85 = (const char *)v193(v51);
        v86 = strlen(v85);
        if ( v86 > 0x3FFFFFFFFFFFFFFFLL - v219 )
          goto LABEL_299;
        sub_2241490(&v218, v85, v86, v87);
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v219) <= 0x2F )
          goto LABEL_299;
        a2 = (__int64)": failed to create the libnvvm compilation unit\n";
        sub_2241490(&v218, ": failed to create the libnvvm compilation unit\n", 48, v88);
      }
      else
      {
        v52 = (__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD))sub_12BC0F0(4660);
        v191 = (void (__fastcall *)(__int64 *))sub_12BC0F0(21257);
        v51 = v52(v209, v46, v47, 0);
        if ( !v51 )
        {
          v54 = *((_DWORD *)a3 + 17);
          v55 = (const char **)a3[9];
          if ( v54 && (!strcmp(*v55, "-opt") || !strcmp(*v55, "-llc")) )
            goto LABEL_48;
          sub_125D200((__int64)v249, v54, (__int64)v55, 8 * v192 == 64);
          v98 = (_QWORD *)v249[0];
          v99 = v249[0] + 16LL * LODWORD(v249[1]);
          if ( v249[0] == v99 )
          {
LABEL_134:
            if ( a4 )
            {
              LOWORD(v236) = 257;
              if ( *a4 )
              {
                v234 = a4;
                LOBYTE(v236) = 3;
              }
              sub_16C2E90(v226, &v234, -1, 1);
              if ( (v227 & 1) != 0 )
              {
                v106 = v226[0];
                if ( LODWORD(v226[0]) )
                {
                  v197 = (void *)v226[1];
                  sub_223E0D0(qword_4FD4BE0, "error in open: ", 15);
                  v107 = strlen(a4);
                  sub_223E0D0(qword_4FD4BE0, a4, v107);
                  sub_223E0D0(qword_4FD4BE0, "\n", 1);
                  (*(void (__fastcall **)(char **, void *, _QWORD))(*(_QWORD *)v197 + 32LL))(&v234, v197, v106);
                  v108 = sub_223E0D0(qword_4FD4BE0, v234, v235);
                  a2 = (__int64)"\n";
                  sub_223E0D0(v108, "\n", 1);
                  if ( v234 != (char *)&v236 )
                  {
                    a2 = (__int64)(v236 + 1);
                    j_j___libc_free_0(v234, v236 + 1);
                  }
                  v191(&v209);
                  if ( (v227 & 1) == 0 && v226[0] )
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v226[0] + 8LL))(v226[0]);
                  v51 = 4;
                  goto LABEL_143;
                }
              }
              v126 = *(void **)(v226[0] + 8);
              v51 = sub_12BCB00(v209, v126, *(_QWORD *)(v226[0] + 16) - (_QWORD)v126, 0);
              if ( (v227 & 1) == 0 && v226[0] )
                (*(void (__fastcall **)(__int64))(*(_QWORD *)v226[0] + 8LL))(v226[0]);
            }
            else
            {
              v126 = &unk_420FD80;
              v51 = sub_12BCB00(v209, &unk_420FD80, 455876, 0);
            }
            if ( !v51 )
            {
              if ( (_QWORD *)v249[0] != &v249[2] )
                _libc_free(v249[0], v126);
LABEL_48:
              v56 = *a3;
              v212 = 0;
              v213 = 0;
              v57 = a3[1];
              v214 = 0;
              v215 = 0;
              v216 = 0;
              v217 = 0;
              if ( a10 )
              {
                v58 = sub_1263BC0(v56, v57);
                v60 = sub_1263BC0(v58, v59);
                LOWORD(v249[2]) = 773;
                v216 = v60;
                v217 = v61;
                v249[1] = ".lnk.bc";
                v249[0] = &v216;
                sub_16E2FC0(v224, v249);
                v249[0] = &v216;
                LOWORD(v249[2]) = 773;
                v249[1] = ".opt.bc";
                sub_16E2FC0(v226, v249);
                if ( a11 )
                  goto LABEL_50;
              }
              else
              {
                v109 = sub_16C40D0(v56, v57, 2);
                v216 = sub_16C40D0(v109, v110, 2);
                v217 = v111;
                LOWORD(v249[2]) = 773;
                v249[1] = ".lnk.bc";
                v249[0] = &v216;
                sub_16E2FC0(v224, v249);
                v249[1] = ".opt.bc";
                LOWORD(v249[2]) = 773;
                v249[0] = &v216;
                sub_16E2FC0(v226, v249);
                if ( a11 || !(_BYTE)a9 )
                  goto LABEL_50;
              }
              v112 = sub_12BC0F0(48879);
              v113 = *((_DWORD *)a3 + 17);
              v190 = (void (__fastcall *)(__int64, __int64 (__fastcall *)(__int64, char *, __int64, __int64), char **, __int64))v112;
              v234 = (char *)*a3;
              v235 = a3[4];
              v236 = a1;
              if ( v113 <= 0 )
              {
                v114 = (const char **)a3[9];
              }
              else
              {
                v114 = (const char **)a3[9];
                if ( !strcmp(*v114, "-lnk") )
                {
                  sub_2240AE0(v224, a3 + 4);
                  v114 = (const char **)a3[9];
                  v113 = *((_DWORD *)a3 + 17);
                }
                else if ( !strcmp(*v114, "-opt") )
                {
                  sub_2240AE0(v224, a3);
                  sub_2240AE0(v226, a3 + 4);
                  v114 = (const char **)a3[9];
                  v113 = *((_DWORD *)a3 + 17);
                }
                else if ( !strcmp(*v114, "-llc") )
                {
                  sub_2240AE0(v226, a3);
                  v114 = (const char **)a3[9];
                  v113 = *((_DWORD *)a3 + 17);
                }
              }
              v115 = v114;
              v116 = (unsigned int)v113;
              v237 = (char *)v224[0];
              v238 = v226[0];
              sub_12D34A0(
                v113,
                (_DWORD)v114,
                57069,
                (unsigned int)&v239,
                (unsigned int)&v243,
                (unsigned int)&v240,
                (__int64)&v244,
                (__int64)v241,
                (__int64)v245,
                (__int64)&v242,
                (__int64)&v246,
                (__int64)v247,
                0,
                (__int64)v248);
              v119 = v244;
              v120 = v240;
              if ( v240 != (_DWORD)v212 || v244 != v213 )
              {
                v116 = (unsigned __int64)&v212;
                v115 = (const char **)&v213;
                sub_12C7AC0(&v212, &v213);
                LODWORD(v212) = v120;
                v213 = v119;
              }
              v121 = v246;
              v122 = v242;
              if ( v242 != (_DWORD)v214 || v246 != v215 )
              {
                v116 = (unsigned __int64)&v214;
                v115 = (const char **)&v215;
                sub_12C7AC0(&v214, &v215);
                LODWORD(v214) = v122;
                v215 = v121;
              }
              if ( !(_BYTE)a9 )
                goto LABEL_159;
              v149 = sub_16E8C20(v116, v115, v117, v118);
              v150 = *(_QWORD *)(v149 + 24);
              v151 = *(_QWORD *)(v149 + 16) - v150;
              if ( v151 <= 2 )
              {
                v149 = sub_16E7EE0(v149, "[ \"", 3);
              }
              else
              {
                *(_BYTE *)(v150 + 2) = 34;
                *(_WORD *)v150 = 8283;
                *(_QWORD *)(v149 + 24) += 3LL;
              }
              v152 = v236;
              if ( v236 )
              {
                src = v236;
                v153 = strlen(v236);
                v154 = *(_DWORD **)(v149 + 24);
                v152 = src;
                v155 = v153;
                if ( v153 <= *(_QWORD *)(v149 + 16) - (_QWORD)v154 )
                {
                  if ( v153 )
                  {
                    memcpy(v154, src, v153);
                    v154 = (_DWORD *)(v155 + *(_QWORD *)(v149 + 24));
                    *(_QWORD *)(v149 + 24) = v154;
                  }
LABEL_230:
                  if ( *(_QWORD *)(v149 + 16) - (_QWORD)v154 <= 6u )
                  {
                    v152 = "\" -lnk ";
                    v154 = (_DWORD *)v149;
                    sub_16E7EE0(v149, "\" -lnk ", 7);
                  }
                  else
                  {
                    *v154 = 1814896674;
                    *((_WORD *)v154 + 2) = 27502;
                    *((_BYTE *)v154 + 6) = 32;
                    *(_QWORD *)(v149 + 24) += 7LL;
                  }
                  if ( a4 )
                  {
                    v156 = sub_16E8C20(v154, v152, v151, v148);
                    v157 = *(__m128i **)(v156 + 24);
                    if ( *(_QWORD *)(v156 + 16) - (_QWORD)v157 <= 0x10u )
                    {
                      v152 = "-nvvmir-library \"";
                      v156 = sub_16E7EE0(v156, "-nvvmir-library \"", 17);
                    }
                    else
                    {
                      si128 = _mm_load_si128((const __m128i *)&xmmword_3F0F5B0);
                      v157[1].m128i_i8[0] = 34;
                      *v157 = si128;
                      *(_QWORD *)(v156 + 24) += 17LL;
                    }
                    v159 = strlen(a4);
                    v154 = *(_DWORD **)(v156 + 24);
                    v160 = v159;
                    v161 = *(_QWORD *)(v156 + 16) - (_QWORD)v154;
                    if ( v160 > v161 )
                    {
                      v152 = a4;
                      v188 = sub_16E7EE0(v156, a4, v160);
                      v154 = *(_DWORD **)(v188 + 24);
                      v156 = v188;
                      v161 = *(_QWORD *)(v188 + 16) - (_QWORD)v154;
                    }
                    else if ( v160 )
                    {
                      v152 = a4;
                      memcpy(v154, a4, v160);
                      v189 = *(_QWORD *)(v156 + 16);
                      v154 = (_DWORD *)(v160 + *(_QWORD *)(v156 + 24));
                      *(_QWORD *)(v156 + 24) = v154;
                      v161 = v189 - (_QWORD)v154;
                    }
                    if ( v161 <= 1 )
                    {
                      v152 = "\" ";
                      v154 = (_DWORD *)v156;
                      sub_16E7EE0(v156, "\" ", 2);
                    }
                    else
                    {
                      *(_WORD *)v154 = 8226;
                      *(_QWORD *)(v156 + 24) += 2LL;
                    }
                  }
                  v164 = sub_16E8C20(v154, v152, v151, v148);
                  v165 = *(_BYTE **)(v164 + 24);
                  if ( *(_BYTE **)(v164 + 16) == v165 )
                  {
                    v164 = sub_16E7EE0(v164, "\"", 1);
                  }
                  else
                  {
                    *v165 = 34;
                    ++*(_QWORD *)(v164 + 24);
                  }
                  if ( v234 )
                  {
                    v205 = v234;
                    v166 = strlen(v234);
                    v167 = *(_WORD **)(v164 + 24);
                    v168 = v166;
                    if ( v166 <= *(_QWORD *)(v164 + 16) - (_QWORD)v167 )
                    {
                      if ( v166 )
                      {
                        memcpy(v167, v205, v166);
                        v167 = (_WORD *)(v168 + *(_QWORD *)(v164 + 24));
                        *(_QWORD *)(v164 + 24) = v167;
                      }
LABEL_246:
                      if ( *(_QWORD *)(v164 + 16) - (_QWORD)v167 <= 5u )
                      {
                        v164 = sub_16E7EE0(v164, "\" -o \"", 6);
                      }
                      else
                      {
                        *(_DWORD *)v167 = 1865228322;
                        v167[2] = 8736;
                        *(_QWORD *)(v164 + 24) += 6LL;
                      }
                      v169 = (__int64)v237;
                      if ( v237 )
                      {
                        v206 = v237;
                        v170 = strlen(v237);
                        v171 = *(_BYTE **)(v164 + 24);
                        v169 = (__int64)v206;
                        v172 = v170;
                        if ( v170 <= *(_QWORD *)(v164 + 16) - (_QWORD)v171 )
                        {
                          if ( v170 )
                          {
                            memcpy(v171, v206, v170);
                            v171 = (_BYTE *)(v172 + *(_QWORD *)(v164 + 24));
                            *(_QWORD *)(v164 + 24) = v171;
                          }
LABEL_252:
                          if ( v171 == *(_BYTE **)(v164 + 16) )
                          {
                            v169 = (__int64)"\"";
                            v171 = (_BYTE *)v164;
                            sub_16E7EE0(v164, "\"", 1);
                          }
                          else
                          {
                            *v171 = 34;
                            ++*(_QWORD *)(v164 + 24);
                          }
                          v173 = 8;
                          v174 = 1;
                          if ( v239 > 1 )
                          {
                            v207 = (char *)v16;
                            srca = a3;
                            do
                            {
                              v178 = sub_16E8C20(v171, v169, v162, v163);
                              v179 = *(_BYTE **)(v178 + 24);
                              if ( (unsigned __int64)v179 < *(_QWORD *)(v178 + 16) )
                              {
                                v162 = v179 + 1;
                                *(_QWORD *)(v178 + 24) = v179 + 1;
                                *v179 = 32;
                              }
                              else
                              {
                                v171 = (_BYTE *)v178;
                                v169 = 32;
                                v178 = sub_16E7DE0(v178, 32);
                              }
                              v175 = *(const void **)(v243 + v173);
                              if ( v175 )
                              {
                                v176 = strlen(*(const char **)(v243 + v173));
                                v171 = *(_BYTE **)(v178 + 24);
                                v177 = v176;
                                if ( v176 > *(_QWORD *)(v178 + 16) - (_QWORD)v171 )
                                {
                                  v169 = (__int64)v175;
                                  v171 = (_BYTE *)v178;
                                  sub_16E7EE0(v178, v175, v176);
                                }
                                else if ( v176 )
                                {
                                  v169 = (__int64)v175;
                                  memcpy(v171, v175, v176);
                                  *(_QWORD *)(v178 + 24) += v177;
                                }
                              }
                              ++v174;
                              v173 += 8;
                            }
                            while ( v239 > v174 );
                            v16 = (__int64)v207;
                            a3 = srca;
                          }
                          v181 = sub_16E8C20(v171, v169, v162, v163);
                          v182 = *(_BYTE **)(v181 + 24);
                          if ( (unsigned __int64)v182 >= *(_QWORD *)(v181 + 16) )
                          {
                            v169 = 32;
                            v181 = sub_16E7DE0(v181, 32);
                          }
                          else
                          {
                            *(_QWORD *)(v181 + 24) = v182 + 1;
                            *v182 = 32;
                          }
                          v183 = *(__m128i **)(v181 + 24);
                          v184 = *(_QWORD *)(v181 + 16) - (_QWORD)v183;
                          if ( v184 <= 0x13 )
                          {
                            v169 = (__int64)"-nvvm-version=nvvm70";
                            sub_16E7EE0(v181, "-nvvm-version=nvvm70", 20);
                          }
                          else
                          {
                            v185 = _mm_load_si128((const __m128i *)&xmmword_3C23BC0);
                            v183[1].m128i_i32[0] = 808938870;
                            *v183 = v185;
                            *(_QWORD *)(v181 + 24) += 20LL;
                          }
                          v186 = sub_16E8C20(v181, v169, v184, v180);
                          v187 = *(_QWORD *)(v186 + 24);
                          if ( (unsigned __int64)(*(_QWORD *)(v186 + 16) - v187) <= 2 )
                          {
                            sub_16E7EE0(v186, " ]\n", 3);
                          }
                          else
                          {
                            *(_BYTE *)(v187 + 2) = 10;
                            *(_WORD *)v187 = 23840;
                            *(_QWORD *)(v186 + 24) += 3LL;
                          }
                          v190(v209, sub_1263280, &v234, 61453);
                          v190(v209, sub_12636E0, &v234, 47710);
                          if ( a10 )
                          {
LABEL_159:
                            v190(
                              v209,
                              (__int64 (__fastcall *)(__int64, char *, __int64, __int64))sub_1268040,
                              &v234,
                              64222);
                            v190(
                              v209,
                              (__int64 (__fastcall *)(__int64, char *, __int64, __int64))sub_1267CC0,
                              &v234,
                              56993);
                          }
LABEL_50:
                          v62 = (void (__fastcall *)(_QWORD *, __int64))sub_12BC0F0(65261);
                          qmemcpy(v249, off_4C6EEE0, 0x128u);
                          v62(v249, 37);
                          v63 = (void (__fastcall *)(__int64, __int64))sub_12BC0F0(48813);
                          v63(v209, 57069);
                          v64 = (__int64 (__fastcall *)(__int64, _QWORD, _QWORD))sub_12BC0F0(17185);
                          v51 = v64(v209, *((unsigned int *)a3 + 17), a3[9]);
                          v65 = (__int64 (__fastcall *)(__int64, unsigned __int64 *))sub_12BC0F0(41856);
                          v67 = v65(v209, &v210);
                          if ( v67 )
                          {
                            if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v219) <= 0xF )
                              goto LABEL_299;
                            sub_2241490(&v218, "libnvvm: error: ", 16, v66);
                            v202 = (char *)v193(v67);
                            v131 = strlen(v202);
                            if ( v131 > 0x3FFFFFFFFFFFFFFFLL - v219 )
                              goto LABEL_299;
                            sub_2241490(&v218, v202, v131, v132);
                            if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v219) <= 0x2A )
                              goto LABEL_299;
                            sub_2241490(&v218, ": failed to get the error/warning messages\n", 43, v133);
                          }
                          else
                          {
                            if ( v210 <= 1 )
                            {
                              if ( v51 )
                              {
LABEL_53:
                                v191(&v209);
LABEL_54:
                                if ( (__int64 *)v226[0] != &v227 )
                                  j_j___libc_free_0(v226[0], v227 + 1);
                                if ( (__int64 *)v224[0] != &v225 )
                                  j_j___libc_free_0(v224[0], v225 + 1);
                                sub_12C7AC0(&v214, &v215);
                                a2 = (__int64)&v213;
                                sub_12C7AC0(&v212, &v213);
                                goto LABEL_107;
                              }
LABEL_188:
                              v134 = (__int64 (__fastcall *)(__int64, unsigned __int64 *))sub_12BC0F0(61451);
                              v136 = v134(v209, &v211);
                              if ( v136 )
                              {
                                if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v219) > 0xF )
                                {
                                  sub_2241490(&v218, "libnvvm: error: ", 16, v135);
                                  v203 = (char *)v193(v136);
                                  v141 = strlen(v203);
                                  if ( v141 <= 0x3FFFFFFFFFFFFFFFLL - v219 )
                                  {
                                    sub_2241490(&v218, v203, v141, v142);
                                    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v219) > 0x1E )
                                    {
                                      sub_2241490(&v218, ": failed to get the PTX output\n", 31, v143);
LABEL_190:
                                      v191(&v209);
                                      if ( !v51 )
                                        v51 = v136;
                                      goto LABEL_54;
                                    }
                                  }
                                }
                              }
                              else
                              {
                                if ( v211 <= 1 )
                                  goto LABEL_190;
                                v144 = sub_2207820(v211);
                                v145 = (__int64 (__fastcall *)(__int64, __int64))sub_12BC0F0(4111);
                                v136 = v145(v209, v144);
                                if ( v136 )
                                {
                                  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v219) > 0xF )
                                  {
                                    sub_2241490(&v218, "libnvvm: error: ", 16, 0x3FFFFFFFFFFFFFFFLL);
                                    v204 = (char *)v193(v136);
                                    v147 = strlen(v204);
                                    if ( v147 <= 0x3FFFFFFFFFFFFFFFLL - v219 )
                                    {
                                      sub_2241490(&v218, v204, v147, 0x3FFFFFFFFFFFFFFFLL);
                                      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v219) > 0x1E )
                                      {
                                        sub_2241490(&v218, ": failed to get the PTX output\n", 31, 0x3FFFFFFFFFFFFFFFLL);
                                        goto LABEL_217;
                                      }
                                    }
                                  }
                                }
                                else if ( v211 <= 0x3FFFFFFFFFFFFFFFLL - (__int64)a6[1] )
                                {
                                  sub_2241490(a6, v144, v211, v146);
LABEL_217:
                                  if ( v144 )
                                    j_j___libc_free_0_0(v144);
                                  goto LABEL_190;
                                }
                              }
LABEL_299:
                              sub_4262D8((__int64)"basic_string::append");
                            }
                            v123 = (const char *)sub_2207820(v210);
                            v124 = (__int64 (__fastcall *)(__int64, const char *))sub_12BC0F0(46903);
                            v67 = v124(v209, v123);
                            if ( !v67 )
                            {
                              v139 = strlen(v123);
                              if ( v139 > 0x3FFFFFFFFFFFFFFFLL - v219 )
                                goto LABEL_299;
                              sub_2241490(&v218, v123, v139, v140);
                              j_j___libc_free_0_0(v123);
                              if ( v51 )
                                goto LABEL_53;
                              goto LABEL_188;
                            }
                            if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v219) <= 0xF )
                              goto LABEL_299;
                            sub_2241490(&v218, "libnvvm: error: ", 16, 0x3FFFFFFFFFFFFFFFLL);
                            v201 = (char *)v193(v67);
                            v125 = strlen(v201);
                            if ( v125 > 0x3FFFFFFFFFFFFFFFLL - v219 )
                              goto LABEL_299;
                            sub_2241490(&v218, v201, v125, 0x3FFFFFFFFFFFFFFFLL);
                            if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v219) <= 0x2A )
                              goto LABEL_299;
                            sub_2241490(&v218, ": failed to get the error/warning messages\n", 43, 0x3FFFFFFFFFFFFFFFLL);
                            if ( v123 )
                              j_j___libc_free_0_0(v123);
                          }
                          if ( v51 )
                          {
                            v51 = v67;
                            v191(&v209);
                            goto LABEL_54;
                          }
                          v51 = v67;
                          goto LABEL_188;
                        }
                        v164 = sub_16E7EE0(v164, v206, v170);
                      }
                      v171 = *(_BYTE **)(v164 + 24);
                      goto LABEL_252;
                    }
                    v164 = sub_16E7EE0(v164, v205, v166);
                  }
                  v167 = *(_WORD **)(v164 + 24);
                  goto LABEL_246;
                }
                v149 = sub_16E7EE0(v149, src, v153);
              }
              v154 = *(_DWORD **)(v149 + 24);
              goto LABEL_230;
            }
          }
          else
          {
            while ( 1 )
            {
              v100 = sub_12BCB00(v209, *v98, v98[1], 0);
              if ( v100 )
                break;
              v98 += 2;
              if ( (_QWORD *)v99 == v98 )
                goto LABEL_134;
            }
            v51 = v100;
          }
          if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v219) > 0xF )
          {
            sub_2241490(&v218, "libnvvm: error: ", 16, v101);
            v102 = (const char *)v193(v51);
            v103 = strlen(v102);
            if ( v103 <= 0x3FFFFFFFFFFFFFFFLL - v219 )
            {
              sub_2241490(&v218, v102, v103, v104);
              if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v219) > 0x2C )
              {
                a2 = (__int64)": failed to link the module with the builtin\n";
                sub_2241490(&v218, ": failed to link the module with the builtin\n", 45, v105);
                v191(&v209);
LABEL_143:
                if ( (_QWORD *)v249[0] != &v249[2] )
                  _libc_free(v249[0], a2);
                goto LABEL_107;
              }
            }
          }
          goto LABEL_299;
        }
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v219) <= 0xF )
          goto LABEL_299;
        sub_2241490(&v218, "libnvvm: error: ", 16, v53);
        v127 = (const char *)v193(v51);
        v128 = strlen(v127);
        if ( v128 > 0x3FFFFFFFFFFFFFFFLL - v219 )
          goto LABEL_299;
        sub_2241490(&v218, v127, v128, v129);
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v219) <= 0x3A )
          goto LABEL_299;
        a2 = (__int64)": failed to add the module to the libnvvm compilation unit\n";
        sub_2241490(&v218, ": failed to add the module to the libnvvm compilation unit\n", 59, v130);
        v191(&v209);
      }
LABEL_107:
      v89 = v219;
      *a8 = v51;
      if ( v89 )
      {
        a2 = sub_2207820(v89 + 1);
        *a7 = (const char *)a2;
        sub_2241570(&v218, a2, v89, 0);
        (*a7)[v89] = 0;
      }
      sub_16E7BC0(&v228);
      if ( (_QWORD *)v221[0] != v222 )
      {
        a2 = v222[0] + 1LL;
        j_j___libc_free_0(v221[0], v222[0] + 1LL);
      }
      v19 = v218;
      if ( v218 != (const char *)v220 )
      {
        a2 = v220[0] + 1LL;
        j_j___libc_free_0(v218, v220[0] + 1LL);
      }
LABEL_9:
      if ( a11 )
        goto LABEL_22;
      if ( !*a7 )
      {
LABEL_20:
        if ( !(_BYTE)a12 && *((_BYTE *)a3 + 66) && a6[1] )
        {
          v74 = **a6;
          LODWORD(v228) = 0;
          v75 = sub_2241E40(v19, a2, v20, v21, v22);
          v76 = (const char *)a3[4];
          v77 = 0;
          v229 = v75;
          if ( v74 != 127 && v74 != -19 )
          {
            if ( v76 )
              v77 = strlen(v76);
            a2 = (__int64)v76;
            sub_16E8AF0(v249, v76, v77, &v228, 1);
            if ( !(_DWORD)v228 )
            {
              v78 = *a6;
              if ( *a6 )
              {
                v79 = strlen(*a6);
                v80 = v79;
                if ( v79 > v249[2] - v249[3] )
                {
                  a2 = (__int64)v78;
                  sub_16E7EE0(v249, v78, v79);
                }
                else if ( v79 )
                {
                  a2 = (__int64)v78;
                  memcpy((void *)v249[3], v78, v79);
                  v249[3] += v80;
                }
              }
LABEL_91:
              v19 = (const char *)v249;
              sub_16E7C30(v249);
              goto LABEL_22;
            }
LABEL_198:
            if ( a1 )
            {
              v137 = strlen(a1);
              sub_223E0D0(qword_4FD4BE0, a1, v137);
            }
            else
            {
              sub_222DC80(
                (char *)qword_4FD4BE0 + *(_QWORD *)(qword_4FD4BE0[0] - 24LL),
                *(_DWORD *)((char *)&qword_4FD4BE0[4] + *(_QWORD *)(qword_4FD4BE0[0] - 24LL)) | 1u);
            }
            sub_223E0D0(qword_4FD4BE0, ": IO error: ", 12);
            (*(void (__fastcall **)(char **, __int64, _QWORD))(*(_QWORD *)v229 + 32LL))(&v234, v229, (unsigned int)v228);
            v138 = sub_223E0D0(qword_4FD4BE0, v234, v235);
            a2 = (__int64)"\n";
            sub_223E0D0(v138, "\n", 1);
            if ( v234 != (char *)&v236 )
            {
              a2 = (__int64)(v236 + 1);
              j_j___libc_free_0(v234, v236 + 1);
            }
            goto LABEL_91;
          }
          if ( v76 )
            v77 = strlen(v76);
          sub_16E8AF0(v249, v76, v77, &v228, 0);
          if ( (_DWORD)v228 )
            goto LABEL_198;
          a2 = (__int64)*a6;
          sub_16E7EE0(v249, *a6, a6[1]);
          v19 = (const char *)v249;
          sub_16E7C30(v249);
        }
LABEL_22:
        result = sub_16DA870(v19, a2, v20, v21, v22, v23);
        if ( result )
          result = sub_16DB5E0();
        goto LABEL_24;
      }
      v24 = sub_16E8CB0(v19, a2, v20);
      v25 = *a7;
      v26 = v24;
      if ( *a7 )
      {
        v27 = strlen(v25);
        v28 = *(_BYTE **)(v26 + 24);
        v29 = v27;
        v30 = *(_BYTE **)(v26 + 16);
        v20 = v30 - v28;
        if ( v29 <= v30 - v28 )
        {
          if ( v29 )
          {
            a2 = (__int64)v25;
            memcpy(v28, v25, v29);
            v30 = *(_BYTE **)(v26 + 16);
            v28 = (_BYTE *)(v29 + *(_QWORD *)(v26 + 24));
            *(_QWORD *)(v26 + 24) = v28;
          }
          if ( v30 != v28 )
            goto LABEL_16;
          goto LABEL_101;
        }
        a2 = (__int64)v25;
        v26 = sub_16E7EE0(v26, v25, v29);
      }
      v28 = *(_BYTE **)(v26 + 24);
      if ( *(_BYTE **)(v26 + 16) != v28 )
      {
LABEL_16:
        *v28 = 10;
        ++*(_QWORD *)(v26 + 24);
LABEL_17:
        v19 = *a7;
        if ( *a7 )
          j_j___libc_free_0_0(v19);
        *a7 = 0;
        goto LABEL_20;
      }
LABEL_101:
      a2 = (__int64)"\n";
      sub_16E7EE0(v26, "\n", 1);
      goto LABEL_17;
    }
    v19 = "LibNVVM";
    v16 = 0;
LABEL_7:
    a2 = 7;
    sub_16DB3F0("LibNVVM", 7, v17, v18);
    goto LABEL_8;
  }
  if ( (_BYTE)a12 )
    goto LABEL_37;
  v16 = 0;
  if ( !*((_BYTE *)a3 + 64) )
    goto LABEL_6;
  v235 = 0;
  v234 = (char *)&v236;
  LOBYTE(v236) = 0;
  v68 = sub_22077B0(8);
  v16 = v68;
  if ( v68 )
    sub_1602D10(v68);
  sub_16033C0(v16, v196);
  v69 = (char *)*a3;
  LOWORD(v249[2]) = 257;
  if ( *v69 )
  {
    v249[0] = v69;
    LOBYTE(v249[2]) = 3;
  }
  sub_16C2E90(&v228, v249, -1, 1);
  if ( (v230 & 1) == 0 || (v81 = (unsigned int)v228) == 0 )
  {
    if ( *((_BYTE *)a3 + 65) )
    {
      v226[0] = (__int64)v228;
      a2 = (__int64)v226;
      v228 = 0;
      sub_12642A0((__int64)v249, v226, (__int64)a3);
      if ( v226[0] )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v226[0] + 8LL))(v226[0]);
      v70 = v249[1];
      v13 = v249[0];
      v14 = v249[1] & 0xFD;
      LOBYTE(v249[1]) &= ~2u;
      if ( (v70 & 1) == 0 )
        goto LABEL_131;
      v249[0] = 0;
      v226[0] = v13 | 1;
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
        a2 = (__int64)"\"\n";
        sub_223E0D0(v72, "\"\n", 2);
        *a8 = -1;
        if ( (v226[0] & 1) != 0 || (v226[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_16BCAE0(v226);
LABEL_73:
        result = LOBYTE(v249[1]);
        if ( (v249[1] & 2) != 0 )
          sub_1264230(v249, (__int64)"\"\n", v32);
        v73 = v249[0];
        if ( (v249[1] & 1) != 0 )
        {
          if ( v249[0] )
            result = (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)v249[0] + 8LL))(v249[0]);
        }
        else if ( v249[0] )
        {
          sub_1633490(v249[0]);
          a2 = 736;
          result = j_j___libc_free_0(v73, 736);
        }
        goto LABEL_77;
      }
    }
    else
    {
      sub_16C2FC0(v223, v228);
      a2 = v16;
      sub_1509BC0((unsigned int)v249, v16, v90, v91, v92, v93, v223[0], v223[1], v223[2], v223[3]);
      v94 = v249[1];
      v14 = v249[1] & 0xFD;
      LOBYTE(v249[1]) &= ~2u;
      if ( (v94 & 1) == 0 )
      {
        v13 = v249[0];
        goto LABEL_131;
      }
      v95 = v249[0];
      v249[0] = 0;
      v14 = v95 | 1;
      v226[0] = v95 | 1;
      if ( (v95 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        if ( a1 )
        {
          v96 = strlen(a1);
          sub_223E0D0(qword_4FD4BE0, a1, v96);
        }
        else
        {
          sub_222DC80(
            (char *)qword_4FD4BE0 + *(_QWORD *)(qword_4FD4BE0[0] - 24LL),
            *(_DWORD *)((char *)&qword_4FD4BE0[4] + *(_QWORD *)(qword_4FD4BE0[0] - 24LL)) | 1u);
        }
        sub_223E0D0(qword_4FD4BE0, ": input file ", 13);
        v97 = sub_223E0D0(qword_4FD4BE0, *a3, a3[1]);
        sub_223E0D0(v97, " read error: \"", 14);
        a2 = (__int64)"\"\n";
        sub_223E0D0(v97, "\"\n", 2);
        *a8 = -1;
        if ( (v226[0] & 1) != 0 || (v226[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_16BCAE0(v226);
        goto LABEL_73;
      }
    }
    v13 = 0;
LABEL_131:
    if ( (v230 & 1) == 0 && v228 )
      (*(void (__fastcall **)(void *))(*(_QWORD *)v228 + 8LL))(v228);
    if ( v234 != (char *)&v236 )
    {
      a2 = (__int64)(v236 + 1);
      j_j___libc_free_0(v234, v236 + 1);
    }
LABEL_6:
    *a8 = 0;
    v17 = (char *)*a3;
    v18 = a3[1];
    v19 = "LibNVVM";
    if ( !sub_16DA870("LibNVVM", a2, v14, v15, a5, a6) )
      goto LABEL_39;
    goto LABEL_7;
  }
  v208 = v229;
  if ( a1 )
  {
    v82 = strlen(a1);
    sub_223E0D0(qword_4FD4BE0, a1, v82);
  }
  else
  {
    sub_222DC80(
      (char *)qword_4FD4BE0 + *(_QWORD *)(qword_4FD4BE0[0] - 24LL),
      *(_DWORD *)((char *)&qword_4FD4BE0[4] + *(_QWORD *)(qword_4FD4BE0[0] - 24LL)) | 1u);
  }
  sub_223E0D0(qword_4FD4BE0, ": error in open ", 16);
  v83 = sub_223E0D0(qword_4FD4BE0, *a3, a3[1]);
  sub_223E0D0(v83, "\n", 1);
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*(_QWORD *)v208 + 32LL))(v249, v208, v81);
  v84 = sub_223E0D0(qword_4FD4BE0, v249[0], v249[1]);
  a2 = (__int64)"\n";
  sub_223E0D0(v84, "\n", 1);
  if ( (_QWORD *)v249[0] != &v249[2] )
  {
    a2 = v249[2] + 1LL;
    j_j___libc_free_0(v249[0], v249[2] + 1LL);
  }
  result = (__int64)a8;
  *a8 = -1;
LABEL_77:
  if ( (v230 & 1) == 0 && v228 )
    result = (*(__int64 (__fastcall **)(void *))(*(_QWORD *)v228 + 8LL))(v228);
  if ( v234 != (char *)&v236 )
  {
    a2 = (__int64)(v236 + 1);
    result = j_j___libc_free_0(v234, v236 + 1);
  }
LABEL_24:
  if ( v16 )
  {
    sub_16025D0(v16, a2, v32, v33, v34, v35);
    return j_j___libc_free_0(v16, 8);
  }
  return result;
}
