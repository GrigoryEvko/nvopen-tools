// Function: sub_F5CD10
// Address: 0xf5cd10
//
__int64 __fastcall sub_F5CD10(__int64 a1, char a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // r15
  unsigned __int64 v5; // rbx
  _QWORD *v6; // r12
  __int64 v7; // rsi
  char v8; // al
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rax
  _QWORD *v12; // rax
  __int64 v13; // r14
  _BYTE *v14; // r15
  _BYTE *v15; // rbx
  __int64 v16; // rdx
  unsigned int v17; // esi
  char v18; // al
  __int64 result; // rax
  _QWORD *v20; // rax
  _BYTE *v21; // r15
  __int64 v22; // rax
  __int64 v23; // r8
  int v24; // r11d
  __int64 v25; // r9
  __int64 v26; // r13
  __int64 v27; // r10
  unsigned __int64 v28; // rsi
  __int64 v29; // r14
  _BYTE *v30; // rbx
  __int64 v31; // r15
  unsigned __int64 v32; // r9
  __int64 v33; // rcx
  __int64 v34; // rcx
  __int64 v35; // rax
  unsigned __int64 v36; // r9
  __int64 v37; // r8
  __int64 v38; // rdi
  __int64 v39; // rcx
  unsigned int *v40; // rax
  unsigned int *v41; // rdx
  char v42; // al
  __int64 v43; // rsi
  __int64 v44; // rax
  __int64 v45; // rdx
  unsigned int v46; // ecx
  _QWORD *v47; // rax
  __int64 v48; // r9
  __int64 v49; // r15
  __int64 v50; // r9
  __int64 v51; // r14
  unsigned int *v52; // r14
  unsigned int *v53; // rbx
  __int64 v54; // rdx
  __int64 v55; // r15
  __int64 v56; // rbx
  unsigned int v57; // r14d
  __int64 *v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  _QWORD *v62; // rax
  char *v63; // r13
  _QWORD *v64; // rax
  _QWORD *v65; // r12
  __m128i *v66; // rsi
  __int64 v67; // rdx
  _QWORD *v68; // rbx
  __int64 v69; // rdx
  unsigned __int8 *v70; // rax
  __int64 v71; // r13
  _QWORD *v72; // rax
  __int64 v73; // r14
  __int64 v74; // r9
  __int64 v75; // r13
  unsigned int *v76; // r13
  unsigned int *v77; // rbx
  __int64 v78; // rdx
  int v79; // eax
  __int64 v80; // rcx
  char **v81; // rdx
  __int64 v82; // r15
  char *v83; // r14
  __int64 v84; // r13
  __int64 v85; // r8
  _QWORD *v86; // rax
  __int64 v87; // r13
  char *v88; // r14
  __int64 v89; // rdx
  unsigned __int64 v90; // rax
  int v91; // edx
  _QWORD *v92; // rdi
  _QWORD *v93; // rax
  __int64 v94; // r12
  _QWORD *v95; // rdi
  _QWORD *v96; // rax
  _QWORD *v97; // r12
  __int64 v98; // rdx
  _QWORD *v99; // rbx
  __int64 *v100; // rax
  __int64 v101; // r14
  _QWORD *v102; // r15
  __int64 v103; // rax
  __int64 v104; // rdx
  __int64 v105; // r10
  _QWORD *v106; // rax
  __int64 v107; // r14
  __int64 v108; // r15
  _BYTE *v109; // r15
  _BYTE *v110; // rbx
  __int64 v111; // rdx
  unsigned int v112; // esi
  unsigned int v113; // r13d
  __int64 v114; // rax
  __int64 v115; // rdx
  __int64 v116; // rdi
  _BYTE *v117; // r15
  __int64 v118; // rcx
  _QWORD **v119; // rdx
  int v120; // ecx
  __int32 v121; // eax
  __int64 *v122; // rax
  __int64 v123; // rax
  _BYTE *v124; // rbx
  _BYTE *v125; // r14
  __int64 v126; // rdx
  unsigned int v127; // esi
  _QWORD *v128; // rax
  __int64 v129; // r14
  _BYTE *v130; // rbx
  _BYTE *v131; // r15
  __int64 v132; // rdx
  unsigned int v133; // esi
  char *v134; // r14
  _QWORD *v135; // rax
  _QWORD *v136; // rax
  unsigned __int64 v137; // [rsp+20h] [rbp-1C0h]
  __int64 v138; // [rsp+20h] [rbp-1C0h]
  __int64 *v139; // [rsp+28h] [rbp-1B8h]
  unsigned __int64 v140; // [rsp+30h] [rbp-1B0h]
  unsigned __int64 v141; // [rsp+30h] [rbp-1B0h]
  __int64 v142; // [rsp+30h] [rbp-1B0h]
  __int64 v143; // [rsp+38h] [rbp-1A8h]
  __int64 v144; // [rsp+38h] [rbp-1A8h]
  __int64 v145; // [rsp+38h] [rbp-1A8h]
  __int64 v146; // [rsp+38h] [rbp-1A8h]
  unsigned __int64 v147; // [rsp+38h] [rbp-1A8h]
  __int64 v148; // [rsp+38h] [rbp-1A8h]
  unsigned __int64 v149; // [rsp+38h] [rbp-1A8h]
  __int64 v151; // [rsp+40h] [rbp-1A0h]
  __int64 v153; // [rsp+48h] [rbp-198h]
  __int64 v155; // [rsp+50h] [rbp-190h]
  unsigned __int64 v156; // [rsp+50h] [rbp-190h]
  unsigned int v157; // [rsp+50h] [rbp-190h]
  unsigned __int64 v158; // [rsp+50h] [rbp-190h]
  unsigned __int8 v160; // [rsp+60h] [rbp-180h]
  __int64 v161; // [rsp+60h] [rbp-180h]
  __int64 v162; // [rsp+60h] [rbp-180h]
  int v163; // [rsp+60h] [rbp-180h]
  unsigned __int64 v164; // [rsp+60h] [rbp-180h]
  __m128i v165; // [rsp+80h] [rbp-160h] BYREF
  const __m128i *v166; // [rsp+90h] [rbp-150h] BYREF
  __m128i *v167; // [rsp+98h] [rbp-148h]
  __m128i *v168; // [rsp+A0h] [rbp-140h]
  __int16 v169; // [rsp+B0h] [rbp-130h]
  __int64 v170; // [rsp+C0h] [rbp-120h] BYREF
  __int64 v171; // [rsp+C8h] [rbp-118h]
  __int64 v172; // [rsp+D0h] [rbp-110h] BYREF
  int v173; // [rsp+D8h] [rbp-108h]
  char v174; // [rsp+DCh] [rbp-104h]
  _WORD v175[32]; // [rsp+E0h] [rbp-100h] BYREF
  _BYTE *v176; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v177; // [rsp+128h] [rbp-B8h]
  _BYTE v178[32]; // [rsp+130h] [rbp-B0h] BYREF
  __int64 v179; // [rsp+150h] [rbp-90h]
  __int64 v180; // [rsp+158h] [rbp-88h]
  __int64 v181; // [rsp+160h] [rbp-80h]
  __int64 v182; // [rsp+168h] [rbp-78h]
  void **v183; // [rsp+170h] [rbp-70h]
  _QWORD *v184; // [rsp+178h] [rbp-68h]
  __int64 v185; // [rsp+180h] [rbp-60h]
  int v186; // [rsp+188h] [rbp-58h]
  __int16 v187; // [rsp+18Ch] [rbp-54h]
  char v188; // [rsp+18Eh] [rbp-52h]
  __int64 v189; // [rsp+190h] [rbp-50h]
  __int64 v190; // [rsp+198h] [rbp-48h]
  void *v191; // [rsp+1A0h] [rbp-40h] BYREF
  _QWORD v192[7]; // [rsp+1A8h] [rbp-38h] BYREF

  v4 = a1 + 48;
  v5 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1 + 48 == v5 )
    goto LABEL_203;
  if ( !v5 )
    BUG();
  v6 = (_QWORD *)(v5 - 24);
  if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 > 0xA )
  {
LABEL_203:
    v188 = 7;
    v182 = sub_BD5C60(0);
    v183 = &v191;
    v184 = v192;
    v176 = v178;
    v177 = 0x200000000LL;
    v191 = &unk_49DA100;
    v185 = 0;
    v186 = 0;
    v192[0] = &unk_49DA0B0;
    v187 = 512;
    v189 = 0;
    v190 = 0;
    v179 = 0;
    v180 = 0;
    LOWORD(v181) = 0;
    sub_D5F1F0((__int64)&v176, 0);
    BUG();
  }
  v182 = sub_BD5C60(v5 - 24);
  v183 = &v191;
  v184 = v192;
  v176 = v178;
  v7 = v5 - 24;
  v191 = &unk_49DA100;
  v177 = 0x200000000LL;
  v185 = 0;
  v192[0] = &unk_49DA0B0;
  v186 = 0;
  v187 = 512;
  v188 = 7;
  v189 = 0;
  v190 = 0;
  v179 = 0;
  v180 = 0;
  LOWORD(v181) = 0;
  sub_D5F1F0((__int64)&v176, v5 - 24);
  v8 = *(_BYTE *)(v5 - 24);
  if ( v8 == 31 )
  {
    if ( (*(_DWORD *)(v5 - 20) & 0x7FFFFFF) != 1 )
    {
      v9 = *(_QWORD *)(v5 - 88);
      v143 = *(_QWORD *)(v5 - 56);
      if ( v9 == v143 )
      {
        sub_AA5980(v9, *(_QWORD *)(v5 + 16), 0);
        v175[0] = 257;
        v128 = sub_BD2C40(72, 1u);
        v129 = (__int64)v128;
        if ( v128 )
          sub_B4C8F0((__int64)v128, v9, 1u, 0, 0);
        (*(void (__fastcall **)(_QWORD *, __int64, __int64 *, __int64, __int64))(*v184 + 16LL))(
          v184,
          v129,
          &v170,
          v180,
          v181);
        if ( v176 != &v176[16 * (unsigned int)v177] )
        {
          v164 = v5;
          v130 = v176;
          v131 = &v176[16 * (unsigned int)v177];
          do
          {
            v132 = *((_QWORD *)v130 + 1);
            v133 = *(_DWORD *)v130;
            v130 += 16;
            sub_B99FD0(v129, v133, v132);
          }
          while ( v131 != v130 );
          v5 = v164;
        }
        v7 = (__int64)v6;
        v170 = 18;
        LODWORD(v171) = 30;
        sub_B47C00(v129, (__int64)v6, (int *)&v170, 3);
        v134 = *(char **)(v5 - 120);
        sub_B43D60(v6);
        if ( !a2 )
          goto LABEL_76;
        v7 = (__int64)a3;
        v172 = 0;
        sub_F5CAB0(v134, a3, 0, (__int64)&v170);
        if ( v172 )
        {
          v7 = (__int64)&v170;
          ((void (__fastcall *)(__int64 *, __int64 *, __int64))v172)(&v170, &v170, 3);
        }
        v18 = a2;
        goto LABEL_19;
      }
      v10 = *(_QWORD *)(v5 - 120);
      if ( *(_BYTE *)v10 == 17 )
      {
        if ( *(_DWORD *)(v10 + 32) <= 0x40u )
          v11 = *(_QWORD *)(v10 + 24);
        else
          v11 = **(_QWORD **)(v10 + 24);
        if ( v11 )
        {
          v143 = *(_QWORD *)(v5 - 88);
          v9 = *(_QWORD *)(v5 - 56);
        }
        sub_AA5980(v143, a1, 0);
        v175[0] = 257;
        v12 = sub_BD2C40(72, 1u);
        v13 = (__int64)v12;
        if ( v12 )
          sub_B4C8F0((__int64)v12, v9, 1u, 0, 0);
        (*(void (__fastcall **)(_QWORD *, __int64, __int64 *, __int64, __int64))(*v184 + 16LL))(
          v184,
          v13,
          &v170,
          v180,
          v181);
        v14 = v176;
        v15 = &v176[16 * (unsigned int)v177];
        if ( v176 != v15 )
        {
          do
          {
            v16 = *((_QWORD *)v14 + 1);
            v17 = *(_DWORD *)v14;
            v14 += 16;
            sub_B99FD0(v13, v17, v16);
          }
          while ( v15 != v14 );
        }
        v7 = (__int64)v6;
        v170 = 18;
        LODWORD(v171) = 30;
        sub_B47C00(v13, (__int64)v6, (int *)&v170, 3);
        sub_B43D60(v6);
        if ( !a4 )
          goto LABEL_76;
        v7 = (__int64)&v170;
        v170 = a1;
        v171 = v143 | 4;
        sub_FFB3D0(a4, &v170, 1);
        v18 = 1;
        goto LABEL_19;
      }
    }
LABEL_18:
    v18 = 0;
    goto LABEL_19;
  }
  if ( v8 == 32 )
  {
    v20 = *(_QWORD **)(v5 - 32);
    v7 = 1;
    v21 = (_BYTE *)*v20;
    v144 = v20[4];
    if ( *(_BYTE *)*v20 != 17 )
      v21 = 0;
    v22 = sub_AA5030(v20[4], 1);
    if ( !v22 )
      BUG();
    v23 = v144;
    v24 = *(_DWORD *)(v5 - 20);
    if ( *(_BYTE *)(v22 - 24) != 36 || (v24 & 0x7FFFFFFu) >> 1 == 1 )
      v25 = v144;
    else
      v25 = *(_QWORD *)(*(_QWORD *)(v5 - 32) + 96LL);
    v26 = 0;
    v27 = ((v24 & 0x7FFFFFFu) >> 1) - 1;
    v18 = 0;
    if ( (v24 & 0x7FFFFFFu) >> 1 == 1 )
    {
LABEL_44:
      if ( v21 )
      {
        if ( !v25 )
          goto LABEL_46;
      }
      else if ( !v25 )
      {
        goto LABEL_121;
      }
      goto LABEL_47;
    }
    v28 = v5;
    v29 = v25;
    v30 = v21;
    v31 = (__int64)v6;
    v32 = v28;
    while ( 1 )
    {
      v7 = *(_QWORD *)(v31 - 8);
      if ( *(_BYTE **)(v7 + 32LL * (unsigned int)(2 * (v26 + 1))) == v30 )
      {
        v117 = v30;
        v118 = 32;
        v5 = v32;
        if ( (_DWORD)v26 != -2 )
          v118 = 32LL * (unsigned int)(2 * v26 + 3);
        v25 = *(_QWORD *)(v7 + v118);
        if ( !v25 )
        {
          if ( !v117 )
          {
LABEL_121:
            if ( (v24 & 0x7FFFFFFu) >> 1 == 2 )
            {
              v166 = (const __m128i *)"cond";
              v169 = 259;
              v100 = *(__int64 **)(v5 - 32);
              v101 = *v100;
              v155 = v100[8];
              v102 = (_QWORD *)(*((__int64 (__fastcall **)(void **, __int64, __int64, __int64, __int64))*v183 + 7))(
                                 v183,
                                 32,
                                 *v100,
                                 v155,
                                 v23);
              if ( !v102 )
              {
                v175[0] = 257;
                v102 = sub_BD2C40(72, unk_3F10FD0);
                if ( v102 )
                {
                  v119 = *(_QWORD ***)(v101 + 8);
                  v120 = *((unsigned __int8 *)v119 + 8);
                  if ( (unsigned int)(v120 - 17) > 1 )
                  {
                    v123 = sub_BCB2A0(*v119);
                  }
                  else
                  {
                    v121 = *((_DWORD *)v119 + 8);
                    v165.m128i_i8[4] = (_BYTE)v120 == 18;
                    v165.m128i_i32[0] = v121;
                    v122 = (__int64 *)sub_BCB2A0(*v119);
                    v123 = sub_BCE1B0(v122, v165.m128i_i64[0]);
                  }
                  sub_B523C0((__int64)v102, v123, 53, 32, v101, v155, (__int64)&v170, 0, 0, 0);
                }
                (*(void (__fastcall **)(_QWORD *, _QWORD *, const __m128i **, __int64, __int64))(*v184 + 16LL))(
                  v184,
                  v102,
                  &v166,
                  v180,
                  v181);
                if ( v176 != &v176[16 * (unsigned int)v177] )
                {
                  v158 = v5;
                  v124 = v176;
                  v125 = &v176[16 * (unsigned int)v177];
                  do
                  {
                    v126 = *((_QWORD *)v124 + 1);
                    v127 = *(_DWORD *)v124;
                    v124 += 16;
                    sub_B99FD0((__int64)v102, v127, v126);
                  }
                  while ( v125 != v124 );
                  v5 = v158;
                }
              }
              v103 = *(_QWORD *)(v5 - 32);
              v104 = *(_QWORD *)(v103 + 32);
              v105 = *(_QWORD *)(v103 + 96);
              v175[0] = 257;
              v151 = v104;
              v153 = v105;
              v106 = sub_BD2C40(72, 3u);
              v107 = (__int64)v106;
              if ( v106 )
                sub_B4C9A0((__int64)v106, v153, v151, (__int64)v102, 3u, 0, 0, 0);
              (*(void (__fastcall **)(_QWORD *, __int64, __int64 *, __int64, __int64))(*v184 + 16LL))(
                v184,
                v107,
                &v170,
                v180,
                v181);
              v108 = 16LL * (unsigned int)v177;
              if ( v176 != &v176[v108] )
              {
                v156 = v5;
                v109 = &v176[v108];
                v110 = v176;
                do
                {
                  v111 = *((_QWORD *)v110 + 1);
                  v112 = *(_DWORD *)v110;
                  v110 += 16;
                  sub_B99FD0(v107, v112, v111);
                }
                while ( v109 != v110 );
                v5 = v156;
              }
              v7 = (__int64)&v170;
              v170 = (__int64)&v172;
              v171 = 0xC00000000LL;
              if ( (unsigned __int8)sub_BC8C10((__int64)v6, (__int64)&v170) && (_DWORD)v171 == 2 )
              {
                v113 = *(_DWORD *)(v170 + 4);
                v157 = *(_DWORD *)v170;
                v166 = (const __m128i *)sub_AA48A0(a1);
                v114 = sub_B8C2F0(&v166, v113, v157, 0);
                v7 = 2;
                sub_B99FD0(v107, 2u, v114);
              }
              if ( (*(_BYTE *)(v5 - 17) & 0x20) != 0 )
              {
                v7 = 14;
                v115 = sub_B91C10((__int64)v6, 14);
                if ( v115 )
                {
                  v7 = 14;
                  sub_B99FD0(v107, 0xEu, v115);
                }
              }
              sub_B43D60(v6);
              v116 = v170;
              if ( (__int64 *)v170 != &v172 )
LABEL_136:
                _libc_free(v116, v7);
              goto LABEL_76;
            }
            goto LABEL_19;
          }
LABEL_46:
          v25 = *(_QWORD *)(*(_QWORD *)(v5 - 32) + 32LL);
          if ( !v25 )
            goto LABEL_121;
        }
LABEL_47:
        v161 = v25;
        v175[0] = 257;
        v47 = sub_BD2C40(72, 1u);
        v48 = v161;
        v49 = (__int64)v47;
        if ( v47 )
        {
          sub_B4C8F0((__int64)v47, v161, 1u, 0, 0);
          v48 = v161;
        }
        v162 = v48;
        (*(void (__fastcall **)(_QWORD *, __int64, __int64 *, __int64, __int64))(*v184 + 16LL))(
          v184,
          v49,
          &v170,
          v180,
          v181);
        v50 = v162;
        v51 = 16LL * (unsigned int)v177;
        v7 = (__int64)&v176[v51];
        if ( v176 != &v176[v51] )
        {
          v52 = (unsigned int *)&v176[v51];
          v147 = v5;
          v53 = (unsigned int *)v176;
          do
          {
            v54 = *((_QWORD *)v53 + 1);
            v7 = *v53;
            v53 += 4;
            sub_B99FD0(v49, v7, v54);
          }
          while ( v52 != v53 );
          v50 = v162;
          v5 = v147;
        }
        v55 = *(_QWORD *)(v5 + 16);
        v148 = v50;
        v170 = 0;
        v171 = (__int64)v175;
        v172 = 8;
        v173 = 0;
        v174 = 1;
        v163 = sub_B46E30((__int64)v6);
        if ( !v163 )
        {
LABEL_66:
          v63 = **(char ***)(v5 - 32);
          sub_B43D60(v6);
          if ( a2 )
          {
            v7 = (__int64)a3;
            v168 = 0;
            sub_F5CAB0(v63, a3, 0, (__int64)&v166);
            if ( v168 )
            {
              v7 = (__int64)&v166;
              ((void (__fastcall *)(const __m128i **, const __m128i **, __int64))v168)(&v166, &v166, 3);
            }
          }
          if ( a4 )
          {
            v166 = 0;
            v167 = 0;
            v168 = 0;
            sub_F58D10(&v166, (unsigned int)(HIDWORD(v172) - v173));
            v64 = (_QWORD *)v171;
            if ( v174 )
              v65 = (_QWORD *)(v171 + 8LL * HIDWORD(v172));
            else
              v65 = (_QWORD *)(v171 + 8LL * (unsigned int)v172);
            v66 = v167;
            if ( (_QWORD *)v171 != v65 )
            {
              while ( 1 )
              {
                v67 = *v64;
                v68 = v64;
                if ( *v64 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v65 == ++v64 )
                  goto LABEL_73;
              }
              if ( v65 != v64 )
              {
                while ( 1 )
                {
                  v165.m128i_i64[0] = v55;
                  v165.m128i_i64[1] = v67 | 4;
                  if ( v66 == v168 )
                  {
                    sub_F38BA0(&v166, v66, &v165);
                    v66 = v167;
                  }
                  else
                  {
                    if ( v66 )
                    {
                      *v66 = _mm_loadu_si128(&v165);
                      v66 = v167;
                    }
                    v167 = ++v66;
                  }
                  v135 = v68 + 1;
                  if ( v68 + 1 == v65 )
                    break;
                  v67 = *v135;
                  for ( ++v68; *v135 >= 0xFFFFFFFFFFFFFFFELL; v68 = v135 )
                  {
                    if ( v65 == ++v135 )
                      goto LABEL_73;
                    v67 = *v135;
                  }
                  if ( v65 == v68 )
                    goto LABEL_73;
                }
              }
            }
            goto LABEL_73;
          }
          goto LABEL_75;
        }
        v137 = v5;
        v56 = v148;
        v57 = 0;
        while ( 2 )
        {
          v7 = v57;
          v60 = sub_B46EC0((__int64)v6, v57);
          if ( v60 != v148 && a4 )
          {
            if ( !v174 )
              goto LABEL_143;
            v62 = (_QWORD *)v171;
            v7 = HIDWORD(v172);
            v58 = (__int64 *)(v171 + 8LL * HIDWORD(v172));
            if ( (__int64 *)v171 == v58 )
            {
LABEL_142:
              if ( HIDWORD(v172) < (unsigned int)v172 )
              {
                v7 = (unsigned int)++HIDWORD(v172);
                *v58 = v60;
                ++v170;
              }
              else
              {
LABEL_143:
                v7 = v60;
                v142 = v60;
                sub_C8CC70((__int64)&v170, v60, (__int64)v58, v59, v60, v61);
                v60 = v142;
              }
            }
            else
            {
              while ( v60 != *v62 )
              {
                if ( v58 == ++v62 )
                  goto LABEL_142;
              }
            }
          }
          if ( v60 == v56 )
          {
            v56 = 0;
          }
          else
          {
            v7 = v55;
            sub_AA5980(v60, v55, 0);
          }
          if ( v163 == ++v57 )
          {
            v5 = v137;
            goto LABEL_66;
          }
          continue;
        }
      }
      v33 = 32;
      if ( (_DWORD)v26 != -2 )
        v33 = 32LL * (unsigned int)(2 * v26 + 3);
      v34 = *(_QWORD *)(v7 + v33);
      if ( v23 != v34 )
        break;
      v140 = v32;
      v145 = v23;
      v35 = sub_BC8A00((__int64)v6);
      v36 = v140;
      v37 = v145;
      v38 = v35;
      if ( ((*(_DWORD *)(v140 - 20) & 0x7FFFFFFu) >> 1) - 1 > 1 )
      {
        if ( v35 )
        {
          v170 = (__int64)&v172;
          v171 = 0x800000000LL;
          sub_BC8BD0(v35, (__int64)&v170);
          v39 = (unsigned int)(v26 + 1);
          *(_DWORD *)v170 += *(_DWORD *)(v170 + 4 * v39);
          v40 = (unsigned int *)(v170 + 4LL * (unsigned int)v171 - 4);
          v41 = (unsigned int *)(v170 + 4 * v39);
          LODWORD(v39) = *v41;
          *v41 = *v40;
          *v40 = v39;
          LODWORD(v171) = v171 - 1;
          v42 = sub_BC8790(v38);
          v43 = v170;
          sub_BC8EC0((__int64)v6, (unsigned int *)v170, (unsigned int)v171, v42);
          v37 = v145;
          v36 = v140;
          if ( (__int64 *)v170 != &v172 )
          {
            _libc_free(v170, v43);
            v36 = v140;
            v37 = v145;
          }
        }
      }
      v141 = v36;
      v146 = v37;
      sub_AA5980(v37, *(_QWORD *)(v36 + 16), 0);
      v7 = v31;
      v44 = sub_B53C80((__int64)v6, v31, v26);
      v32 = v141;
      v23 = v146;
      v31 = v44;
      v24 = *(_DWORD *)(v141 - 20);
      v46 = (v24 & 0x7FFFFFFu) >> 1;
      v27 = v46 - 1;
      if ( ***(_BYTE ***)(v141 - 32) == 17 )
      {
        v26 = 0;
        v30 = **(_BYTE ***)(v141 - 32);
        v31 = (__int64)v6;
        v18 = 1;
        if ( v46 == 1 )
        {
LABEL_43:
          v21 = v30;
          v5 = v32;
          v25 = v29;
          goto LABEL_44;
        }
      }
      else
      {
        v26 = v45;
        v18 = 1;
LABEL_32:
        if ( v26 == v27 )
          goto LABEL_43;
      }
    }
    ++v26;
    if ( v34 != v29 )
      v29 = 0;
    goto LABEL_32;
  }
  if ( v8 != 33 )
    goto LABEL_18;
  v70 = sub_BD3990(**(unsigned __int8 ***)(v5 - 32), v7);
  v139 = (__int64 *)v70;
  if ( *v70 != 4 )
    goto LABEL_18;
  v71 = *((_QWORD *)v70 - 4);
  v171 = (__int64)v175;
  v138 = v71;
  v170 = 0;
  v172 = 8;
  v173 = 0;
  v174 = 1;
  v169 = 257;
  v72 = sub_BD2C40(72, 1u);
  v73 = (__int64)v72;
  if ( v72 )
    sub_B4C8F0((__int64)v72, v71, 1u, 0, 0);
  (*(void (__fastcall **)(_QWORD *, __int64, const __m128i **, __int64, __int64))(*v184 + 16LL))(
    v184,
    v73,
    &v166,
    v180,
    v181);
  v75 = 16LL * (unsigned int)v177;
  v7 = (__int64)&v176[v75];
  if ( v176 != &v176[v75] )
  {
    v149 = v5;
    v76 = (unsigned int *)&v176[v75];
    v77 = (unsigned int *)v176;
    do
    {
      v78 = *((_QWORD *)v77 + 1);
      v7 = *v77;
      v77 += 4;
      sub_B99FD0(v73, v7, v78);
    }
    while ( v76 != v77 );
    v5 = v149;
  }
  v79 = *(_DWORD *)(v5 - 20) & 0x7FFFFFF;
  if ( v79 == 1 )
  {
    v87 = v138;
    v81 = *(char ***)(v5 - 32);
    goto LABEL_101;
  }
  v7 = (__int64)&v170;
  v80 = v138;
  v81 = *(char ***)(v5 - 32);
  v82 = 32 * ((unsigned int)(v79 - 2) + 2LL);
  v83 = (char *)v138;
  v84 = 32;
  do
  {
    v85 = (__int64)v81[(unsigned __int64)v84 / 8];
    if ( v138 == v85 || !a4 )
      goto LABEL_97;
    if ( !v174 )
    {
LABEL_146:
      v7 = (__int64)v81[(unsigned __int64)v84 / 8];
      sub_C8CC70((__int64)&v170, v7, (__int64)v81, v80, v85, v74);
      v81 = *(char ***)(v5 - 32);
      v85 = v7;
      goto LABEL_97;
    }
    v86 = (_QWORD *)v171;
    v7 = v171 + 8LL * HIDWORD(v172);
    if ( v171 == v7 )
    {
LABEL_145:
      if ( HIDWORD(v172) >= (unsigned int)v172 )
        goto LABEL_146;
      ++HIDWORD(v172);
      *(_QWORD *)v7 = v85;
      ++v170;
      v81 = *(char ***)(v5 - 32);
    }
    else
    {
      while ( v85 != *v86 )
      {
        if ( (_QWORD *)v7 == ++v86 )
          goto LABEL_145;
      }
    }
LABEL_97:
    if ( v81[(unsigned __int64)v84 / 8] == v83 )
    {
      v83 = 0;
    }
    else
    {
      v7 = a1;
      sub_AA5980(v85, a1, 0);
      v81 = *(char ***)(v5 - 32);
    }
    v84 += 32;
  }
  while ( v82 != v84 );
  v4 = a1 + 48;
  v87 = (__int64)v83;
LABEL_101:
  v88 = *v81;
  sub_B43D60(v6);
  if ( a2 )
  {
    v7 = (__int64)a3;
    v168 = 0;
    sub_F5CAB0(v88, a3, 0, (__int64)&v166);
    if ( v168 )
    {
      v7 = (__int64)&v166;
      ((void (__fastcall *)(const __m128i **, const __m128i **, __int64))v168)(&v166, &v166, 3);
    }
  }
  if ( !v139[2] )
    sub_ACFDF0(v139, v7, v89);
  if ( v87 )
  {
    v90 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v4 == v90 )
    {
      v92 = 0;
    }
    else
    {
      if ( !v90 )
        BUG();
      v91 = *(unsigned __int8 *)(v90 - 24);
      v92 = 0;
      v93 = (_QWORD *)(v90 - 24);
      if ( (unsigned int)(v91 - 30) < 0xB )
        v92 = v93;
    }
    sub_B43D60(v92);
    v94 = sub_AA48A0(a1);
    sub_B43C20((__int64)&v166, a1);
    v7 = unk_3F148B8;
    v95 = sub_BD2C40(72, unk_3F148B8);
    if ( v95 )
    {
      v7 = v94;
      sub_B4C8A0((__int64)v95, v94, (__int64)v166, (unsigned __int16)v167);
    }
  }
  if ( a4 )
  {
    v166 = 0;
    v167 = 0;
    v168 = 0;
    sub_F58D10(&v166, (unsigned int)(HIDWORD(v172) - v173));
    v96 = (_QWORD *)v171;
    if ( v174 )
      v97 = (_QWORD *)(v171 + 8LL * HIDWORD(v172));
    else
      v97 = (_QWORD *)(v171 + 8LL * (unsigned int)v172);
    v66 = v167;
    if ( (_QWORD *)v171 != v97 )
    {
      while ( 1 )
      {
        v98 = *v96;
        v99 = v96;
        if ( *v96 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v97 == ++v96 )
          goto LABEL_73;
      }
      if ( v97 != v96 )
      {
        do
        {
          v165.m128i_i64[0] = a1;
          v165.m128i_i64[1] = v98 | 4;
          if ( v66 == v168 )
          {
            sub_F38BA0(&v166, v66, &v165);
            v66 = v167;
          }
          else
          {
            if ( v66 )
            {
              *v66 = _mm_loadu_si128(&v165);
              v66 = v167;
            }
            v167 = ++v66;
          }
          v136 = v99 + 1;
          if ( v99 + 1 == v97 )
            break;
          v98 = *v136;
          for ( ++v99; *v136 >= 0xFFFFFFFFFFFFFFFELL; v99 = v136 )
          {
            if ( v97 == ++v136 )
              goto LABEL_73;
            v98 = *v136;
          }
        }
        while ( v97 != v99 );
      }
    }
LABEL_73:
    v69 = (char *)v66 - (char *)v166;
    v7 = (__int64)v166;
    sub_FFB3D0(a4, v166, v69 >> 4);
    if ( v166 )
    {
      v7 = (char *)v168 - (char *)v166;
      j_j___libc_free_0(v166, (char *)v168 - (char *)v166);
    }
  }
LABEL_75:
  if ( !v174 )
  {
    v116 = v171;
    goto LABEL_136;
  }
LABEL_76:
  v18 = 1;
LABEL_19:
  v160 = v18;
  nullsub_61();
  v191 = &unk_49DA100;
  nullsub_63();
  result = v160;
  if ( v176 != v178 )
  {
    _libc_free(v176, v7);
    return v160;
  }
  return result;
}
