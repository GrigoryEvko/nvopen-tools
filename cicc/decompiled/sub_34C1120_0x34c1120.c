// Function: sub_34C1120
// Address: 0x34c1120
//
__int64 __fastcall sub_34C1120(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  bool v3; // r12
  unsigned int v4; // edi
  __int64 *v5; // rsi
  unsigned int v6; // r8d
  unsigned int v7; // ecx
  __int64 *v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // ecx
  __int64 *v11; // rdx
  __int64 v12; // r10
  __int64 v13; // rdi
  __int64 (*v14)(); // rax
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 (*v17)(); // rax
  char v18; // r15
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdi
  __int64 (*v22)(); // rax
  __int64 *v23; // r11
  __int64 *v24; // r13
  __int64 v25; // r15
  __int64 v26; // rdi
  __int64 (*v27)(); // rax
  __int64 *v28; // r15
  __int64 **v29; // r13
  __int64 *v30; // rdi
  __int64 *v32; // rbx
  __int64 v33; // r12
  __int64 v34; // r15
  __int64 v35; // rax
  __int64 v36; // r13
  __int64 *v37; // r14
  __int64 v38; // rbx
  __int64 v39; // rdx
  __int64 v40; // r13
  __int64 v41; // rdi
  void (__fastcall *v42)(__int64, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 **, _QWORD); // rax
  __int64 v43; // rsi
  __int64 v44; // rdi
  __int64 (*v45)(); // rax
  bool v46; // bl
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // rdi
  __int64 (*v57)(); // rax
  __int64 v58; // rdi
  __int64 (*v59)(); // rax
  __int64 v60; // rdi
  __int64 (*v61)(); // rax
  __int64 *v62; // r10
  __int64 *v63; // r13
  __int64 *v64; // r15
  __int64 v65; // rdi
  __int64 (*v66)(); // rax
  bool v67; // al
  __int64 *v68; // rdi
  __int64 v69; // rax
  __int64 (*v70)(); // r8
  int v71; // eax
  int v72; // edx
  __int64 v73; // rdi
  __int64 (*v74)(); // rax
  __int64 v75; // rax
  unsigned __int64 v76; // rdx
  __int64 v77; // rdi
  __int64 (*v78)(); // rax
  __int64 v79; // rdi
  int v80; // eax
  int v81; // r9d
  int v82; // r9d
  __int64 *v83; // rbx
  __int64 *v84; // r12
  __int64 v85; // rdi
  __int64 *v86; // rbx
  __int64 v87; // r15
  __int64 v88; // r12
  unsigned __int64 v89; // rax
  __int64 v90; // r13
  __int64 *v91; // r14
  unsigned __int64 v92; // rbx
  bool v93; // bl
  __int64 v94; // rax
  __int64 *v95; // r13
  __int64 *v96; // r12
  __int64 v97; // rcx
  __int64 v98; // r8
  __int64 v99; // r9
  unsigned __int64 v100; // r12
  int v101; // eax
  char v102; // al
  int v103; // eax
  char v104; // al
  unsigned __int64 v105; // rbx
  __int64 v106; // r13
  __int64 v107; // rsi
  __int64 v108; // rdx
  __int64 v109; // rax
  int v110; // ecx
  __int64 v111; // r12
  __int64 v112; // rdi
  __int64 v113; // rax
  __int64 v114; // r15
  bool v115; // dl
  __int64 v116; // rcx
  __int64 v117; // r12
  __int64 *v118; // rdx
  __int64 *v119; // r12
  __int64 v120; // rcx
  __int64 v121; // r8
  __int64 v122; // r9
  __int64 v123; // rdi
  unsigned __int64 *v124; // r12
  unsigned __int64 v125; // rdx
  __int64 v126; // rax
  __int64 v127; // rdx
  __int64 v128; // rcx
  __int64 v129; // r8
  __int64 v130; // r9
  _QWORD *v131; // rax
  __int64 v132; // r14
  _QWORD *v133; // r13
  _QWORD *v134; // r12
  unsigned __int64 *v135; // rax
  unsigned __int64 v136; // rcx
  __int64 v137; // rdi
  __int64 v138; // rdi
  unsigned __int8 v139; // al
  __int64 v140; // rdi
  __int64 v141; // [rsp+0h] [rbp-350h]
  unsigned __int8 v142; // [rsp+Fh] [rbp-341h]
  char v143; // [rsp+10h] [rbp-340h]
  __int64 v144; // [rsp+10h] [rbp-340h]
  unsigned __int8 v145; // [rsp+10h] [rbp-340h]
  unsigned int v146; // [rsp+28h] [rbp-328h]
  unsigned int v147; // [rsp+2Ch] [rbp-324h]
  unsigned __int8 v148; // [rsp+30h] [rbp-320h]
  bool v149; // [rsp+40h] [rbp-310h]
  unsigned __int8 v150; // [rsp+40h] [rbp-310h]
  __int64 v151; // [rsp+40h] [rbp-310h]
  _QWORD *v152; // [rsp+48h] [rbp-308h]
  __int64 v153; // [rsp+50h] [rbp-300h]
  unsigned __int64 v154; // [rsp+58h] [rbp-2F8h]
  __int64 *v155; // [rsp+58h] [rbp-2F8h]
  unsigned __int8 v156; // [rsp+58h] [rbp-2F8h]
  unsigned __int64 v157; // [rsp+60h] [rbp-2F0h]
  __int64 v158; // [rsp+60h] [rbp-2F0h]
  __int64 v159; // [rsp+68h] [rbp-2E8h]
  char v160; // [rsp+70h] [rbp-2E0h]
  void (__fastcall *v161)(__int64, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 **, _QWORD); // [rsp+70h] [rbp-2E0h]
  __int64 v163; // [rsp+80h] [rbp-2D0h]
  __int64 v164; // [rsp+80h] [rbp-2D0h]
  __int64 v165; // [rsp+88h] [rbp-2C8h]
  __int64 *v166; // [rsp+88h] [rbp-2C8h]
  __int64 *i; // [rsp+88h] [rbp-2C8h]
  bool v168; // [rsp+88h] [rbp-2C8h]
  __int64 *j; // [rsp+88h] [rbp-2C8h]
  __int64 v170; // [rsp+88h] [rbp-2C8h]
  unsigned __int8 v171; // [rsp+88h] [rbp-2C8h]
  __int64 v172; // [rsp+98h] [rbp-2B8h] BYREF
  __int64 *v173; // [rsp+A0h] [rbp-2B0h] BYREF
  __int64 v174; // [rsp+A8h] [rbp-2A8h] BYREF
  __int64 v175; // [rsp+B0h] [rbp-2A0h] BYREF
  __int64 v176; // [rsp+B8h] [rbp-298h] BYREF
  __int64 v177; // [rsp+C0h] [rbp-290h] BYREF
  __int64 v178; // [rsp+C8h] [rbp-288h] BYREF
  __int64 *v179; // [rsp+D0h] [rbp-280h] BYREF
  __int64 v180; // [rsp+D8h] [rbp-278h]
  _BYTE v181[48]; // [rsp+E0h] [rbp-270h] BYREF
  _BYTE *v182; // [rsp+110h] [rbp-240h] BYREF
  __int64 v183; // [rsp+118h] [rbp-238h]
  _BYTE v184[160]; // [rsp+120h] [rbp-230h] BYREF
  _BYTE *v185; // [rsp+1C0h] [rbp-190h] BYREF
  __int64 v186; // [rsp+1C8h] [rbp-188h]
  _BYTE v187[160]; // [rsp+1D0h] [rbp-180h] BYREF
  __int64 *v188; // [rsp+270h] [rbp-E0h] BYREF
  __int64 v189; // [rsp+278h] [rbp-D8h]
  _BYTE v190[208]; // [rsp+280h] [rbp-D0h] BYREF

  v2 = a2;
  v148 = 0;
  v152 = *(_QWORD **)(a2 + 32);
  v146 = *(_DWORD *)(v152[1] + 544LL) - 42;
  v159 = a2 + 48;
  v147 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  while ( 1 )
  {
    v3 = 1;
    v153 = *(_QWORD *)(v2 + 8);
    if ( *(_DWORD *)(a1 + 88) && *(_QWORD **)(v2 + 8) != v152 + 40 )
    {
      v4 = *(_DWORD *)(a1 + 96);
      v5 = *(__int64 **)(a1 + 80);
      if ( v4 )
      {
        v6 = v4 - 1;
        v7 = (v4 - 1) & v147;
        v8 = &v5[2 * v7];
        v9 = *v8;
        if ( v2 != *v8 )
        {
          v71 = 1;
          while ( v9 != -4096 )
          {
            v82 = v71 + 1;
            v7 = v6 & (v71 + v7);
            v8 = &v5[2 * v7];
            v9 = *v8;
            if ( v2 == *v8 )
              goto LABEL_6;
            v71 = v82;
          }
          v8 = &v5[2 * v4];
        }
LABEL_6:
        v10 = v6 & (((unsigned int)v153 >> 9) ^ ((unsigned int)v153 >> 4));
        v11 = &v5[2 * v10];
        v12 = *v11;
        if ( v153 == *v11 )
          goto LABEL_7;
        v72 = 1;
        while ( v12 != -4096 )
        {
          v81 = v72 + 1;
          v10 = v6 & (v72 + v10);
          v11 = &v5[2 * v10];
          v12 = *v11;
          if ( v153 == *v11 )
            goto LABEL_7;
          v72 = v81;
        }
        v11 = &v5[2 * v4];
        v5 = v8;
      }
      else
      {
        v11 = *(__int64 **)(a1 + 80);
      }
      v8 = v5;
LABEL_7:
      v3 = *((_DWORD *)v8 + 2) == *((_DWORD *)v11 + 2);
    }
    v183 = 0x400000000LL;
    v172 = 0;
    v182 = v184;
    v173 = 0;
    v13 = *(_QWORD *)(a1 + 136);
    v160 = 1;
    v14 = *(__int64 (**)())(*(_QWORD *)v13 + 344LL);
    if ( v14 != sub_2DB1AE0 )
      v160 = ((__int64 (__fastcall *)(__int64, __int64, __int64 *, __int64 **, _BYTE **, __int64))v14)(
               v13,
               v2,
               &v172,
               &v173,
               &v182,
               1);
    if ( v159 == sub_2E319B0(v2, 1) && !*(_BYTE *)(v2 + 216) && !*(_BYTE *)(v2 + 217) && !*(_QWORD *)(v2 + 224) && v3 )
    {
      v32 = *(__int64 **)(v2 + 112);
      v33 = *(_QWORD *)(a1 + 136);
      for ( i = &v32[*(unsigned int *)(v2 + 120)]; i != v32; ++v32 )
      {
        v34 = *v32;
        if ( *(_DWORD *)(*v32 + 72) == 1 )
        {
          v35 = sub_2E31210(*v32, *(_QWORD *)(v34 + 56));
          v36 = *(_QWORD *)(v2 + 56);
          if ( v36 != v159 )
          {
            v163 = v2;
            v37 = v32;
            v38 = v35;
            do
            {
              if ( (unsigned __int16)(*(_WORD *)(v36 + 68) - 14) <= 4u )
                (*(void (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v33 + 248LL))(
                  v33,
                  v34,
                  v38,
                  v36);
              v36 = *(_QWORD *)(v36 + 8);
            }
            while ( v36 != v159 );
            v32 = v37;
            v2 = v163;
          }
        }
      }
      v86 = *(__int64 **)(v2 + 64);
      v87 = v33;
      for ( j = &v86[*(unsigned int *)(v2 + 72)]; j != v86; ++v86 )
      {
        v88 = *v86;
        if ( *(_DWORD *)(*v86 + 120) == 1 )
        {
          v89 = sub_2E313E0(*v86);
          v90 = *(_QWORD *)(v2 + 56);
          if ( v90 != v159 )
          {
            v164 = v2;
            v91 = v86;
            v92 = v89;
            do
            {
              if ( (unsigned __int16)(*(_WORD *)(v90 + 68) - 14) <= 4u )
                (*(void (__fastcall **)(__int64, __int64, unsigned __int64, __int64))(*(_QWORD *)v87 + 248LL))(
                  v87,
                  v88,
                  v92,
                  v90);
              v90 = *(_QWORD *)(v90 + 8);
            }
            while ( v90 != v159 );
            v86 = v91;
            v2 = v164;
          }
        }
      }
      if ( *(_DWORD *)(v2 + 72) )
      {
        if ( (_QWORD *)v153 != v152 + 40 && !*(_BYTE *)(v153 + 216) )
        {
          v93 = sub_2E322C0(v2, v153);
          if ( v93 )
          {
            while ( 1 )
            {
              v94 = *(unsigned int *)(v2 + 72);
              if ( !(_DWORD)v94 )
                break;
              sub_2E337A0(*(_QWORD *)(*(_QWORD *)(v2 + 64) + 8 * v94 - 8), v2, v153);
            }
            v95 = *(__int64 **)(v2 + 112);
            v96 = &v95[*(unsigned int *)(v2 + 120)];
            while ( v95 != v96 )
            {
              if ( *v95 != v153 && !sub_2E322C0(v153, *v95) )
                sub_2E34060(v153, v2, v95, v97, v98, v99);
              ++v95;
            }
            v148 = v93;
            v137 = v152[8];
            if ( v137 )
              sub_2E79DC0(v137, v2, v153);
          }
        }
      }
      goto LABEL_51;
    }
    v15 = *(_QWORD *)v2;
    v186 = 0x400000000LL;
    v174 = 0;
    v175 = 0;
    v157 = v15 & 0xFFFFFFFFFFFFFFF8LL;
    v185 = v187;
    v16 = *(_QWORD *)(a1 + 136);
    v17 = *(__int64 (**)())(*(_QWORD *)v16 + 344LL);
    if ( v17 == sub_2DB1AE0
      || (v18 = ((__int64 (__fastcall *)(__int64, unsigned __int64, __int64 *, __int64 *, _BYTE **, __int64))v17)(
                  v16,
                  v157,
                  &v174,
                  &v175,
                  &v185,
                  1)) != 0 )
    {
      v18 = 1;
      goto LABEL_13;
    }
    v50 = v174;
    if ( v174 )
    {
      v51 = v175;
      if ( v174 == v175 )
      {
        v60 = *(_QWORD *)(a1 + 136);
        v61 = *(__int64 (**)())(*(_QWORD *)v60 + 232LL);
        if ( v61 != sub_2FDC480
          && !((unsigned __int8 (__fastcall *)(__int64, unsigned __int64, __int64, __int64))v61)(v60, v157, v174, v174) )
        {
          goto LABEL_49;
        }
        sub_2E32880((__int64 *)&v188, v157);
        (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD))(**(_QWORD **)(a1 + 136) + 360LL))(
          *(_QWORD *)(a1 + 136),
          v157,
          0);
        LODWORD(v186) = 0;
        if ( v174 != v2 )
          (*(void (__fastcall **)(_QWORD, unsigned __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 **, _QWORD))(**(_QWORD **)(a1 + 136) + 368LL))(
            *(_QWORD *)(a1 + 136),
            v157,
            v174,
            0,
            v185,
            0,
            &v188,
            0);
        if ( !v188 )
          goto LABEL_107;
        goto LABEL_106;
      }
LABEL_83:
      if ( v2 != v50 )
        goto LABEL_84;
      if ( !v51 )
      {
        v68 = *(__int64 **)(a1 + 136);
        v69 = *v68;
        v70 = *(__int64 (**)())(*v68 + 232);
        if ( v70 != sub_2FDC480 )
        {
          if ( !((unsigned __int8 (__fastcall *)(__int64 *, unsigned __int64, __int64, __int64))v70)(v68, v157, v2, v2) )
            goto LABEL_49;
          v68 = *(__int64 **)(a1 + 136);
          v69 = *v68;
        }
        (*(void (__fastcall **)(__int64 *, unsigned __int64, _QWORD))(v69 + 360))(v68, v157, 0);
        goto LABEL_107;
      }
      if ( v2 != v51 )
      {
        v189 = 0x400000000LL;
        v188 = (__int64 *)v190;
        if ( (_DWORD)v186 )
        {
          sub_34BE650((__int64)&v188, (__int64)&v185, (unsigned int)v186, v47, v48, v49);
          v58 = *(_QWORD *)(a1 + 136);
          v59 = *(__int64 (**)())(*(_QWORD *)v58 + 880LL);
          if ( v59 == sub_2DB1B20 )
          {
LABEL_178:
            if ( v188 != (__int64 *)v190 )
              _libc_free((unsigned __int64)v188);
            goto LABEL_85;
          }
        }
        else
        {
          v58 = *(_QWORD *)(a1 + 136);
          v59 = *(__int64 (**)())(*(_QWORD *)v58 + 880LL);
          if ( v59 == sub_2DB1B20 )
            goto LABEL_85;
        }
        if ( !((unsigned __int8 (__fastcall *)(__int64, __int64 **))v59)(v58, &v188) )
        {
          sub_2E32880((__int64 *)&v179, v157);
          (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD))(**(_QWORD **)(a1 + 136) + 360LL))(
            *(_QWORD *)(a1 + 136),
            v157,
            0);
          (*(void (__fastcall **)(_QWORD, unsigned __int64, __int64, _QWORD, __int64 *, _QWORD, __int64 **, _QWORD))(**(_QWORD **)(a1 + 136) + 368LL))(
            *(_QWORD *)(a1 + 136),
            v157,
            v175,
            0,
            v188,
            (unsigned int)v189,
            &v179,
            0);
          goto LABEL_153;
        }
        goto LABEL_178;
      }
LABEL_147:
      sub_2E32880((__int64 *)&v188, v157);
      (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD))(**(_QWORD **)(a1 + 136) + 360LL))(
        *(_QWORD *)(a1 + 136),
        v157,
        0);
      (*(void (__fastcall **)(_QWORD, unsigned __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 **))(**(_QWORD **)(a1 + 136) + 368LL))(
        *(_QWORD *)(a1 + 136),
        v157,
        v174,
        0,
        v185,
        (unsigned int)v186,
        &v188);
      if ( !v188 )
        goto LABEL_107;
LABEL_106:
      sub_B91220((__int64)&v188, (__int64)v188);
      goto LABEL_107;
    }
    if ( !(_DWORD)v186 && *(_DWORD *)(v2 + 72) == 1 && *(_DWORD *)(v157 + 120) == 1 )
    {
      v67 = sub_2E322C0(v157, v2);
      if ( v67 && !*(_BYTE *)(v2 + 217) && !*(_QWORD *)(v2 + 224) && !*(_BYTE *)(v2 + 216) )
      {
        v171 = v67;
        v105 = v157 + 48;
        if ( v157 + 48 != (*(_QWORD *)(v157 + 48) & 0xFFFFFFFFFFFFFFF8LL) )
        {
          v188 = (__int64 *)(v157 + 48);
          sub_34C10E0((unsigned __int64 *)&v188);
          v106 = *(_QWORD *)(v2 + 56);
          while ( 1 )
          {
            v107 = (__int64)v188;
            if ( *(__int64 **)(v157 + 56) == v188
              || v106 == v159
              || !sub_34BE730(*((_WORD *)v188 + 34))
              || !sub_34BE730(*(_WORD *)(v106 + 68))
              || !sub_2E88AF0(v106, v107, 0) )
            {
              break;
            }
            v108 = v106;
            if ( (*(_BYTE *)v106 & 4) == 0 )
            {
              v109 = v106;
              do
              {
                v110 = *(_DWORD *)(v109 + 44);
                v108 = v109;
                v109 = *(_QWORD *)(v109 + 8);
              }
              while ( (v110 & 8) != 0 );
            }
            v111 = *(_QWORD *)(v108 + 8);
            sub_34C10E0((unsigned __int64 *)&v188);
            v112 = v106;
            v106 = v111;
            sub_2E88E20(v112);
          }
        }
        v124 = *(unsigned __int64 **)(v2 + 56);
        if ( v124 != (unsigned __int64 *)v159 && v105 != v159 )
        {
          sub_2E310C0((__int64 *)(v157 + 40), (__int64 *)(v2 + 40), *(_QWORD *)(v2 + 56), v159);
          v125 = *(_QWORD *)(v2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)((*v124 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v159;
          *(_QWORD *)(v2 + 48) = *(_QWORD *)(v2 + 48) & 7LL | *v124 & 0xFFFFFFFFFFFFFFF8LL;
          v126 = *(_QWORD *)(v157 + 48);
          *(_QWORD *)(v125 + 8) = v105;
          *v124 = v126 & 0xFFFFFFFFFFFFFFF8LL | *v124 & 7;
          *(_QWORD *)((v126 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v124;
          *(_QWORD *)(v157 + 48) = v125 | *(_QWORD *)(v157 + 48) & 7LL;
        }
        sub_2E33590(v157, *(__int64 **)(v157 + 112), 0);
        sub_2E340B0(v157, v2, v127, v128, v129, v130);
        v148 = v171;
        goto LABEL_49;
      }
      v50 = v174;
      v51 = v175;
      goto LABEL_83;
    }
    v51 = v175;
LABEL_84:
    if ( v2 == v51 )
      goto LABEL_147;
LABEL_85:
    if ( !*(_DWORD *)(v2 + 120) && (_DWORD)v186 && !v175 && v174 == v153 && !sub_2E32580((__int64 *)v2) )
    {
      if ( v153 != (v152[40] & 0xFFFFFFFFFFFFFFF8LL) )
        goto LABEL_91;
      v170 = v174;
      v100 = sub_2E31A10(v174, 1);
      v154 = sub_2E31A10(v2, 1);
      if ( v154 != v159 && v100 != v170 + 48 )
      {
        if ( sub_2E322C0(v170, v2)
          || !sub_2E322C0(v2, v170)
          && ((v55 = v154, v101 = *(_DWORD *)(v154 + 44), (v101 & 4) != 0) || (v101 & 8) == 0
            ? (v102 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v154 + 16) + 24LL) >> 7)
            : (v102 = sub_2E88A90(v154, 128, 1)),
              v102
           && ((v103 = *(_DWORD *)(v100 + 44), (v103 & 4) != 0) || (v103 & 8) == 0
             ? (v104 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v100 + 16) + 24LL) >> 7)
             : (v104 = sub_2E88A90(v100, 128, 1)),
               !v104)) )
        {
LABEL_91:
          v189 = 0x400000000LL;
          v188 = (__int64 *)v190;
          if ( (_DWORD)v186 )
          {
            sub_34BE650((__int64)&v188, (__int64)&v185, v52, v53, v54, v55);
            v56 = *(_QWORD *)(a1 + 136);
            v57 = *(__int64 (**)())(*(_QWORD *)v56 + 880LL);
            if ( v57 == sub_2DB1B20 )
            {
LABEL_220:
              if ( v188 != (__int64 *)v190 )
                _libc_free((unsigned __int64)v188);
              goto LABEL_93;
            }
          }
          else
          {
            v56 = *(_QWORD *)(a1 + 136);
            v57 = *(__int64 (**)())(*(_QWORD *)v56 + 880LL);
            if ( v57 == sub_2DB1B20 )
            {
LABEL_93:
              v18 = 0;
              goto LABEL_13;
            }
          }
          if ( !((unsigned __int8 (__fastcall *)(__int64, __int64 **))v57)(v56, &v188) )
          {
            sub_2E32880((__int64 *)&v179, v157);
            (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD))(**(_QWORD **)(a1 + 136) + 360LL))(
              *(_QWORD *)(a1 + 136),
              v157,
              0);
            (*(void (__fastcall **)(_QWORD, unsigned __int64, __int64, _QWORD, __int64 *, _QWORD, __int64 **, _QWORD))(**(_QWORD **)(a1 + 136) + 368LL))(
              *(_QWORD *)(a1 + 136),
              v157,
              v2,
              0,
              v188,
              (unsigned int)v189,
              &v179,
              0);
            sub_2E320F0((__int64 *)v2, v152[40] & 0xFFFFFFFFFFFFFFF8LL);
            sub_9C6650(&v179);
            if ( v188 != (__int64 *)v190 )
              _libc_free((unsigned __int64)v188);
            v148 = 1;
            goto LABEL_49;
          }
          goto LABEL_220;
        }
      }
    }
LABEL_13:
    if ( v159 == sub_2E319B0(v2, 1) )
      goto LABEL_15;
    v165 = sub_2E319B0(v2, 1);
    v21 = *(_QWORD *)(a1 + 136);
    v22 = *(__int64 (**)())(*(_QWORD *)v21 + 944LL);
    if ( v22 == sub_2FDC700 )
      goto LABEL_15;
    v142 = ((__int64 (__fastcall *)(__int64, __int64))v22)(v21, v165);
    if ( !v142 )
      goto LABEL_15;
    v62 = *(__int64 **)(v2 + 64);
    v179 = (__int64 *)v181;
    v180 = 0x600000000LL;
    if ( v62 == &v62[*(unsigned int *)(v2 + 72)] )
      goto LABEL_15;
    v63 = &v62[*(unsigned int *)(v2 + 72)];
    v143 = v18;
    v64 = v62;
    do
    {
      while ( 1 )
      {
        v65 = *(_QWORD *)(a1 + 136);
        v188 = (__int64 *)v190;
        v177 = 0;
        v178 = 0;
        v189 = 0x400000000LL;
        v66 = *(__int64 (**)())(*(_QWORD *)v65 + 344LL);
        if ( v66 != sub_2DB1AE0 )
        {
          if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, __int64 **, __int64))v66)(
                  v65,
                  *v64,
                  &v177,
                  &v178,
                  &v188,
                  1) )
          {
            if ( (_DWORD)v189 )
            {
              if ( v177 == v2 && v178 != v2 )
              {
                v73 = *(_QWORD *)(a1 + 136);
                v74 = *(__int64 (**)())(*(_QWORD *)v73 + 952LL);
                if ( v74 != sub_2FDC710 )
                {
                  if ( ((unsigned __int8 (__fastcall *)(__int64, __int64 **, __int64))v74)(v73, &v188, v165) )
                  {
                    (*(void (__fastcall **)(_QWORD, __int64, __int64 **, __int64))(**(_QWORD **)(a1 + 136) + 960LL))(
                      *(_QWORD *)(a1 + 136),
                      *v64,
                      &v188,
                      v165);
                    v75 = (unsigned int)v180;
                    v19 = *v64;
                    v76 = (unsigned int)v180 + 1LL;
                    if ( v76 > HIDWORD(v180) )
                    {
                      v141 = *v64;
                      sub_C8D5F0((__int64)&v179, v181, v76, 8u, v19, v20);
                      v75 = (unsigned int)v180;
                      v19 = v141;
                    }
                    v179[v75] = v19;
                    LODWORD(v180) = v180 + 1;
                  }
                }
              }
            }
          }
          if ( v188 != (__int64 *)v190 )
            break;
        }
        if ( v63 == ++v64 )
          goto LABEL_122;
      }
      _libc_free((unsigned __int64)v188);
      ++v64;
    }
    while ( v63 != v64 );
LABEL_122:
    v18 = v143;
    if ( (_DWORD)v180 )
      break;
    if ( v179 != (__int64 *)v181 )
      _libc_free((unsigned __int64)v179);
LABEL_15:
    if ( v160 || !v172 )
      goto LABEL_21;
    if ( v173 == (__int64 *)v2 && v172 != v2 && v173 )
    {
      v189 = 0x400000000LL;
      v188 = (__int64 *)v190;
      if ( (_DWORD)v183 )
      {
        sub_34BE650((__int64)&v188, (__int64)&v182, (__int64)v173, (unsigned int)v183, v19, v20);
        v77 = *(_QWORD *)(a1 + 136);
        v78 = *(__int64 (**)())(*(_QWORD *)v77 + 880LL);
        if ( v78 != sub_2DB1B20 )
          goto LABEL_151;
      }
      else
      {
        v77 = *(_QWORD *)(a1 + 136);
        v78 = *(__int64 (**)())(*(_QWORD *)v77 + 880LL);
        if ( v78 == sub_2DB1B20 )
          goto LABEL_19;
LABEL_151:
        if ( !((unsigned __int8 (__fastcall *)(__int64, __int64 **))v78)(v77, &v188) )
        {
          sub_2E32880((__int64 *)&v179, v2);
          (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 136) + 360LL))(
            *(_QWORD *)(a1 + 136),
            v2,
            0);
          (*(void (__fastcall **)(_QWORD, __int64, __int64 *, __int64, __int64 *, _QWORD, __int64 **, _QWORD))(**(_QWORD **)(a1 + 136) + 368LL))(
            *(_QWORD *)(a1 + 136),
            v2,
            v173,
            v172,
            v188,
            (unsigned int)v189,
            &v179,
            0);
LABEL_153:
          if ( v179 )
            sub_B91220((__int64)&v179, (__int64)v179);
          goto LABEL_36;
        }
      }
      if ( v188 != (__int64 *)v190 )
        _libc_free((unsigned __int64)v188);
      if ( !v172 )
        goto LABEL_21;
    }
LABEL_19:
    if ( (_DWORD)v183 || v173 )
      goto LABEL_21;
    v79 = sub_2E319B0(v2, 1);
    v80 = *(_DWORD *)(v79 + 44);
    if ( (v80 & 4) != 0 || (v80 & 8) == 0 )
      v168 = (*(_QWORD *)(*(_QWORD *)(v79 + 16) + 24LL) & 0x400LL) != 0;
    else
      v168 = sub_2E88A90(v79, 1024, 1);
    if ( !v168 || v172 == v2 || *(_BYTE *)(v2 + 217) || *(_QWORD *)(v2 + 224) || *(_BYTE *)(v2 + 216) )
      goto LABEL_21;
    sub_2E32880(&v176, v2);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 136) + 360LL))(*(_QWORD *)(a1 + 136), v2, 0);
    if ( v159 == sub_2E319B0(v2, 1) )
    {
      v131 = *(_QWORD **)(v2 + 56);
      if ( v131 != (_QWORD *)v159 )
      {
        v151 = v2;
        v132 = v2 + 40;
        v133 = v131;
        do
        {
          v134 = v133;
          v133 = (_QWORD *)v133[1];
          sub_2E31080(v132, (__int64)v134);
          v135 = (unsigned __int64 *)v134[1];
          v136 = *v134 & 0xFFFFFFFFFFFFFFF8LL;
          *v135 = v136 | *v135 & 7;
          *(_QWORD *)(v136 + 8) = v135;
          *v134 &= 7uLL;
          v134[1] = 0;
          sub_2E310F0(v132);
        }
        while ( (_QWORD *)v159 != v133 );
        v2 = v151;
      }
    }
    if ( v159 != (*(_QWORD *)(v2 + 48) & 0xFFFFFFFFFFFFFFF8LL) )
      goto LABEL_165;
    if ( !sub_2E32580((__int64 *)v157) )
      goto LABEL_261;
    if ( !v18 || !sub_2E322C0(v157, v2) )
    {
      if ( sub_2E322C0(v157, v2) && v174 != v2 && v175 != v2 )
      {
        if ( v174 )
          v175 = v2;
        else
          v174 = v2;
        sub_2E32880((__int64 *)&v188, v157);
        (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD))(**(_QWORD **)(a1 + 136) + 360LL))(
          *(_QWORD *)(a1 + 136),
          v157,
          0);
        (*(void (__fastcall **)(_QWORD, unsigned __int64, __int64, __int64, _BYTE *, _QWORD, __int64 **, _QWORD))(**(_QWORD **)(a1 + 136) + 368LL))(
          *(_QWORD *)(a1 + 136),
          v157,
          v174,
          v175,
          v185,
          (unsigned int)v186,
          &v188,
          0);
        sub_9C6650(&v188);
      }
LABEL_261:
      v150 = 0;
      v113 = *(unsigned int *)(v2 + 72);
      v114 = 0;
      v115 = 0;
      while ( v113 != v114 )
      {
        v116 = *(_QWORD *)(v2 + 64);
        v117 = *(_QWORD *)(v116 + 8 * v114);
        if ( v2 == v117 )
        {
          ++v114;
          v150 = v168;
        }
        else
        {
          sub_2E337A0(*(_QWORD *)(v116 + 8 * v114), v2, v172);
          v118 = *(__int64 **)(v2 + 112);
          v155 = &v118[*(unsigned int *)(v2 + 120)];
          if ( v155 != v118 )
          {
            v144 = v117;
            v119 = *(__int64 **)(v2 + 112);
            do
            {
              if ( *v119 != v172 && !sub_2E322C0(v172, *v119) )
                sub_2E34060(v172, v2, v119, v120, v121, v122);
              ++v119;
            }
            while ( v119 != v155 );
            v117 = v144;
          }
          v123 = *(_QWORD *)(a1 + 136);
          v177 = 0;
          v188 = (__int64 *)v190;
          v178 = 0;
          v189 = 0x400000000LL;
          if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64 *, __int64 *, __int64 **, __int64))(*(_QWORD *)v123 + 344LL))(
                  v123,
                  v117,
                  &v177,
                  &v178,
                  &v188,
                  1) )
          {
            if ( v177 )
            {
              if ( v177 == v178 )
              {
                v139 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 136) + 232LL))(
                         *(_QWORD *)(a1 + 136),
                         v117,
                         v177,
                         v177);
                if ( v139 )
                {
                  v145 = v139;
                  sub_2E32880((__int64 *)&v179, v117);
                  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 136) + 360LL))(
                    *(_QWORD *)(a1 + 136),
                    v117,
                    0);
                  v140 = *(_QWORD *)(a1 + 136);
                  LODWORD(v189) = 0;
                  (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64 *, _QWORD, __int64 **, _QWORD))(*(_QWORD *)v140 + 368LL))(
                    v140,
                    v117,
                    v177,
                    0,
                    v188,
                    0,
                    &v179,
                    0);
                  sub_9C6650(&v179);
                  v148 = v145;
                }
              }
            }
          }
          if ( v188 != (__int64 *)v190 )
            _libc_free((unsigned __int64)v188);
          v115 = v168;
          v113 = *(unsigned int *)(v2 + 72);
        }
      }
      v156 = v115;
      v138 = v152[8];
      if ( v138 )
        sub_2E79DC0(v138, v2, v172);
      if ( v156 )
      {
        if ( !v150 )
        {
          sub_9C6650(&v176);
          v148 = v156;
          goto LABEL_49;
        }
        v148 = v150;
      }
    }
LABEL_165:
    (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 *, _QWORD))(**(_QWORD **)(a1 + 136)
                                                                                                 + 368LL))(
      *(_QWORD *)(a1 + 136),
      v2,
      v172,
      0,
      v182,
      (unsigned int)v183,
      &v176,
      0);
    sub_9C6650(&v176);
LABEL_21:
    if ( sub_2E32580((__int64 *)v157) )
      goto LABEL_49;
    v149 = sub_2E32580((__int64 *)v2);
    if ( *(_BYTE *)(v2 + 216) || (v23 = *(__int64 **)(v2 + 64), v166 = &v23[*(unsigned int *)(v2 + 72)], v23 == v166) )
    {
LABEL_38:
      if ( v149 )
        goto LABEL_49;
      if ( v160 )
      {
LABEL_47:
        v178 = 0;
        v189 = 0x400000000LL;
        v179 = 0;
        v188 = (__int64 *)v190;
        if ( (_QWORD *)v153 != v152 + 40 && !*(_BYTE *)(v153 + 216) )
        {
          v44 = *(_QWORD *)(a1 + 136);
          v45 = *(__int64 (**)())(*(_QWORD *)v44 + 344LL);
          if ( v45 != sub_2DB1AE0 )
          {
            if ( !((unsigned __int8 (__fastcall *)(__int64, unsigned __int64, __int64 *, __int64 **, __int64 **, __int64))v45)(
                    v44,
                    v157,
                    &v178,
                    &v179,
                    &v188,
                    1)
              && (v46 = sub_2E322C0(v157, v153)) )
            {
              sub_2E320F0((__int64 *)v2, v152[40] & 0xFFFFFFFFFFFFFFF8LL);
              if ( v188 != (__int64 *)v190 )
                _libc_free((unsigned __int64)v188);
              v148 = v46;
            }
            else if ( v188 != (__int64 *)v190 )
            {
              _libc_free((unsigned __int64)v188);
            }
          }
        }
        goto LABEL_49;
      }
      v28 = v173;
      v189 = v172;
      v188 = v173;
      v29 = &v188;
      if ( !v173 )
        goto LABEL_44;
      while ( 1 )
      {
        v30 = (__int64 *)(*v28 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (__int64 *)v2 != v30 && (__int64 *)v2 != v28 && !sub_2E32580(v30) )
        {
          sub_2E32080((__int64 *)v2, v28);
          goto LABEL_107;
        }
LABEL_44:
        if ( v190 == (_BYTE *)++v29 )
          goto LABEL_47;
        while ( 1 )
        {
          v28 = *v29;
          if ( *v29 )
            break;
          if ( v190 == (_BYTE *)++v29 )
            goto LABEL_47;
        }
      }
    }
    v24 = *(__int64 **)(v2 + 64);
    while ( 1 )
    {
      v25 = *v24;
      v188 = (__int64 *)v190;
      v177 = 0;
      v178 = 0;
      v189 = 0x400000000LL;
      if ( v2 != v25 )
        break;
LABEL_27:
      if ( v166 == ++v24 )
        goto LABEL_38;
    }
    if ( sub_2E32580((__int64 *)v25) )
      goto LABEL_25;
    v26 = *(_QWORD *)(a1 + 136);
    v27 = *(__int64 (**)())(*(_QWORD *)v26 + 344LL);
    if ( v27 == sub_2DB1AE0
      || ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, __int64 **, __int64))v27)(
           v26,
           v25,
           &v177,
           &v178,
           &v188,
           1)
      || v177 != v2 && v178 != v2 )
    {
      goto LABEL_25;
    }
    if ( !v149 )
      goto LABEL_35;
    if ( v172 && v173 || *(_DWORD *)(v25 + 24) > *(_DWORD *)(v2 + 24) )
    {
LABEL_25:
      if ( v188 != (__int64 *)v190 )
        _libc_free((unsigned __int64)v188);
      goto LABEL_27;
    }
    LODWORD(v183) = 0;
    v39 = *(_QWORD *)(v2 + 8);
    v40 = *(_QWORD *)(a1 + 136);
    if ( v146 <= 1 && (_BYTE)qword_503A988 )
    {
      v158 = *(_QWORD *)(v2 + 8);
      v161 = *(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 **, _QWORD))(*(_QWORD *)v40 + 368LL);
      sub_2E32880((__int64 *)&v179, v2);
      v161(v40, v2, v158, 0, v182, (unsigned int)v183, &v179, 0);
      v43 = (__int64)v179;
      if ( v179 )
LABEL_74:
        sub_B91220((__int64)&v179, v43);
    }
    else
    {
      v41 = *(_QWORD *)(a1 + 136);
      v42 = *(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _BYTE *, _QWORD, __int64 **, _QWORD))(*(_QWORD *)v40 + 368LL);
      v179 = 0;
      v42(v41, v2, v39, 0, v182, 0, &v179, 0);
      v43 = (__int64)v179;
      if ( v179 )
        goto LABEL_74;
    }
LABEL_35:
    sub_2E320F0((__int64 *)v2, v25);
LABEL_36:
    if ( v188 != (__int64 *)v190 )
      _libc_free((unsigned __int64)v188);
LABEL_107:
    if ( v185 != v187 )
      _libc_free((unsigned __int64)v185);
    if ( v182 != v184 )
      _libc_free((unsigned __int64)v182);
    v148 = 1;
  }
  v83 = &v179[(unsigned int)v180];
  v84 = v179;
  do
  {
    v85 = *v84++;
    sub_2E33650(v85, v2);
  }
  while ( v83 != v84 );
  if ( v179 != (__int64 *)v181 )
    _libc_free((unsigned __int64)v179);
  v148 = v142;
LABEL_49:
  if ( v185 != v187 )
    _libc_free((unsigned __int64)v185);
LABEL_51:
  if ( v182 != v184 )
    _libc_free((unsigned __int64)v182);
  return v148;
}
