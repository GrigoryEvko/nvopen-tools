// Function: sub_2FC9E30
// Address: 0x2fc9e30
//
__int64 __fastcall sub_2FC9E30(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 *a4, _BYTE *a5)
{
  __int64 *v5; // rax
  __int64 (*v6)(); // rax
  __int64 v7; // rdi
  __int64 (*v8)(); // rax
  __int64 (*v9)(); // rax
  __int64 v10; // r15
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rbx
  unsigned __int64 v14; // r12
  int v15; // ecx
  __int64 v16; // r12
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // rax
  const char *v21; // rsi
  __int64 v22; // r9
  const char *v23; // r12
  unsigned int *v24; // rax
  int v25; // ecx
  unsigned int *v26; // rdx
  __int64 v27; // r12
  __int64 v28; // rax
  char v29; // al
  __int16 v30; // cx
  _QWORD *v31; // rax
  __int64 v32; // rbx
  __int64 v33; // r12
  unsigned int *v34; // r12
  unsigned int *v35; // r13
  __int64 v36; // rdx
  unsigned int v37; // esi
  __int64 v38; // rax
  __int16 v39; // cx
  unsigned int *v40; // rdi
  __int64 v42; // r13
  __int64 v43; // rcx
  __int64 v44; // rax
  __int64 *v45; // rbx
  const char *v46; // rsi
  __int64 v47; // r9
  const char *v48; // r14
  unsigned int *v49; // rax
  int v50; // ecx
  unsigned int *v51; // rdx
  __int64 *v52; // rax
  __int64 *v53; // r14
  __int64 v54; // rbx
  unsigned __int8 v55; // al
  unsigned int v56; // ebx
  _QWORD *v57; // rax
  unsigned int *v58; // r14
  unsigned int *v59; // rbx
  __int64 v60; // rdx
  unsigned int v61; // esi
  __int64 v62; // rax
  char v63; // bl
  __int64 *v64; // rax
  const char *v65; // rsi
  __int64 v66; // r8
  __int64 v67; // r9
  const char *v68; // r12
  unsigned int *v69; // rax
  int v70; // edx
  unsigned int *v71; // rcx
  __int64 v72; // rax
  __int64 v73; // r14
  __int64 v74; // rax
  char v75; // al
  char v76; // bl
  unsigned __int8 *v77; // rax
  __int64 v78; // r9
  unsigned __int8 *v79; // r12
  unsigned int *v80; // r14
  unsigned int *v81; // rbx
  __int64 v82; // rdx
  unsigned int v83; // esi
  __int64 (__fastcall *v84)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v85; // rbx
  unsigned int v86; // r12d
  unsigned int v87; // r14d
  __int64 v88; // r8
  __int64 v89; // rax
  __int64 v90; // rdx
  unsigned __int64 v91; // rax
  __int64 v92; // r12
  unsigned __int8 *v93; // r14
  __int64 v94; // rsi
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // rdx
  int v98; // ecx
  int v99; // eax
  _QWORD *v100; // rdi
  __int64 *v101; // rax
  __int64 v102; // rsi
  unsigned int *v103; // r14
  unsigned int *v104; // r12
  __int64 v105; // rdx
  unsigned int v106; // esi
  __int64 v107; // rax
  __int64 *v108; // r14
  __int64 *v109; // rax
  __int64 v110; // rax
  _QWORD *v111; // rax
  const char *v112; // rbx
  unsigned int *v113; // rax
  int v114; // ecx
  unsigned int *v115; // rsi
  __int64 *v116; // rax
  unsigned __int64 v117; // rax
  __int64 v118; // rax
  __int64 v119; // rdx
  __int64 v120; // rbx
  unsigned __int64 v121; // r14
  _QWORD *v122; // r14
  __int64 v123; // rbx
  unsigned int *v124; // rbx
  unsigned int *v125; // r12
  __int64 v126; // rdx
  unsigned int v127; // esi
  __int64 v128; // rbx
  __int64 *v129; // rax
  unsigned __int64 v130; // rax
  __int64 v131; // rax
  __int64 v132; // rdx
  char *v133; // rax
  signed __int64 v134; // rdx
  __int64 v135; // rax
  __int64 v136; // r9
  __int64 v137; // rdx
  unsigned __int64 v138; // r8
  unsigned __int64 v139; // r8
  unsigned __int64 v140; // rbx
  unsigned __int64 v141; // rbx
  unsigned __int64 v142; // r8
  unsigned __int64 v143; // rbx
  unsigned __int64 v144; // rdx
  unsigned __int64 v145; // r8
  __int64 v146; // [rsp-10h] [rbp-230h]
  __int64 v147; // [rsp-10h] [rbp-230h]
  __int64 v148; // [rsp+8h] [rbp-218h]
  __int64 v149; // [rsp+10h] [rbp-210h]
  __int64 v150; // [rsp+18h] [rbp-208h]
  __int64 v151; // [rsp+20h] [rbp-200h]
  __int64 v152; // [rsp+28h] [rbp-1F8h]
  __int64 v153; // [rsp+28h] [rbp-1F8h]
  unsigned __int64 v154; // [rsp+30h] [rbp-1F0h]
  __int64 v157; // [rsp+50h] [rbp-1D0h]
  char v159; // [rsp+65h] [rbp-1BBh]
  __int16 v160; // [rsp+66h] [rbp-1BAh]
  __int64 v163; // [rsp+78h] [rbp-1A8h]
  __int64 v164; // [rsp+80h] [rbp-1A0h]
  __int64 *v165; // [rsp+88h] [rbp-198h]
  __int64 v166; // [rsp+90h] [rbp-190h]
  unsigned __int8 v167; // [rsp+98h] [rbp-188h]
  __int64 v168; // [rsp+98h] [rbp-188h]
  __int64 v169; // [rsp+98h] [rbp-188h]
  __int64 v170; // [rsp+98h] [rbp-188h]
  __int64 v171; // [rsp+A0h] [rbp-180h]
  __int64 v172; // [rsp+A8h] [rbp-178h]
  __int64 v173; // [rsp+D0h] [rbp-150h]
  __int64 v174; // [rsp+D8h] [rbp-148h]
  char v175; // [rsp+EFh] [rbp-131h] BYREF
  unsigned int v176; // [rsp+F0h] [rbp-130h]
  int v177; // [rsp+F4h] [rbp-12Ch]
  __int64 v178; // [rsp+F8h] [rbp-128h]
  char *v179; // [rsp+100h] [rbp-120h] BYREF
  __int64 v180; // [rsp+108h] [rbp-118h]
  _BYTE v181[16]; // [rsp+110h] [rbp-110h] BYREF
  __int16 v182; // [rsp+120h] [rbp-100h]
  const char *v183; // [rsp+130h] [rbp-F0h] BYREF
  __int64 v184; // [rsp+138h] [rbp-E8h]
  _QWORD v185[2]; // [rsp+140h] [rbp-E0h] BYREF
  __int16 v186; // [rsp+150h] [rbp-D0h]
  unsigned int *v187; // [rsp+160h] [rbp-C0h] BYREF
  __int64 v188; // [rsp+168h] [rbp-B8h]
  _BYTE v189[32]; // [rsp+170h] [rbp-B0h] BYREF
  __int64 v190; // [rsp+190h] [rbp-90h]
  __int64 *v191; // [rsp+198h] [rbp-88h]
  __int64 v192; // [rsp+1A0h] [rbp-80h]
  __int64 *v193; // [rsp+1A8h] [rbp-78h]
  void **v194; // [rsp+1B0h] [rbp-70h]
  _QWORD *v195; // [rsp+1B8h] [rbp-68h]
  __int64 v196; // [rsp+1C0h] [rbp-60h]
  int v197; // [rsp+1C8h] [rbp-58h]
  __int16 v198; // [rsp+1CCh] [rbp-54h]
  char v199; // [rsp+1CEh] [rbp-52h]
  __int64 v200; // [rsp+1D0h] [rbp-50h]
  __int64 v201; // [rsp+1D8h] [rbp-48h]
  void *v202; // [rsp+1E0h] [rbp-40h] BYREF
  _QWORD v203[7]; // [rsp+1E8h] [rbp-38h] BYREF

  v6 = *(__int64 (**)())(*(_QWORD *)a1 + 16LL);
  if ( v6 == sub_23CE270
    || (v163 = *(_QWORD *)(a2 + 40), v7 = v6(), v8 = *(__int64 (**)())(*(_QWORD *)v7 + 144LL), v8 == sub_2C8F680) )
  {
LABEL_199:
    BUG();
  }
  v166 = ((__int64 (__fastcall *)(__int64))v8)(v7);
  v9 = *(__int64 (**)())(*(_QWORD *)v166 + 944LL);
  if ( v9 == sub_2FC91F0 || (v159 = ((__int64 (__fastcall *)(__int64))v9)(v166)) == 0 )
  {
    v159 = qword_5026208;
    if ( (_BYTE)qword_5026208 )
      v159 = ((*(_BYTE *)(a1 + 865) >> 3) ^ 1) & 1;
  }
  v171 = 0;
  v10 = 0;
  v172 = a2 + 72;
  v174 = *(_QWORD *)(a2 + 80);
  if ( v174 != a2 + 72 )
  {
    while ( 1 )
    {
      v11 = v174;
      v12 = v174;
      v164 = v174 - 24;
      v174 = *(_QWORD *)(v174 + 8);
      if ( v171 == v164 )
        goto LABEL_37;
      v13 = v11 + 24;
      v14 = *(_QWORD *)(v12 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v13 == v14 )
        goto LABEL_93;
      if ( !v14 )
        goto LABEL_195;
      v15 = *(unsigned __int8 *)(v14 - 24);
      if ( (unsigned int)(v15 - 30) > 0xA )
LABEL_93:
        BUG();
      v16 = v14 - 24;
      if ( (_BYTE)v15 == 30 )
        break;
      if ( (_BYTE)qword_5026128 )
        goto LABEL_37;
      v42 = *(_QWORD *)(v12 + 32);
      if ( v42 == v13 )
        goto LABEL_37;
      while ( 1 )
      {
        if ( !v42 )
          goto LABEL_195;
        v16 = v42 - 24;
        if ( (unsigned __int8)(*(_BYTE *)(v42 - 24) - 34) <= 0x33u )
        {
          v43 = 0x8000000000041LL;
          if ( _bittest64(&v43, (unsigned int)*(unsigned __int8 *)(v42 - 24) - 34) )
          {
            if ( ((unsigned __int8)sub_A73ED0((_QWORD *)(v42 + 48), 36) || (unsigned __int8)sub_B49560(v42 - 24, 36))
              && !(unsigned __int8)sub_A73ED0((_QWORD *)(v42 + 48), 41)
              && !(unsigned __int8)sub_B49560(v42 - 24, 41) )
            {
              break;
            }
          }
        }
        v42 = *(_QWORD *)(v42 + 8);
        if ( v42 == v13 )
          goto LABEL_37;
      }
      if ( !*a4 )
        goto LABEL_51;
LABEL_13:
      if ( v159 )
        return *a4;
      if ( !v10 )
      {
        v94 = *(_QWORD *)(a2 + 80);
        if ( v172 == v94 )
          goto LABEL_199;
        while ( 1 )
        {
          if ( !v94 )
            goto LABEL_197;
          v95 = *(_QWORD *)(v94 + 32);
          if ( v95 != v94 + 24 )
            break;
LABEL_107:
          v94 = *(_QWORD *)(v94 + 8);
          if ( v172 == v94 )
            goto LABEL_199;
        }
        while ( 1 )
        {
          if ( !v95 )
            goto LABEL_195;
          if ( *(_BYTE *)(v95 - 24) == 85 )
          {
            v96 = *(_QWORD *)(v95 - 56);
            if ( v96 )
            {
              if ( !*(_BYTE *)v96
                && *(_QWORD *)(v96 + 24) == *(_QWORD *)(v95 + 56)
                && (*(_BYTE *)(v96 + 33) & 0x20) != 0
                && *(_DWORD *)(v96 + 36) == 341 )
              {
                break;
              }
            }
          }
          v95 = *(_QWORD *)(v95 + 8);
          if ( v94 + 24 == v95 )
            goto LABEL_107;
        }
        v10 = *(_QWORD *)(v95 + 32 * (1LL - (*(_DWORD *)(v95 - 20) & 0x7FFFFFF)) - 24);
      }
      *a5 = 1;
      v17 = sub_B46BC0(v16, 0);
      if ( v17 )
      {
        if ( *(_BYTE *)v17 == 85 && (*(_WORD *)(v17 + 2) & 3u) - 1 <= 1 )
        {
          v16 = v17;
        }
        else
        {
          v18 = sub_B46BC0(v17, 0);
          if ( v18 && *(_BYTE *)v18 == 85 && (*(_WORD *)(v18 + 2) & 3u) - 1 <= 1 )
            v16 = v18;
        }
      }
      v165 = (__int64 *)(v16 + 24);
      v19 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v166 + 952LL))(v166, v163);
      if ( v19 )
      {
        v193 = (__int64 *)sub_BD5C60(v16);
        v194 = &v202;
        v195 = v203;
        v187 = (unsigned int *)v189;
        v202 = &unk_49DA100;
        v198 = 512;
        v188 = 0x200000000LL;
        v190 = 0;
        v191 = 0;
        v196 = 0;
        v197 = 0;
        v199 = 7;
        v200 = 0;
        v201 = 0;
        LOWORD(v192) = 0;
        v203[0] = &unk_49DA0B0;
        v20 = *(_QWORD *)(v16 + 40);
        v191 = (__int64 *)(v16 + 24);
        v190 = v20;
        v21 = *(const char **)sub_B46C60(v16);
        v183 = v21;
        if ( !v21 || (sub_B96E90((__int64)&v183, (__int64)v21, 1), (v23 = v183) == 0) )
        {
          sub_93FB40((__int64)&v187, 0);
          v23 = v183;
          goto LABEL_110;
        }
        v24 = v187;
        v25 = v188;
        v26 = &v187[4 * (unsigned int)v188];
        if ( v187 == v26 )
        {
LABEL_112:
          if ( (unsigned int)v188 >= (unsigned __int64)HIDWORD(v188) )
          {
            v139 = (unsigned int)v188 + 1LL;
            v140 = v150 & 0xFFFFFFFF00000000LL;
            v150 &= 0xFFFFFFFF00000000LL;
            if ( HIDWORD(v188) < v139 )
            {
              sub_C8D5F0((__int64)&v187, v189, v139, 0x10u, v139, v22);
              v26 = &v187[4 * (unsigned int)v188];
            }
            *(_QWORD *)v26 = v140;
            *((_QWORD *)v26 + 1) = v23;
            v23 = v183;
            LODWORD(v188) = v188 + 1;
          }
          else
          {
            if ( v26 )
            {
              *v26 = 0;
              *((_QWORD *)v26 + 1) = v23;
              v25 = v188;
              v23 = v183;
            }
            LODWORD(v188) = v25 + 1;
          }
LABEL_110:
          if ( v23 )
LABEL_29:
            sub_B91220((__int64)&v183, (__int64)v23);
          v179 = "Guard";
          v182 = 259;
          v27 = sub_BCE3C0(v193, 0);
          v28 = sub_AA4E30(v190);
          v29 = sub_AE5020(v28, v27);
          HIBYTE(v30) = HIBYTE(v160);
          v186 = 257;
          LOBYTE(v30) = v29;
          v160 = v30;
          v31 = sub_BD2C40(80, 1u);
          v32 = (__int64)v31;
          if ( v31 )
            sub_B4D190((__int64)v31, v27, v10, (__int64)&v183, 1, v160, 0, 0);
          (*(void (__fastcall **)(_QWORD *, __int64, char **, __int64 *, __int64))(*v195 + 16LL))(
            v195,
            v32,
            &v179,
            v191,
            v192);
          v33 = 4LL * (unsigned int)v188;
          if ( v187 != &v187[v33] )
          {
            v34 = &v187[v33];
            v35 = v187;
            do
            {
              v36 = *((_QWORD *)v35 + 1);
              v37 = *v35;
              v35 += 4;
              sub_B99FD0(v32, v37, v36);
            }
            while ( v34 != v35 );
          }
          v186 = 257;
          v179 = (char *)v32;
          v38 = sub_921880(&v187, *(_QWORD *)(v19 + 24), v19, (int)&v179, 1, (__int64)&v183, 0);
          v39 = *(_WORD *)(v38 + 2);
          *(_QWORD *)(v38 + 72) = *(_QWORD *)(v19 + 120);
          *(_WORD *)(v38 + 2) = v39 & 0xF003 | (4 * ((*(_WORD *)(v19 + 2) >> 4) & 0x3FF));
          nullsub_61();
          v202 = &unk_49DA100;
          nullsub_63();
          v40 = v187;
          if ( v187 == (unsigned int *)v189 )
            goto LABEL_37;
LABEL_36:
          _libc_free((unsigned __int64)v40);
          goto LABEL_37;
        }
        while ( 1 )
        {
          v22 = *v24;
          if ( !(_DWORD)v22 )
            break;
          v24 += 4;
          if ( v26 == v24 )
            goto LABEL_112;
        }
        *((_QWORD *)v24 + 1) = v183;
        goto LABEL_29;
      }
      if ( !v171 )
      {
        v168 = *(_QWORD *)(a2 + 40);
        v107 = sub_B2BE50(a2);
        v189[17] = 1;
        v108 = (__int64 *)v107;
        v189[16] = 3;
        v187 = (unsigned int *)"CallStackCheckFailBlk";
        v171 = sub_22077B0(0x50u);
        if ( v171 )
          sub_AA4D50(v171, (__int64)v108, (__int64)&v187, a2, 0);
        v109 = (__int64 *)sub_AA48A0(v171);
        v190 = v171;
        v193 = v109;
        v194 = &v202;
        v195 = v203;
        v198 = 512;
        v202 = &unk_49DA100;
        v203[0] = &unk_49DA0B0;
        v187 = (unsigned int *)v189;
        v188 = 0x200000000LL;
        LOWORD(v192) = 0;
        v196 = 0;
        v197 = 0;
        v199 = 7;
        v200 = 0;
        v201 = 0;
        v191 = (__int64 *)(v171 + 48);
        if ( sub_B92180(a2) )
        {
          v110 = sub_B92180(a2);
          v111 = sub_B01860(v108, 0, 0, v110, 0, 0, 0, 1);
          sub_B10CB0(&v183, (__int64)v111);
          v112 = v183;
          if ( !v183 )
          {
            sub_93FB40((__int64)&v187, 0);
            v112 = v183;
            goto LABEL_186;
          }
          v113 = v187;
          v114 = v188;
          v115 = &v187[4 * (unsigned int)v188];
          if ( v187 != v115 )
          {
            while ( *v113 )
            {
              v113 += 4;
              if ( v115 == v113 )
                goto LABEL_182;
            }
            *((_QWORD *)v113 + 1) = v183;
            goto LABEL_153;
          }
LABEL_182:
          if ( (unsigned int)v188 >= (unsigned __int64)HIDWORD(v188) )
          {
            v144 = (unsigned int)v188 + 1LL;
            v145 = v149 & 0xFFFFFFFF00000000LL;
            v149 &= 0xFFFFFFFF00000000LL;
            if ( HIDWORD(v188) < v144 )
            {
              v154 = v145;
              sub_C8D5F0((__int64)&v187, v189, v144, 0x10u, v145, v147);
              v145 = v154;
              v115 = &v187[4 * (unsigned int)v188];
            }
            *(_QWORD *)v115 = v145;
            *((_QWORD *)v115 + 1) = v112;
            v112 = v183;
            LODWORD(v188) = v188 + 1;
          }
          else
          {
            if ( v115 )
            {
              *v115 = 0;
              *((_QWORD *)v115 + 1) = v112;
              v114 = v188;
              v112 = v183;
            }
            LODWORD(v188) = v114 + 1;
          }
LABEL_186:
          if ( v112 )
LABEL_153:
            sub_B91220((__int64)&v183, (__int64)v112);
        }
        v179 = v181;
        v180 = 0x100000000LL;
        if ( *(_DWORD *)(a1 + 556) == 11 )
        {
          v128 = sub_BCE3C0(v108, 0);
          v129 = (__int64 *)sub_BCB120(v108);
          v185[0] = v128;
          v183 = (const char *)v185;
          v184 = 0x100000001LL;
          v130 = sub_BCF480(v129, v185, 1, 0);
          v131 = sub_BA8C10(v168, (__int64)"__stack_smash_handler", 0x15u, v130, 0);
          v120 = v132;
          if ( v183 != (const char *)v185 )
          {
            v170 = v131;
            _libc_free((unsigned __int64)v183);
            v131 = v170;
          }
          v121 = v131;
          v183 = "SSH";
          v186 = 259;
          v133 = (char *)sub_BD5D20(a2);
          v135 = sub_B33830((__int64)&v187, v133, v134, (__int64)&v183, 0, 0, 1);
          v137 = (unsigned int)v180;
          v138 = (unsigned int)v180 + 1LL;
          if ( v138 > HIDWORD(v180) )
          {
            v153 = v135;
            sub_C8D5F0((__int64)&v179, v181, (unsigned int)v180 + 1LL, 8u, v138, v136);
            v137 = (unsigned int)v180;
            v135 = v153;
          }
          *(_QWORD *)&v179[8 * v137] = v135;
          LODWORD(v180) = v180 + 1;
        }
        else
        {
          v116 = (__int64 *)sub_BCB120(v108);
          v183 = (const char *)v185;
          v184 = 0;
          v117 = sub_BCF480(v116, v185, 0, 0);
          v118 = sub_BA8C10(v168, (__int64)"__stack_chk_fail", 0x10u, v117, 0);
          v120 = v119;
          if ( v183 != (const char *)v185 )
          {
            v169 = v118;
            _libc_free((unsigned __int64)v183);
            v118 = v169;
          }
          v121 = v118;
        }
        sub_B2CD30(v120, 36);
        v186 = 257;
        sub_921880(&v187, v121, v120, (int)v179, v180, (__int64)&v183, 0);
        v186 = 257;
        v122 = sub_BD2C40(72, unk_3F148B8);
        if ( v122 )
          sub_B4C8A0((__int64)v122, (__int64)v193, 0, 0);
        (*(void (__fastcall **)(_QWORD *, _QWORD *, const char **, __int64 *, __int64))(*v195 + 16LL))(
          v195,
          v122,
          &v183,
          v191,
          v192);
        v123 = 4LL * (unsigned int)v188;
        if ( v187 != &v187[v123] )
        {
          v152 = v16;
          v124 = &v187[v123];
          v125 = v187;
          do
          {
            v126 = *((_QWORD *)v125 + 1);
            v127 = *v125;
            v125 += 4;
            sub_B99FD0((__int64)v122, v127, v126);
          }
          while ( v124 != v125 );
          v16 = v152;
        }
        if ( v179 != v181 )
          _libc_free((unsigned __int64)v179);
        nullsub_61();
        v202 = &unk_49DA100;
        nullsub_63();
        if ( v187 != (unsigned int *)v189 )
          _libc_free((unsigned __int64)v187);
      }
      v64 = (__int64 *)sub_BD5C60(v16);
      v199 = 7;
      v193 = v64;
      v187 = (unsigned int *)v189;
      v194 = &v202;
      v188 = 0x200000000LL;
      v195 = v203;
      v198 = 512;
      LOWORD(v192) = 0;
      v190 = 0;
      v196 = 0;
      v202 = &unk_49DA100;
      v191 = 0;
      v197 = 0;
      v200 = 0;
      v201 = 0;
      v203[0] = &unk_49DA0B0;
      v190 = *(_QWORD *)(v16 + 40);
      v191 = v165;
      v65 = *(const char **)sub_B46C60(v16);
      v183 = v65;
      if ( v65 && (sub_B96E90((__int64)&v183, (__int64)v65, 1), (v68 = v183) != 0) )
      {
        v69 = v187;
        v70 = v188;
        v71 = &v187[4 * (unsigned int)v188];
        if ( v187 != v71 )
        {
          while ( *v69 )
          {
            v69 += 4;
            if ( v71 == v69 )
              goto LABEL_118;
          }
          *((_QWORD *)v69 + 1) = v183;
LABEL_76:
          sub_B91220((__int64)&v183, (__int64)v68);
          goto LABEL_77;
        }
LABEL_118:
        if ( (unsigned int)v188 >= (unsigned __int64)HIDWORD(v188) )
        {
          v141 = v151 & 0xFFFFFFFF00000000LL;
          v151 &= 0xFFFFFFFF00000000LL;
          if ( HIDWORD(v188) < (unsigned __int64)(unsigned int)v188 + 1 )
          {
            sub_C8D5F0((__int64)&v187, v189, (unsigned int)v188 + 1LL, 0x10u, v66, v67);
            v71 = &v187[4 * (unsigned int)v188];
          }
          *(_QWORD *)v71 = v141;
          *((_QWORD *)v71 + 1) = v68;
          v68 = v183;
          LODWORD(v188) = v188 + 1;
        }
        else
        {
          if ( v71 )
          {
            *v71 = 0;
            *((_QWORD *)v71 + 1) = v68;
            v70 = v188;
            v68 = v183;
          }
          LODWORD(v188) = v70 + 1;
        }
      }
      else
      {
        sub_93FB40((__int64)&v187, 0);
        v68 = v183;
      }
      if ( v68 )
        goto LABEL_76;
LABEL_77:
      v72 = sub_2FC95F0(v166, v163, (__int64)&v187, 0);
      v182 = 257;
      v173 = v72;
      v73 = sub_BCE3C0(v193, 0);
      v74 = sub_AA4E30(v190);
      v75 = sub_AE5020(v74, v73);
      v186 = 257;
      v76 = v75;
      v77 = (unsigned __int8 *)sub_BD2C40(80, 1u);
      v79 = v77;
      if ( v77 )
      {
        sub_B4D190((__int64)v77, v73, v10, (__int64)&v183, 1, v76, 0, 0);
        v78 = v146;
      }
      (*(void (__fastcall **)(_QWORD *, unsigned __int8 *, char **, __int64 *, __int64, __int64))(*v195 + 16LL))(
        v195,
        v79,
        &v179,
        v191,
        v192,
        v78);
      v80 = v187;
      v81 = &v187[4 * (unsigned int)v188];
      if ( v187 != v81 )
      {
        do
        {
          v82 = *((_QWORD *)v80 + 1);
          v83 = *v80;
          v80 += 4;
          sub_B99FD0((__int64)v79, v83, v82);
        }
        while ( v81 != v80 );
      }
      v182 = 257;
      v84 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, unsigned __int8 *))*((_QWORD *)*v194 + 7);
      if ( v84 != sub_928890 )
      {
        v85 = v84((__int64)v194, 33u, (_BYTE *)v173, v79);
LABEL_85:
        if ( v85 )
          goto LABEL_86;
        goto LABEL_125;
      }
      if ( *(_BYTE *)v173 <= 0x15u && *v79 <= 0x15u )
      {
        v85 = sub_AAB310(0x21u, (unsigned __int8 *)v173, v79);
        goto LABEL_85;
      }
LABEL_125:
      v186 = 257;
      v85 = (__int64)sub_BD2C40(72, unk_3F10FD0);
      if ( v85 )
      {
        v97 = *(_QWORD *)(v173 + 8);
        v98 = *(unsigned __int8 *)(v97 + 8);
        if ( (unsigned int)(v98 - 17) > 1 )
        {
          v102 = sub_BCB2A0(*(_QWORD **)v97);
        }
        else
        {
          v99 = *(_DWORD *)(v97 + 32);
          v100 = *(_QWORD **)v97;
          BYTE4(v178) = (_BYTE)v98 == 18;
          LODWORD(v178) = v99;
          v101 = (__int64 *)sub_BCB2A0(v100);
          v102 = sub_BCE1B0(v101, v178);
        }
        sub_B523C0(v85, v102, 53, 33, v173, (__int64)v79, (__int64)&v183, 0, 0, 0);
      }
      (*(void (__fastcall **)(_QWORD *, __int64, char **, __int64 *, __int64))(*v195 + 16LL))(
        v195,
        v85,
        &v179,
        v191,
        v192);
      v103 = v187;
      v104 = &v187[4 * (unsigned int)v188];
      if ( v187 != v104 )
      {
        do
        {
          v105 = *((_QWORD *)v103 + 1);
          v106 = *v103;
          v103 += 4;
          sub_B99FD0(v85, v106, v105);
        }
        while ( v104 != v103 );
        if ( byte_5025188[0] )
          goto LABEL_87;
        goto LABEL_132;
      }
LABEL_86:
      if ( byte_5025188[0] )
        goto LABEL_87;
LABEL_132:
      if ( (unsigned int)sub_2207590((__int64)byte_5025188) )
      {
        sub_F02DB0(dword_5025190, 0xFFFFFu, 0x100000u);
        sub_2207640((__int64)byte_5025188);
        v86 = dword_5025190[0];
        if ( byte_5025188[0] )
          goto LABEL_88;
        goto LABEL_134;
      }
LABEL_87:
      v86 = dword_5025190[0];
      if ( byte_5025188[0] )
        goto LABEL_88;
LABEL_134:
      if ( (unsigned int)sub_2207590((__int64)byte_5025188) )
      {
        sub_F02DB0(dword_5025190, 0xFFFFFu, 0x100000u);
        sub_2207640((__int64)byte_5025188);
      }
LABEL_88:
      v87 = 0x80000000 - dword_5025190[0];
      v183 = (const char *)sub_B2BE50(a2);
      v88 = sub_B8C2F0(&v183, v87, v86, 0);
      v89 = v157;
      LOWORD(v89) = 0;
      v157 = v89;
      sub_F38250(v85, v165, v89, 0, v88, a3, 0, v171);
      v90 = *(_QWORD *)(v85 + 40);
      v91 = *(_QWORD *)(v90 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v91 == v90 + 48 )
        goto LABEL_198;
      if ( !v91 )
LABEL_195:
        BUG();
      v92 = v91 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v91 - 24) - 30 > 0xA )
LABEL_198:
        BUG();
      v93 = *(unsigned __int8 **)(v91 - 88);
      v183 = "SP_return";
      v186 = 259;
      sub_BD6B50(v93, &v183);
      sub_AA4AF0((__int64)v93, v164);
      *(_WORD *)(v85 + 2) = sub_B52870(*(_WORD *)(v85 + 2) & 0x3F) | *(_WORD *)(v85 + 2) & 0xFFC0;
      sub_B4CC70(v92);
      nullsub_61();
      v202 = &unk_49DA100;
      nullsub_63();
      v40 = v187;
      if ( v187 != (unsigned int *)v189 )
        goto LABEL_36;
LABEL_37:
      if ( v172 == v174 )
        return *a4;
    }
    if ( *a4 )
      goto LABEL_13;
LABEL_51:
    v175 = 0;
    *a4 = 1;
    v44 = *(_QWORD *)(a2 + 80);
    if ( !v44 )
LABEL_197:
      BUG();
    v45 = *(__int64 **)(v44 + 32);
    if ( !v45 )
    {
      v5 = (__int64 *)sub_BD5C60(0);
      v199 = 7;
      v193 = v5;
      v194 = &v202;
      v195 = v203;
      v187 = (unsigned int *)v189;
      v188 = 0x200000000LL;
      v202 = &unk_49DA100;
      v196 = 0;
      v197 = 0;
      v198 = 512;
      v200 = 0;
      v201 = 0;
      v190 = 0;
      v191 = 0;
      LOWORD(v192) = 0;
      v203[0] = &unk_49DA0B0;
      BUG();
    }
    v193 = (__int64 *)sub_BD5C60((__int64)(v45 - 3));
    v194 = &v202;
    v195 = v203;
    v187 = (unsigned int *)v189;
    v202 = &unk_49DA100;
    v190 = 0;
    v188 = 0x200000000LL;
    v191 = 0;
    v196 = 0;
    v197 = 0;
    v198 = 512;
    v199 = 7;
    v200 = 0;
    v201 = 0;
    LOWORD(v192) = 0;
    v203[0] = &unk_49DA0B0;
    v190 = v45[2];
    v191 = v45;
    v46 = *(const char **)sub_B46C60((__int64)(v45 - 3));
    v183 = v46;
    if ( v46 && (sub_B96E90((__int64)&v183, (__int64)v46, 1), (v48 = v183) != 0) )
    {
      v49 = v187;
      v50 = v188;
      v51 = &v187[4 * (unsigned int)v188];
      if ( v187 != v51 )
      {
        while ( 1 )
        {
          v47 = *v49;
          if ( !(_DWORD)v47 )
            break;
          v49 += 4;
          if ( v51 == v49 )
            goto LABEL_136;
        }
        *((_QWORD *)v49 + 1) = v183;
LABEL_60:
        sub_B91220((__int64)&v183, (__int64)v48);
LABEL_61:
        v52 = (__int64 *)sub_BD5C60(v16);
        v53 = (__int64 *)sub_BCE3C0(v52, 0);
        v182 = 259;
        v179 = "StackGuardSlot";
        v54 = sub_AA4E30(v190);
        v55 = sub_AE5260(v54, (__int64)v53);
        v56 = *(_DWORD *)(v54 + 4);
        v167 = v55;
        v186 = 257;
        v57 = sub_BD2C40(80, 1u);
        v10 = (__int64)v57;
        if ( v57 )
          sub_B4CCA0((__int64)v57, v53, v56, 0, v167, (__int64)&v183, 0, 0);
        (*(void (__fastcall **)(_QWORD *, __int64, char **, __int64 *, __int64))(*v195 + 16LL))(
          v195,
          v10,
          &v179,
          v191,
          v192);
        v58 = v187;
        v59 = &v187[4 * (unsigned int)v188];
        if ( v187 != v59 )
        {
          do
          {
            v60 = *((_QWORD *)v58 + 1);
            v61 = *v58;
            v58 += 4;
            sub_B99FD0(v10, v61, v60);
          }
          while ( v59 != v58 );
        }
        v62 = sub_2FC95F0(v166, v163, (__int64)&v187, &v175);
        v177 = 0;
        v186 = 257;
        v179 = (char *)v62;
        v180 = v10;
        sub_B33D10((__int64)&v187, 0x155u, 0, 0, (int)&v179, 2, v176, (__int64)&v183);
        v63 = v175;
        nullsub_61();
        v202 = &unk_49DA100;
        nullsub_63();
        if ( v187 != (unsigned int *)v189 )
          _libc_free((unsigned __int64)v187);
        v159 &= v63;
        goto LABEL_13;
      }
LABEL_136:
      if ( (unsigned int)v188 >= (unsigned __int64)HIDWORD(v188) )
      {
        v142 = (unsigned int)v188 + 1LL;
        v143 = v148 & 0xFFFFFFFF00000000LL;
        v148 &= 0xFFFFFFFF00000000LL;
        if ( HIDWORD(v188) < v142 )
        {
          sub_C8D5F0((__int64)&v187, v189, v142, 0x10u, v142, v47);
          v51 = &v187[4 * (unsigned int)v188];
        }
        *(_QWORD *)v51 = v143;
        *((_QWORD *)v51 + 1) = v48;
        v48 = v183;
        LODWORD(v188) = v188 + 1;
      }
      else
      {
        if ( v51 )
        {
          *v51 = 0;
          *((_QWORD *)v51 + 1) = v48;
          v50 = v188;
          v48 = v183;
        }
        LODWORD(v188) = v50 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v187, 0);
      v48 = v183;
    }
    if ( !v48 )
      goto LABEL_61;
    goto LABEL_60;
  }
  return *a4;
}
