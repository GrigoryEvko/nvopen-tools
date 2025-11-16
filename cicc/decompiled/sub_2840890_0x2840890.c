// Function: sub_2840890
// Address: 0x2840890
//
char __fastcall sub_2840890(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5)
{
  __int64 v5; // r13
  unsigned __int64 v6; // rax
  __int64 v8; // r14
  int *v9; // r12
  __int64 *v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rcx
  bool v13; // al
  _QWORD *v14; // r8
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // rdx
  __int64 v24; // rax
  unsigned int v25; // edx
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rcx
  int v28; // ecx
  bool v29; // zf
  int v30; // eax
  __int64 v31; // rax
  unsigned int v32; // edx
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rcx
  int v35; // ecx
  int v36; // eax
  _QWORD *v37; // rax
  __int64 *v38; // rax
  __int64 v39; // rsi
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r9
  __int64 v43; // r8
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  _QWORD *v55; // rax
  _QWORD *v56; // rax
  unsigned int v57; // eax
  __int64 v58; // rax
  __int64 v59; // r9
  __int64 v60; // r12
  unsigned __int64 v61; // rbx
  __int64 v62; // rdx
  unsigned int v63; // esi
  __int64 v64; // r14
  unsigned __int64 v65; // rbx
  __int64 v66; // rdx
  unsigned int v67; // esi
  bool v68; // al
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // r8
  char v72; // al
  bool v73; // al
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r8
  char v77; // al
  bool v78; // al
  __int64 v79; // rdx
  __int64 v80; // rcx
  __int64 v81; // r8
  char v82; // al
  bool v83; // al
  __int64 v84; // rdx
  __int64 v85; // rcx
  __int64 v86; // r9
  __int64 v87; // r8
  char v88; // al
  __int64 v89; // rcx
  __int64 v90; // r9
  _QWORD *v91; // r9
  _QWORD *v92; // rax
  __int64 v93; // rax
  __int64 v94; // r9
  __int64 v95; // r12
  unsigned __int64 v96; // rbx
  __int64 v97; // rdx
  unsigned int v98; // esi
  __int64 v99; // r14
  unsigned __int64 v100; // rbx
  __int64 v101; // rdx
  unsigned int v102; // esi
  __int64 v104; // [rsp+8h] [rbp-1E8h]
  unsigned int v105; // [rsp+8h] [rbp-1E8h]
  unsigned int v106; // [rsp+8h] [rbp-1E8h]
  _QWORD *v107; // [rsp+10h] [rbp-1E0h]
  _QWORD *v108; // [rsp+18h] [rbp-1D8h]
  __int64 v109; // [rsp+20h] [rbp-1D0h]
  unsigned __int64 v110; // [rsp+20h] [rbp-1D0h]
  unsigned __int64 v111; // [rsp+20h] [rbp-1D0h]
  __int64 v112; // [rsp+20h] [rbp-1D0h]
  __int64 v113; // [rsp+20h] [rbp-1D0h]
  __int64 v114; // [rsp+20h] [rbp-1D0h]
  __int64 v115; // [rsp+20h] [rbp-1D0h]
  __int64 v116; // [rsp+20h] [rbp-1D0h]
  unsigned int v117; // [rsp+28h] [rbp-1C8h]
  __int64 v118; // [rsp+28h] [rbp-1C8h]
  __int64 v119; // [rsp+28h] [rbp-1C8h]
  __int64 v120; // [rsp+30h] [rbp-1C0h]
  unsigned __int64 v121; // [rsp+30h] [rbp-1C0h]
  __int64 v122; // [rsp+30h] [rbp-1C0h]
  __int64 v123; // [rsp+30h] [rbp-1C0h]
  __int64 v124; // [rsp+30h] [rbp-1C0h]
  __int64 v125; // [rsp+30h] [rbp-1C0h]
  __int64 v126; // [rsp+30h] [rbp-1C0h]
  __int64 v127; // [rsp+38h] [rbp-1B8h]
  __int64 v128; // [rsp+38h] [rbp-1B8h]
  __int64 *v129; // [rsp+38h] [rbp-1B8h]
  __int64 v130; // [rsp+38h] [rbp-1B8h]
  __int64 v131; // [rsp+40h] [rbp-1B0h]
  __int64 v132; // [rsp+40h] [rbp-1B0h]
  __int64 v133; // [rsp+48h] [rbp-1A8h]
  __int64 *v134; // [rsp+48h] [rbp-1A8h]
  _QWORD *v135; // [rsp+48h] [rbp-1A8h]
  _QWORD *v136; // [rsp+48h] [rbp-1A8h]
  __int64 v137; // [rsp+48h] [rbp-1A8h]
  __int64 v138; // [rsp+48h] [rbp-1A8h]
  unsigned int v139; // [rsp+48h] [rbp-1A8h]
  unsigned __int64 v140; // [rsp+48h] [rbp-1A8h]
  __int64 v141; // [rsp+48h] [rbp-1A8h]
  __int64 v142; // [rsp+50h] [rbp-1A0h]
  __int64 v143; // [rsp+50h] [rbp-1A0h]
  __int64 v144; // [rsp+50h] [rbp-1A0h]
  __int64 v145; // [rsp+50h] [rbp-1A0h]
  __int64 v146; // [rsp+58h] [rbp-198h]
  __int64 v147; // [rsp+58h] [rbp-198h]
  unsigned __int64 v148; // [rsp+58h] [rbp-198h]
  __int64 v149; // [rsp+58h] [rbp-198h]
  int *v150; // [rsp+58h] [rbp-198h]
  _QWORD *v151; // [rsp+58h] [rbp-198h]
  __int64 v152; // [rsp+58h] [rbp-198h]
  __int64 v153; // [rsp+58h] [rbp-198h]
  int *v154; // [rsp+58h] [rbp-198h]
  _QWORD *v155; // [rsp+58h] [rbp-198h]
  _QWORD *v156; // [rsp+60h] [rbp-190h]
  __int64 v157; // [rsp+60h] [rbp-190h]
  __int64 v158; // [rsp+60h] [rbp-190h]
  _QWORD *v159; // [rsp+60h] [rbp-190h]
  _QWORD *v160; // [rsp+60h] [rbp-190h]
  const void *v161; // [rsp+68h] [rbp-188h]
  int v164; // [rsp+80h] [rbp-170h] BYREF
  _QWORD *v165; // [rsp+88h] [rbp-168h]
  __int64 v166; // [rsp+90h] [rbp-160h]
  char v167; // [rsp+98h] [rbp-158h]
  _BYTE v168[32]; // [rsp+A0h] [rbp-150h] BYREF
  __int16 v169; // [rsp+C0h] [rbp-130h]
  _BYTE v170[32]; // [rsp+D0h] [rbp-120h] BYREF
  __int16 v171; // [rsp+F0h] [rbp-100h]
  __int64 v172; // [rsp+100h] [rbp-F0h] BYREF
  __int64 v173; // [rsp+108h] [rbp-E8h]
  __int16 v174; // [rsp+120h] [rbp-D0h]
  _QWORD *v175; // [rsp+130h] [rbp-C0h] BYREF
  __int64 v176; // [rsp+138h] [rbp-B8h]
  _QWORD v177[4]; // [rsp+140h] [rbp-B0h] BYREF
  __int64 v178; // [rsp+160h] [rbp-90h]
  __int64 v179; // [rsp+168h] [rbp-88h]
  __int64 v180; // [rsp+170h] [rbp-80h]
  __int64 v181; // [rsp+178h] [rbp-78h]
  void **v182; // [rsp+180h] [rbp-70h]
  _QWORD *v183; // [rsp+188h] [rbp-68h]
  __int64 v184; // [rsp+190h] [rbp-60h]
  int v185; // [rsp+198h] [rbp-58h]
  __int16 v186; // [rsp+19Ch] [rbp-54h]
  char v187; // [rsp+19Eh] [rbp-52h]
  __int64 v188; // [rsp+1A0h] [rbp-50h]
  __int64 v189; // [rsp+1A8h] [rbp-48h]
  void *v190; // [rsp+1B0h] [rbp-40h] BYREF
  _QWORD v191[7]; // [rsp+1B8h] [rbp-38h] BYREF

  v5 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  LOBYTE(v6) = a3 + 16;
  v161 = (const void *)(a3 + 16);
  if ( v5 != *(_QWORD *)a2 )
  {
    v8 = a3;
    v9 = &v164;
    v10 = *(__int64 **)a2;
    while ( 1 )
    {
      if ( *(_BYTE *)*v10 != 82 )
        goto LABEL_3;
      LOBYTE(v6) = sub_283FFC0((__int64)v9, a1, *v10);
      if ( !v167 )
        goto LABEL_3;
      if ( v164 != 36 )
        goto LABEL_3;
      if ( v165[5] != 2 )
        goto LABEL_3;
      v156 = v165;
      v146 = v166;
      v131 = sub_D33D80(v165, *(_QWORD *)(a1 + 16), v11, v12, (__int64)v165);
      v13 = sub_D96900(v131);
      v14 = v156;
      if ( !v13 )
      {
        LOBYTE(v6) = sub_D96960(v131);
        if ( !(_BYTE)v6 )
          goto LABEL_3;
        v14 = v156;
        if ( !(_BYTE)qword_5000348 )
          goto LABEL_3;
      }
      v109 = (__int64)v14;
      v142 = sub_D95540(*(_QWORD *)v14[4]);
      v127 = *(_QWORD *)(a1 + 16);
      v133 = *(_QWORD *)(a1 + 72);
      v120 = *(_QWORD *)(a1 + 48);
      v117 = *(_DWORD *)(a1 + 64);
      v157 = *(_QWORD *)(a1 + 80);
      v15 = sub_D95540(**(_QWORD **)(v133 + 32));
      v18 = v109;
      if ( v142 != v15 )
      {
        v104 = v109;
        v110 = sub_9208B0(v120, v15);
        v176 = v19;
        v175 = (_QWORD *)v110;
        v6 = sub_9208B0(v120, v142);
        v172 = v6;
        v173 = v20;
        if ( v110 < v6 )
          goto LABEL_3;
        if ( !(_BYTE)qword_5000428 )
          goto LABEL_3;
        LOBYTE(v6) = v157;
        if ( *(_WORD *)(v157 + 24) )
          goto LABEL_3;
        v6 = **(_QWORD **)(v133 + 32);
        v111 = v6;
        if ( *(_WORD *)(v6 + 24) )
          goto LABEL_3;
        v6 = sub_DC1950(v127, v133, v117);
        v175 = (_QWORD *)v6;
        if ( !BYTE4(v6) )
          goto LABEL_3;
        v21 = sub_9208B0(v120, v142);
        v22 = v104;
        v175 = (_QWORD *)v21;
        v121 = v21;
        v176 = v23;
        v24 = *(_QWORD *)(v111 + 32);
        v25 = *(_DWORD *)(v24 + 32);
        if ( v25 > 0x40 )
        {
          v106 = *(_DWORD *)(v24 + 32);
          v116 = v22;
          LODWORD(v6) = sub_C444A0(v24 + 24);
          v25 = v106;
          v22 = v116;
        }
        else
        {
          v26 = *(_QWORD *)(v24 + 24);
          _BitScanReverse64(&v27, v26);
          v28 = v27 ^ 0x3F;
          v29 = v26 == 0;
          v30 = 64;
          if ( !v29 )
            v30 = v28;
          LODWORD(v6) = v25 + v30 - 64;
        }
        if ( v121 <= v25 - (unsigned int)v6 )
          goto LABEL_3;
        v31 = *(_QWORD *)(v157 + 32);
        v32 = *(_DWORD *)(v31 + 32);
        if ( v32 > 0x40 )
        {
          v105 = *(_DWORD *)(v31 + 32);
          v114 = v22;
          LODWORD(v6) = sub_C444A0(v31 + 24);
          v32 = v105;
          v22 = v114;
        }
        else
        {
          v33 = *(_QWORD *)(v31 + 24);
          _BitScanReverse64(&v34, v33);
          v35 = v34 ^ 0x3F;
          v29 = v33 == 0;
          v36 = 64;
          if ( !v29 )
            v36 = v35;
          LODWORD(v6) = v32 + v36 - 64;
        }
        v112 = v22;
        if ( v121 <= v32 - (unsigned int)v6 )
          goto LABEL_3;
        v6 = (unsigned __int64)sub_DC5200(v127, v133, v142, 0);
        v133 = v6;
        if ( *(_WORD *)(v6 + 24) != 8 )
          goto LABEL_3;
        v37 = sub_DC5200(v127, v157, v142, 0);
        v18 = v112;
        v157 = (__int64)v37;
      }
      v143 = v18;
      v6 = sub_D33D80((_QWORD *)v133, *(_QWORD *)(a1 + 16), v16, v17, v18);
      if ( v131 != v6 )
        goto LABEL_3;
      v29 = !sub_D96900(v131);
      v38 = *(__int64 **)(v143 + 32);
      if ( v29 )
      {
        v130 = sub_D95540(*v38);
        v122 = v143;
        v145 = **(_QWORD **)(v143 + 32);
        v132 = **(_QWORD **)(v133 + 32);
        v68 = sub_DADE90(*(_QWORD *)(a1 + 16), v145, *(_QWORD *)(a1 + 40));
        v71 = v122;
        if ( !v68 )
        {
          if ( *(_WORD *)(v145 + 24) != 15 )
            goto LABEL_72;
          v72 = sub_2840090((__int64 *)a1, v145, v69, v70);
          v71 = v122;
          if ( !v72 )
            goto LABEL_72;
        }
        v123 = v71;
        v73 = sub_DADE90(*(_QWORD *)(a1 + 16), v146, *(_QWORD *)(a1 + 40));
        v76 = v123;
        if ( !v73 )
        {
          if ( *(_WORD *)(v146 + 24) != 15 )
            goto LABEL_72;
          v77 = sub_2840090((__int64 *)a1, v146, v74, v75);
          v76 = v123;
          if ( !v77 )
            goto LABEL_72;
        }
        v124 = v76;
        v78 = sub_DADE90(*(_QWORD *)(a1 + 16), v132, *(_QWORD *)(a1 + 40));
        v81 = v124;
        if ( !v78 )
        {
          if ( *(_WORD *)(v132 + 24) != 15 )
            goto LABEL_72;
          v82 = sub_2840090((__int64 *)a1, v132, v79, v80);
          v81 = v124;
          if ( !v82 )
            goto LABEL_72;
        }
        if ( ((v125 = v81, v83 = sub_DADE90(*(_QWORD *)(a1 + 16), v157, *(_QWORD *)(a1 + 40)), v87 = v125, v83)
           || *(_WORD *)(v157 + 24) == 15 && (v88 = sub_2840090((__int64 *)a1, v157, v84, v85), v87 = v125, v88))
          && (v126 = v87, (unsigned __int8)sub_F80650(a4, v132, a5, v85, v87, v86))
          && (unsigned __int8)sub_F80650(a4, v157, a5, v89, v126, v90) )
        {
          v91 = sub_DCC620(v133, *(__int64 **)(a1 + 16));
          LOBYTE(v6) = 0;
          if ( (_QWORD *)v126 == v91 )
          {
            v139 = sub_B53250(v117);
            v152 = sub_2840320(a1, a4, a5, 0x24u, v145, v146);
            v92 = sub_DA2C50(*(_QWORD *)(a1 + 16), v130, 1, 0);
            v173 = sub_2840320(a1, a4, a5, v139, v157, (__int64)v92);
            v119 = v173;
            v172 = v152;
            v140 = sub_28401D0(a1, a5, (__int64)&v172, 2);
            v93 = sub_BD5C60(v140);
            v175 = v177;
            v181 = v93;
            v176 = 0x200000000LL;
            v182 = &v190;
            v183 = v191;
            v184 = 0;
            v190 = &unk_49DA100;
            v185 = 0;
            v186 = 512;
            v191[0] = &unk_49DA0B0;
            v187 = 7;
            v188 = 0;
            v189 = 0;
            v178 = 0;
            v179 = 0;
            LOWORD(v180) = 0;
            sub_D5F1F0((__int64)&v175, v140);
            v169 = 257;
            v171 = 257;
            v94 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v182 + 2))(v182, 28, v152, v119);
            if ( !v94 )
            {
              v174 = 257;
              v115 = sub_B504D0(28, v152, v119, (__int64)&v172, 0, 0);
              (*(void (__fastcall **)(_QWORD *, __int64, _BYTE *, __int64, __int64))(*v183 + 16LL))(
                v183,
                v115,
                v170,
                v179,
                v180);
              v141 = v8;
              v99 = a1;
              v100 = (unsigned __int64)v175;
              v155 = &v175[2 * (unsigned int)v176];
              while ( v155 != (_QWORD *)v100 )
              {
                v101 = *(_QWORD *)(v100 + 8);
                v102 = *(_DWORD *)v100;
                v100 += 16LL;
                sub_B99FD0(v115, v102, v101);
              }
              v94 = v115;
              a1 = v99;
              v8 = v141;
            }
            v153 = v94;
            v174 = 257;
            v108 = sub_BD2C40(72, unk_3F10A14);
            if ( v108 )
              sub_B549F0((__int64)v108, v153, (__int64)&v172, 0, 0);
            (*(void (__fastcall **)(_QWORD *, _QWORD *, _BYTE *, __int64, __int64))(*v183 + 16LL))(
              v183,
              v108,
              v168,
              v179,
              v180);
            v154 = v9;
            v95 = a1;
            v96 = (unsigned __int64)v175;
            v160 = &v175[2 * (unsigned int)v176];
            while ( v160 != (_QWORD *)v96 )
            {
              v97 = *(_QWORD *)(v96 + 8);
              v98 = *(_DWORD *)v96;
              v96 += 16LL;
              sub_B99FD0((__int64)v108, v98, v97);
            }
            a1 = v95;
            v9 = v154;
            nullsub_61();
            v190 = &unk_49DA100;
            nullsub_63();
            if ( v175 != v177 )
              _libc_free((unsigned __int64)v175);
            LOBYTE(v6) = 1;
          }
        }
        else
        {
LABEL_72:
          LOBYTE(v6) = 0;
        }
        v42 = (__int64)v108;
      }
      else
      {
        if ( ((v128 = sub_D95540(*v38),
               v39 = **(_QWORD **)(v143 + 32),
               v144 = **(_QWORD **)(v133 + 32),
               sub_DADE90(*(_QWORD *)(a1 + 16), v39, *(_QWORD *)(a1 + 40)))
           || *(_WORD *)(v39 + 24) == 15 && (unsigned __int8)sub_2840090((__int64 *)a1, v39, v40, v41))
          && (sub_DADE90(*(_QWORD *)(a1 + 16), v146, *(_QWORD *)(a1 + 40))
           || *(_WORD *)(v146 + 24) == 15 && (unsigned __int8)sub_2840090((__int64 *)a1, v146, v44, v45))
          && (sub_DADE90(*(_QWORD *)(a1 + 16), v144, *(_QWORD *)(a1 + 40))
           || *(_WORD *)(v144 + 24) == 15 && (unsigned __int8)sub_2840090((__int64 *)a1, v144, v46, v47))
          && (sub_DADE90(*(_QWORD *)(a1 + 16), v157, *(_QWORD *)(a1 + 40))
           || *(_WORD *)(v157 + 24) == 15 && (unsigned __int8)sub_2840090((__int64 *)a1, v157, v48, v49))
          && (unsigned __int8)sub_F80650(a4, v144, a5, v49, v50, v51)
          && (unsigned __int8)sub_F80650(a4, v157, a5, v52, v53, v54) )
        {
          v134 = *(__int64 **)(a1 + 16);
          v55 = sub_DA2C50((__int64)v134, v128, 1, 0);
          v129 = v134;
          v135 = sub_DCC810(v134, v144, (__int64)v55, 0, 0);
          v56 = sub_DCC810(*(__int64 **)(a1 + 16), v146, v39, 0, 0);
          v175 = v177;
          v177[0] = v56;
          v177[1] = v135;
          v176 = 0x200000002LL;
          v136 = sub_DC7EB0(v129, (__int64)&v175, 0, 0);
          if ( v175 != v177 )
            _libc_free((unsigned __int64)v175);
          v57 = sub_B53250(v117);
          v137 = sub_2840320(a1, a4, a5, v57, v157, (__int64)v136);
          v172 = sub_2840320(a1, a4, a5, 0x24u, v39, v146);
          v118 = v172;
          v173 = v137;
          v148 = sub_28401D0(a1, a5, (__int64)&v172, 2);
          v58 = sub_BD5C60(v148);
          v187 = 7;
          v181 = v58;
          v175 = v177;
          v184 = 0;
          v176 = 0x200000000LL;
          v182 = &v190;
          v183 = v191;
          v185 = 0;
          v190 = &unk_49DA100;
          v186 = 512;
          v188 = 0;
          v191[0] = &unk_49DA0B0;
          v189 = 0;
          v178 = 0;
          v179 = 0;
          LOWORD(v180) = 0;
          sub_D5F1F0((__int64)&v175, v148);
          v169 = 257;
          v171 = 257;
          v59 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v182 + 2))(v182, 28, v118, v137);
          if ( !v59 )
          {
            v174 = 257;
            v113 = sub_B504D0(28, v118, v137, (__int64)&v172, 0, 0);
            (*(void (__fastcall **)(_QWORD *, __int64, _BYTE *, __int64, __int64))(*v183 + 16LL))(
              v183,
              v113,
              v170,
              v179,
              v180);
            v138 = v8;
            v64 = a1;
            v65 = (unsigned __int64)v175;
            v151 = &v175[2 * (unsigned int)v176];
            while ( v151 != (_QWORD *)v65 )
            {
              v66 = *(_QWORD *)(v65 + 8);
              v67 = *(_DWORD *)v65;
              v65 += 16LL;
              sub_B99FD0(v113, v67, v66);
            }
            v59 = v113;
            a1 = v64;
            v8 = v138;
          }
          v149 = v59;
          v174 = 257;
          v107 = sub_BD2C40(72, unk_3F10A14);
          if ( v107 )
            sub_B549F0((__int64)v107, v149, (__int64)&v172, 0, 0);
          (*(void (__fastcall **)(_QWORD *, _QWORD *, _BYTE *, __int64, __int64))(*v183 + 16LL))(
            v183,
            v107,
            v168,
            v179,
            v180);
          v150 = v9;
          v60 = a1;
          v61 = (unsigned __int64)v175;
          v159 = &v175[2 * (unsigned int)v176];
          while ( v159 != (_QWORD *)v61 )
          {
            v62 = *(_QWORD *)(v61 + 8);
            v63 = *(_DWORD *)v61;
            v61 += 16LL;
            sub_B99FD0((__int64)v107, v63, v62);
          }
          a1 = v60;
          v9 = v150;
          nullsub_61();
          v190 = &unk_49DA100;
          nullsub_63();
          if ( v175 != v177 )
            _libc_free((unsigned __int64)v175);
          LOBYTE(v6) = 1;
        }
        else
        {
          LOBYTE(v6) = 0;
        }
        v42 = (__int64)v107;
      }
      if ( (_BYTE)v6 )
      {
        v6 = *(unsigned int *)(v8 + 8);
        v43 = *v10;
        if ( v6 + 1 > *(unsigned int *)(v8 + 12) )
        {
          v147 = v42;
          v158 = *v10;
          sub_C8D5F0(v8, v161, v6 + 1, 8u, v43, v42);
          v6 = *(unsigned int *)(v8 + 8);
          v42 = v147;
          v43 = v158;
        }
        ++v10;
        *(_QWORD *)(*(_QWORD *)v8 + 8 * v6) = v43;
        ++*(_DWORD *)(v8 + 8);
        *(v10 - 1) = v42;
        if ( (__int64 *)v5 == v10 )
          return v6;
      }
      else
      {
LABEL_3:
        if ( (__int64 *)v5 == ++v10 )
          return v6;
      }
    }
  }
  return v6;
}
