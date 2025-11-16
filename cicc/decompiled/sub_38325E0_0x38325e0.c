// Function: sub_38325E0
// Address: 0x38325e0
//
unsigned __int8 *__fastcall sub_38325E0(__int64 a1, _QWORD *a2, __m128i a3)
{
  _QWORD *v3; // r14
  const __m128i *v5; // rax
  __int64 v6; // r11
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v12; // rax
  __int64 v13; // r11
  unsigned __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 (__fastcall *v16)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v17; // rsi
  unsigned __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // r14
  int *v22; // rbx
  int v23; // r9d
  __int64 v25; // r8
  int v26; // edx
  unsigned int v27; // edi
  int *v28; // rcx
  int v29; // r10d
  __int64 v30; // rcx
  __int64 v31; // rsi
  __int64 v32; // r8
  unsigned __int8 *v33; // r14
  __int16 v35; // cx
  __int64 v36; // rdx
  __int16 v37; // ax
  __int64 v38; // rdx
  __int64 v39; // r12
  int v40; // r9d
  __int64 v41; // rcx
  __int64 v42; // rsi
  __int64 v43; // r8
  int *v44; // rbx
  char v45; // si
  __int64 v46; // rdi
  int v47; // ecx
  unsigned int v48; // r8d
  __int64 v49; // rax
  int v50; // r10d
  __int64 v51; // rax
  __int32 v52; // edx
  int v53; // edx
  __int64 v54; // r14
  __int64 v55; // rax
  __int64 v56; // rdx
  unsigned int v57; // eax
  unsigned int v58; // eax
  __int64 v59; // rdx
  int v60; // r9d
  int v61; // r9d
  __int16 v62; // cx
  __m128i v63; // rax
  __int64 v64; // rax
  __int64 v65; // r8
  __int64 v66; // rdx
  unsigned __int16 *v67; // rdx
  __int64 v68; // rcx
  unsigned __int64 v69; // rax
  __int64 v70; // rsi
  unsigned int v71; // eax
  __int64 v72; // rdx
  __int64 *v73; // r8
  int v74; // eax
  __int64 v75; // r9
  unsigned int v76; // edx
  __int64 *v77; // r8
  __int64 v78; // r10
  __int64 v79; // r14
  unsigned int v80; // ecx
  __int64 v81; // r12
  __int64 v82; // rax
  unsigned int v83; // edx
  unsigned __int8 *v84; // rax
  _QWORD *v85; // r14
  unsigned int v86; // edx
  __int128 v87; // rax
  __int64 v88; // r9
  int v89; // r9d
  int *v90; // rbx
  __int64 v92; // rdi
  int v93; // edx
  unsigned int v94; // r8d
  int *v95; // rcx
  int v96; // r10d
  __int64 v97; // rcx
  __int64 v98; // r8
  unsigned __int16 *v99; // rbx
  __int64 v100; // rdx
  __int64 v101; // rdx
  __int64 v102; // rbx
  __m128i v103; // rax
  __int64 v104; // rdx
  unsigned int v105; // r15d
  __int64 *v106; // r14
  int v107; // eax
  __int64 v108; // r8
  unsigned int v109; // ebx
  __int128 v110; // rax
  _QWORD *v111; // rdi
  _QWORD *v112; // rax
  __int64 v113; // r8
  _QWORD *v114; // r9
  _QWORD *v115; // r14
  __int64 v116; // rdx
  __int64 v117; // r15
  int v118; // r9d
  int v119; // eax
  __int64 v120; // rdx
  __int64 v121; // rax
  __int64 v122; // rdx
  __int64 v123; // rdx
  __int64 v124; // rax
  __int64 v125; // rdx
  bool v126; // al
  __int64 v127; // r14
  int v128; // r9d
  __int64 v129; // rdx
  __int64 v130; // r15
  __int64 v131; // rdx
  __m128i v132; // rax
  __int8 v133; // al
  unsigned int v134; // eax
  _QWORD *v135; // r12
  __int128 v136; // rax
  __int64 v137; // r9
  __int64 v138; // rdx
  __int64 v139; // rax
  __int64 v140; // rdx
  bool v141; // al
  int v142; // eax
  __int64 v143; // rax
  bool v144; // al
  int j; // ecx
  __int64 v146; // rax
  int v147; // eax
  int i; // ecx
  __int64 v149; // rdx
  __int64 v150; // rdx
  int *v151; // rcx
  int v152; // r9d
  int *v153; // rcx
  __int128 v154; // [rsp-30h] [rbp-1D0h]
  __int128 v155; // [rsp-20h] [rbp-1C0h]
  __int64 v156; // [rsp+8h] [rbp-198h]
  unsigned int v157; // [rsp+10h] [rbp-190h]
  __int16 v158; // [rsp+12h] [rbp-18Eh]
  __int16 v159; // [rsp+1Ah] [rbp-186h]
  __int16 v160; // [rsp+20h] [rbp-180h]
  char v161; // [rsp+20h] [rbp-180h]
  __int64 *v162; // [rsp+20h] [rbp-180h]
  __int64 v163; // [rsp+20h] [rbp-180h]
  __int64 v164; // [rsp+20h] [rbp-180h]
  unsigned __int64 v165; // [rsp+20h] [rbp-180h]
  unsigned int v166; // [rsp+28h] [rbp-178h]
  __int64 v167; // [rsp+28h] [rbp-178h]
  _QWORD *v168; // [rsp+28h] [rbp-178h]
  __int16 v169; // [rsp+28h] [rbp-178h]
  __int16 v170; // [rsp+28h] [rbp-178h]
  __int16 v171; // [rsp+28h] [rbp-178h]
  __int16 v172; // [rsp+28h] [rbp-178h]
  __int16 v173; // [rsp+28h] [rbp-178h]
  __int16 v174; // [rsp+28h] [rbp-178h]
  __int16 v175; // [rsp+28h] [rbp-178h]
  __int64 v176; // [rsp+30h] [rbp-170h]
  unsigned __int64 v177; // [rsp+30h] [rbp-170h]
  __int64 v178; // [rsp+30h] [rbp-170h]
  __int128 v179; // [rsp+30h] [rbp-170h]
  __m128i v180; // [rsp+40h] [rbp-160h]
  __int64 v181; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v182; // [rsp+D8h] [rbp-C8h]
  __m128i v183; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v184; // [rsp+F0h] [rbp-B0h] BYREF
  unsigned __int64 v185; // [rsp+F8h] [rbp-A8h]
  unsigned int v186; // [rsp+100h] [rbp-A0h] BYREF
  __int64 v187; // [rsp+108h] [rbp-98h]
  __int64 v188; // [rsp+110h] [rbp-90h] BYREF
  int v189; // [rsp+118h] [rbp-88h]
  _QWORD v190[2]; // [rsp+120h] [rbp-80h] BYREF
  __m128i v191; // [rsp+130h] [rbp-70h] BYREF
  unsigned __int64 v192; // [rsp+140h] [rbp-60h] BYREF
  __int64 v193; // [rsp+148h] [rbp-58h]
  unsigned __int64 v194; // [rsp+150h] [rbp-50h] BYREF
  __int64 v195; // [rsp+158h] [rbp-48h]
  __int64 v196; // [rsp+160h] [rbp-40h]

  v3 = a2;
  v5 = (const __m128i *)a2[5];
  v6 = *(_QWORD *)a1;
  v7 = 16LL * v5->m128i_u32[2];
  v176 = v5->m128i_i64[0];
  v180 = _mm_loadu_si128(v5);
  v8 = v7 + *(_QWORD *)(v5->m128i_i64[0] + 48);
  v9 = *(_QWORD *)(v8 + 8);
  LOWORD(v181) = *(_WORD *)v8;
  v10 = *(_QWORD *)(a1 + 8);
  v182 = v9;
  v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v6 + 592LL);
  if ( v11 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v194, v6, *(_QWORD *)(v10 + 64), v181, v182);
    v183.m128i_i16[0] = v195;
    v183.m128i_i64[1] = v196;
  }
  else
  {
    v183.m128i_i32[0] = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v11)(
                          v6,
                          *(_QWORD *)(v10 + 64),
                          (unsigned int)v181);
    v183.m128i_i64[1] = v20;
  }
  v12 = a2[6];
  v13 = *(_QWORD *)a1;
  v14 = *(_QWORD *)(v12 + 8);
  LOWORD(v184) = *(_WORD *)v12;
  v15 = *(_QWORD *)(a1 + 8);
  v185 = v14;
  v16 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v13 + 592LL);
  if ( v16 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v194, v13, *(_QWORD *)(v15 + 64), v184, v185);
    LOWORD(v186) = v195;
    v187 = v196;
  }
  else
  {
    v186 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v16)(v13, *(_QWORD *)(v15 + 64), (unsigned int)v184);
    v187 = v19;
  }
  v17 = a2[10];
  v188 = v17;
  if ( v17 )
    sub_B96E90((__int64)&v188, v17, 1);
  v18 = *(_QWORD *)a1;
  v189 = *((_DWORD *)v3 + 18);
  sub_2FE6CC0((__int64)&v194, v18, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL), (unsigned __int16)v181, v182);
  switch ( (char)v194 )
  {
    case 1:
      v35 = v186;
      v191 = _mm_loadu_si128(&v183);
      if ( v183.m128i_i16[0] != (_WORD)v186 )
        goto LABEL_78;
      if ( (_WORD)v186 )
        goto LABEL_28;
      if ( v191.m128i_i64[1] == v187 )
        goto LABEL_81;
LABEL_78:
      v169 = v186;
      v194 = sub_2D5B750((unsigned __int16 *)&v191);
      v195 = v120;
      v121 = sub_2D5B750((unsigned __int16 *)&v186);
      v35 = v169;
      v192 = v121;
      v193 = v122;
      if ( v121 != v194 || (_BYTE)v193 != (_BYTE)v195 )
        goto LABEL_19;
      if ( v169 )
      {
LABEL_28:
        if ( (unsigned __int16)(v35 - 17) <= 0xD3u )
          goto LABEL_23;
      }
      else
      {
LABEL_81:
        v170 = v35;
        if ( sub_30070B0((__int64)&v186) )
          goto LABEL_23;
        v35 = v170;
      }
      if ( v183.m128i_i16[0] )
      {
        if ( (unsigned __int16)(v183.m128i_i16[0] - 17) <= 0xD3u )
          goto LABEL_19;
        goto LABEL_31;
      }
      v175 = v35;
      v144 = sub_30070B0((__int64)&v183);
      v35 = v175;
      if ( !v144 )
      {
LABEL_31:
        v39 = *(_QWORD *)(a1 + 8);
        sub_37AE0F0(a1, v180.m128i_u64[0], v180.m128i_i64[1]);
        v41 = v186;
        v42 = 234;
        v43 = v187;
        goto LABEL_25;
      }
      goto LABEL_19;
    case 3:
      v39 = *(_QWORD *)(a1 + 8);
      sub_3805E70(a1, v180.m128i_u64[0], v180.m128i_i64[1]);
      goto LABEL_24;
    case 5:
      if ( (_WORD)v186 )
      {
        if ( (unsigned __int16)(v186 - 17) <= 0xD3u )
          goto LABEL_23;
LABEL_35:
        v21 = *(_QWORD *)(a1 + 8);
        LODWORD(v194) = sub_375D5B0(a1, v180.m128i_u64[0], v180.m128i_i64[1]);
        v44 = sub_3805BC0(a1 + 1256, (int *)&v194);
        sub_37593F0(a1, v44);
        v45 = *(_BYTE *)(a1 + 512) & 1;
        if ( v45 )
        {
          v46 = a1 + 520;
          v47 = 7;
        }
        else
        {
          v143 = *(unsigned int *)(a1 + 528);
          v46 = *(_QWORD *)(a1 + 520);
          if ( !(_DWORD)v143 )
            goto LABEL_116;
          v47 = v143 - 1;
        }
        v48 = v47 & (37 * *v44);
        v49 = v46 + 24LL * v48;
        v50 = *(_DWORD *)v49;
        if ( *v44 == *(_DWORD *)v49 )
        {
LABEL_38:
          sub_375A6A0(a1, *(_QWORD *)(v49 + 8), *(_DWORD *)(v49 + 16), a3);
          goto LABEL_13;
        }
        v147 = 1;
        while ( v50 != -1 )
        {
          v152 = v147 + 1;
          v48 = v47 & (v147 + v48);
          v49 = v46 + 24LL * v48;
          v50 = *(_DWORD *)v49;
          if ( *v44 == *(_DWORD *)v49 )
            goto LABEL_38;
          v147 = v152;
        }
        if ( v45 )
        {
          v146 = 192;
          goto LABEL_117;
        }
        v143 = *(unsigned int *)(a1 + 528);
LABEL_116:
        v146 = 24 * v143;
LABEL_117:
        v49 = v46 + v146;
        goto LABEL_38;
      }
      if ( !sub_30070B0((__int64)&v186) )
        goto LABEL_35;
      goto LABEL_23;
    case 6:
      if ( (_WORD)v186 )
      {
        if ( (unsigned __int16)(v186 - 17) <= 0xD3u )
          goto LABEL_23;
      }
      else if ( sub_30070B0((__int64)&v186) )
      {
        goto LABEL_23;
      }
      v51 = v3[5];
      v191.m128i_i64[0] = 0;
      v191.m128i_i32[2] = 0;
      v192 = 0;
      LODWORD(v193) = 0;
      sub_375E8D0(a1, *(_QWORD *)v51, *(_QWORD *)(v51 + 8), (__int64)&v191, (__int64)&v192);
      v191.m128i_i64[0] = (__int64)sub_375A6A0(a1, v191.m128i_i64[0], v191.m128i_u32[2], a3);
      v191.m128i_i32[2] = v52;
      v192 = (unsigned __int64)sub_375A6A0(a1, v192, v193, a3);
      LODWORD(v193) = v53;
      if ( *(_BYTE *)sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 8) + 40LL)) )
      {
        a3 = _mm_loadu_si128(&v191);
        v191.m128i_i64[0] = v192;
        v191.m128i_i32[2] = v193;
        v192 = a3.m128i_i64[0];
        LODWORD(v193) = a3.m128i_i32[2];
      }
      v54 = *(_QWORD *)(a1 + 8);
      sub_375AFE0((__int64 *)a1, v191.m128i_i64[0], v191.m128i_i64[1], v192, v193, a3);
      v55 = sub_2D5B750((unsigned __int16 *)&v186);
      v195 = v56;
      v194 = v55;
      v57 = sub_CA1930(&v194);
      v58 = sub_327FC40(*(_QWORD **)(*(_QWORD *)(a1 + 8) + 64LL), v57);
      sub_33FAF80(v54, 215, (__int64)&v188, v58, v59, v60, a3);
      v33 = sub_33FAF80(*(_QWORD *)(a1 + 8), 234, (__int64)&v188, v186, v187, v61, a3);
      goto LABEL_15;
    case 7:
      v62 = v186;
      v191 = _mm_loadu_si128(&v183);
      if ( v183.m128i_i16[0] != (_WORD)v186 )
      {
        v171 = v186;
        v194 = sub_2D5B750((unsigned __int16 *)&v191);
        v195 = v123;
        v124 = sub_2D5B750((unsigned __int16 *)&v186);
        v62 = v171;
        v192 = v124;
        v193 = v125;
        if ( v124 != v194 )
          goto LABEL_90;
        goto LABEL_94;
      }
      if ( (_WORD)v186 )
        goto LABEL_46;
      if ( v191.m128i_i64[1] != v187 )
      {
        v173 = v186;
        v194 = sub_2D5B750((unsigned __int16 *)&v191);
        v195 = v138;
        v139 = sub_2D5B750((unsigned __int16 *)&v186);
        v62 = v173;
        v192 = v139;
        v193 = v140;
        if ( v194 != v139 )
          goto LABEL_103;
LABEL_94:
        if ( (_BYTE)v193 == (_BYTE)v195 )
        {
          if ( v62 )
          {
LABEL_46:
            if ( (unsigned __int16)(v62 - 17) <= 0xD3u )
              goto LABEL_47;
LABEL_97:
            v127 = *(_QWORD *)(a1 + 8);
            sub_379AB60(a1, v180.m128i_u64[0], v180.m128i_i64[1]);
            v33 = sub_33FAF80(v127, 234, (__int64)&v188, v186, v187, v128, a3);
            v130 = v129;
            if ( *(_BYTE *)sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 8) + 40LL)) )
            {
              v192 = sub_2D5B750((unsigned __int16 *)&v181);
              v193 = v131;
              v132.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v183);
              v191 = v132;
              v132.m128i_i64[1] = v132.m128i_i64[0];
              v133 = v191.m128i_i8[8];
              v194 = v132.m128i_i64[1] - v192;
              if ( v192 )
                v133 = v193;
              LOBYTE(v195) = v133;
              v134 = sub_CA1930(&v194);
              v135 = *(_QWORD **)(a1 + 8);
              *(_QWORD *)&v136 = sub_3400E40((__int64)v135, v134, v186, v187, (__int64)&v188, a3);
              *((_QWORD *)&v155 + 1) = v130;
              *(_QWORD *)&v155 = v33;
              v33 = sub_3406EB0(v135, 0xC0u, (__int64)&v188, v186, v187, v137, v155, v136);
            }
            goto LABEL_15;
          }
          goto LABEL_96;
        }
LABEL_90:
        if ( v62 )
        {
          if ( (unsigned __int16)(v62 - 17) > 0xD3u )
            goto LABEL_21;
          goto LABEL_47;
        }
LABEL_103:
        v174 = v62;
        v141 = sub_30070B0((__int64)&v186);
        v62 = v174;
        if ( !v141 )
          goto LABEL_21;
        goto LABEL_47;
      }
LABEL_96:
      v172 = v62;
      v126 = sub_30070B0((__int64)&v186);
      v62 = v172;
      if ( !v126 )
        goto LABEL_97;
LABEL_47:
      v160 = v62;
      v63.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v183);
      v191 = v63;
      v64 = sub_2D5B750((unsigned __int16 *)&v184);
      v35 = v160;
      v192 = v64;
      v193 = v66;
      if ( v191.m128i_i8[8] == (_BYTE)v66 )
      {
        v18 = v191.m128i_i64[0] / v192;
        if ( !(v191.m128i_i64[0] % v192) )
        {
          if ( (_WORD)v184 )
          {
            v67 = word_4456340;
            LOBYTE(v3) = (unsigned __int16)(v184 - 176) <= 0x34u;
            v68 = (unsigned int)v3;
            LODWORD(v69) = word_4456340[(unsigned __int16)v184 - 1];
          }
          else
          {
            v165 = v191.m128i_i64[0] / v192;
            v69 = sub_3007240((__int64)&v184);
            LODWORD(v18) = v165;
            v68 = HIDWORD(v69);
            LOBYTE(v3) = BYTE4(v69);
          }
          v70 = (unsigned int)(v69 * v18);
          v161 = v68;
          v71 = sub_3281170(&v184, v70, (__int64)v67, v68, v65);
          v156 = v72;
          v73 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 64LL);
          v166 = v71;
          LODWORD(v194) = v70;
          BYTE4(v194) = v161;
          v162 = v73;
          if ( (_BYTE)v3 )
          {
            LOWORD(v74) = sub_2D43AD0(v71, v70);
            v76 = v166;
            v77 = v162;
            v78 = v156;
          }
          else
          {
            LOWORD(v74) = sub_2D43050(v71, v70);
            v78 = v156;
            v77 = v162;
            v76 = v166;
          }
          v79 = 0;
          if ( !(_WORD)v74 )
          {
            v74 = sub_3009450(v77, v76, v78, v194, (__int64)v77, v75);
            v158 = HIWORD(v74);
            v79 = v149;
          }
          v18 = *(_QWORD *)a1;
          HIWORD(v80) = v158;
          LOWORD(v80) = v74;
          v157 = v80;
          sub_2FE6CC0(
            (__int64)&v194,
            *(_QWORD *)a1,
            *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL),
            (unsigned __int16)v74,
            v79);
          v35 = v186;
          if ( !(_BYTE)v194 )
          {
            v81 = *(_QWORD *)(a1 + 8);
            v82 = sub_379AB60(a1, v180.m128i_u64[0], v180.m128i_i64[1]);
            v84 = sub_33FB890(v81, v157, v79, v82, v83, a3);
            v85 = *(_QWORD **)(a1 + 8);
            v180.m128i_i64[0] = (__int64)v84;
            v180.m128i_i64[1] = v86 | v180.m128i_i64[1] & 0xFFFFFFFF00000000LL;
            *(_QWORD *)&v87 = sub_3400EE0((__int64)v85, 0, (__int64)&v188, 0, a3);
            sub_3406EB0(v85, 0xA1u, (__int64)&v188, (unsigned int)v184, v185, v88, *(_OWORD *)&v180, v87);
            v33 = sub_33FAF80(*(_QWORD *)(a1 + 8), 215, (__int64)&v188, v186, v187, v89, a3);
            goto LABEL_15;
          }
        }
      }
LABEL_19:
      if ( v35 )
      {
        if ( (unsigned __int16)(v35 - 17) <= 0xD3u )
          goto LABEL_23;
      }
      else if ( sub_30070B0((__int64)&v186) )
      {
        goto LABEL_23;
      }
LABEL_21:
      v36 = v7 + *(_QWORD *)(v176 + 48);
      v37 = *(_WORD *)v36;
      v38 = *(_QWORD *)(v36 + 8);
      LOWORD(v194) = v37;
      v195 = v38;
      if ( v37 )
      {
        if ( (unsigned __int16)(v37 - 17) > 0xD3u )
          goto LABEL_23;
      }
      else if ( !sub_30070B0((__int64)&v194) )
      {
        goto LABEL_23;
      }
      if ( !*(_BYTE *)sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 8) + 40LL)) )
      {
        v99 = (unsigned __int16 *)(*(_QWORD *)(v176 + 48) + v7);
        v100 = *v99;
        v195 = *((_QWORD *)v99 + 1);
        LOWORD(v194) = v100;
        LODWORD(v190[0]) = sub_3281170(&v194, v18, v100, v97, v98);
        v102 = v101;
        v190[1] = v101;
        v103.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)v190);
        v191 = v103;
        v192 = sub_2D5B750((unsigned __int16 *)&v186);
        v193 = v104;
        if ( (_BYTE)v104 == v191.m128i_i8[8] && !(v192 % v191.m128i_i64[0]) )
        {
          v105 = v190[0];
          v177 = v192 / v191.m128i_i64[0];
          v106 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 64LL);
          LOWORD(v107) = sub_2D43050(v190[0], v192 / v191.m128i_i64[0]);
          v108 = 0;
          if ( !(_WORD)v107 )
          {
            v107 = sub_3009400(v106, v105, v102, v177, 0);
            v159 = HIWORD(v107);
            v108 = v150;
          }
          HIWORD(v109) = v159;
          v178 = v108;
          LOWORD(v109) = v107;
          sub_2FE6CC0(
            (__int64)&v194,
            *(_QWORD *)a1,
            *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL),
            (unsigned __int16)v107,
            v108);
          if ( !(_BYTE)v194 )
          {
            v167 = v178;
            v163 = *(_QWORD *)(a1 + 8);
            *(_QWORD *)&v110 = sub_3400EE0(v163, 0, (__int64)&v188, 0, a3);
            v111 = *(_QWORD **)(a1 + 8);
            v179 = v110;
            v194 = 0;
            LODWORD(v195) = 0;
            v112 = sub_33F17F0(v111, 51, (__int64)&v194, v109, v167);
            v113 = v167;
            v114 = (_QWORD *)v163;
            v115 = v112;
            v117 = v116;
            if ( v194 )
            {
              v164 = v167;
              v168 = v114;
              sub_B91220((__int64)&v194, v194);
              v113 = v164;
              v114 = v168;
            }
            *((_QWORD *)&v154 + 1) = v117;
            *(_QWORD *)&v154 = v115;
            sub_340F900(v114, 0xA0u, (__int64)&v188, v109, v113, (__int64)v114, v154, *(_OWORD *)&v180, v179);
            v33 = sub_33FAF80(*(_QWORD *)(a1 + 8), 234, (__int64)&v188, v186, v187, v118, a3);
            goto LABEL_15;
          }
        }
      }
LABEL_23:
      v39 = *(_QWORD *)(a1 + 8);
      sub_375AC00(a1, v180.m128i_u64[0], v180.m128i_u64[1], v184, v185);
LABEL_24:
      v41 = v186;
      v42 = 215;
      v43 = v187;
LABEL_25:
      v33 = sub_33FAF80(v39, v42, (__int64)&v188, v41, v43, v40, a3);
LABEL_15:
      if ( v188 )
        sub_B91220((__int64)&v188, v188);
      return v33;
    case 8:
      if ( (_WORD)v186 )
      {
        if ( (unsigned __int16)(v186 - 17) <= 0xD3u )
          goto LABEL_23;
      }
      else if ( sub_30070B0((__int64)&v186) )
      {
        goto LABEL_23;
      }
      v21 = *(_QWORD *)(a1 + 8);
      LODWORD(v194) = sub_375D5B0(a1, v180.m128i_u64[0], v180.m128i_i64[1]);
      v90 = sub_3805BC0(a1 + 984, (int *)&v194);
      sub_37593F0(a1, v90);
      if ( *(_BYTE *)(a1 + 512) & 1 )
      {
        v92 = a1 + 520;
        v93 = 7;
      }
      else
      {
        v142 = *(_DWORD *)(a1 + 528);
        v92 = *(_QWORD *)(a1 + 520);
        if ( !v142 )
        {
LABEL_62:
          v32 = v187;
          v31 = 237;
          v30 = v186;
          goto LABEL_14;
        }
        v93 = v142 - 1;
      }
      v94 = v93 & (37 * *v90);
      v95 = (int *)(v92 + 24LL * v94);
      v96 = *v95;
      if ( *v90 != *v95 )
      {
        for ( i = 1; v96 != -1; i = v23 )
        {
          v23 = i + 1;
          v94 = v93 & (i + v94);
          v153 = (int *)(v92 + 24LL * v94);
          v96 = *v153;
          if ( *v90 == *v153 )
            break;
        }
      }
      goto LABEL_62;
    case 9:
      v21 = *(_QWORD *)(a1 + 8);
      LODWORD(v194) = sub_375D5B0(a1, v180.m128i_u64[0], v180.m128i_i64[1]);
      v22 = sub_3805BC0(a1 + 1064, (int *)&v194);
      sub_37593F0(a1, v22);
      if ( *(_BYTE *)(a1 + 512) & 1 )
      {
        v25 = a1 + 520;
        v26 = 7;
      }
      else
      {
        v119 = *(_DWORD *)(a1 + 528);
        v25 = *(_QWORD *)(a1 + 520);
        if ( !v119 )
          goto LABEL_13;
        v26 = v119 - 1;
      }
      v27 = v26 & (37 * *v22);
      v28 = (int *)(v25 + 24LL * v27);
      v29 = *v28;
      if ( *v22 != *v28 )
      {
        for ( j = 1; v29 != -1; j = v23 )
        {
          v23 = j + 1;
          v27 = v26 & (j + v27);
          v151 = (int *)(v25 + 24LL * v27);
          v29 = *v151;
          if ( *v22 == *v151 )
            break;
        }
      }
LABEL_13:
      v30 = v186;
      v31 = 215;
      v32 = v187;
LABEL_14:
      v33 = sub_33FAF80(v21, v31, (__int64)&v188, v30, v32, v23, a3);
      goto LABEL_15;
    case 10:
      sub_C64ED0("Scalarization of scalable vectors is not supported.", 1u);
    default:
      v35 = v186;
      goto LABEL_19;
  }
}
