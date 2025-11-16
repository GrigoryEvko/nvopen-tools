// Function: sub_379D060
// Address: 0x379d060
//
unsigned __int8 *__fastcall sub_379D060(
        __int64 *a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7)
{
  __int64 v7; // rbx
  int v8; // eax
  __int64 v10; // rsi
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int, __int64); // r9
  __int16 *v12; // rdx
  unsigned __int16 v13; // di
  __int64 v14; // r8
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // r8
  __int32 v19; // eax
  __int64 v20; // rdx
  unsigned int v21; // r13d
  unsigned __int16 v22; // ax
  __m128i v23; // xmm2
  __int16 *v24; // rax
  __int16 v25; // dx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  unsigned int *v29; // rax
  unsigned int *v30; // rdx
  unsigned int *i; // rdx
  unsigned __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rbx
  unsigned __int64 v35; // r12
  unsigned __int64 v36; // r13
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  unsigned __int64 *v39; // rax
  unsigned int *v40; // rax
  __int64 v41; // r15
  unsigned __int64 v42; // rdx
  unsigned __int64 v43; // r12
  __int64 v44; // r13
  __int64 v45; // rax
  __int16 v46; // cx
  bool v47; // al
  unsigned int v48; // eax
  __int64 v49; // rcx
  unsigned __int16 *v50; // rdx
  __int64 v51; // r8
  unsigned __int64 v52; // rsi
  int v53; // eax
  unsigned __int16 v54; // ax
  __int64 v55; // rdx
  __int64 *v56; // rdi
  __int64 v57; // rax
  __int64 v58; // r9
  __int64 v59; // rdx
  __int64 v60; // r8
  __int64 v61; // rsi
  __int128 v62; // rax
  _QWORD *v63; // rdi
  __int128 v64; // rax
  __int64 v65; // r8
  _QWORD *v66; // r9
  __int64 v67; // rax
  int v68; // edx
  __int64 v69; // r14
  unsigned __int8 *v70; // r12
  unsigned __int64 v71; // r13
  __int64 v72; // rax
  unsigned __int64 v73; // rdx
  unsigned __int8 **v74; // rax
  __int64 v75; // rax
  __int64 v76; // r15
  unsigned __int8 *v77; // rcx
  unsigned __int8 *v78; // r12
  __int64 v79; // r13
  unsigned __int16 *v80; // rax
  int v81; // ebx
  __int64 v82; // rax
  bool v83; // al
  unsigned __int16 *v84; // rdx
  __int64 v85; // rcx
  unsigned __int64 v86; // rsi
  __int64 v87; // r15
  unsigned __int16 v88; // ax
  unsigned int v89; // ebx
  __int64 v90; // rax
  __int64 v91; // r9
  __int64 v92; // r8
  __int64 v93; // rbx
  _QWORD *v94; // r15
  __int128 v95; // rax
  __int64 v96; // r9
  unsigned int v97; // edx
  __int64 v98; // rdx
  __int64 v99; // rdx
  _BYTE *v100; // rdx
  _QWORD *v101; // rdi
  unsigned __int8 *v102; // r12
  unsigned __int64 v103; // r13
  unsigned int *v104; // rax
  unsigned int v105; // edx
  __int64 v106; // rax
  unsigned __int64 v107; // rdx
  unsigned __int8 **v108; // rax
  unsigned int v109; // r15d
  __int64 *v110; // r12
  unsigned __int16 v111; // ax
  __int64 v112; // rbx
  __int64 v113; // r8
  __int64 v114; // r13
  unsigned __int8 *v115; // r14
  unsigned __int64 v116; // r15
  __int64 v117; // rax
  unsigned __int64 v118; // rdx
  unsigned __int8 **v119; // rax
  __int64 v120; // rax
  __int64 v121; // r12
  unsigned __int8 *v122; // rcx
  unsigned __int8 *v123; // r14
  __int64 v124; // r15
  __int64 v125; // rax
  __int16 v126; // si
  __int64 v127; // rax
  bool v128; // al
  _QWORD *v129; // r12
  __int64 v130; // rdx
  __int64 v131; // rcx
  __int64 v132; // r8
  __int64 v133; // r9
  unsigned __int8 *v134; // r10
  __int64 v135; // r11
  __int64 v136; // r8
  __int64 v137; // rax
  unsigned int v138; // edx
  __int64 v139; // rdx
  _BYTE *v140; // rdx
  _QWORD *v141; // rdi
  unsigned __int8 *v142; // r13
  unsigned int *v143; // rax
  unsigned int v144; // edx
  __int64 v145; // rax
  unsigned __int64 v146; // rdx
  unsigned __int8 **v147; // rax
  __int64 v148; // rdx
  __int64 *v149; // r15
  __int64 v150; // rdx
  unsigned __int8 *v151; // rcx
  __int64 v152; // r8
  unsigned int v153; // edx
  unsigned __int8 *v154; // r12
  __int64 v155; // rdx
  __int64 v156; // rax
  int v157; // edx
  unsigned int v158; // eax
  __int64 v159; // rcx
  __int64 v160; // r8
  __int64 v161; // r9
  __int64 v162; // rdx
  __int128 v163; // [rsp-20h] [rbp-4C0h]
  __int128 v164; // [rsp-20h] [rbp-4C0h]
  __int128 v165; // [rsp-20h] [rbp-4C0h]
  __int128 v166; // [rsp-10h] [rbp-4B0h]
  __int128 v167; // [rsp-10h] [rbp-4B0h]
  __int128 v168; // [rsp-10h] [rbp-4B0h]
  __int128 v169; // [rsp-10h] [rbp-4B0h]
  __int64 v170; // [rsp+10h] [rbp-490h]
  unsigned __int64 v171; // [rsp+30h] [rbp-470h]
  __int64 v173; // [rsp+60h] [rbp-440h]
  unsigned __int16 v174; // [rsp+72h] [rbp-42Eh]
  unsigned int v175; // [rsp+74h] [rbp-42Ch]
  __int128 v176; // [rsp+80h] [rbp-420h]
  char v177; // [rsp+80h] [rbp-420h]
  unsigned int v178; // [rsp+80h] [rbp-420h]
  char v179; // [rsp+90h] [rbp-410h]
  unsigned __int64 v180; // [rsp+98h] [rbp-408h]
  __int64 v181; // [rsp+A0h] [rbp-400h]
  unsigned int v182; // [rsp+A0h] [rbp-400h]
  unsigned int v183; // [rsp+A8h] [rbp-3F8h]
  unsigned int v184; // [rsp+ACh] [rbp-3F4h]
  unsigned int v185; // [rsp+B0h] [rbp-3F0h]
  __int64 v186; // [rsp+B8h] [rbp-3E8h]
  int v187; // [rsp+D0h] [rbp-3D0h]
  __int16 v188; // [rsp+D2h] [rbp-3CEh]
  __int64 v189; // [rsp+D8h] [rbp-3C8h]
  unsigned __int64 v190; // [rsp+E0h] [rbp-3C0h]
  __int64 v191; // [rsp+E0h] [rbp-3C0h]
  __int64 v192; // [rsp+E0h] [rbp-3C0h]
  __int64 v193; // [rsp+E0h] [rbp-3C0h]
  __int64 v194; // [rsp+E0h] [rbp-3C0h]
  char v195; // [rsp+E0h] [rbp-3C0h]
  unsigned __int64 v196; // [rsp+E8h] [rbp-3B8h]
  __int64 v197; // [rsp+E8h] [rbp-3B8h]
  __int64 v198; // [rsp+E8h] [rbp-3B8h]
  _QWORD *v199; // [rsp+E8h] [rbp-3B8h]
  char v200; // [rsp+E8h] [rbp-3B8h]
  __int128 v201; // [rsp+F0h] [rbp-3B0h]
  unsigned __int8 *v202; // [rsp+F0h] [rbp-3B0h]
  __int64 *v203; // [rsp+F0h] [rbp-3B0h]
  __int64 v204; // [rsp+F0h] [rbp-3B0h]
  unsigned __int8 *v205; // [rsp+F0h] [rbp-3B0h]
  unsigned int v206; // [rsp+F0h] [rbp-3B0h]
  unsigned __int8 *v207; // [rsp+F0h] [rbp-3B0h]
  unsigned __int64 v208; // [rsp+F0h] [rbp-3B0h]
  __int64 v209; // [rsp+F8h] [rbp-3A8h]
  __int64 v211; // [rsp+108h] [rbp-398h]
  __int64 v212; // [rsp+150h] [rbp-350h]
  __int64 v213; // [rsp+160h] [rbp-340h] BYREF
  int v214; // [rsp+168h] [rbp-338h]
  __m128i v215; // [rsp+170h] [rbp-330h] BYREF
  __m128i v216; // [rsp+180h] [rbp-320h] BYREF
  __m128i v217; // [rsp+190h] [rbp-310h] BYREF
  __int16 v218; // [rsp+1A0h] [rbp-300h]
  __int64 v219; // [rsp+1A8h] [rbp-2F8h]
  _BYTE *v220; // [rsp+1B0h] [rbp-2F0h] BYREF
  __int64 v221; // [rsp+1B8h] [rbp-2E8h]
  _BYTE v222[64]; // [rsp+1C0h] [rbp-2E0h] BYREF
  _BYTE *v223; // [rsp+200h] [rbp-2A0h] BYREF
  __int64 v224; // [rsp+208h] [rbp-298h]
  _BYTE v225[64]; // [rsp+210h] [rbp-290h] BYREF
  unsigned int *v226; // [rsp+250h] [rbp-250h] BYREF
  __int64 v227; // [rsp+258h] [rbp-248h]
  _BYTE v228[256]; // [rsp+260h] [rbp-240h] BYREF
  unsigned __int8 **v229; // [rsp+360h] [rbp-140h] BYREF
  __int64 v230; // [rsp+368h] [rbp-138h]
  _QWORD v231[38]; // [rsp+370h] [rbp-130h] BYREF

  v8 = *(_DWORD *)(a2 + 24);
  v183 = v8;
  if ( v8 <= 146 )
  {
    if ( v8 > 140 )
      return sub_378E190(a1, a2, a3, a4, a5, a6);
  }
  else if ( (unsigned int)(v8 - 147) <= 1 )
  {
    return sub_378F130(a1, a2, a7);
  }
  v10 = *(_QWORD *)(a2 + 80);
  v175 = *(_DWORD *)(a2 + 64);
  v213 = v10;
  if ( v10 )
    sub_B96E90((__int64)&v213, v10, 1);
  v214 = *(_DWORD *)(a2 + 72);
  v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v12 = *(__int16 **)(a2 + 48);
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  v15 = a1[1];
  v16 = *(_QWORD *)(v15 + 64);
  if ( v11 == sub_2D56A50 )
  {
    v16 = *a1;
    sub_2FE6CC0((__int64)&v229, *a1, *(_QWORD *)(v15 + 64), v13, v14);
    LOWORD(v19) = v230;
    v20 = v231[0];
    v215.m128i_i16[0] = v230;
    v215.m128i_i64[1] = v231[0];
  }
  else
  {
    v19 = v11(*a1, v16, v13, v14);
    v215.m128i_i32[0] = v19;
    v215.m128i_i64[1] = v20;
  }
  if ( (_WORD)v19 )
  {
    v173 = 0;
    v174 = word_4456580[(unsigned __int16)v19 - 1];
  }
  else
  {
    v174 = sub_3009970((__int64)&v215, v16, v20, v17, v18);
    v173 = v162;
  }
  v216 = _mm_load_si128(&v215);
  v21 = sub_3281500(&v216, v16);
  v22 = v216.m128i_i16[0];
  while ( !v22 || !*(_QWORD *)(*a1 + 8LL * v22 + 112) )
  {
    if ( v21 == 1 )
      goto LABEL_123;
    v21 >>= 1;
    v16 = v21;
    v149 = *(__int64 **)(a1[1] + 64);
    v22 = sub_2D43050(v174, v21);
    v150 = 0;
    if ( !v22 )
    {
      v16 = v174;
      v22 = sub_3009400(v149, v174, v173, v21, 0);
    }
    v216.m128i_i16[0] = v22;
    v216.m128i_i64[1] = v150;
  }
  v185 = v21;
  if ( v21 == 1 )
  {
LABEL_123:
    v158 = sub_3281500(&v215, v16);
    v154 = sub_377B740((__int64)a1, (_QWORD *)a2, v158, a7, v159, v160, v161);
    goto LABEL_114;
  }
  v23 = _mm_load_si128(&v216);
  v220 = v222;
  v221 = 0x400000000LL;
  v24 = *(__int16 **)(a2 + 48);
  v25 = *v24;
  v230 = *((_QWORD *)v24 + 1);
  LOWORD(v229) = v25;
  v184 = sub_3281500(&v229, v16);
  v29 = (unsigned int *)v228;
  v30 = (unsigned int *)v228;
  v226 = (unsigned int *)v228;
  v227 = 0x1000000000LL;
  if ( v184 )
  {
    if ( v184 > 0x10uLL )
    {
      sub_C8D5F0((__int64)&v226, v228, v184, 0x10u, v27, v28);
      v30 = v226;
      v29 = &v226[4 * (unsigned int)v227];
    }
    v26 = 4LL * v184;
    for ( i = &v30[v26]; i != v29; v29 += 4 )
    {
      if ( v29 )
      {
        *(_QWORD *)v29 = 0;
        v29[2] = 0;
      }
    }
    LODWORD(v227) = v184;
  }
  v229 = (unsigned __int8 **)v231;
  v230 = 0x1000000000LL;
  sub_3050D50((__int64)&v220, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), v26 * 4, v27, v28);
  if ( v175 > 1 )
  {
    v181 = v7;
    v34 = 40;
    while ( 1 )
    {
      v40 = (unsigned int *)(v34 + *(_QWORD *)(a2 + 40));
      v41 = v40[2];
      v42 = *(_QWORD *)v40;
      v43 = *(_QWORD *)v40;
      v44 = *((_QWORD *)v40 + 1);
      v45 = *(_QWORD *)(*(_QWORD *)v40 + 48LL) + 16 * v41;
      v46 = *(_WORD *)v45;
      v32 = *(_QWORD *)(v45 + 8);
      v217.m128i_i16[0] = v46;
      v217.m128i_i64[1] = v32;
      if ( v46 )
      {
        if ( (unsigned __int16)(v46 - 17) > 0xD3u )
          goto LABEL_26;
      }
      else
      {
        v190 = v42;
        v196 = v32;
        v47 = sub_30070B0((__int64)&v217);
        v46 = 0;
        v32 = v196;
        v42 = v190;
        if ( !v47 )
          goto LABEL_26;
      }
      HIWORD(v48) = v188;
      LOWORD(v48) = v46;
      sub_2FE6CC0((__int64)&v223, *a1, *(_QWORD *)(a1[1] + 64), v48, v32);
      if ( (_BYTE)v223 != 7 )
      {
        if ( v215.m128i_i16[0] )
        {
          v50 = word_4456340;
          LOBYTE(v49) = (unsigned __int16)(v215.m128i_i16[0] - 176) <= 0x34u;
          v51 = (unsigned int)v49;
          v52 = word_4456340[v215.m128i_u16[0] - 1];
          v53 = v217.m128i_u16[0];
          if ( v217.m128i_i16[0] )
            goto LABEL_34;
LABEL_121:
          v195 = v51;
          v200 = v49;
          v54 = sub_3009970((__int64)&v217, v52, (__int64)v50, v49, v51);
          LOBYTE(v51) = v195;
          LOBYTE(v49) = v200;
        }
        else
        {
          v52 = sub_3007240((__int64)&v215);
          v53 = v217.m128i_u16[0];
          v51 = HIDWORD(v52);
          v49 = HIDWORD(v52);
          if ( !v217.m128i_i16[0] )
            goto LABEL_121;
LABEL_34:
          v54 = word_4456580[v53 - 1];
          v55 = 0;
        }
        v191 = v55;
        v56 = *(__int64 **)(a1[1] + 64);
        v197 = v54;
        LODWORD(v223) = v52;
        BYTE4(v223) = v51;
        if ( (_BYTE)v49 )
        {
          LOWORD(v57) = sub_2D43AD0(v54, v52);
          v58 = v197;
          v59 = v191;
          v60 = 0;
          if ( (_WORD)v57 )
          {
LABEL_37:
            v61 = v189;
            LOWORD(v61) = v57;
            v192 = a1[1];
            v189 = v61;
            v198 = v60;
            *(_QWORD *)&v62 = sub_3400EE0(v192, 0, (__int64)&v213, 0, a7);
            v63 = (_QWORD *)a1[1];
            v201 = v62;
            v223 = 0;
            LODWORD(v224) = 0;
            *(_QWORD *)&v64 = sub_33F17F0(v63, 51, (__int64)&v223, v61, v198);
            v65 = v198;
            v66 = (_QWORD *)v192;
            if ( v223 )
            {
              v176 = v64;
              v193 = v198;
              v199 = v66;
              sub_B91220((__int64)&v223, (__int64)v223);
              v64 = v176;
              v65 = v193;
              v66 = v199;
            }
            *((_QWORD *)&v163 + 1) = v44;
            *(_QWORD *)&v163 = v43;
            v67 = sub_340F900(v66, 0xA0u, (__int64)&v213, v61, v65, (__int64)v66, v64, v163, v201);
            LODWORD(v41) = v68;
            v42 = v67;
            v41 = (unsigned int)v41;
            goto LABEL_26;
          }
        }
        else
        {
          LOWORD(v57) = sub_2D43050(v54, v52);
          v59 = v191;
          v58 = v197;
          v60 = 0;
          if ( (_WORD)v57 )
            goto LABEL_37;
        }
        v57 = sub_3009450(v56, (unsigned int)v58, v59, (__int64)v223, 0, v58);
        v189 = v57;
        v60 = v155;
        goto LABEL_37;
      }
      v156 = sub_379AB60((__int64)a1, v43, v44);
      LODWORD(v41) = v157;
      v42 = v156;
      v41 = (unsigned int)v41;
LABEL_26:
      v35 = v42;
      v36 = v41 | v44 & 0xFFFFFFFF00000000LL;
      v37 = (unsigned int)v221;
      v38 = (unsigned int)v221 + 1LL;
      if ( v38 > HIDWORD(v221) )
      {
        sub_C8D5F0((__int64)&v220, v222, v38, 0x10u, v32, v33);
        v37 = (unsigned int)v221;
      }
      v39 = (unsigned __int64 *)&v220[16 * v37];
      v34 += 40;
      *v39 = v35;
      v39[1] = v36;
      LODWORD(v221) = v221 + 1;
      if ( 40LL * (v175 - 2) + 80 == v34 )
      {
        v7 = v181;
        break;
      }
    }
  }
  v194 = v7;
  v187 = 0;
  v182 = 0;
  v211 = 16 * (v175 - 1 + 1LL);
LABEL_42:
  if ( v184 )
  {
    if ( v185 <= v184 )
    {
      while ( 1 )
      {
        v223 = v225;
        v224 = 0x400000000LL;
        if ( v175 )
          break;
        v100 = v225;
        v75 = 0;
LABEL_63:
        a7 = _mm_load_si128(&v216);
        v101 = (_QWORD *)a1[1];
        *((_QWORD *)&v166 + 1) = v75;
        *(_QWORD *)&v166 = v100;
        v218 = 1;
        v217 = a7;
        v219 = 0;
        v102 = sub_3411BE0(v101, v183, (__int64)&v213, (unsigned __int16 *)&v217, 2, v33, v166);
        v103 = v180 & 0xFFFFFFFF00000000LL | 1;
        v104 = &v226[4 * v182];
        v180 = v103;
        *(_QWORD *)v104 = v102;
        v104[2] = v105;
        v106 = (unsigned int)v230;
        v107 = (unsigned int)v230 + 1LL;
        if ( v107 > HIDWORD(v230) )
        {
          sub_C8D5F0((__int64)&v229, v231, v107, 0x10u, v32, v33);
          v106 = (unsigned int)v230;
        }
        v108 = &v229[2 * v106];
        *v108 = v102;
        v108[1] = (unsigned __int8 *)v103;
        v187 += v185;
        v184 -= v185;
        LODWORD(v230) = v230 + 1;
        if ( v223 != v225 )
          _libc_free((unsigned __int64)v223);
        ++v182;
        if ( v185 > v184 )
          goto LABEL_68;
      }
      v69 = 0;
      while ( 1 )
      {
        v76 = *(unsigned int *)&v220[v69 + 8];
        v77 = *(unsigned __int8 **)&v220[v69];
        v78 = v77;
        v79 = *(_QWORD *)&v220[v69 + 8];
        v80 = (unsigned __int16 *)(*((_QWORD *)v77 + 6) + 16 * v76);
        v81 = *v80;
        v82 = *((_QWORD *)v80 + 1);
        v217.m128i_i16[0] = v81;
        v217.m128i_i64[1] = v82;
        if ( (_WORD)v81 )
        {
          if ( (unsigned __int16)(v81 - 17) <= 0xD3u )
            goto LABEL_52;
        }
        else
        {
          v202 = v77;
          v83 = sub_30070B0((__int64)&v217);
          v77 = v202;
          if ( v83 )
          {
LABEL_52:
            if ( v216.m128i_i16[0] )
            {
              v84 = word_4456340;
              LOBYTE(v32) = (unsigned __int16)(v216.m128i_i16[0] - 176) <= 0x34u;
              v85 = (unsigned int)v32;
              v86 = word_4456340[v216.m128i_u16[0] - 1];
              if ( (_WORD)v81 )
                goto LABEL_54;
LABEL_59:
              v179 = v32;
              v177 = v85;
              v88 = sub_3009970((__int64)&v217, v86, (__int64)v84, v85, v32);
              LOBYTE(v32) = v179;
              LOBYTE(v85) = v177;
              v87 = v98;
            }
            else
            {
              v86 = sub_3007240((__int64)&v216);
              v85 = HIDWORD(v86);
              v32 = HIDWORD(v86);
              if ( !(_WORD)v81 )
                goto LABEL_59;
LABEL_54:
              v87 = 0;
              v88 = word_4456580[v81 - 1];
            }
            LODWORD(v212) = v86;
            v89 = v88;
            BYTE4(v212) = v85;
            v203 = *(__int64 **)(a1[1] + 64);
            if ( (_BYTE)v32 )
            {
              LOWORD(v90) = sub_2D43AD0(v88, v86);
              v92 = 0;
              if ( (_WORD)v90 )
              {
LABEL_57:
                v93 = v194;
                v204 = v92;
                LOWORD(v93) = v90;
                v194 = v93;
                v94 = (_QWORD *)a1[1];
                *(_QWORD *)&v95 = sub_3400EE0((__int64)v94, v187, (__int64)&v213, 0, a7);
                *((_QWORD *)&v164 + 1) = v79;
                *(_QWORD *)&v164 = v78;
                v77 = sub_3406EB0(v94, 0xA1u, (__int64)&v213, (unsigned int)v93, v204, v96, v164, v95);
                v76 = v97;
                goto LABEL_47;
              }
            }
            else
            {
              LOWORD(v90) = sub_2D43050(v88, v86);
              v92 = 0;
              if ( (_WORD)v90 )
                goto LABEL_57;
            }
            v90 = sub_3009450(v203, v89, v87, v212, 0, v91);
            v194 = v90;
            v92 = v99;
            goto LABEL_57;
          }
        }
LABEL_47:
        v70 = v77;
        v71 = v76 | v79 & 0xFFFFFFFF00000000LL;
        v72 = (unsigned int)v224;
        v73 = (unsigned int)v224 + 1LL;
        if ( v73 > HIDWORD(v224) )
        {
          sub_C8D5F0((__int64)&v223, v225, v73, 0x10u, v32, v33);
          v72 = (unsigned int)v224;
        }
        v74 = (unsigned __int8 **)&v223[16 * v72];
        v69 += 16;
        *v74 = v70;
        v74[1] = (unsigned __int8 *)v71;
        v75 = (unsigned int)(v224 + 1);
        LODWORD(v224) = v224 + 1;
        if ( v211 == v69 )
        {
          v100 = v223;
          goto LABEL_63;
        }
      }
    }
LABEL_68:
    v109 = v185;
    while ( 1 )
    {
      v109 >>= 1;
      v110 = *(__int64 **)(a1[1] + 64);
      v111 = sub_2D43050(v174, v109);
      if ( v111 )
      {
        v216.m128i_i16[0] = v111;
        v216.m128i_i64[1] = 0;
      }
      else
      {
        v111 = sub_3009400(v110, v174, v173, v109, 0);
        v216.m128i_i16[0] = v111;
        v216.m128i_i64[1] = v139;
        if ( !v111 )
          goto LABEL_87;
      }
      if ( *(_QWORD *)(*a1 + 8LL * v111 + 112) )
      {
        v185 = v109;
        if ( v109 == 1 )
        {
LABEL_73:
          v33 = v184;
          if ( !v184 )
            goto LABEL_106;
          v112 = v170;
          v186 = v187;
          v178 = v182 + v184;
          while ( 2 )
          {
            v113 = v175;
            v223 = v225;
            v224 = 0x400000000LL;
            if ( v175 )
            {
              v114 = 0;
              while ( 1 )
              {
                v121 = *(unsigned int *)&v220[v114 + 8];
                v122 = *(unsigned __int8 **)&v220[v114];
                v123 = v122;
                v124 = *(_QWORD *)&v220[v114 + 8];
                v125 = *((_QWORD *)v122 + 6) + 16 * v121;
                v126 = *(_WORD *)v125;
                v127 = *(_QWORD *)(v125 + 8);
                v217.m128i_i16[0] = v126;
                v217.m128i_i64[1] = v127;
                if ( v126 )
                {
                  if ( (unsigned __int16)(v126 - 17) <= 0xD3u )
                    goto LABEL_83;
                }
                else
                {
                  v205 = v122;
                  v128 = sub_30070B0((__int64)&v217);
                  v122 = v205;
                  if ( v128 )
                  {
LABEL_83:
                    v129 = (_QWORD *)a1[1];
                    v134 = sub_3400EE0((__int64)v129, v186, (__int64)&v213, 0, a7);
                    v135 = v130;
                    if ( v217.m128i_i16[0] )
                    {
                      v136 = 0;
                      LOWORD(v137) = word_4456580[v217.m128i_u16[0] - 1];
                    }
                    else
                    {
                      v207 = v134;
                      v209 = v130;
                      v137 = sub_3009970((__int64)&v217, v186, v130, v131, v132);
                      v134 = v207;
                      v135 = v209;
                      v112 = v137;
                      v136 = v148;
                    }
                    *((_QWORD *)&v167 + 1) = v135;
                    LOWORD(v112) = v137;
                    *(_QWORD *)&v167 = v134;
                    *((_QWORD *)&v165 + 1) = v124;
                    *(_QWORD *)&v165 = v123;
                    v122 = sub_3406EB0(v129, 0x9Eu, (__int64)&v213, (unsigned int)v112, v136, v133, v165, v167);
                    v121 = v138;
                  }
                }
                v115 = v122;
                v116 = v121 | v124 & 0xFFFFFFFF00000000LL;
                v117 = (unsigned int)v224;
                v118 = (unsigned int)v224 + 1LL;
                if ( v118 > HIDWORD(v224) )
                {
                  sub_C8D5F0((__int64)&v223, v225, v118, 0x10u, v113, v33);
                  v117 = (unsigned int)v224;
                }
                v119 = (unsigned __int8 **)&v223[16 * v117];
                v114 += 16;
                *v119 = v115;
                v119[1] = (unsigned __int8 *)v116;
                v120 = (unsigned int)(v224 + 1);
                LODWORD(v224) = v224 + 1;
                if ( v211 == v114 )
                {
                  v140 = v223;
                  goto LABEL_91;
                }
              }
            }
            v140 = v225;
            v120 = 0;
LABEL_91:
            v217.m128i_i16[0] = v174;
            v217.m128i_i64[1] = v173;
            v141 = (_QWORD *)a1[1];
            *((_QWORD *)&v168 + 1) = v120;
            *(_QWORD *)&v168 = v140;
            v218 = 1;
            v219 = 0;
            v142 = sub_3411BE0(v141, v183, (__int64)&v213, (unsigned __int16 *)&v217, 2, v33, v168);
            v32 = v171 & 0xFFFFFFFF00000000LL | 1;
            v33 = v182 + 1;
            v143 = &v226[4 * v182];
            v171 = v32;
            *(_QWORD *)v143 = v142;
            v143[2] = v144;
            v145 = (unsigned int)v230;
            v146 = (unsigned int)v230 + 1LL;
            if ( v146 > HIDWORD(v230) )
            {
              v208 = v32;
              sub_C8D5F0((__int64)&v229, v231, v146, 0x10u, v32, v33);
              v145 = (unsigned int)v230;
              v33 = v182 + 1;
              v32 = v208;
            }
            v147 = &v229[2 * v145];
            *v147 = v142;
            v147[1] = (unsigned __int8 *)v32;
            LODWORD(v230) = v230 + 1;
            if ( v223 != v225 )
            {
              v206 = v33;
              _libc_free((unsigned __int64)v223);
              v33 = v206;
            }
            ++v186;
            if ( (_DWORD)v33 != v178 )
            {
              v182 = v33;
              continue;
            }
            break;
          }
          v187 += v184;
          v170 = v112;
          v182 = v33;
LABEL_106:
          v184 = 0;
          v185 = 1;
        }
        goto LABEL_42;
      }
LABEL_87:
      if ( v109 == 1 )
        goto LABEL_73;
    }
  }
  if ( (unsigned int)v230 == 1 )
  {
    v151 = *v229;
    v152 = *((unsigned int *)v229 + 2);
  }
  else
  {
    *((_QWORD *)&v169 + 1) = (unsigned int)v230;
    *(_QWORD *)&v169 = v229;
    v151 = sub_33FC220((_QWORD *)a1[1], 2, (__int64)&v213, 1, 0, (__int64)&v213, v169);
    v152 = v153;
  }
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v151, v152);
  v154 = sub_37753B0(
           (_QWORD *)a1[1],
           *a1,
           &v226,
           v182,
           v216.m128i_u32[0],
           v216.m128i_i64[1],
           a7,
           v23.m128i_i64[0],
           v23.m128i_i64[1],
           v215.m128i_u32[0],
           v215.m128i_i64[1]);
  if ( v229 != v231 )
    _libc_free((unsigned __int64)v229);
  if ( v226 != (unsigned int *)v228 )
    _libc_free((unsigned __int64)v226);
  if ( v220 != v222 )
    _libc_free((unsigned __int64)v220);
LABEL_114:
  if ( v213 )
    sub_B91220((__int64)&v213, v213);
  return v154;
}
