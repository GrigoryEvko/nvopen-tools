// Function: sub_1F94350
// Address: 0x1f94350
//
__int64 *__fastcall sub_1F94350(__int64 **a1, __int64 a2, double a3, double a4, __m128i a5)
{
  char *v5; // rax
  char v6; // cl
  const void **v7; // r14
  __int64 v8; // rax
  __int64 v9; // r15
  unsigned int v10; // edi
  __int128 v11; // xmm0
  __m128i v12; // xmm1
  __int64 v13; // rbx
  __int64 *result; // rax
  __int64 v15; // r13
  bool v17; // al
  __int64 v18; // r9
  __int64 v19; // rsi
  int v20; // r11d
  char v21; // cl
  __int16 v22; // dx
  int v23; // edx
  int v24; // eax
  const __m128i *v25; // roff
  __int64 v26; // r9
  __int64 v27; // rsi
  char v28; // r8
  __int64 v29; // rax
  char v30; // r14
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rsi
  __int64 *v36; // r13
  __int64 v37; // rax
  _QWORD *v38; // r14
  const __m128i *v39; // r8
  __int64 v40; // rdx
  __int64 v41; // rax
  char v42; // cl
  __int64 v43; // rdx
  unsigned __int64 v44; // r9
  __int64 v45; // rdx
  const __m128i *v46; // rbx
  unsigned __int64 v47; // rax
  _BYTE *v48; // rcx
  int v49; // esi
  __m128 *v50; // rdx
  __int64 v51; // rdx
  unsigned __int64 v52; // rax
  _BYTE *v53; // rax
  __int64 v54; // rsi
  __int64 *v55; // r13
  __int128 v56; // rcx
  __int16 v57; // ax
  __int64 v58; // rax
  __int64 v59; // rdx
  unsigned __int8 *v60; // rax
  __int64 v61; // r8
  __int64 v62; // rax
  int v63; // r14d
  unsigned __int8 v64; // al
  unsigned __int16 *v65; // rcx
  int v66; // eax
  int v67; // r14d
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  int v72; // eax
  __int64 v73; // rcx
  __int64 v74; // r8
  __int64 v75; // rdx
  __int64 v76; // rax
  bool v77; // al
  __int64 v78; // rcx
  __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rdx
  int v82; // eax
  __int64 v83; // rdx
  _QWORD *v84; // rax
  __int64 v85; // rsi
  __int64 *v86; // r13
  __int64 *v87; // r12
  __int16 *v88; // rdx
  __int16 *v89; // r13
  __int64 v90; // rsi
  __int64 *v91; // r10
  __int64 v92; // r15
  __int64 v93; // rax
  __int64 v94; // rdx
  unsigned __int64 v95; // r8
  __int64 v96; // rsi
  bool v97; // al
  char v98; // si
  __int64 v99; // r10
  char v100; // al
  int v101; // r14d
  int v102; // eax
  bool v103; // al
  char v104; // al
  __int64 v105; // rdx
  unsigned int v106; // eax
  char v107; // al
  __int64 v108; // rdx
  int v109; // eax
  __int64 *v110; // r13
  const void **v111; // rbx
  unsigned int v112; // r15d
  __int64 *v113; // r12
  __int64 v114; // rdx
  __int64 v115; // rbx
  int v116; // eax
  unsigned __int64 v117; // [rsp-30h] [rbp-1B0h]
  __int16 *v118; // [rsp-28h] [rbp-1A8h]
  __int128 v119; // [rsp-20h] [rbp-1A0h]
  __m128i v120; // [rsp-10h] [rbp-190h]
  int v121; // [rsp+0h] [rbp-180h]
  int v122; // [rsp+0h] [rbp-180h]
  __int64 v123; // [rsp+8h] [rbp-178h]
  __int64 v124; // [rsp+8h] [rbp-178h]
  int v125; // [rsp+10h] [rbp-170h]
  int v126; // [rsp+10h] [rbp-170h]
  __int16 v127; // [rsp+10h] [rbp-170h]
  __int16 v128; // [rsp+10h] [rbp-170h]
  __int64 v129; // [rsp+10h] [rbp-170h]
  __int64 v130; // [rsp+18h] [rbp-168h]
  __int64 v131; // [rsp+18h] [rbp-168h]
  int v132; // [rsp+18h] [rbp-168h]
  int v133; // [rsp+18h] [rbp-168h]
  int v134; // [rsp+18h] [rbp-168h]
  char v135; // [rsp+1Ch] [rbp-164h]
  char v136; // [rsp+1Ch] [rbp-164h]
  char v137; // [rsp+1Ch] [rbp-164h]
  int v138; // [rsp+1Ch] [rbp-164h]
  __int128 v139; // [rsp+20h] [rbp-160h]
  int v140; // [rsp+20h] [rbp-160h]
  unsigned __int8 v141; // [rsp+20h] [rbp-160h]
  char v142; // [rsp+20h] [rbp-160h]
  __int64 v143; // [rsp+30h] [rbp-150h]
  __int64 v144; // [rsp+30h] [rbp-150h]
  __int64 v145; // [rsp+30h] [rbp-150h]
  __int64 v146; // [rsp+30h] [rbp-150h]
  int v147; // [rsp+30h] [rbp-150h]
  int v148; // [rsp+38h] [rbp-148h]
  __int64 *v149; // [rsp+38h] [rbp-148h]
  __int64 v150; // [rsp+38h] [rbp-148h]
  __int64 v151; // [rsp+38h] [rbp-148h]
  char v152; // [rsp+40h] [rbp-140h]
  __m128i v153; // [rsp+40h] [rbp-140h]
  int v154; // [rsp+40h] [rbp-140h]
  unsigned int v155; // [rsp+40h] [rbp-140h]
  int v156; // [rsp+50h] [rbp-130h]
  const __m128i *v157; // [rsp+60h] [rbp-120h]
  int v158; // [rsp+70h] [rbp-110h]
  __int64 v159; // [rsp+70h] [rbp-110h]
  __int64 v160; // [rsp+70h] [rbp-110h]
  int v161; // [rsp+70h] [rbp-110h]
  char v162; // [rsp+70h] [rbp-110h]
  const __m128i *v163; // [rsp+70h] [rbp-110h]
  __int64 v165; // [rsp+80h] [rbp-100h]
  __int128 v166; // [rsp+80h] [rbp-100h]
  __int64 *v167; // [rsp+80h] [rbp-100h]
  __int64 *v168; // [rsp+80h] [rbp-100h]
  __int64 v169; // [rsp+80h] [rbp-100h]
  __int64 *v170; // [rsp+80h] [rbp-100h]
  __int64 v171; // [rsp+80h] [rbp-100h]
  __int64 *v172; // [rsp+80h] [rbp-100h]
  __int64 v173; // [rsp+88h] [rbp-F8h]
  unsigned int v174; // [rsp+90h] [rbp-F0h] BYREF
  const void **v175; // [rsp+98h] [rbp-E8h]
  char v176[8]; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v177; // [rsp+A8h] [rbp-D8h]
  __int64 v178; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v179; // [rsp+B8h] [rbp-C8h]
  _BYTE *v180; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v181; // [rsp+C8h] [rbp-B8h]
  _BYTE v182[176]; // [rsp+D0h] [rbp-B0h] BYREF

  v5 = *(char **)(a2 + 40);
  v6 = *v5;
  v7 = (const void **)*((_QWORD *)v5 + 1);
  v8 = *(_QWORD *)(a2 + 32);
  LOBYTE(v174) = v6;
  v9 = *(_QWORD *)(v8 + 40);
  v10 = *(_DWORD *)(v8 + 48);
  v175 = v7;
  v11 = (__int128)_mm_loadu_si128((const __m128i *)(v8 + 40));
  v12 = _mm_loadu_si128((const __m128i *)(v8 + 80));
  v13 = *(_QWORD *)v8;
  if ( *(_WORD *)(v9 + 24) == 48 )
    return *(__int64 **)v8;
  v15 = *(_QWORD *)(v8 + 80);
  v152 = v6;
  v158 = *(_DWORD *)(v8 + 8);
  v148 = *(_DWORD *)(v8 + 88);
  v17 = sub_1D18C00(v13, 1, v158);
  v19 = *(unsigned __int16 *)(v13 + 24);
  v20 = v158;
  v21 = v152;
  if ( v17 && (_WORD)v19 == 108 )
  {
    v154 = v158;
    v162 = v21;
    v93 = sub_1F94350(a1, v13);
    v21 = v162;
    v20 = v154;
    v95 = v93;
    v18 = v94;
    if ( v93 )
    {
      v96 = *(_QWORD *)(a2 + 72);
      v180 = (_BYTE *)v96;
      v36 = *a1;
      if ( v96 )
      {
        v173 = v94;
        v171 = v93;
        sub_1623A60((__int64)&v180, v96, 2);
        v95 = v171;
        v18 = v173;
      }
      v120 = v12;
      LODWORD(v181) = *(_DWORD *)(a2 + 64);
      v119 = v11;
      v118 = (__int16 *)v18;
      v117 = v95;
      goto LABEL_89;
    }
    v19 = *(unsigned __int16 *)(v13 + 24);
  }
  if ( (_WORD)v19 == 48 )
  {
    v57 = *(_WORD *)(v9 + 24);
    if ( v57 == 109 )
    {
      v75 = *(_QWORD *)(v9 + 32);
      if ( *(_QWORD *)(v75 + 40) == v15 && *(_DWORD *)(v75 + 48) == v148 )
      {
        v76 = *(_QWORD *)(*(_QWORD *)v75 + 40LL) + 16LL * *(unsigned int *)(v75 + 8);
        if ( *(_BYTE *)v76 == v21 && (*(const void ***)(v76 + 8) == v7 || v21) )
          return *(__int64 **)v75;
      }
      goto LABEL_9;
    }
    if ( v57 != 158 )
      goto LABEL_9;
    v58 = **(_QWORD **)(v9 + 32);
    if ( *(_WORD *)(v58 + 24) != 109 )
      goto LABEL_9;
    v59 = *(_QWORD *)(v58 + 32);
    if ( v15 != *(_QWORD *)(v59 + 40) || *(_DWORD *)(v59 + 48) != v148 )
      goto LABEL_9;
    v60 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v59 + 40LL) + 16LL * *(unsigned int *)(v59 + 8));
    v61 = *v60;
    v62 = *((_QWORD *)v60 + 1);
    LOBYTE(v178) = v61;
    v160 = v62;
    v179 = v62;
    if ( (_BYTE)v61 )
    {
      v63 = word_42FA680[(unsigned __int8)(v61 - 14)];
    }
    else
    {
      HIDWORD(v130) = v20;
      v142 = v21;
      v151 = v59;
      v116 = sub_1F58D30((__int64)&v178);
      v19 = (unsigned int)v19;
      v59 = v151;
      v21 = v142;
      v61 = 0;
      v63 = v116;
      v20 = HIDWORD(v130);
    }
    if ( v21 )
    {
      v64 = v21 - 14;
      v65 = word_42FA680;
      v66 = word_42FA680[v64];
    }
    else
    {
      v147 = v20;
      v141 = v61;
      v150 = v59;
      v66 = sub_1F58D30((__int64)&v174);
      v20 = v147;
      v61 = v141;
      v59 = v150;
      v19 = (unsigned int)v19;
    }
    if ( v66 != v63
      || (v140 = v20,
          v149 = (__int64 *)v59,
          LOBYTE(v180) = v61,
          v181 = v160,
          v67 = sub_1D159A0((char *)&v180, v19, v59, (__int64)v65, v61, v18, v121, v123, v125, v130),
          v72 = sub_1D159A0((char *)&v174, v19, v68, v69, v70, v71, v122, v124, v126, v131),
          v20 = v140,
          v67 != v72) )
    {
LABEL_9:
      v23 = (unsigned __int16)v19;
      goto LABEL_10;
    }
    v73 = *v149;
    v74 = v149[1];
    return (__int64 *)sub_1D32840(
                        *a1,
                        v174,
                        v175,
                        v73,
                        v74,
                        *(double *)&v11,
                        *(double *)v12.m128i_i64,
                        *(double *)a5.m128i_i64);
  }
  if ( (_WORD)v19 != 158 )
    goto LABEL_16;
  v22 = *(_WORD *)(v9 + 24);
  if ( v22 != 158 )
    goto LABEL_9;
  v25 = *(const __m128i **)(v13 + 32);
  v26 = v25->m128i_i64[0];
  v153 = _mm_loadu_si128(v25);
  v139 = (__int128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(v9 + 32));
  v159 = 16LL * v25->m128i_u32[2];
  v27 = *(_QWORD *)(v25->m128i_i64[0] + 40) + v159;
  v28 = *(_BYTE *)v27;
  v29 = *(_QWORD *)(**(_QWORD **)(v9 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v9 + 32) + 8LL);
  v177 = *(_QWORD *)(v27 + 8);
  v176[0] = v28;
  v30 = *(_BYTE *)v29;
  v31 = *(_QWORD *)(v29 + 8);
  LOBYTE(v178) = v30;
  v179 = v31;
  if ( v28 )
  {
    if ( (unsigned __int8)(v28 - 14) > 0x5Fu )
    {
LABEL_15:
      LOWORD(v19) = v22;
      goto LABEL_16;
    }
  }
  else
  {
    v127 = v22;
    v132 = v20;
    v135 = v21;
    v143 = v26;
    v97 = sub_1F58D20((__int64)v176);
    v26 = v143;
    v21 = v135;
    v20 = v132;
    v22 = v127;
    v28 = 0;
    if ( !v97 )
      goto LABEL_15;
  }
  if ( v30 )
  {
    if ( (unsigned __int8)(v30 - 14) > 0x5Fu )
      goto LABEL_15;
    switch ( v30 )
    {
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
        v98 = 3;
        break;
      case 33:
      case 34:
      case 35:
      case 36:
      case 37:
      case 38:
      case 39:
      case 40:
      case 68:
      case 69:
      case 70:
      case 71:
      case 72:
      case 73:
        v98 = 4;
        break;
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 74:
      case 75:
      case 76:
      case 77:
      case 78:
      case 79:
        v98 = 5;
        break;
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 80:
      case 81:
      case 82:
      case 83:
      case 84:
      case 85:
        v98 = 6;
        break;
      case 55:
        v98 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v98 = 8;
        break;
      case 89:
      case 90:
      case 91:
      case 92:
      case 93:
      case 101:
      case 102:
      case 103:
      case 104:
      case 105:
        v98 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v98 = 10;
        break;
      default:
        v98 = 2;
        break;
    }
    v99 = 0;
  }
  else
  {
    v128 = v22;
    v133 = v20;
    v136 = v21;
    v144 = v26;
    v103 = sub_1F58D20((__int64)&v178);
    v20 = v133;
    v22 = v128;
    if ( !v103 )
      goto LABEL_15;
    v104 = sub_1F596B0((__int64)&v178);
    v28 = v176[0];
    v20 = v133;
    v21 = v136;
    v26 = v144;
    v98 = v104;
    v99 = v105;
  }
  if ( v28 )
  {
    switch ( v28 )
    {
      case 14:
      case 15:
      case 16:
      case 17:
      case 18:
      case 19:
      case 20:
      case 21:
      case 22:
      case 23:
      case 56:
      case 57:
      case 58:
      case 59:
      case 60:
      case 61:
        v100 = 2;
        break;
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
        v100 = 3;
        break;
      case 33:
      case 34:
      case 35:
      case 36:
      case 37:
      case 38:
      case 39:
      case 40:
      case 68:
      case 69:
      case 70:
      case 71:
      case 72:
      case 73:
        v100 = 4;
        break;
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 74:
      case 75:
      case 76:
      case 77:
      case 78:
      case 79:
        v100 = 5;
        break;
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 80:
      case 81:
      case 82:
      case 83:
      case 84:
      case 85:
        v100 = 6;
        break;
      case 55:
        v100 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v100 = 8;
        break;
      case 89:
      case 90:
      case 91:
      case 92:
      case 93:
      case 101:
      case 102:
      case 103:
      case 104:
      case 105:
        v100 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v100 = 10;
        break;
      default:
        BUG();
    }
    if ( v98 != v100 )
      goto LABEL_106;
    goto LABEL_102;
  }
  v129 = v99;
  v134 = v20;
  v137 = v21;
  v145 = v26;
  v107 = sub_1F596B0((__int64)v176);
  v26 = v145;
  v21 = v137;
  v20 = v134;
  if ( v98 == v107 && (v98 || v129 == v108) )
  {
    v28 = v176[0];
    if ( !v176[0] )
    {
      v109 = sub_1F58D30((__int64)v176);
      v26 = v145;
      v21 = v137;
      v20 = v134;
      v101 = v109;
      goto LABEL_103;
    }
LABEL_102:
    v101 = word_42FA680[(unsigned __int8)(v28 - 14)];
LABEL_103:
    if ( v21 )
    {
      v102 = word_42FA680[(unsigned __int8)(v21 - 14)];
    }
    else
    {
      v138 = v20;
      v146 = v26;
      v102 = sub_1F58D30((__int64)&v174);
      v20 = v138;
      v26 = v146;
    }
    if ( v102 != v101 )
      goto LABEL_106;
    v110 = *a1;
    v111 = *(const void ***)(*(_QWORD *)(v26 + 40) + v159 + 8);
    v112 = *(unsigned __int8 *)(*(_QWORD *)(v26 + 40) + v159);
    v180 = *(_BYTE **)(a2 + 72);
    if ( v180 )
      sub_1F6CA20((__int64 *)&v180);
    LODWORD(v181) = *(_DWORD *)(a2 + 64);
    v113 = sub_1D3A900(
             v110,
             0x6Cu,
             (__int64)&v180,
             v112,
             v111,
             0,
             (__m128)v11,
             *(double *)v12.m128i_i64,
             a5,
             v153.m128i_u64[0],
             (__int16 *)v153.m128i_i64[1],
             v139,
             v12.m128i_i64[0],
             v12.m128i_i64[1]);
    v115 = v114;
    if ( v180 )
      sub_161E7C0((__int64)&v180, (__int64)v180);
    v74 = v115;
    v73 = (__int64)v113;
    return (__int64 *)sub_1D32840(
                        *a1,
                        v174,
                        v175,
                        v73,
                        v74,
                        *(double *)&v11,
                        *(double *)v12.m128i_i64,
                        *(double *)a5.m128i_i64);
  }
LABEL_106:
  LOWORD(v19) = *(_WORD *)(v13 + 24);
LABEL_16:
  v23 = (unsigned __int16)v19;
  if ( (_WORD)v19 == 108 )
  {
    v32 = *(_QWORD *)(v13 + 32);
    v33 = *(_QWORD *)(v9 + 40) + 16LL * v10;
    v34 = *(_QWORD *)(*(_QWORD *)(v32 + 40) + 40LL) + 16LL * *(unsigned int *)(v32 + 48);
    if ( *(_BYTE *)v34 == *(_BYTE *)v33
      && (*(_QWORD *)(v34 + 8) == *(_QWORD *)(v33 + 8) || *(_BYTE *)v34)
      && v15 == *(_QWORD *)(v32 + 80)
      && v148 == *(_DWORD *)(v32 + 88) )
    {
      v35 = *(_QWORD *)(a2 + 72);
      v180 = (_BYTE *)v35;
      v36 = *a1;
      if ( v35 )
      {
        v165 = v32;
        sub_1623A60((__int64)&v180, v35, 2);
        v32 = v165;
      }
      v120 = v12;
      LODWORD(v181) = *(_DWORD *)(a2 + 64);
      v119 = v11;
      v118 = *(__int16 **)(v32 + 8);
      v117 = *(_QWORD *)v32;
LABEL_89:
      result = sub_1D3A900(
                 v36,
                 0x6Cu,
                 (__int64)&v180,
                 v174,
                 v175,
                 0,
                 (__m128)v11,
                 *(double *)v12.m128i_i64,
                 a5,
                 v117,
                 v118,
                 v119,
                 v120.m128i_i64[0],
                 v120.m128i_i64[1]);
LABEL_90:
      if ( v180 )
      {
        v172 = result;
        sub_161E7C0((__int64)&v180, (__int64)v180);
        return v172;
      }
      return result;
    }
  }
LABEL_10:
  v24 = *(unsigned __int16 *)(v15 + 24);
  if ( v24 != 32 && v24 != 10 )
    return 0;
  v37 = *(_QWORD *)(v15 + 88);
  v38 = *(_QWORD **)(v37 + 24);
  if ( *(_DWORD *)(v37 + 32) > 0x40u )
    v38 = (_QWORD *)*v38;
  if ( v23 == 108 )
  {
    v161 = v20;
    v77 = sub_1D18C00(v13, 1, v20);
    v20 = v161;
    if ( v77 )
    {
      v78 = *(_QWORD *)(v13 + 32);
      v79 = *(_QWORD *)(*(_QWORD *)(v78 + 40) + 40LL) + 16LL * *(unsigned int *)(v78 + 48);
      v80 = *(_QWORD *)(v9 + 40) + 16LL * v10;
      if ( *(_BYTE *)v80 == *(_BYTE *)v79 && (*(_QWORD *)(v80 + 8) == *(_QWORD *)(v79 + 8) || *(_BYTE *)v80) )
      {
        v81 = *(_QWORD *)(v78 + 80);
        v82 = *(unsigned __int16 *)(v81 + 24);
        if ( v82 == 10 || v82 == 32 )
        {
          v83 = *(_QWORD *)(v81 + 88);
          v84 = *(_QWORD **)(v83 + 24);
          if ( *(_DWORD *)(v83 + 32) > 0x40u )
            v84 = (_QWORD *)*v84;
          if ( (unsigned int)v38 < (unsigned int)v84 )
          {
            v85 = *(_QWORD *)(a2 + 72);
            v180 = (_BYTE *)v85;
            v86 = *a1;
            if ( v85 )
            {
              v169 = v78;
              sub_1623A60((__int64)&v180, v85, 2);
              v78 = v169;
            }
            LODWORD(v181) = *(_DWORD *)(a2 + 64);
            v87 = sub_1D3A900(
                    v86,
                    0x6Cu,
                    (__int64)&v180,
                    v174,
                    v175,
                    0,
                    (__m128)v11,
                    *(double *)v12.m128i_i64,
                    a5,
                    *(_QWORD *)v78,
                    *(__int16 **)(v78 + 8),
                    v11,
                    v12.m128i_i64[0],
                    v12.m128i_i64[1]);
            v89 = v88;
            if ( v180 )
              sub_161E7C0((__int64)&v180, (__int64)v180);
            sub_1F81BC0((__int64)a1, (__int64)v87);
            v90 = *(_QWORD *)(v13 + 72);
            v91 = *a1;
            v92 = *(_QWORD *)(v13 + 32);
            v180 = (_BYTE *)v90;
            if ( v90 )
            {
              v170 = v91;
              sub_1623A60((__int64)&v180, v90, 2);
              v91 = v170;
            }
            LODWORD(v181) = *(_DWORD *)(v13 + 64);
            result = sub_1D3A900(
                       v91,
                       0x6Cu,
                       (__int64)&v180,
                       v174,
                       v175,
                       0,
                       (__m128)v11,
                       *(double *)v12.m128i_i64,
                       a5,
                       (unsigned __int64)v87,
                       v89,
                       *(_OWORD *)(v92 + 40),
                       *(_QWORD *)(v92 + 80),
                       *(_QWORD *)(v92 + 88));
            goto LABEL_90;
          }
        }
      }
    }
  }
  if ( *(_WORD *)(v13 + 24) != 107 )
    return 0;
  if ( !sub_1D18C00(v13, 1, v20) )
    return 0;
  v39 = *(const __m128i **)(v13 + 32);
  v40 = *(_QWORD *)(v9 + 40) + 16LL * v10;
  v41 = *(_QWORD *)(v39->m128i_i64[0] + 40) + 16LL * v39->m128i_u32[2];
  v42 = *(_BYTE *)v40;
  if ( *(_BYTE *)v41 != *(_BYTE *)v40 )
    return 0;
  v43 = *(_QWORD *)(v40 + 8);
  if ( *(_QWORD *)(v41 + 8) != v43 && !v42 )
    return 0;
  LOBYTE(v180) = v42;
  v181 = v43;
  if ( v42 )
  {
    v44 = word_42FA680[(unsigned __int8)(v42 - 14)];
  }
  else
  {
    v157 = v39;
    v106 = sub_1F58D30((__int64)&v180);
    v39 = v157;
    v44 = v106;
  }
  v45 = 40LL * *(unsigned int *)(v13 + 56);
  v180 = v182;
  v46 = (const __m128i *)((char *)v39 + v45);
  v181 = 0x800000000LL;
  v47 = 0xCCCCCCCCCCCCCCCDLL * (v45 >> 3);
  if ( (unsigned __int64)v45 > 0x140 )
  {
    v155 = v44;
    v163 = v39;
    v156 = -858993459 * (v45 >> 3);
    sub_16CD150((__int64)&v180, v182, 0xCCCCCCCCCCCCCCCDLL * (v45 >> 3), 16, (int)v39, v44);
    v49 = v181;
    v48 = v180;
    LODWORD(v47) = v156;
    v39 = v163;
    v44 = v155;
    v50 = (__m128 *)&v180[16 * (unsigned int)v181];
  }
  else
  {
    v48 = v182;
    v49 = 0;
    v50 = (__m128 *)v182;
  }
  if ( v39 != v46 )
  {
    do
    {
      if ( v50 )
      {
        a5 = _mm_loadu_si128(v39);
        *v50 = (__m128)a5;
      }
      v39 = (const __m128i *)((char *)v39 + 40);
      ++v50;
    }
    while ( v46 != v39 );
    v48 = v180;
    v49 = v181;
  }
  LODWORD(v181) = v49 + v47;
  v51 = *(_QWORD *)(v15 + 88);
  v52 = *(_QWORD *)(v51 + 24);
  if ( *(_DWORD *)(v51 + 32) > 0x40u )
    v52 = *(_QWORD *)v52;
  v53 = &v48[16 * (v52 / v44)];
  *(_QWORD *)v53 = v9;
  *((_DWORD *)v53 + 2) = v10;
  v54 = *(_QWORD *)(a2 + 72);
  *(_QWORD *)&v56 = v180;
  v55 = *a1;
  *((_QWORD *)&v56 + 1) = (unsigned int)v181;
  v178 = v54;
  if ( v54 )
  {
    *(_QWORD *)&v166 = v180;
    *((_QWORD *)&v166 + 1) = (unsigned int)v181;
    sub_1623A60((__int64)&v178, v54, 2);
    v56 = v166;
  }
  LODWORD(v179) = *(_DWORD *)(a2 + 64);
  result = sub_1D359D0(v55, 107, (__int64)&v178, v174, v175, 0, *(double *)&v11, *(double *)v12.m128i_i64, a5, v56);
  if ( v178 )
  {
    v167 = result;
    sub_161E7C0((__int64)&v178, v178);
    result = v167;
  }
  if ( v180 != v182 )
  {
    v168 = result;
    _libc_free((unsigned __int64)v180);
    return v168;
  }
  return result;
}
