// Function: sub_21C80B0
// Address: 0x21c80b0
//
__int64 __fastcall sub_21C80B0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 *v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int32 v10; // ebx
  __int64 v11; // rbx
  char v12; // r13
  unsigned int v13; // r15d
  __int64 v15; // rcx
  int v16; // r8d
  __int64 v17; // rdx
  __int64 v18; // r15
  unsigned int v19; // eax
  __int64 v20; // rdx
  _QWORD *v21; // rax
  __int16 v22; // ax
  int v23; // esi
  unsigned int v24; // eax
  unsigned int v25; // eax
  __int16 v26; // ax
  unsigned __int8 v27; // si
  int v28; // r8d
  int v29; // ecx
  int v30; // edx
  __int16 v31; // r15
  __int64 v32; // rax
  __int64 v33; // rdi
  int v34; // edx
  __int64 v35; // rax
  __int64 v36; // rdi
  int v37; // edx
  __int64 v38; // rax
  __int64 v39; // rdi
  int v40; // edx
  __int64 v41; // rax
  __int64 v42; // rdi
  int v43; // edx
  __int64 v44; // rax
  __int64 v45; // r9
  __m128i v46; // xmm0
  _QWORD *v47; // rdi
  int v48; // edx
  __int64 v49; // rdx
  __int64 v50; // r13
  _QWORD *v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int16 v56; // ax
  unsigned __int8 v57; // si
  int v58; // r8d
  int v59; // ecx
  int v60; // edx
  __int64 v61; // rax
  __int64 v62; // rdi
  int v63; // edx
  __int64 v64; // rax
  __int64 v65; // rdi
  int v66; // edx
  __int64 v67; // rax
  __int64 v68; // rdi
  int v69; // edx
  __int64 v70; // rax
  __int64 v71; // rdi
  int v72; // edx
  __int64 v73; // rax
  __m128i v74; // xmm1
  __m128i v75; // xmm2
  int v76; // edx
  __int16 v77; // ax
  unsigned __int8 v78; // si
  int v79; // r8d
  int v80; // ecx
  int v81; // edx
  __int64 v82; // rax
  __int64 v83; // rdi
  int v84; // edx
  __int64 v85; // rax
  __int64 v86; // rdi
  int v87; // edx
  __int64 v88; // rax
  __int64 v89; // rdi
  int v90; // edx
  __int64 v91; // rax
  __int64 v92; // rdi
  int v93; // edx
  __int64 v94; // rax
  __m128i v95; // xmm3
  __m128i v96; // xmm4
  int v97; // edx
  __int16 v98; // ax
  __int16 v99; // ax
  unsigned __int8 v100; // si
  int v101; // r8d
  int v102; // ecx
  int v103; // edx
  __int16 v104; // r13
  __int64 v105; // rax
  __int64 v106; // rdi
  int v107; // edx
  __int64 v108; // rax
  __int64 v109; // rdi
  int v110; // edx
  __int64 v111; // rax
  __int64 v112; // rdi
  int v113; // edx
  __int64 v114; // rax
  __int64 v115; // rdi
  int v116; // edx
  __int64 v117; // rax
  __int64 v118; // rcx
  int v119; // r8d
  _QWORD *v120; // rdi
  int v121; // edx
  __int64 v122; // r9
  __int64 v123; // r15
  unsigned int v124; // eax
  __int16 v125; // ax
  int v126; // [rsp+0h] [rbp-160h]
  int v127; // [rsp+0h] [rbp-160h]
  int v128; // [rsp+0h] [rbp-160h]
  int v129; // [rsp+0h] [rbp-160h]
  unsigned int v130; // [rsp+18h] [rbp-148h]
  unsigned int v131; // [rsp+1Ch] [rbp-144h]
  int v132; // [rsp+20h] [rbp-140h]
  bool v133; // [rsp+2Ah] [rbp-136h]
  char v134; // [rsp+2Bh] [rbp-135h]
  unsigned int v135; // [rsp+2Ch] [rbp-134h]
  __int64 v136; // [rsp+30h] [rbp-130h]
  unsigned int v137; // [rsp+38h] [rbp-128h]
  __int32 v138; // [rsp+3Ch] [rbp-124h]
  __int64 v139; // [rsp+48h] [rbp-118h]
  __int64 v140; // [rsp+50h] [rbp-110h] BYREF
  __int64 v141; // [rsp+58h] [rbp-108h] BYREF
  __int64 v142; // [rsp+60h] [rbp-100h] BYREF
  __int64 v143; // [rsp+68h] [rbp-F8h] BYREF
  __m128i v144; // [rsp+70h] [rbp-F0h] BYREF
  __m128i v145; // [rsp+80h] [rbp-E0h] BYREF
  __m128i v146; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v147; // [rsp+A0h] [rbp-C0h] BYREF
  int v148; // [rsp+A8h] [rbp-B8h]
  __int64 v149; // [rsp+B0h] [rbp-B0h] BYREF
  int v150; // [rsp+B8h] [rbp-A8h]
  __int64 v151; // [rsp+C0h] [rbp-A0h]
  int v152; // [rsp+C8h] [rbp-98h]
  __int64 v153; // [rsp+D0h] [rbp-90h]
  int v154; // [rsp+D8h] [rbp-88h]
  __int64 v155; // [rsp+E0h] [rbp-80h]
  int v156; // [rsp+E8h] [rbp-78h]
  __int64 v157; // [rsp+F0h] [rbp-70h]
  int v158; // [rsp+F8h] [rbp-68h]
  __m128i v159; // [rsp+100h] [rbp-60h]
  __m128i v160; // [rsp+110h] [rbp-50h]
  __int64 v161; // [rsp+120h] [rbp-40h]
  __int32 v162; // [rsp+128h] [rbp-38h]

  v7 = *(__int64 **)(a2 + 32);
  v8 = *(_QWORD *)(a2 + 72);
  v9 = *v7;
  v10 = *((_DWORD *)v7 + 2);
  v147 = v8;
  v144.m128i_i64[0] = 0;
  v136 = v9;
  v138 = v10;
  v11 = v7[5];
  v139 = v7[6];
  v144.m128i_i32[2] = 0;
  v145.m128i_i64[0] = 0;
  v145.m128i_i32[2] = 0;
  v146.m128i_i64[0] = 0;
  v146.m128i_i32[2] = 0;
  if ( v8 )
    sub_1623A60((__int64)&v147, v8, 2);
  v12 = *(_BYTE *)(a2 + 88);
  v13 = 0;
  v148 = *(_DWORD *)(a2 + 64);
  if ( !v12 )
    goto LABEL_4;
  v137 = sub_21BD7A0(*(_QWORD **)(a2 + 104));
  v17 = v137;
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 480) + 252LL) > 0x1Fu && v137 == 1 )
  {
    if ( (*(_BYTE *)(a2 + 26) & 0x40) != 0 || (unsigned __int8)sub_21BD8E0(a2, *(__int64 **)(a1 + 256)) )
    {
      v13 = sub_21C5A60((__int64 *)a1, a2, a3, a4, a5, v17, v15, v16);
      goto LABEL_4;
    }
    v123 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 32LL));
    v124 = sub_1E340A0(*(_QWORD *)(a2 + 104));
    v132 = 8 * sub_15A9520(v123, v124);
    v133 = (*(_BYTE *)(a2 + 26) & 8) != 0;
  }
  else
  {
    v18 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 32LL));
    v19 = sub_1E340A0(*(_QWORD *)(a2 + 104));
    v132 = 8 * sub_15A9520(v18, v19);
    if ( v137 == 3 || (v133 = 0, v137 <= 1) )
      v133 = (*(_BYTE *)(a2 + 26) & 8) != 0;
  }
  if ( (unsigned __int8)(v12 - 14) <= 0x5Fu )
  {
    switch ( v12 )
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
        v12 = 3;
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
        v12 = 4;
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
        v12 = 5;
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
        v12 = 6;
        break;
      case 55:
        v12 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v12 = 8;
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
        v12 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v12 = 10;
        break;
      default:
        v12 = 2;
        break;
    }
  }
  v20 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL * (unsigned int)(*(_DWORD *)(a2 + 56) - 1)) + 88LL);
  v21 = *(_QWORD **)(v20 + 24);
  if ( *(_DWORD *)(v20 + 32) > 0x40u )
    v21 = (_QWORD *)*v21;
  v135 = 1;
  if ( (_DWORD)v21 != 2 )
  {
    if ( (unsigned __int8)(v12 - 8) <= 5u || (v135 = 0, (unsigned __int8)(v12 - 86) <= 0x17u) )
      v135 = (v12 == 8) + 2;
  }
  v22 = *(_WORD *)(a2 + 24);
  if ( v22 == 659 )
  {
    v131 = 2;
  }
  else
  {
    v13 = 0;
    if ( v22 != 660 )
      goto LABEL_4;
    v131 = 4;
  }
  v134 = **(_BYTE **)(a2 + 40);
  if ( v134 == 86 )
  {
    v134 = 5;
    v135 = 3;
    v130 = 32;
  }
  else
  {
    v23 = 8;
    v24 = sub_21BD810(v12);
    if ( v24 >= 8 )
      v23 = v24;
    v130 = v23;
  }
  LOBYTE(v25) = sub_21C2A00(a1, v11, v139, (__int64)&v144);
  v13 = v25;
  if ( (_BYTE)v25 )
  {
    v26 = *(_WORD *)(a2 + 24);
    if ( v26 == 659 )
    {
      v27 = v134;
      v28 = 2994;
      v29 = 2982;
      v30 = 3018;
      v143 = 0x100000B9ALL;
      v142 = 0x100000B82LL;
      v141 = 0x100000B76LL;
      v140 = 0x100000BBELL;
      v126 = 2958;
    }
    else
    {
      if ( v26 != 660 )
        goto LABEL_37;
      BYTE4(v143) = 0;
      v27 = v134;
      v28 = 3000;
      v29 = 2988;
      v30 = 3024;
      v142 = 0x100000B88LL;
      BYTE4(v140) = 0;
      v141 = 0x100000B7CLL;
      v126 = 2964;
    }
    sub_21BD570(
      (__int64)&v149,
      v27,
      v30,
      v29,
      v28,
      (__int64)&v140,
      (__int64)&v141,
      (__int64)&v142,
      v126,
      (__int64)&v143);
    if ( BYTE4(v149) )
    {
      v31 = v149;
      v32 = sub_1D38BB0(*(_QWORD *)(a1 + 272), v133, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
      v33 = *(_QWORD *)(a1 + 272);
      v150 = v34;
      v149 = v32;
      v35 = sub_1D38BB0(v33, v137, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
      v36 = *(_QWORD *)(a1 + 272);
      v152 = v37;
      v151 = v35;
      v38 = sub_1D38BB0(v36, v131, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
      v39 = *(_QWORD *)(a1 + 272);
      v154 = v40;
      v153 = v38;
      v41 = sub_1D38BB0(v39, v135, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
      v42 = *(_QWORD *)(a1 + 272);
      v156 = v43;
      v155 = v41;
      v44 = sub_1D38BB0(v42, v130, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
      v46 = _mm_loadu_si128(&v144);
      v157 = v44;
      v47 = *(_QWORD **)(a1 + 272);
      v158 = v48;
      v49 = 7;
      v160.m128i_i64[0] = v136;
      v159 = v46;
      v160.m128i_i32[2] = v138;
LABEL_30:
      v50 = sub_1D23DE0(v47, v31, (__int64)&v147, *(_QWORD *)(a2 + 40), *(_DWORD *)(a2 + 60), v45, &v149, v49);
      goto LABEL_31;
    }
LABEL_37:
    v13 = 0;
    goto LABEL_4;
  }
  if ( v132 != 64 )
  {
    if ( (unsigned __int8)sub_21C2BA0(
                            a1,
                            v11,
                            v11,
                            v139,
                            (__int64)&v146,
                            (__int64)&v145,
                            a3,
                            *(double *)a4.m128i_i64,
                            a5) )
      goto LABEL_42;
    if ( (unsigned __int8)sub_21C2F60(
                            a1,
                            v11,
                            v11,
                            v139,
                            (__int64)&v146,
                            (__int64)&v145,
                            a3,
                            *(double *)a4.m128i_i64,
                            a5) )
    {
      v77 = *(_WORD *)(a2 + 24);
      if ( v77 == 659 )
      {
        v78 = v134;
        v79 = 2991;
        v80 = 2979;
        v81 = 3015;
        v143 = 0x100000B97LL;
        v142 = 0x100000B7FLL;
        v141 = 0x100000B73LL;
        v140 = 0x100000BBBLL;
        v128 = 2955;
      }
      else
      {
        if ( v77 != 660 )
          goto LABEL_4;
        BYTE4(v143) = 0;
        v78 = v134;
        v79 = 2997;
        v80 = 2985;
        v81 = 3021;
        v142 = 0x100000B85LL;
        BYTE4(v140) = 0;
        v141 = 0x100000B79LL;
        v128 = 2961;
      }
LABEL_53:
      sub_21BD570(
        (__int64)&v149,
        v78,
        v81,
        v80,
        v79,
        (__int64)&v140,
        (__int64)&v141,
        (__int64)&v142,
        v128,
        (__int64)&v143);
      if ( !BYTE4(v149) )
        goto LABEL_4;
      v31 = v149;
      v82 = sub_1D38BB0(*(_QWORD *)(a1 + 272), v133, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
      v83 = *(_QWORD *)(a1 + 272);
      v150 = v84;
      v149 = v82;
      v85 = sub_1D38BB0(v83, v137, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
      v86 = *(_QWORD *)(a1 + 272);
      v152 = v87;
      v151 = v85;
      v88 = sub_1D38BB0(v86, v131, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
      v89 = *(_QWORD *)(a1 + 272);
      v154 = v90;
      v153 = v88;
      v91 = sub_1D38BB0(v89, v135, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
      v92 = *(_QWORD *)(a1 + 272);
      v156 = v93;
      v155 = v91;
      v94 = sub_1D38BB0(v92, v130, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
      v95 = _mm_loadu_si128(&v146);
      v96 = _mm_loadu_si128(&v145);
      v157 = v94;
      v158 = v97;
      v159 = v95;
      v160 = v96;
      goto LABEL_47;
    }
    v99 = *(_WORD *)(a2 + 24);
    if ( v99 == 659 )
    {
      v100 = v134;
      v101 = 2989;
      v102 = 2977;
      v103 = 3013;
      v143 = 0x100000B95LL;
      v142 = 0x100000B7DLL;
      v141 = 0x100000B71LL;
      v140 = 0x100000BB9LL;
      v129 = 2953;
    }
    else
    {
      if ( v99 != 660 )
        goto LABEL_4;
      BYTE4(v143) = 0;
      v100 = v134;
      v101 = 2995;
      v102 = 2983;
      v103 = 3019;
      v142 = 0x100000B83LL;
      BYTE4(v140) = 0;
      v141 = 0x100000B77LL;
      v129 = 2959;
    }
    goto LABEL_74;
  }
  if ( !(unsigned __int8)sub_21C2BC0(
                           a1,
                           v11,
                           v11,
                           v139,
                           (__int64)&v146,
                           (__int64)&v145,
                           a3,
                           *(double *)a4.m128i_i64,
                           a5) )
  {
    if ( (unsigned __int8)sub_21C2F80(
                            a1,
                            v11,
                            v11,
                            v139,
                            (__int64)&v146,
                            (__int64)&v145,
                            a3,
                            *(double *)a4.m128i_i64,
                            a5) )
    {
      v98 = *(_WORD *)(a2 + 24);
      if ( v98 == 659 )
      {
        v78 = v134;
        v79 = 2992;
        v80 = 2980;
        v81 = 3016;
        v143 = 0x100000B98LL;
        v142 = 0x100000B80LL;
        v141 = 0x100000B74LL;
        v140 = 0x100000BBCLL;
        v128 = 2956;
      }
      else
      {
        if ( v98 != 660 )
          goto LABEL_4;
        BYTE4(v143) = 0;
        v78 = v134;
        v79 = 2998;
        v80 = 2986;
        v81 = 3022;
        v142 = 0x100000B86LL;
        BYTE4(v140) = 0;
        v141 = 0x100000B7ALL;
        v128 = 2962;
      }
      goto LABEL_53;
    }
    v125 = *(_WORD *)(a2 + 24);
    if ( v125 == 659 )
    {
      v100 = v134;
      v101 = 2990;
      v102 = 2978;
      v103 = 3014;
      v143 = 0x100000B96LL;
      v142 = 0x100000B7ELL;
      v141 = 0x100000B72LL;
      v140 = 0x100000BBALL;
      v129 = 2954;
    }
    else
    {
      if ( v125 != 660 )
        goto LABEL_4;
      BYTE4(v143) = 0;
      v100 = v134;
      v101 = 2996;
      v102 = 2984;
      v103 = 3020;
      v142 = 0x100000B84LL;
      BYTE4(v140) = 0;
      v141 = 0x100000B78LL;
      v129 = 2960;
    }
LABEL_74:
    sub_21BD570(
      (__int64)&v149,
      v100,
      v103,
      v102,
      v101,
      (__int64)&v140,
      (__int64)&v141,
      (__int64)&v142,
      v129,
      (__int64)&v143);
    if ( !BYTE4(v149) )
      goto LABEL_4;
    v104 = v149;
    v105 = sub_1D38BB0(*(_QWORD *)(a1 + 272), v133, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
    v106 = *(_QWORD *)(a1 + 272);
    v150 = v107;
    v149 = v105;
    v108 = sub_1D38BB0(v106, v137, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
    v109 = *(_QWORD *)(a1 + 272);
    v152 = v110;
    v151 = v108;
    v111 = sub_1D38BB0(v109, v131, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
    v112 = *(_QWORD *)(a1 + 272);
    v154 = v113;
    v153 = v111;
    v114 = sub_1D38BB0(v112, v135, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
    v115 = *(_QWORD *)(a1 + 272);
    v156 = v116;
    v155 = v114;
    v117 = sub_1D38BB0(v115, v130, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
    v118 = *(_QWORD *)(a2 + 40);
    v119 = *(_DWORD *)(a2 + 60);
    v157 = v117;
    v120 = *(_QWORD **)(a1 + 272);
    v158 = v121;
    v159.m128i_i64[1] = v139;
    v160.m128i_i64[0] = v136;
    v159.m128i_i64[0] = v11;
    v160.m128i_i32[2] = v138;
    v50 = sub_1D23DE0(v120, v104, (__int64)&v147, v118, v119, v122, &v149, 7);
LABEL_31:
    v13 = 1;
    v51 = (_QWORD *)sub_1E0A240(*(_QWORD *)(a1 + 256), 1);
    *v51 = *(_QWORD *)(a2 + 104);
    *(_QWORD *)(v50 + 88) = v51;
    *(_QWORD *)(v50 + 96) = v51 + 1;
    sub_1D444E0(*(_QWORD *)(a1 + 272), a2, v50);
    sub_1D49010(v50);
    sub_1D2DC70(*(const __m128i **)(a1 + 272), a2, v52, v53, v54, v55);
    goto LABEL_4;
  }
LABEL_42:
  v56 = *(_WORD *)(a2 + 24);
  if ( v56 == 659 )
  {
    v57 = v134;
    v58 = 2993;
    v59 = 2981;
    v60 = 3017;
    v143 = 0x100000B99LL;
    v142 = 0x100000B81LL;
    v141 = 0x100000B75LL;
    v140 = 0x100000BBDLL;
    v127 = 2957;
  }
  else
  {
    if ( v56 != 660 )
      goto LABEL_4;
    BYTE4(v143) = 0;
    v57 = v134;
    v58 = 2999;
    v59 = 2987;
    v60 = 3023;
    v142 = 0x100000B87LL;
    BYTE4(v140) = 0;
    v141 = 0x100000B7BLL;
    v127 = 2963;
  }
  sub_21BD570((__int64)&v149, v57, v60, v59, v58, (__int64)&v140, (__int64)&v141, (__int64)&v142, v127, (__int64)&v143);
  if ( BYTE4(v149) )
  {
    v31 = v149;
    v61 = sub_1D38BB0(*(_QWORD *)(a1 + 272), v133, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
    v62 = *(_QWORD *)(a1 + 272);
    v150 = v63;
    v149 = v61;
    v64 = sub_1D38BB0(v62, v137, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
    v65 = *(_QWORD *)(a1 + 272);
    v152 = v66;
    v151 = v64;
    v67 = sub_1D38BB0(v65, v131, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
    v68 = *(_QWORD *)(a1 + 272);
    v154 = v69;
    v153 = v67;
    v70 = sub_1D38BB0(v68, v135, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
    v71 = *(_QWORD *)(a1 + 272);
    v156 = v72;
    v155 = v70;
    v73 = sub_1D38BB0(v71, v130, (__int64)&v147, 5, 0, 1, a3, *(double *)a4.m128i_i64, a5, 0);
    v74 = _mm_loadu_si128(&v146);
    v75 = _mm_loadu_si128(&v145);
    v157 = v73;
    v158 = v76;
    v159 = v74;
    v160 = v75;
LABEL_47:
    v49 = 8;
    v47 = *(_QWORD **)(a1 + 272);
    v161 = v136;
    v162 = v138;
    goto LABEL_30;
  }
LABEL_4:
  if ( v147 )
    sub_161E7C0((__int64)&v147, v147);
  return v13;
}
