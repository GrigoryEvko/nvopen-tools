// Function: sub_378A4B0
// Address: 0x378a4b0
//
unsigned __int8 *__fastcall sub_378A4B0(__int64 *a1, unsigned __int64 a2, __m128i a3)
{
  int v3; // ebx
  int v5; // edx
  __int64 v6; // rax
  unsigned __int64 *v7; // rax
  unsigned __int16 *v8; // rax
  int v9; // r13d
  __int64 v10; // r14
  __int64 v11; // rax
  unsigned __int16 v12; // dx
  __int64 v13; // rax
  int v14; // eax
  int v15; // r13d
  __int64 v16; // r14
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // rdx
  __int64 v21; // r13
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rsi
  __m128i v30; // xmm0
  unsigned int v31; // r14d
  int v32; // eax
  __int64 v33; // r8
  __int16 v34; // ax
  __int64 v35; // r9
  __int64 v36; // rsi
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  unsigned __int64 v40; // rax
  __int64 v41; // rdx
  unsigned __int16 v42; // cx
  bool v43; // di
  unsigned int v44; // esi
  int v45; // esi
  unsigned int v46; // r13d
  __int64 v47; // rdx
  __int64 v48; // rdx
  __int64 v49; // rdx
  unsigned __int64 v50; // rax
  unsigned __int8 *result; // rax
  unsigned int v52; // esi
  unsigned __int16 v53; // ax
  __int64 v54; // r9
  __int64 *v55; // rdi
  __m128i v56; // rax
  int v57; // r9d
  __int64 v58; // rbx
  __int64 v59; // rsi
  unsigned __int8 *v60; // rax
  __int64 v61; // r8
  unsigned __int8 *v62; // r13
  unsigned __int8 *v63; // rbx
  unsigned int v64; // edx
  int v65; // r9d
  unsigned __int8 *v66; // rax
  unsigned int v67; // r9d
  unsigned __int8 *v68; // r11
  unsigned int v69; // edx
  unsigned int v70; // eax
  __int64 v71; // rdx
  __int64 v72; // r9
  __int64 v73; // rdx
  int v74; // r9d
  unsigned __int8 *v75; // r10
  int v76; // eax
  __int64 v77; // r13
  __int64 v78; // r12
  __int64 (__fastcall *v79)(__int64, __int64, unsigned int); // rbx
  __int64 v80; // rax
  _DWORD *v81; // rax
  unsigned __int8 *v82; // r10
  __int64 v83; // r11
  int v84; // edx
  unsigned __int16 v85; // ax
  __int128 v86; // rax
  __int64 v87; // r9
  __int64 v88; // rax
  __int64 v89; // rdx
  __int64 v90; // r10
  __int64 v91; // rbx
  __int64 v92; // r8
  __int64 v93; // r13
  __int64 (__fastcall *v94)(__int64, __int64, unsigned int); // rbx
  __int64 v95; // rax
  int v96; // edx
  unsigned __int16 v97; // ax
  unsigned __int8 *v98; // rax
  __m128i v99; // xmm5
  int v100; // edx
  __int64 v101; // r9
  unsigned __int8 *v102; // rbx
  _QWORD *v103; // rdi
  __m128i v104; // xmm2
  __m128i v105; // xmm1
  unsigned __int8 *v106; // rax
  __m128i v107; // xmm4
  unsigned int v108; // esi
  _QWORD *v109; // rdi
  unsigned int v110; // edx
  __m128i v111; // xmm3
  unsigned int v112; // edx
  __int64 v113; // r9
  unsigned __int8 *v114; // rax
  __int32 v115; // edx
  __int128 v116; // [rsp-20h] [rbp-1E0h]
  __int128 v117; // [rsp-20h] [rbp-1E0h]
  __int128 v118; // [rsp-20h] [rbp-1E0h]
  __int128 v119; // [rsp-20h] [rbp-1E0h]
  __int128 v120; // [rsp-10h] [rbp-1D0h]
  __int128 v121; // [rsp-10h] [rbp-1D0h]
  __int128 v122; // [rsp-10h] [rbp-1D0h]
  __int64 v123; // [rsp+0h] [rbp-1C0h]
  unsigned int v124; // [rsp+8h] [rbp-1B8h]
  unsigned int v125; // [rsp+8h] [rbp-1B8h]
  __int64 v126; // [rsp+10h] [rbp-1B0h]
  __int32 v127; // [rsp+10h] [rbp-1B0h]
  __int64 v128; // [rsp+10h] [rbp-1B0h]
  unsigned __int64 v129; // [rsp+18h] [rbp-1A8h]
  __int64 v130; // [rsp+18h] [rbp-1A8h]
  unsigned int v131; // [rsp+18h] [rbp-1A8h]
  unsigned __int8 *v132; // [rsp+18h] [rbp-1A8h]
  unsigned int v133; // [rsp+20h] [rbp-1A0h]
  char v134; // [rsp+26h] [rbp-19Ah]
  char v135; // [rsp+27h] [rbp-199h]
  unsigned __int16 v136; // [rsp+28h] [rbp-198h]
  __int64 v137; // [rsp+28h] [rbp-198h]
  __int64 *v138; // [rsp+30h] [rbp-190h]
  unsigned __int8 *v139; // [rsp+30h] [rbp-190h]
  __int64 v140; // [rsp+38h] [rbp-188h]
  __int64 v141; // [rsp+40h] [rbp-180h]
  __int64 v142; // [rsp+40h] [rbp-180h]
  __int64 v143; // [rsp+48h] [rbp-178h]
  unsigned __int8 *v145; // [rsp+50h] [rbp-170h]
  __int128 v146; // [rsp+50h] [rbp-170h]
  unsigned __int8 *v147; // [rsp+50h] [rbp-170h]
  __int64 v148; // [rsp+58h] [rbp-168h]
  __int64 v149; // [rsp+A0h] [rbp-120h]
  __int64 v150; // [rsp+A8h] [rbp-118h]
  __int64 v151; // [rsp+B0h] [rbp-110h]
  int v152; // [rsp+B4h] [rbp-10Ch]
  __m128i v153; // [rsp+C0h] [rbp-100h] BYREF
  __m128i v154; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v155; // [rsp+E0h] [rbp-E0h] BYREF
  int v156; // [rsp+E8h] [rbp-D8h]
  __m128i v157; // [rsp+F0h] [rbp-D0h] BYREF
  __m128i v158; // [rsp+100h] [rbp-C0h] BYREF
  __m128i v159; // [rsp+110h] [rbp-B0h] BYREF
  __int64 v160; // [rsp+120h] [rbp-A0h]
  __int64 v161; // [rsp+128h] [rbp-98h]
  __int64 v162; // [rsp+130h] [rbp-90h]
  __int64 v163; // [rsp+138h] [rbp-88h]
  __m128i v164; // [rsp+140h] [rbp-80h] BYREF
  __int16 v165; // [rsp+150h] [rbp-70h]
  __int64 v166; // [rsp+158h] [rbp-68h]
  __m128i v167; // [rsp+160h] [rbp-60h] BYREF
  __m128i v168; // [rsp+170h] [rbp-50h]
  unsigned __int8 *v169; // [rsp+180h] [rbp-40h]
  int v170; // [rsp+188h] [rbp-38h]

  v5 = *(_DWORD *)(a2 + 24);
  if ( v5 > 239 )
  {
    v6 = (unsigned int)(v5 - 242) < 2 ? 0x28 : 0;
  }
  else
  {
    v6 = 40;
    if ( v5 <= 237 )
      v6 = (unsigned int)(v5 - 101) < 0x30 ? 0x28 : 0;
  }
  v7 = (unsigned __int64 *)(*(_QWORD *)(a2 + 40) + v6);
  v129 = *v7;
  v126 = v7[1];
  v8 = *(unsigned __int16 **)(*v7 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  v11 = *(_QWORD *)(a2 + 48);
  v153.m128i_i64[1] = v10;
  v153.m128i_i16[0] = v9;
  v12 = *(_WORD *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  v154.m128i_i16[0] = v12;
  v154.m128i_i64[1] = v13;
  if ( !v12 )
  {
    v151 = sub_3007240((__int64)&v154);
    v133 = v151;
    v135 = BYTE4(v151);
    HIDWORD(v149) = HIDWORD(v151);
    v134 = sub_3007030((__int64)&v154);
    if ( (_WORD)v9 )
      goto LABEL_8;
LABEL_20:
    if ( !sub_30070B0((__int64)&v153) )
    {
      v167.m128i_i64[1] = v10;
      v167.m128i_i16[0] = 0;
      goto LABEL_22;
    }
    LOWORD(v9) = sub_3009970((__int64)&v153, a2, v22, v23, v24);
LABEL_42:
    v167.m128i_i16[0] = v9;
    v167.m128i_i64[1] = v48;
    if ( (_WORD)v9 )
      goto LABEL_10;
LABEL_22:
    v25 = sub_3007260((__int64)&v167);
    v15 = v154.m128i_u16[0];
    v160 = v25;
    LODWORD(v16) = v25;
    v161 = v26;
    if ( !v154.m128i_i16[0] )
      goto LABEL_13;
LABEL_23:
    if ( (unsigned __int16)(v15 - 17) <= 0xD3u )
    {
      v27 = 0;
      LOWORD(v15) = word_4456580[v15 - 1];
LABEL_25:
      v164.m128i_i16[0] = v15;
      v164.m128i_i64[1] = v27;
      if ( (_WORD)v15 )
        goto LABEL_15;
      goto LABEL_26;
    }
LABEL_24:
    v27 = v154.m128i_i64[1];
    goto LABEL_25;
  }
  v135 = (unsigned __int16)(v12 - 176) <= 0x34u;
  LOBYTE(v152) = v135;
  v133 = word_4456340[v12 - 1];
  HIDWORD(v149) = v152;
  v134 = (unsigned __int16)(v12 - 126) <= 0x31u || (unsigned __int16)(v12 - 10) <= 6u;
  if ( !v134 )
    v134 = (unsigned __int16)(v12 - 208) <= 0x14u;
  if ( !(_WORD)v9 )
    goto LABEL_20;
LABEL_8:
  if ( (unsigned __int16)(v9 - 17) <= 0xD3u )
  {
    v48 = 0;
    LOWORD(v9) = word_4456580[v9 - 1];
    goto LABEL_42;
  }
  v167.m128i_i16[0] = v9;
  v167.m128i_i64[1] = v10;
LABEL_10:
  v14 = (unsigned __int16)v9;
  if ( (_WORD)v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
    goto LABEL_121;
  v15 = v154.m128i_u16[0];
  v16 = *(_QWORD *)&byte_444C4A0[16 * v14 - 16];
  if ( v154.m128i_i16[0] )
    goto LABEL_23;
LABEL_13:
  if ( !sub_30070B0((__int64)&v154) )
    goto LABEL_24;
  LOWORD(v15) = sub_3009970((__int64)&v154, a2, v17, v18, v19);
  v164.m128i_i64[1] = v20;
  v164.m128i_i16[0] = v15;
  if ( (_WORD)v15 )
  {
LABEL_15:
    if ( (_WORD)v15 != 1 && (unsigned __int16)(v15 - 504) > 7u )
    {
      v21 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v15 - 16];
      goto LABEL_27;
    }
LABEL_121:
    BUG();
  }
LABEL_26:
  v162 = sub_3007260((__int64)&v164);
  LODWORD(v21) = v162;
  v163 = v28;
LABEL_27:
  sub_33D0340((__int64)&v167, a1[1], v154.m128i_i64);
  sub_2FE6CC0((__int64)&v167, *a1, *(_QWORD *)(a1[1] + 64), v167.m128i_u16[0], v167.m128i_i64[1]);
  if ( !v167.m128i_i8[0] )
    return sub_37856A0(a1, a2, a3);
  v124 = v16;
  if ( 2 * (int)v21 >= (unsigned int)v16 )
    return sub_37856A0(a1, a2, a3);
  v29 = *(_QWORD *)(a2 + 80);
  v155 = v29;
  if ( v29 )
    sub_B96E90((__int64)&v155, v29, 1);
  v30 = _mm_loadu_si128(&v153);
  HIWORD(v31) = HIWORD(v3);
  v32 = *(_DWORD *)(a2 + 72);
  v157 = v30;
  v33 = v30.m128i_i64[1];
  v156 = v32;
  v34 = v153.m128i_i16[0];
  while ( 1 )
  {
    LOWORD(v31) = v34;
    v36 = *a1;
    sub_2FE6CC0((__int64)&v167, *a1, *(_QWORD *)(a1[1] + 64), v31, v33);
    if ( v167.m128i_i8[0] != 6 )
      break;
    v138 = *(__int64 **)(a1[1] + 64);
    LOWORD(v40) = v157.m128i_i16[0];
    if ( v157.m128i_i16[0] )
    {
      v141 = 0;
      v41 = v157.m128i_u16[0] - 1;
      v42 = word_4456580[v41];
LABEL_37:
      v43 = (unsigned __int16)(v40 - 176) <= 0x34u;
      v44 = word_4456340[v41];
      LOBYTE(v40) = v43;
      goto LABEL_38;
    }
    v42 = sub_3009970((__int64)&v157, v36, v37, v38, v39);
    LOWORD(v40) = v157.m128i_i16[0];
    v141 = v49;
    if ( v157.m128i_i16[0] )
    {
      v41 = v157.m128i_u16[0] - 1;
      goto LABEL_37;
    }
    v136 = v42;
    v50 = sub_3007240((__int64)&v157);
    v42 = v136;
    v44 = v50;
    v40 = HIDWORD(v50);
    v43 = v40;
LABEL_38:
    v45 = v44 >> 1;
    v167.m128i_i8[4] = v40;
    v46 = v42;
    v167.m128i_i32[0] = v45;
    if ( v43 )
    {
      v34 = sub_2D43AD0(v42, v45);
      v33 = 0;
      if ( v34 )
        goto LABEL_33;
    }
    else
    {
      v34 = sub_2D43050(v42, v45);
      v33 = 0;
      if ( v34 )
        goto LABEL_33;
    }
    v34 = sub_3009450(v138, v46, v141, v167.m128i_i64[0], 0, v35);
    v33 = v47;
LABEL_33:
    v157.m128i_i16[0] = v34;
    v157.m128i_i64[1] = v33;
  }
  sub_2FE6CC0((__int64)&v167, *a1, *(_QWORD *)(a1[1] + 64), v157.m128i_u16[0], v157.m128i_i64[1]);
  if ( v167.m128i_i8[0] == 5 )
  {
    result = sub_37856A0(a1, a2, v30);
    goto LABEL_88;
  }
  v158.m128i_i64[0] = 0;
  v158.m128i_i32[2] = 0;
  v159.m128i_i64[0] = 0;
  v159.m128i_i32[2] = 0;
  sub_375E8D0((__int64)a1, v129, v126, (__int64)&v158, (__int64)&v159);
  v52 = v124 >> 1;
  if ( !v134 )
  {
    v55 = *(__int64 **)(a1[1] + 64);
    switch ( v52 )
    {
      case 1u:
        LOWORD(v88) = 2;
        break;
      case 2u:
        LOWORD(v88) = 3;
        break;
      case 4u:
        LOWORD(v88) = 4;
        break;
      case 8u:
        LOWORD(v88) = 5;
        break;
      case 0x10u:
        LOWORD(v88) = 6;
        break;
      case 0x20u:
        LOWORD(v88) = 7;
        break;
      case 0x40u:
        LOWORD(v88) = 8;
        break;
      case 0x80u:
        LOWORD(v88) = 9;
        break;
      default:
        v88 = sub_3007020(v55, v52);
        v90 = v89;
        v123 = v88;
        v55 = *(__int64 **)(a1[1] + 64);
LABEL_85:
        v91 = v123;
        v137 = v90;
        LOWORD(v91) = v88;
        v54 = v91;
        goto LABEL_57;
    }
    v90 = 0;
    goto LABEL_85;
  }
  switch ( v52 )
  {
    case 0x10u:
      v53 = 11;
      break;
    case 0x20u:
      v53 = 12;
      break;
    case 0x40u:
      v53 = 13;
      break;
    case 0x50u:
      v53 = 14;
      break;
    case 0x80u:
      v53 = 15;
      break;
    default:
      BUG();
  }
  v137 = 0;
  v54 = v53;
  v55 = *(__int64 **)(a1[1] + 64);
LABEL_57:
  v130 = v54;
  LODWORD(v150) = v133 >> 1;
  BYTE4(v150) = v135;
  v56.m128i_i64[0] = sub_327FD70(v55, v54, v137, v150);
  v57 = v130;
  v58 = v56.m128i_i64[1];
  v59 = *(unsigned int *)(a2 + 24);
  if ( (int)v59 > 239 )
  {
    if ( (unsigned int)(v59 - 242) > 1 )
      goto LABEL_60;
  }
  else if ( (int)v59 <= 237 && (unsigned int)(v59 - 101) > 0x2F )
  {
LABEL_60:
    v125 = v130;
    v131 = v56.m128i_i32[0];
    v60 = sub_33FAF80(a1[1], v59, (__int64)&v155, v56.m128i_u32[0], v56.m128i_i64[1], v57, v30);
    v61 = v58;
    v62 = v60;
    v63 = 0;
    v143 = v64;
    v66 = sub_33FAF80(a1[1], *(unsigned int *)(a2 + 24), (__int64)&v155, v131, v61, v65, v30);
    v67 = v125;
    v68 = v66;
    v127 = 0;
    v140 = v69;
    goto LABEL_61;
  }
  v103 = (_QWORD *)a1[1];
  v104 = _mm_loadu_si128(&v158);
  v128 = v56.m128i_i64[0];
  v105 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  *((_QWORD *)&v121 + 1) = 2;
  *(_QWORD *)&v121 = &v167;
  v165 = 1;
  v167 = v105;
  v168 = v104;
  v164 = v56;
  v166 = 0;
  v106 = sub_3411BE0(v103, v59, (__int64)&v155, (unsigned __int16 *)&v164, 2, v130, v121);
  v107 = _mm_loadu_si128(&v159);
  v62 = v106;
  v108 = *(_DWORD *)(a2 + 24);
  v109 = (_QWORD *)a1[1];
  v143 = v110;
  v111 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  *((_QWORD *)&v118 + 1) = 2;
  *(_QWORD *)&v118 = &v167;
  v165 = 1;
  v164.m128i_i64[0] = v128;
  v164.m128i_i64[1] = v58;
  v167 = v111;
  v168 = v107;
  v166 = 0;
  v139 = sub_3411BE0(v109, v108, (__int64)&v155, (unsigned __int16 *)&v164, 2, 1, v118);
  v140 = v112;
  *((_QWORD *)&v122 + 1) = 1;
  *(_QWORD *)&v122 = v139;
  *((_QWORD *)&v119 + 1) = 1;
  *(_QWORD *)&v119 = v62;
  v114 = sub_3406EB0((_QWORD *)a1[1], 2u, (__int64)&v155, 1, 0, v113, v119, v122);
  v68 = v139;
  v67 = v130;
  v127 = v115;
  v63 = v114;
LABEL_61:
  v132 = v68;
  LODWORD(v149) = v133;
  BYTE4(v149) = v135;
  v70 = sub_327FD70(*(__int64 **)(a1[1] + 64), v67, v137, v149);
  *((_QWORD *)&v120 + 1) = v140;
  *(_QWORD *)&v120 = v132;
  *((_QWORD *)&v116 + 1) = v143;
  *(_QWORD *)&v116 = v62;
  v75 = sub_3406EB0((_QWORD *)a1[1], 0x9Fu, (__int64)&v155, v70, v71, v72, v116, v120);
  v76 = *(_DWORD *)(a2 + 24);
  if ( v76 > 239 )
  {
    if ( (unsigned int)(v76 - 242) <= 1 )
      goto LABEL_94;
LABEL_64:
    v77 = a1[1];
    if ( v134 )
    {
      v78 = *a1;
      v145 = v75;
      v148 = v73;
      v79 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v78 + 32LL);
      v80 = sub_2E79000(*(__int64 **)(v77 + 40));
      if ( v79 == sub_2D42F30 )
      {
        v81 = sub_AE2980(v80, 0);
        v82 = v145;
        v83 = v148;
        v84 = v81[1];
        v85 = 2;
        if ( v84 != 1 )
        {
          v85 = 3;
          if ( v84 != 2 )
          {
            v85 = 4;
            if ( v84 != 4 )
            {
              v85 = 5;
              if ( v84 != 8 )
              {
                v85 = 6;
                if ( v84 != 16 )
                {
                  v85 = 7;
                  if ( v84 != 32 )
                  {
                    v85 = 8;
                    if ( v84 != 64 )
                      v85 = 9 * (v84 == 128);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        v85 = v79(v78, v80, 0);
        v82 = v145;
        v83 = v148;
      }
      *(_QWORD *)&v146 = v82;
      *((_QWORD *)&v146 + 1) = v83;
      *(_QWORD *)&v86 = sub_3400BD0(v77, 0, (__int64)&v155, v85, 0, 1u, v30, 0);
      result = sub_3406EB0((_QWORD *)v77, 0xE6u, (__int64)&v155, v154.m128i_u32[0], v154.m128i_i64[1], v87, v146, v86);
    }
    else
    {
      result = sub_33FAF80(v77, 216, (__int64)&v155, v154.m128i_u32[0], v154.m128i_i64[1], v74, v30);
    }
  }
  else
  {
    if ( v76 <= 237 && (unsigned int)(v76 - 101) > 0x2F )
      goto LABEL_64;
LABEL_94:
    v92 = *a1;
    v167.m128i_i64[0] = (__int64)v63;
    v93 = a1[1];
    v168.m128i_i64[0] = (__int64)v75;
    v167.m128i_i32[2] = v127;
    v168.m128i_i64[1] = v73;
    v142 = v92;
    v94 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v92 + 32LL);
    v95 = sub_2E79000(*(__int64 **)(v93 + 40));
    if ( v94 == sub_2D42F30 )
    {
      v96 = sub_AE2980(v95, 0)[1];
      v97 = 2;
      if ( v96 != 1 )
      {
        v97 = 3;
        if ( v96 != 2 )
        {
          v97 = 4;
          if ( v96 != 4 )
          {
            v97 = 5;
            if ( v96 != 8 )
            {
              v97 = 6;
              if ( v96 != 16 )
              {
                v97 = 7;
                if ( v96 != 32 )
                {
                  v97 = 8;
                  if ( v96 != 64 )
                    v97 = 9 * (v96 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v97 = v94(v142, v95, 0);
    }
    v98 = sub_3400BD0(v93, 0, (__int64)&v155, v97, 0, 1u, v30, 0);
    v99 = _mm_loadu_si128(&v154);
    v170 = v100;
    *((_QWORD *)&v117 + 1) = 3;
    *(_QWORD *)&v117 = &v167;
    v165 = 1;
    v164 = v99;
    v169 = v98;
    v166 = 0;
    v102 = sub_3411BE0((_QWORD *)v93, 0x91u, (__int64)&v155, (unsigned __int16 *)&v164, 2, v101, v117);
    sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v102, 1);
    result = v102;
  }
LABEL_88:
  if ( v155 )
  {
    v147 = result;
    sub_B91220((__int64)&v155, v155);
    return v147;
  }
  return result;
}
