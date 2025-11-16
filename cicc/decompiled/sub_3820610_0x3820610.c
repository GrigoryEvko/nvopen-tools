// Function: sub_3820610
// Address: 0x3820610
//
void __fastcall sub_3820610(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int16 *v6; // rax
  __int64 v7; // rdx
  unsigned __int16 v8; // si
  __int64 v9; // r8
  __int64 (__fastcall *v10)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v11; // rsi
  __m128i v12; // xmm0
  __m128i v13; // xmm1
  __int64 v14; // rax
  __int16 v15; // bx
  unsigned __int16 v16; // cx
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rax
  __int16 v21; // cx
  __m128i *v22; // rcx
  int v23; // edx
  _QWORD *v24; // rdi
  _QWORD *v25; // rbx
  int v26; // edx
  int v27; // r15d
  __int64 v28; // rdx
  __int128 v29; // rax
  __int64 v30; // rax
  __int16 v31; // dx
  int v32; // edx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rdx
  char v36; // al
  unsigned int v37; // eax
  __int64 v38; // rax
  __int64 v39; // rdx
  unsigned __int128 v40; // rax
  __int64 v41; // rbx
  _QWORD *v42; // rdi
  unsigned __int8 *v43; // rax
  __int64 v44; // r8
  __int64 *v45; // rdi
  unsigned int v46; // edx
  unsigned __int64 v47; // rax
  __int16 v48; // dx
  unsigned __int64 v49; // rcx
  char v50; // si
  __int64 v51; // rbx
  int v52; // edx
  unsigned int v53; // edx
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rdx
  int v57; // ebx
  unsigned __int128 v58; // rax
  unsigned __int128 v59; // rax
  int v60; // eax
  __int64 v61; // rdx
  int v62; // edx
  _QWORD *v63; // rdi
  unsigned int v64; // edx
  __int16 v65; // ax
  __int16 v66; // r15
  __int64 v67; // r8
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // r9
  unsigned __int64 v71; // rcx
  __int64 v72; // rbx
  char v73; // si
  int v74; // edx
  unsigned int v75; // edx
  __int64 v76; // rdx
  __int64 v77; // r15
  __int64 v78; // rax
  int v79; // edx
  unsigned __int16 v80; // ax
  __int128 v81; // rax
  __int64 v82; // r9
  __int128 v83; // rax
  __int64 v84; // r9
  int v85; // edx
  __int64 v86; // r15
  __int64 v87; // rax
  int v88; // edx
  unsigned __int16 v89; // ax
  __int128 v90; // rax
  __int64 v91; // rax
  __int128 v92; // rax
  __int64 v93; // r9
  int v94; // edx
  int v95; // edx
  int v96; // ecx
  __int64 v97; // rbx
  __int64 v98; // rdx
  int v99; // edx
  __int64 v100; // rax
  __int16 v101; // dx
  __int64 v102; // rax
  __int64 v103; // rdx
  int v104; // eax
  __int64 v105; // r15
  int v106; // ebx
  __int64 v107; // rax
  int v108; // edx
  unsigned __int16 v109; // ax
  __int128 v110; // rax
  __int64 v111; // r9
  int v112; // edx
  __int64 v113; // rcx
  __int64 v114; // rdx
  __int64 v115; // [rsp-28h] [rbp-268h]
  __int128 v116; // [rsp-20h] [rbp-260h]
  __int128 v117; // [rsp-20h] [rbp-260h]
  __int128 v118; // [rsp-10h] [rbp-250h]
  __int128 v119; // [rsp-10h] [rbp-250h]
  unsigned __int64 v120; // [rsp+8h] [rbp-238h]
  __int64 *v121; // [rsp+10h] [rbp-230h]
  unsigned __int16 v122; // [rsp+20h] [rbp-220h]
  __int64 v123; // [rsp+20h] [rbp-220h]
  unsigned int v124; // [rsp+20h] [rbp-220h]
  __int64 v125; // [rsp+28h] [rbp-218h]
  __int64 v126; // [rsp+30h] [rbp-210h]
  __int16 v127; // [rsp+38h] [rbp-208h]
  __int64 v128; // [rsp+38h] [rbp-208h]
  int v129; // [rsp+44h] [rbp-1FCh]
  __int64 (__fastcall *v130)(__int64, __int64, unsigned int); // [rsp+48h] [rbp-1F8h]
  __int64 v131; // [rsp+48h] [rbp-1F8h]
  __int64 v132; // [rsp+48h] [rbp-1F8h]
  __int64 (__fastcall *v135)(__int64, __int64, unsigned int); // [rsp+58h] [rbp-1E8h]
  __int64 v136; // [rsp+58h] [rbp-1E8h]
  __int64 v137; // [rsp+60h] [rbp-1E0h]
  unsigned __int8 *v138; // [rsp+60h] [rbp-1E0h]
  __int64 (__fastcall *v139)(__int64, __int64, unsigned int); // [rsp+60h] [rbp-1E0h]
  unsigned __int64 v140; // [rsp+68h] [rbp-1D8h]
  unsigned __int8 *v141; // [rsp+70h] [rbp-1D0h]
  unsigned __int8 *v142; // [rsp+70h] [rbp-1D0h]
  unsigned __int64 v143; // [rsp+78h] [rbp-1C8h]
  __m128i *v144; // [rsp+B0h] [rbp-190h]
  __m128i *v145; // [rsp+F0h] [rbp-150h]
  __int64 v146; // [rsp+160h] [rbp-E0h] BYREF
  __int64 v147; // [rsp+168h] [rbp-D8h]
  __int64 v148; // [rsp+170h] [rbp-D0h] BYREF
  int v149; // [rsp+178h] [rbp-C8h]
  __int64 v150; // [rsp+180h] [rbp-C0h]
  __int64 v151; // [rsp+188h] [rbp-B8h]
  __int64 v152; // [rsp+190h] [rbp-B0h] BYREF
  __int64 v153; // [rsp+198h] [rbp-A8h]
  __int64 v154; // [rsp+1A0h] [rbp-A0h] BYREF
  __int64 v155; // [rsp+1A8h] [rbp-98h]
  __int128 v156; // [rsp+1B0h] [rbp-90h] BYREF
  __int64 v157; // [rsp+1C0h] [rbp-80h]
  unsigned __int128 v158; // [rsp+1D0h] [rbp-70h] BYREF
  __int64 v159; // [rsp+1E0h] [rbp-60h]
  __m128i v160; // [rsp+1F0h] [rbp-50h] BYREF
  __m128i v161; // [rsp+200h] [rbp-40h]

  if ( *(_DWORD *)(a2 + 24) == 298 && (*(_BYTE *)(a2 + 33) & 0xC) == 0 && (*(_WORD *)(a2 + 32) & 0x380) == 0 )
  {
    sub_3846760();
    return;
  }
  v6 = *(__int16 **)(a2 + 48);
  v7 = a1[1];
  v8 = *v6;
  v9 = *((_QWORD *)v6 + 1);
  v10 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v10 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v160, *a1, *(_QWORD *)(v7 + 64), v8, v9);
    LOWORD(v146) = v160.m128i_i16[4];
    v147 = v161.m128i_i64[0];
  }
  else
  {
    LODWORD(v146) = v10(*a1, *(_QWORD *)(v7 + 64), v8, v9);
    v147 = v98;
  }
  v11 = *(_QWORD *)(a2 + 80);
  v12 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v13 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL));
  v148 = v11;
  v129 = (*(_BYTE *)(a2 + 33) >> 2) & 3;
  v14 = *(_QWORD *)(a2 + 112);
  v15 = *(_WORD *)(v14 + 32);
  v160 = _mm_loadu_si128((const __m128i *)(v14 + 40));
  v161 = _mm_loadu_si128((const __m128i *)(v14 + 56));
  if ( v11 )
    sub_B96E90((__int64)&v148, v11, 1);
  v16 = *(_WORD *)(a2 + 96);
  v17 = *(_QWORD *)(a2 + 104);
  v149 = *(_DWORD *)(a2 + 72);
  LOWORD(v152) = v16;
  v153 = v17;
  if ( v16 == (_WORD)v146 && (v16 || v17 == v147) )
    goto LABEL_9;
  LOWORD(v154) = v146;
  v122 = v16;
  v155 = v147;
  *(_QWORD *)&v158 = sub_2D5B750((unsigned __int16 *)&v154);
  *((_QWORD *)&v158 + 1) = v28;
  *(_QWORD *)&v29 = sub_2D5B750((unsigned __int16 *)&v152);
  v16 = v122;
  v156 = v29;
  if ( BYTE8(v29) )
  {
    if ( !BYTE8(v158) )
      goto LABEL_21;
  }
  if ( (unsigned __int64)v156 <= (unsigned __int64)v158 )
  {
LABEL_9:
    v18 = v17;
    v19 = *(_QWORD *)(a2 + 112);
    v20 = v16;
    LOBYTE(v21) = *(_BYTE *)(v19 + 34);
    HIBYTE(v21) = 1;
    v22 = sub_33F1DB0(
            (__int64 *)a1[1],
            v129,
            (__int64)&v148,
            v146,
            v147,
            v21,
            *(_OWORD *)&v12,
            v13.m128i_i64[0],
            v13.m128i_i64[1],
            *(_OWORD *)v19,
            *(_QWORD *)(v19 + 16),
            v20,
            v18,
            v15,
            (__int64)&v160);
    *(_QWORD *)a3 = v22;
    v141 = (unsigned __int8 *)v22;
    *(_DWORD *)(a3 + 8) = v23;
    v143 = v12.m128i_i64[1] & 0xFFFFFFFF00000000LL | 1;
    if ( v129 == 2 )
    {
      v100 = v22[3].m128i_i64[0] + 16LL * *(unsigned int *)(a3 + 8);
      v101 = *(_WORD *)v100;
      v102 = *(_QWORD *)(v100 + 8);
      LOWORD(v158) = v101;
      *((_QWORD *)&v158 + 1) = v102;
      v154 = sub_2D5B750((unsigned __int16 *)&v158);
      v155 = v103;
      *(_QWORD *)&v158 = v154;
      BYTE8(v158) = v103;
      v104 = sub_CA1930(&v158);
      v105 = a1[1];
      v106 = v104;
      v132 = *a1;
      v139 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)*a1 + 32LL);
      v107 = sub_2E79000(*(__int64 **)(v105 + 40));
      if ( v139 == sub_2D42F30 )
      {
        v108 = sub_AE2980(v107, 0)[1];
        v109 = 2;
        if ( v108 != 1 )
        {
          v109 = 3;
          if ( v108 != 2 )
          {
            v109 = 4;
            if ( v108 != 4 )
            {
              v109 = 5;
              if ( v108 != 8 )
              {
                v109 = 6;
                if ( v108 != 16 )
                {
                  v109 = 7;
                  if ( v108 != 32 )
                  {
                    v109 = 8;
                    if ( v108 != 64 )
                      v109 = 9 * (v108 == 128);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        v109 = v139(v132, v107, 0);
      }
      *(_QWORD *)&v110 = sub_3400BD0(v105, (unsigned int)(v106 - 1), (__int64)&v148, v109, 0, 0, v12, 0);
      *(_QWORD *)a4 = sub_3406EB0(
                        (_QWORD *)v105,
                        0xBFu,
                        (__int64)&v148,
                        (unsigned int)v146,
                        v147,
                        v111,
                        *(_OWORD *)a3,
                        v110);
      *(_DWORD *)(a4 + 8) = v112;
    }
    else
    {
      v24 = (_QWORD *)a1[1];
      if ( v129 == 3 )
      {
        *(_QWORD *)a4 = sub_3400BD0((__int64)v24, 0, (__int64)&v148, (unsigned int)v146, v147, 0, v12, 0);
        *(_DWORD *)(a4 + 8) = v99;
      }
      else
      {
        *(_QWORD *)&v158 = 0;
        DWORD2(v158) = 0;
        v25 = sub_33F17F0(v24, 51, (__int64)&v158, v146, v147);
        v27 = v26;
        if ( (_QWORD)v158 )
          sub_B91220((__int64)&v158, v158);
        *(_QWORD *)a4 = v25;
        *(_DWORD *)(a4 + 8) = v27;
      }
    }
  }
  else
  {
LABEL_21:
    v127 = v15;
    if ( !*(_BYTE *)sub_2E79000(*(__int64 **)(a1[1] + 40)) )
    {
      v30 = *(_QWORD *)(a2 + 112);
      LOBYTE(v31) = *(_BYTE *)(v30 + 34);
      HIBYTE(v31) = 1;
      *(_QWORD *)a3 = sub_33F1F00(
                        (__int64 *)a1[1],
                        (unsigned int)v146,
                        v147,
                        (__int64)&v148,
                        v12.m128i_i64[0],
                        v12.m128i_i64[1],
                        v13.m128i_i64[0],
                        v13.m128i_i64[1],
                        *(_OWORD *)v30,
                        *(_QWORD *)(v30 + 16),
                        v31,
                        v15,
                        (__int64)&v160,
                        0);
      *(_DWORD *)(a3 + 8) = v32;
      *(_QWORD *)&v156 = sub_2D5B750((unsigned __int16 *)&v146);
      v33 = *(_QWORD *)(a2 + 104);
      *((_QWORD *)&v156 + 1) = v34;
      LOWORD(v34) = *(_WORD *)(a2 + 96);
      v153 = v33;
      LOWORD(v152) = v34;
      v154 = sub_2D5B750((unsigned __int16 *)&v152);
      v155 = v35;
      v36 = v35;
      *(_QWORD *)&v158 = v154 - v156;
      if ( (_QWORD)v156 )
        v36 = BYTE8(v156);
      BYTE8(v158) = v36;
      v37 = sub_CA1930(&v158);
      v38 = sub_327FC40(*(_QWORD **)(a1[1] + 64), v37);
      v125 = v39;
      v123 = v38;
      *(_QWORD *)&v40 = sub_2D5B750((unsigned __int16 *)&v146);
      v158 = v40;
      v41 = (unsigned int)((unsigned __int64)sub_CA1930(&v158) >> 3);
      LOBYTE(v151) = 0;
      v42 = (_QWORD *)a1[1];
      v150 = v41;
      v43 = sub_3409320(v42, v13.m128i_i64[0], v13.m128i_i64[1], v41, 0, (__int64)&v148, v12, 0);
      v44 = *(_QWORD *)(a2 + 112);
      v45 = (__int64 *)a1[1];
      v137 = (__int64)v43;
      v47 = v46 | v13.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      LOBYTE(v48) = *(_BYTE *)(v44 + 34);
      HIBYTE(v48) = 1;
      v49 = *(_QWORD *)v44 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v49 )
      {
        v50 = *(_BYTE *)(v44 + 20);
        v51 = *(_QWORD *)(v44 + 8) + v41;
        if ( (*(_QWORD *)v44 & 4) != 0 )
        {
          *((_QWORD *)&v156 + 1) = v51;
          BYTE4(v157) = v50;
          *(_QWORD *)&v156 = v49 | 4;
          LODWORD(v157) = *(_DWORD *)(v49 + 12);
        }
        else
        {
          *(_QWORD *)&v156 = *(_QWORD *)v44 & 0xFFFFFFFFFFFFFFF8LL;
          *((_QWORD *)&v156 + 1) = v51;
          BYTE4(v157) = v50;
          v113 = *(_QWORD *)(v49 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v113 + 8) - 17 <= 1 )
            v113 = **(_QWORD **)(v113 + 16);
          LODWORD(v157) = *(_DWORD *)(v113 + 8) >> 8;
        }
      }
      else
      {
        v96 = *(_DWORD *)(v44 + 16);
        v97 = *(_QWORD *)(v44 + 8) + v41;
        BYTE4(v157) = 0;
        *(_QWORD *)&v156 = 0;
        *((_QWORD *)&v156 + 1) = v97;
        LODWORD(v157) = v96;
      }
      v145 = sub_33F1DB0(
               v45,
               v129,
               (__int64)&v148,
               v146,
               v147,
               v48,
               *(_OWORD *)&v12,
               v137,
               v47,
               v156,
               v157,
               v123,
               v125,
               v127,
               (__int64)&v160);
      *(_QWORD *)a4 = v145;
      *(_DWORD *)(a4 + 8) = v52;
      *((_QWORD *)&v118 + 1) = 1;
      *(_QWORD *)&v118 = v145;
      *((_QWORD *)&v116 + 1) = 1;
      *(_QWORD *)&v116 = *(_QWORD *)a3;
      v142 = sub_3406EB0((_QWORD *)a1[1], 2u, (__int64)&v148, 1, 0, a1[1], v116, v118);
      sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v142, v53 | v12.m128i_i64[1] & 0xFFFFFFFF00000000LL);
      goto LABEL_15;
    }
    v54 = *(_QWORD *)(a2 + 104);
    LOWORD(v154) = *(_WORD *)(a2 + 96);
    v155 = v54;
    v55 = sub_2D5B750((unsigned __int16 *)&v154);
    *((_QWORD *)&v158 + 1) = v56;
    *(_QWORD *)&v158 = (unsigned __int64)(v55 + 7) >> 3;
    v57 = sub_CA1930(&v158);
    *(_QWORD *)&v58 = sub_2D5B750((unsigned __int16 *)&v146);
    v158 = v58;
    v120 = (unsigned __int64)sub_CA1930(&v158) >> 3;
    v121 = (__int64 *)a1[1];
    v124 = 8 * (v57 - v120);
    LOBYTE(v57) = *(_BYTE *)(*(_QWORD *)(a2 + 112) + 34LL);
    *(_QWORD *)&v59 = sub_2D5B750((unsigned __int16 *)&v154);
    BYTE1(v57) = 1;
    v158 = v59;
    v60 = sub_CA1930(&v158);
    v115 = sub_327FC40(*(_QWORD **)(a1[1] + 64), v60 - v124);
    *(_QWORD *)a4 = sub_33F1DB0(
                      v121,
                      v129,
                      (__int64)&v148,
                      v146,
                      v147,
                      v57,
                      *(_OWORD *)&v12,
                      v13.m128i_i64[0],
                      v13.m128i_i64[1],
                      *(_OWORD *)*(_QWORD *)(a2 + 112),
                      *(_QWORD *)(*(_QWORD *)(a2 + 112) + 16LL),
                      v115,
                      v61,
                      v127,
                      (__int64)&v160);
    LOBYTE(v153) = 0;
    *(_DWORD *)(a4 + 8) = v62;
    v63 = (_QWORD *)a1[1];
    v152 = (unsigned int)v120;
    v138 = sub_3409320(v63, v13.m128i_i64[0], v13.m128i_i64[1], (unsigned int)v120, 0, (__int64)&v148, v12, 0);
    v126 = a1[1];
    v140 = v64 | v13.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    LOBYTE(v65) = *(_BYTE *)(*(_QWORD *)(a2 + 112) + 34LL);
    HIBYTE(v65) = 1;
    v66 = v65;
    v67 = sub_327FC40(*(_QWORD **)(v126 + 64), v124);
    v68 = *(_QWORD *)(a2 + 112);
    v70 = v69;
    v71 = *(_QWORD *)v68 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v71 )
    {
      v72 = *(_QWORD *)(v68 + 8) + (unsigned int)v120;
      v73 = *(_BYTE *)(v68 + 20);
      if ( (*(_QWORD *)v68 & 4) != 0 )
      {
        *((_QWORD *)&v158 + 1) = *(_QWORD *)(v68 + 8) + (unsigned int)v120;
        BYTE4(v159) = v73;
        *(_QWORD *)&v158 = v71 | 4;
        LODWORD(v159) = *(_DWORD *)(v71 + 12);
      }
      else
      {
        *(_QWORD *)&v158 = *(_QWORD *)v68 & 0xFFFFFFFFFFFFFFF8LL;
        *((_QWORD *)&v158 + 1) = v72;
        BYTE4(v159) = v73;
        v114 = *(_QWORD *)(v71 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v114 + 8) - 17 <= 1 )
          v114 = **(_QWORD **)(v114 + 16);
        LODWORD(v159) = *(_DWORD *)(v114 + 8) >> 8;
      }
    }
    else
    {
      v95 = *(_DWORD *)(v68 + 16);
      v158 = __PAIR128__(*(_QWORD *)(v68 + 8) + (unsigned __int64)(unsigned int)v120, 0);
      LODWORD(v159) = v95;
      BYTE4(v159) = 0;
    }
    v144 = sub_33F1DB0(
             (__int64 *)v126,
             3,
             (__int64)&v148,
             v146,
             v147,
             v66,
             *(_OWORD *)&v12,
             (__int64)v138,
             v140,
             v158,
             v159,
             v67,
             v70,
             v127,
             (__int64)&v160);
    *(_QWORD *)a3 = v144;
    *(_DWORD *)(a3 + 8) = v74;
    *((_QWORD *)&v119 + 1) = 1;
    *(_QWORD *)&v119 = *(_QWORD *)a4;
    *((_QWORD *)&v117 + 1) = 1;
    *(_QWORD *)&v117 = v144;
    v141 = sub_3406EB0((_QWORD *)a1[1], 2u, (__int64)&v148, 1, 0, a1[1], v117, v119);
    v143 = v75 | v12.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v156 = sub_2D5B750((unsigned __int16 *)&v146);
    *((_QWORD *)&v156 + 1) = v76;
    if ( v124 < (unsigned __int64)sub_CA1930(&v156) )
    {
      v77 = a1[1];
      v128 = *a1;
      v130 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)*a1 + 32LL);
      v78 = sub_2E79000(*(__int64 **)(v77 + 40));
      if ( v130 == sub_2D42F30 )
      {
        v79 = sub_AE2980(v78, 0)[1];
        v80 = 2;
        if ( v79 != 1 )
        {
          v80 = 3;
          if ( v79 != 2 )
          {
            v80 = 4;
            if ( v79 != 4 )
            {
              v80 = 5;
              if ( v79 != 8 )
              {
                v80 = 6;
                if ( v79 != 16 )
                {
                  v80 = 7;
                  if ( v79 != 32 )
                  {
                    v80 = 8;
                    if ( v79 != 64 )
                      v80 = 9 * (v79 == 128);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        v80 = v130(v128, v78, 0);
      }
      *(_QWORD *)&v81 = sub_3400BD0(v77, v124, (__int64)&v148, v80, 0, 0, v12, 0);
      *(_QWORD *)&v83 = sub_3406EB0(
                          (_QWORD *)v77,
                          0xBEu,
                          (__int64)&v148,
                          (unsigned int)v146,
                          v147,
                          v82,
                          *(_OWORD *)a4,
                          v81);
      *(_QWORD *)a3 = sub_3406EB0(
                        (_QWORD *)v77,
                        0xBBu,
                        (__int64)&v148,
                        (unsigned int)v146,
                        v147,
                        v84,
                        *(_OWORD *)a3,
                        v83);
      *(_DWORD *)(a3 + 8) = v85;
      v86 = a1[1];
      v131 = *a1;
      v135 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)*a1 + 32LL);
      v87 = sub_2E79000(*(__int64 **)(v86 + 40));
      if ( v135 == sub_2D42F30 )
      {
        v88 = sub_AE2980(v87, 0)[1];
        v89 = 2;
        if ( v88 != 1 )
        {
          v89 = 3;
          if ( v88 != 2 )
          {
            v89 = 4;
            if ( v88 != 4 )
            {
              v89 = 5;
              if ( v88 != 8 )
              {
                v89 = 6;
                if ( v88 != 16 )
                {
                  v89 = 7;
                  if ( v88 != 32 )
                  {
                    v89 = 8;
                    if ( v88 != 64 )
                      v89 = 9 * (v88 == 128);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        v89 = v135(v131, v87, 0);
      }
      v136 = v89;
      *(_QWORD *)&v90 = sub_2D5B750((unsigned __int16 *)&v146);
      v156 = v90;
      v91 = sub_CA1930(&v156);
      *(_QWORD *)&v92 = sub_3400BD0(v86, v91 - v124, (__int64)&v148, v136, 0, 0, v12, 0);
      *(_QWORD *)a4 = sub_3406EB0(
                        (_QWORD *)v86,
                        (unsigned int)(v129 != 2) + 191,
                        (__int64)&v148,
                        (unsigned int)v146,
                        v147,
                        v93,
                        *(_OWORD *)a4,
                        v92);
      *(_DWORD *)(a4 + 8) = v94;
    }
  }
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v141, v143);
LABEL_15:
  if ( v148 )
    sub_B91220((__int64)&v148, v148);
}
