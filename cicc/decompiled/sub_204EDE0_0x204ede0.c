// Function: sub_204EDE0
// Address: 0x204ede0
//
__int64 *__fastcall sub_204EDE0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 **a6,
        __m128i a7,
        __m128i a8,
        __m128i a9,
        __int64 a10)
{
  __int64 v10; // rbx
  unsigned int v11; // r15d
  __int64 *v12; // r13
  __int64 v15; // r10
  _BYTE *v17; // rax
  _BYTE *v18; // rdx
  unsigned int v19; // eax
  unsigned int v20; // r14d
  __int64 v21; // r15
  unsigned __int8 v22; // bl
  __int64 v23; // r12
  __int64 (__fastcall *v24)(__int64, __int64, __int64, __int64, __int64); // rax
  __int64 v25; // rax
  __int64 *v26; // r10
  __int64 v27; // r11
  __int64 *v28; // r14
  __int64 v29; // rax
  int v30; // edx
  const void ***v31; // r12
  __int64 v32; // r8
  __int64 v33; // rdx
  __int64 v34; // r9
  __int64 *v35; // rax
  unsigned int v36; // edx
  __int64 v37; // r12
  __int64 v38; // r14
  _BYTE *v39; // rax
  int v40; // eax
  bool v41; // bl
  __int64 v42; // rax
  int v43; // r14d
  __int64 v44; // rax
  __int64 v45; // rax
  int v46; // edx
  const void ***v47; // r12
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rdx
  __int64 v51; // r9
  __int64 *v52; // rax
  __int64 v53; // rax
  int v54; // eax
  __int64 v55; // r8
  __int64 v56; // r9
  _DWORD *v57; // rdx
  int v58; // esi
  unsigned int v59; // eax
  __int64 v60; // rcx
  unsigned __int64 v61; // rax
  unsigned int v62; // eax
  unsigned int v63; // esi
  __int64 v64; // rax
  unsigned __int64 v65; // rdx
  __int64 v66; // rbx
  __int64 v67; // rsi
  __int128 v68; // rax
  unsigned int v69; // edi
  __int64 *v70; // rax
  int v71; // edx
  int v72; // esi
  __int64 *v73; // rdx
  __int64 v74; // rax
  char v75; // al
  __int64 *v76; // rax
  int v77; // edx
  int v78; // esi
  __int64 *v79; // rdx
  unsigned __int64 v80; // rax
  unsigned __int64 v81; // r14
  __int64 v82; // r15
  const void ***v83; // rax
  int v84; // edx
  __int64 v85; // r9
  unsigned int v86; // eax
  __int64 v87; // rax
  int v88; // edx
  int v89; // esi
  __int64 v90; // rdx
  __int64 v91; // rax
  _BYTE *v92; // rax
  _BYTE *i; // rbx
  unsigned int v94; // esi
  __int64 v95; // rax
  _QWORD *v96; // rbx
  __int64 v97; // rax
  unsigned __int8 v98; // al
  __int128 v99; // [rsp-10h] [rbp-2F0h]
  __int128 v100; // [rsp-10h] [rbp-2F0h]
  __int128 v101; // [rsp-10h] [rbp-2F0h]
  __int64 v102; // [rsp+8h] [rbp-2D8h]
  unsigned int v103; // [rsp+18h] [rbp-2C8h]
  __int64 v104; // [rsp+28h] [rbp-2B8h]
  __int64 *v105; // [rsp+30h] [rbp-2B0h]
  __int64 v106; // [rsp+38h] [rbp-2A8h]
  __int64 v107; // [rsp+40h] [rbp-2A0h]
  __int64 v108; // [rsp+48h] [rbp-298h]
  unsigned int v110; // [rsp+70h] [rbp-270h]
  __int64 v111; // [rsp+78h] [rbp-268h]
  __int64 v113; // [rsp+88h] [rbp-258h]
  __int64 v114; // [rsp+98h] [rbp-248h]
  unsigned int v115; // [rsp+B0h] [rbp-230h]
  unsigned int v116; // [rsp+B4h] [rbp-22Ch]
  __int64 *v118; // [rsp+C0h] [rbp-220h]
  unsigned int v119; // [rsp+C8h] [rbp-218h]
  __int64 v120; // [rsp+C8h] [rbp-218h]
  int v121; // [rsp+D0h] [rbp-210h]
  __int64 *v122; // [rsp+D0h] [rbp-210h]
  __int64 v123; // [rsp+D0h] [rbp-210h]
  int v124; // [rsp+D8h] [rbp-208h]
  int v125; // [rsp+D8h] [rbp-208h]
  __int64 *v126; // [rsp+D8h] [rbp-208h]
  __int64 v127; // [rsp+E0h] [rbp-200h]
  __int64 v128; // [rsp+E0h] [rbp-200h]
  __int64 v129; // [rsp+E0h] [rbp-200h]
  __int64 v130; // [rsp+E0h] [rbp-200h]
  __int64 v131; // [rsp+E8h] [rbp-1F8h]
  int v132; // [rsp+E8h] [rbp-1F8h]
  __int64 v133; // [rsp+E8h] [rbp-1F8h]
  __int64 *v134; // [rsp+E8h] [rbp-1F8h]
  __int64 *v135; // [rsp+E8h] [rbp-1F8h]
  __int64 v136; // [rsp+E8h] [rbp-1F8h]
  __int64 *v137; // [rsp+E8h] [rbp-1F8h]
  unsigned int v138; // [rsp+100h] [rbp-1E0h]
  unsigned __int8 v139; // [rsp+107h] [rbp-1D9h]
  __int64 v140; // [rsp+120h] [rbp-1C0h]
  unsigned int v141; // [rsp+128h] [rbp-1B8h]
  __int64 *v142; // [rsp+130h] [rbp-1B0h]
  unsigned __int64 v143; // [rsp+138h] [rbp-1A8h]
  unsigned __int8 v144; // [rsp+17Bh] [rbp-165h] BYREF
  unsigned int v145; // [rsp+17Ch] [rbp-164h] BYREF
  __int64 v146; // [rsp+180h] [rbp-160h] BYREF
  __int64 v147; // [rsp+188h] [rbp-158h]
  __int64 v148; // [rsp+190h] [rbp-150h] BYREF
  _QWORD *v149; // [rsp+198h] [rbp-148h]
  __int64 v150; // [rsp+1A0h] [rbp-140h] BYREF
  __int64 v151; // [rsp+1A8h] [rbp-138h]
  _QWORD *v152; // [rsp+1B0h] [rbp-130h]
  __int64 v153; // [rsp+1B8h] [rbp-128h]
  __int64 *v154; // [rsp+1C0h] [rbp-120h]
  int v155; // [rsp+1C8h] [rbp-118h]
  _BYTE *v156; // [rsp+1D0h] [rbp-110h] BYREF
  __int64 v157; // [rsp+1D8h] [rbp-108h]
  _BYTE v158[64]; // [rsp+1E0h] [rbp-100h] BYREF
  _BYTE *v159; // [rsp+220h] [rbp-C0h] BYREF
  __int64 v160; // [rsp+228h] [rbp-B8h]
  _BYTE v161[176]; // [rsp+230h] [rbp-B0h] BYREF

  v11 = *(_DWORD *)(a1 + 8);
  if ( !v11 )
    return 0;
  v15 = a1;
  v106 = a2[2];
  v17 = v158;
  v156 = v158;
  v157 = 0x400000000LL;
  if ( v11 > 4 )
  {
    sub_16CD150((__int64)&v156, v158, v11, 16, (int)a5, (int)a6);
    v17 = v156;
    v15 = a1;
  }
  LODWORD(v157) = v11;
  v18 = &v17[16 * v11];
  do
  {
    if ( v17 )
    {
      *(_QWORD *)v17 = 0;
      *((_DWORD *)v17 + 2) = 0;
    }
    v17 += 16;
  }
  while ( v18 != v17 );
  v159 = v161;
  v160 = 0x800000000LL;
  v19 = *(_DWORD *)(v15 + 8);
  if ( v19 )
  {
    v108 = *(unsigned int *)(v15 + 8);
    v20 = 0;
    v21 = v15;
    v113 = 0;
    v105 = a5;
    v131 = v10;
    while ( 1 )
    {
      v111 = 16 * v113;
      a7 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)v21 + 16 * v113));
      v115 = *(_DWORD *)(*(_QWORD *)(v21 + 136) + 4 * v113);
      v22 = *(_BYTE *)(*(_QWORD *)(v21 + 80) + v113);
      v139 = v22;
      if ( !*(_BYTE *)(v21 + 172) )
        goto LABEL_16;
      v23 = a2[6];
      v24 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v106 + 384LL);
      if ( v24 != sub_1F42DB0 )
        break;
      LOBYTE(v146) = *(_BYTE *)(*(_QWORD *)(v21 + 80) + v113);
      v147 = 0;
      if ( v22 )
      {
        v139 = *(_BYTE *)(v106 + v22 + 1155);
        goto LABEL_16;
      }
      if ( !sub_1F58D20((__int64)&v146) )
      {
        sub_1F40D10((__int64)&v150, v106, v23, v146, v147);
        v96 = v152;
        LOBYTE(v148) = v151;
        v149 = v152;
        if ( (_BYTE)v151 )
        {
          v98 = *(_BYTE *)(v106 + (unsigned __int8)v151 + 1155);
        }
        else if ( sub_1F58D20((__int64)&v148) )
        {
          LOBYTE(v150) = 0;
          v151 = 0;
          v144 = 0;
          sub_1F426C0(v106, v23, (unsigned int)v148, (__int64)v96, (__int64)&v150, &v145, &v144);
          v98 = v144;
        }
        else
        {
          sub_1F40D10((__int64)&v150, v106, v23, v148, (__int64)v149);
          v97 = v102;
          LOBYTE(v97) = v151;
          v102 = v97;
          v98 = sub_1D5E9F0(v106, v23, (unsigned int)v97, (__int64)v152);
        }
        goto LABEL_87;
      }
      LOBYTE(v150) = 0;
      v151 = 0;
      LOBYTE(v145) = 0;
      sub_1F426C0(v106, v23, (unsigned int)v146, 0, (__int64)&v150, (unsigned int *)&v148, &v145);
      v139 = v145;
LABEL_16:
      v25 = (unsigned int)v160;
      if ( v115 >= (unsigned __int64)(unsigned int)v160 )
      {
        if ( v115 > (unsigned __int64)(unsigned int)v160 )
        {
          if ( v115 > (unsigned __int64)HIDWORD(v160) )
          {
            sub_16CD150((__int64)&v159, v161, v115, 16, (int)a5, (int)a6);
            v25 = (unsigned int)v160;
          }
          v92 = &v159[16 * v25];
          for ( i = &v159[16 * v115]; i != v92; v92 += 16 )
          {
            if ( v92 )
            {
              *(_QWORD *)v92 = 0;
              *((_DWORD *)v92 + 2) = 0;
            }
          }
          LODWORD(v160) = v115;
        }
      }
      else
      {
        LODWORD(v160) = v115;
      }
      v141 = v20;
      v140 = 0;
      v116 = v115 + v20;
      if ( v115 )
      {
        v26 = v105;
        v27 = v131;
        while ( 1 )
        {
          v127 = *v26;
          v42 = *(_QWORD *)(v21 + 104);
          v132 = *((_DWORD *)v26 + 2);
          if ( a6 )
          {
            v28 = *a6;
            LOBYTE(v27) = v139;
            v118 = v26;
            v138 = *(_DWORD *)(v42 + 4LL * v141);
            v119 = v27;
            v124 = *((_DWORD *)a6 + 2);
            v29 = sub_1D25E70((__int64)a2, (unsigned int)v27, 0, 1, 0, v138, 111, 0);
            v121 = v30;
            v31 = (const void ***)v29;
            v150 = v127;
            LODWORD(v151) = v132;
            v152 = sub_1D2A660(a2, v138, v119, 0, v32, v138);
            v153 = v33;
            v155 = v124;
            v154 = v28;
            *((_QWORD *)&v99 + 1) = 2 - ((v28 == 0) - 1LL);
            *(_QWORD *)&v99 = &v150;
            v35 = sub_1D36D80(a2, 47, a4, v31, v121, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9, v34, v99);
            *((_DWORD *)a6 + 2) = 2;
            v37 = (__int64)v35;
            v38 = v36;
            *a6 = v35;
            v26 = v118;
          }
          else
          {
            v43 = *(_DWORD *)(v42 + 4LL * v141);
            v44 = v114;
            LOBYTE(v44) = v139;
            v120 = v27;
            v122 = v26;
            v114 = v44;
            v45 = sub_1D252B0((__int64)a2, (unsigned int)v44, 0, 1, 0);
            v125 = v46;
            v47 = (const void ***)v45;
            v150 = v127;
            LODWORD(v151) = v132;
            v152 = sub_1D2A660(a2, v43, v114, 0, v48, v49);
            v153 = v50;
            *((_QWORD *)&v100 + 1) = 2;
            *(_QWORD *)&v100 = &v150;
            v52 = sub_1D36D80(a2, 47, a4, v47, v125, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9, v51, v100);
            v26 = v122;
            v37 = (__int64)v52;
            v38 = v36;
            v27 = v120;
          }
          v143 = v36 | v143 & 0xFFFFFFFF00000000LL;
          v39 = &v159[v140];
          *v26 = v37;
          *((_DWORD *)v26 + 2) = 1;
          *(_QWORD *)v39 = v37;
          *((_DWORD *)v39 + 2) = v38;
          v40 = *(_DWORD *)(*(_QWORD *)(v21 + 104) + 4LL * v141);
          if ( v40 >= 0 )
            goto LABEL_23;
          v41 = (unsigned __int8)(v139 - 14) <= 0x5Fu
             || (unsigned __int8)(v139 - 14) > 0x47u && (unsigned __int8)(v139 - 2) > 5u;
          if ( v41 )
            goto LABEL_23;
          v53 = v40 & 0x7FFFFFFF;
          if ( (unsigned int)v53 >= *(_DWORD *)(a3 + 952) )
            goto LABEL_23;
          v133 = *(_QWORD *)(a3 + 944) + 40 * v53;
          if ( *(char *)(v133 + 3) >= 0 )
            goto LABEL_23;
          v54 = sub_2045180(v139);
          v57 = (_DWORD *)v133;
          v58 = v54;
          v59 = *(_DWORD *)(v133 + 16);
          if ( v59 > 0x40 )
          {
            v123 = v27;
            v126 = v26;
            LODWORD(v61) = sub_16A5810(v133 + 8);
            v27 = v123;
            v26 = v126;
            v57 = (_DWORD *)v133;
            if ( v58 == (_DWORD)v61 )
              goto LABEL_52;
          }
          else
          {
            v60 = 64 - v59;
            v61 = ~(*(_QWORD *)(v133 + 8) << (64 - (unsigned __int8)v59));
            if ( !v61 )
            {
              if ( v58 == 64 )
              {
LABEL_52:
                v86 = v103;
                LOBYTE(v86) = v139;
                v136 = v27;
                v142 = v26;
                v87 = sub_1D38BB0((__int64)a2, 0, a4, v86, 0, 0, a7, *(double *)a8.m128i_i64, a9, 0);
                v26 = v142;
                v89 = v88;
                v90 = v87;
                v91 = (__int64)v159;
                v27 = v136;
                *(_QWORD *)&v159[v140] = v90;
                *(_DWORD *)(v91 + v140 + 8) = v89;
                goto LABEL_23;
              }
              LODWORD(v61) = 64;
LABEL_66:
              v94 = v58 - v61;
              if ( v94 == 32 )
              {
                LOBYTE(v95) = 5;
              }
              else if ( v94 > 0x20 )
              {
                if ( v94 == 64 )
                {
                  LOBYTE(v95) = 6;
                }
                else
                {
                  if ( v94 != 128 )
                    goto LABEL_73;
                  LOBYTE(v95) = 7;
                }
              }
              else if ( v94 == 8 )
              {
                LOBYTE(v95) = 3;
              }
              else
              {
                LOBYTE(v95) = 4;
                if ( v94 != 16 )
                {
                  LOBYTE(v95) = 2;
                  if ( v94 != 1 )
                  {
LABEL_73:
                    v130 = v27;
                    v137 = v26;
                    v95 = sub_1F58CC0((_QWORD *)a2[6], v94);
                    v27 = v130;
                    v26 = v137;
                    v104 = v95;
                    goto LABEL_71;
                  }
                }
              }
              v65 = 0;
LABEL_71:
              v60 = v104;
              LOBYTE(v60) = v95;
              v104 = v60;
              v67 = v60;
              goto LABEL_39;
            }
            _BitScanReverse64(&v61, v61);
            LODWORD(v61) = v61 ^ 0x3F;
            if ( v58 == (_DWORD)v61 )
              goto LABEL_52;
          }
          if ( (_DWORD)v61 )
            goto LABEL_66;
          v62 = *v57 & 0x7FFFFFFF;
          if ( v62 > 1 )
          {
            v63 = v58 - v62 + 1;
            if ( v63 == 32 )
            {
              LOBYTE(v64) = 5;
            }
            else if ( v63 > 0x20 )
            {
              if ( v63 == 64 )
              {
                LOBYTE(v64) = 6;
              }
              else
              {
                if ( v63 != 128 )
                  goto LABEL_41;
                LOBYTE(v64) = 7;
              }
            }
            else if ( v63 == 8 )
            {
              LOBYTE(v64) = 3;
            }
            else
            {
              LOBYTE(v64) = 4;
              if ( v63 != 16 )
              {
                LOBYTE(v64) = 2;
                if ( v63 != 1 )
                {
LABEL_41:
                  v129 = v27;
                  v135 = v26;
                  v64 = sub_1F58CC0((_QWORD *)a2[6], v63);
                  v27 = v129;
                  v26 = v135;
                  v107 = v64;
LABEL_38:
                  v66 = v107;
                  LOBYTE(v66) = v64;
                  v107 = v66;
                  v67 = v66;
                  v41 = 1;
LABEL_39:
                  v128 = v27;
                  v134 = v26;
                  *(_QWORD *)&v68 = sub_1D2EF30(a2, v67, v65, v60, v55, v56);
                  v69 = v110;
                  LOBYTE(v69) = v139;
                  v143 = v38 | v143 & 0xFFFFFFFF00000000LL;
                  v70 = sub_1D332F0(
                          a2,
                          (unsigned int)!v41 + 3,
                          a4,
                          v69,
                          0,
                          0,
                          *(double *)a7.m128i_i64,
                          *(double *)a8.m128i_i64,
                          a9,
                          v37,
                          v143,
                          v68);
                  v26 = v134;
                  v72 = v71;
                  v73 = v70;
                  v74 = (__int64)v159;
                  v27 = v128;
                  *(_QWORD *)&v159[v140] = v73;
                  *(_DWORD *)(v74 + v140 + 8) = v72;
                  goto LABEL_23;
                }
              }
            }
            v65 = 0;
            goto LABEL_38;
          }
LABEL_23:
          ++v141;
          v140 += 16;
          if ( v141 == v116 )
          {
            v131 = v27;
            goto LABEL_43;
          }
        }
      }
      v116 = v20;
LABEL_43:
      v75 = *(_BYTE *)(v21 + 172);
      BYTE4(v150) = 0;
      BYTE4(v148) = v75;
      if ( v75 )
        LODWORD(v148) = *(_DWORD *)(v21 + 168);
      v76 = sub_204AFD0(
              a2,
              a4,
              (__int64)v159,
              v115,
              v139,
              a10,
              a7,
              a8,
              a9,
              *(_OWORD *)&a7,
              (__int64)&v148,
              (unsigned int *)&v150);
      v20 = v116;
      v78 = v77;
      v79 = v76;
      v80 = (unsigned __int64)v156;
      ++v113;
      *(_QWORD *)&v156[v111] = v79;
      *(_DWORD *)(v80 + v111 + 8) = v78;
      LODWORD(v160) = 0;
      if ( v108 == v113 )
      {
        v19 = *(_DWORD *)(v21 + 8);
        v15 = v21;
        goto LABEL_47;
      }
    }
    v98 = v24(v106, a2[6], *(unsigned int *)(v21 + 168), v22, 0);
LABEL_87:
    v139 = v98;
    goto LABEL_16;
  }
LABEL_47:
  v81 = (unsigned __int64)v156;
  v82 = (unsigned int)v157;
  v83 = (const void ***)sub_1D25C30((__int64)a2, *(unsigned __int8 **)v15, v19);
  *((_QWORD *)&v101 + 1) = v82;
  *(_QWORD *)&v101 = v81;
  v12 = sub_1D36D80(a2, 51, a4, v83, v84, *(double *)a7.m128i_i64, *(double *)a8.m128i_i64, a9, v85, v101);
  if ( v159 != v161 )
    _libc_free((unsigned __int64)v159);
  if ( v156 != v158 )
    _libc_free((unsigned __int64)v156);
  return v12;
}
