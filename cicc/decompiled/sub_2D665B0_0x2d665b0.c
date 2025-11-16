// Function: sub_2D665B0
// Address: 0x2d665b0
//
__int64 __fastcall sub_2D665B0(__int64 *a1, __int64 a2, int a3, unsigned int a4, _BYTE *a5)
{
  __int64 v5; // r15
  __int64 v6; // r13
  unsigned int v9; // r14d
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 (__fastcall *v13)(int, int, int, int, int, int, __int64); // r14
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // rdi
  unsigned __int8 v17; // bl
  unsigned __int8 *v18; // rcx
  const __m128i *v19; // rax
  unsigned int v20; // r12d
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  unsigned int v24; // ebx
  bool v25; // al
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned int v29; // esi
  unsigned __int64 v30; // rdi
  __int64 v31; // rdx
  __int64 v32; // rax
  int v33; // eax
  __int16 v34; // bx
  __int64 v35; // rax
  unsigned __int8 **v36; // rax
  unsigned int v37; // edx
  unsigned __int8 *v38; // rsi
  const __m128i *v39; // r12
  __int64 v40; // rdx
  __int64 v41; // r14
  unsigned __int64 v42; // rbx
  __int64 v43; // rax
  _BYTE **v44; // rax
  bool v45; // zf
  __int64 v46; // rax
  __int64 v47; // rax
  unsigned int v48; // r12d
  __int64 v49; // rax
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // rax
  __m128i *v53; // rax
  __int64 v54; // rdi
  unsigned __int64 v55; // rax
  _QWORD *v56; // rax
  __int64 i; // rdx
  __int64 v58; // rax
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // rax
  __m128i *v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rsi
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // rdx
  __int64 v70; // rdi
  __int64 (*v71)(); // rax
  __int64 v72; // rbx
  unsigned __int64 v73; // rdx
  unsigned __int64 v74; // r8
  int v75; // r12d
  signed __int64 v76; // r13
  unsigned __int64 v77; // r14
  __int64 v78; // r11
  __int64 v79; // rbx
  unsigned __int64 v80; // r15
  __int64 v81; // rbx
  __int64 v82; // rax
  bool v83; // cc
  _QWORD *v84; // rax
  __int64 v85; // rax
  __int64 v86; // rdx
  unsigned __int64 v87; // rdx
  unsigned __int64 v88; // rsi
  unsigned __int64 v89; // rcx
  __int64 v90; // rdx
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // rcx
  int v94; // eax
  __int64 v95; // rcx
  unsigned __int64 v96; // rdx
  __int64 v97; // rax
  __int64 v98; // rax
  __int64 v99; // rax
  _QWORD *v100; // rax
  const __m128i *v101; // rax
  __int64 v102; // rdx
  unsigned __int64 v103; // rdi
  unsigned __int8 **v104; // rax
  __int64 v105; // rcx
  __int64 v106; // r8
  __int64 v107; // r9
  __m128i *v108; // rax
  __int64 v109; // rbx
  __int64 v110; // rax
  __int64 v111; // rcx
  __int64 v112; // r8
  __int64 v113; // r9
  __m128i *v114; // rax
  __int64 v115; // rdx
  __int64 v116; // rax
  __int64 v117; // r14
  __int64 v118; // rax
  __int64 v119; // rdx
  __int64 v120; // rax
  __int64 v121; // rdx
  __m128i *v122; // rax
  __int64 v123; // rdx
  __int64 v124; // rax
  __int64 v125; // rdi
  __int64 (*v126)(); // rax
  unsigned __int8 *v127; // rsi
  char v128; // cl
  char v129; // al
  unsigned __int8 **v130; // rax
  unsigned __int8 *v131; // rcx
  unsigned __int8 v132; // al
  __int64 v133; // rdi
  unsigned __int64 v134; // rax
  __int64 v135; // rdx
  _QWORD *v136; // rbx
  __int64 v137; // rax
  unsigned int v138; // [rsp+4h] [rbp-DCh]
  unsigned __int64 v139; // [rsp+8h] [rbp-D8h]
  __int64 v140; // [rsp+10h] [rbp-D0h]
  unsigned int v141; // [rsp+18h] [rbp-C8h]
  unsigned int v142; // [rsp+1Ch] [rbp-C4h]
  int v145; // [rsp+2Ch] [rbp-B4h]
  unsigned __int64 v146; // [rsp+30h] [rbp-B0h]
  __int64 v147; // [rsp+30h] [rbp-B0h]
  __int64 v148; // [rsp+30h] [rbp-B0h]
  __int64 v149; // [rsp+30h] [rbp-B0h]
  __int64 v150; // [rsp+30h] [rbp-B0h]
  __int64 v151; // [rsp+30h] [rbp-B0h]
  int v152; // [rsp+38h] [rbp-A8h]
  __int64 **v154; // [rsp+40h] [rbp-A0h]
  __int64 v155; // [rsp+40h] [rbp-A0h]
  unsigned int v156; // [rsp+40h] [rbp-A0h]
  __int64 v157; // [rsp+48h] [rbp-98h]
  __int64 v158; // [rsp+48h] [rbp-98h]
  unsigned __int8 **v159; // [rsp+48h] [rbp-98h]
  _QWORD v160[2]; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int128 v161; // [rsp+60h] [rbp-80h] BYREF
  __m128i v162; // [rsp+70h] [rbp-70h] BYREF
  __m128i v163; // [rsp+80h] [rbp-60h] BYREF
  __m128i v164; // [rsp+90h] [rbp-50h] BYREF
  __int64 v165; // [rsp+A0h] [rbp-40h]

  v5 = (__int64)a1;
  v6 = a2;
  if ( a5 )
    *a5 = 0;
  switch ( a3 )
  {
    case 13:
      v39 = (const __m128i *)a1[12];
      v40 = a1[15];
      v41 = 0;
      v161 = (unsigned __int128)_mm_loadu_si128(v39);
      v162 = _mm_loadu_si128(v39 + 1);
      v163 = _mm_loadu_si128(v39 + 2);
      v164 = _mm_loadu_si128(v39 + 3);
      v165 = v39[4].m128i_i64[0];
      v42 = *(unsigned int *)(*a1 + 8);
      v43 = *(unsigned int *)(v40 + 8);
      v152 = *(_DWORD *)(*a1 + 8);
      if ( (_DWORD)v43 )
        v41 = *(_QWORD *)(*(_QWORD *)v40 + 8 * v43 - 8);
      v44 = (_BYTE **)sub_986520(a2);
      v155 = 32;
      v158 = 0;
      if ( **v44 == 17 )
      {
        v45 = *v44[4] == 17;
        v46 = 32;
        if ( !v45 )
          v46 = 0;
        v155 = v46;
        v47 = 32;
        if ( v45 )
          v47 = 0;
        v158 = v47;
      }
      v39[4].m128i_i8[0] = 0;
      v48 = a4 + 1;
      v49 = sub_986520(a2);
      if ( (unsigned __int8)sub_2D65BF0((__int64)a1, *(unsigned __int8 **)(v49 + v158), a4 + 1) )
      {
        v52 = sub_986520(a2);
        if ( (unsigned __int8)sub_2D65BF0((__int64)a1, *(unsigned __int8 **)(v52 + v155), v48) )
          return 1;
      }
      v53 = (__m128i *)a1[12];
      *v53 = _mm_loadu_si128((const __m128i *)&v161);
      v53[1] = _mm_loadu_si128(&v162);
      v53[2] = _mm_loadu_si128(&v163);
      v53[3] = _mm_loadu_si128(&v164);
      v53[4].m128i_i8[0] = v165;
      v54 = *a1;
      v55 = *(unsigned int *)(*(_QWORD *)v5 + 8LL);
      if ( v42 != v55 )
      {
        if ( v42 >= v55 )
        {
          if ( v42 > *(unsigned int *)(v54 + 12) )
          {
            v151 = *(_QWORD *)v5;
            sub_C8D5F0(v54, (const void *)(v54 + 16), v42, 8u, v50, v51);
            v54 = v151;
            v55 = *(unsigned int *)(v151 + 8);
          }
          v56 = (_QWORD *)(*(_QWORD *)v54 + 8 * v55);
          for ( i = *(_QWORD *)v54 + 8 * v42; (_QWORD *)i != v56; ++v56 )
          {
            if ( v56 )
              *v56 = 0;
          }
        }
        *(_DWORD *)(v54 + 8) = v152;
      }
      sub_2D57BD0(*(__int64 **)(v5 + 120), v41);
      v58 = sub_986520(a2);
      if ( (unsigned __int8)sub_2D65BF0(v5, *(unsigned __int8 **)(v58 + v155), v48) )
      {
        v62 = sub_986520(a2);
        if ( (unsigned __int8)sub_2D65BF0(v5, *(unsigned __int8 **)(v62 + v158), v48) )
          return 1;
      }
      v63 = *(__m128i **)(v5 + 96);
      *v63 = _mm_loadu_si128((const __m128i *)&v161);
      v63[1] = _mm_loadu_si128(&v162);
      v63[2] = _mm_loadu_si128(&v163);
      v63[3] = _mm_loadu_si128(&v164);
      v64 = (unsigned __int8)v165;
      v63[4].m128i_i8[0] = v165;
      sub_2D65B70(*(_QWORD *)v5, v42, v64, v59, v60, v61);
      v65 = v41;
      v9 = 0;
      sub_2D57BD0(*(__int64 **)(v5 + 120), v65);
      return v9;
    case 17:
    case 25:
      *(_BYTE *)(a1[12] + 64) = 0;
      v27 = sub_986520(a2);
      v28 = *(_QWORD *)(v27 + 32);
      if ( *(_BYTE *)v28 != 17 )
        return 0;
      v29 = *(_DWORD *)(v28 + 32);
      if ( v29 > 0x40 )
        return 0;
      v30 = *(_QWORD *)(v28 + 24);
      if ( a3 == 25 )
      {
        v128 = v29 - 1;
        if ( v30 <= v29 - 1 )
          v128 = *(_QWORD *)(v28 + 24);
        v31 = 1LL << v128;
      }
      else
      {
        v31 = 0;
        if ( v29 )
          v31 = (__int64)(v30 << (64 - (unsigned __int8)v29)) >> (64 - (unsigned __int8)v29);
      }
      return sub_2D66030(v5, *(char **)v27, v31, a4);
    case 34:
      v72 = sub_986520(a2) + 32;
      v74 = sub_BB5290(a2) & 0xFFFFFFFFFFFFFFF9LL | 4;
      v145 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      if ( v145 == 1 )
      {
        v140 = 0;
        v101 = (const __m128i *)a1[12];
LABEL_142:
        v101->m128i_i64[1] += v140;
        v127 = *(unsigned __int8 **)sub_986520(v6);
        v9 = sub_2D65BF0(v5, v127, a4 + 1);
        if ( (_BYTE)v9 )
        {
          if ( (*(_BYTE *)(v6 + 1) & 2) == 0 )
          {
            *(_BYTE *)(*(_QWORD *)(v5 + 96) + 64LL) = 0;
            return v9;
          }
          return 1;
        }
        *(_QWORD *)(*(_QWORD *)(v5 + 96) + 8LL) -= v140;
        if ( byte_5016F48 )
        {
          if ( *(_BYTE *)v6 == 63 )
          {
            v129 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v5 + 8) + 1720LL))(*(_QWORD *)(v5 + 8));
            if ( v140 > 0 && a4 == 0 )
            {
              if ( v129 )
              {
                v130 = (unsigned __int8 **)sub_986520(v6);
                v131 = *v130;
                v132 = **v130;
                if ( v132 <= 0x1Cu )
                {
                  if ( v132 != 22 && v132 > 3u )
                    return 0;
                  v133 = *(_QWORD *)(sub_B43CB0(v6) + 80);
                  if ( v133 )
                    v133 -= 24;
                }
                else
                {
                  if ( (unsigned int)v132 - 67 <= 0xC || v132 == 63 )
                    return 0;
                  v133 = *((_QWORD *)v131 + 5);
                }
                v134 = (unsigned int)*(unsigned __int8 *)sub_986580(v133) - 39;
                if ( (unsigned int)v134 > 0x38 || (v135 = 0x100060000000001LL, !_bittest64(&v135, v134)) )
                {
                  v136 = *(_QWORD **)(v5 + 128);
                  sub_2D57220(v136, v6);
                  v136[3] = v140;
                }
                return v9;
              }
            }
          }
        }
        return 0;
      }
      v75 = 2;
      v76 = v74;
      v159 = (unsigned __int8 **)v72;
      v138 = 0;
      v142 = -1;
      v140 = 0;
      break;
    case 39:
    case 40:
      if ( *(_BYTE *)a2 <= 0x1Cu )
        return 0;
      v13 = sub_2D5AD50((char *)a2, a1[13], a1[1], a1[14], (__int64)a5);
      if ( !v13 )
        return 0;
      v157 = 0;
      v14 = a1[15];
      v15 = *(unsigned int *)(v14 + 8);
      if ( (_DWORD)v15 )
      {
        v14 = *(_QWORD *)v14;
        v157 = *(_QWORD *)(v14 + 8 * v15 - 8);
      }
      v16 = (__int64 *)a1[1];
      LODWORD(v160[0]) = 0;
      v17 = sub_2D5C100(v16, (unsigned __int8 *)a2, v14, v11, v12);
      v18 = (unsigned __int8 *)v13(
                                 a2,
                                 *(_QWORD *)(v5 + 120),
                                 *(_QWORD *)(v5 + 112),
                                 (int)v160,
                                 0,
                                 0,
                                 *(_QWORD *)(v5 + 8));
      if ( a5 )
        *a5 = 1;
      v19 = *(const __m128i **)(v5 + 96);
      v154 = (__int64 **)v18;
      v161 = (unsigned __int128)_mm_loadu_si128(v19);
      v162 = _mm_loadu_si128(v19 + 1);
      v163 = _mm_loadu_si128(v19 + 2);
      v164 = _mm_loadu_si128(v19 + 3);
      v165 = v19[4].m128i_i64[0];
      v20 = *(_DWORD *)(*(_QWORD *)v5 + 8LL);
      v9 = sub_2D65BF0(v5, v18, a4);
      if ( !(_BYTE)v9
        || (v24 = *(_DWORD *)(*(_QWORD *)v5 + 8LL) - v20 + (v17 ^ 1), v24 < LODWORD(v160[0]))
        || (v21 = (__int64)v154, v24 <= LODWORD(v160[0]))
        && (v25 = sub_2D5BD10(*(_QWORD *)(v5 + 8), *(_QWORD *)(v5 + 24), v154), v21 = (__int64)v154, !v25) )
      {
        v122 = *(__m128i **)(v5 + 96);
        *v122 = _mm_loadu_si128((const __m128i *)&v161);
        v122[1] = _mm_loadu_si128(&v162);
        v122[2] = _mm_loadu_si128(&v163);
        v122[3] = _mm_loadu_si128(&v164);
        v123 = (unsigned __int8)v165;
        v122[4].m128i_i8[0] = v165;
        sub_2D65B70(*(_QWORD *)v5, v20, v123, v21, v22, v23);
        sub_2D57BD0(*(__int64 **)(v5 + 120), v157);
        return 0;
      }
      v26 = *(_QWORD *)(v5 + 96);
      if ( a2 == *(_QWORD *)(v26 + 48) )
        *(_QWORD *)(v26 + 48) = v21;
      return v9;
    case 47:
      goto LABEL_31;
    case 48:
      v32 = *(_QWORD *)(a2 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v32 + 8) - 17 <= 1 )
        v32 = **(_QWORD **)(v32 + 16);
      v33 = sub_AE2980(a1[3], *(_DWORD *)(v32 + 8) >> 8)[1];
      if ( v33 == 1 )
      {
        v34 = 2;
      }
      else if ( v33 == 2 )
      {
        v34 = 3;
      }
      else if ( v33 == 4 )
      {
        v34 = 4;
      }
      else if ( v33 == 8 )
      {
        v34 = 5;
      }
      else if ( v33 == 16 )
      {
        v34 = 6;
      }
      else if ( v33 == 32 )
      {
        v34 = 7;
      }
      else
      {
        if ( v33 != 64 )
        {
          if ( v33 == 128 )
          {
            v137 = sub_986520(a2);
            if ( (unsigned __int16)sub_2D5BAE0(a1[1], a1[3], *(__int64 **)(*(_QWORD *)v137 + 8LL), 0) != 9 )
              return 0;
          }
          else
          {
            v120 = sub_986520(a2);
            if ( (unsigned __int16)sub_2D5BAE0(a1[1], a1[3], *(__int64 **)(*(_QWORD *)v120 + 8LL), 0) || v121 )
              return 0;
          }
LABEL_31:
          v36 = (unsigned __int8 **)sub_986520(a2);
          v37 = a4;
          v38 = *v36;
          return sub_2D65BF0(v5, v38, v37);
        }
        v34 = 8;
      }
      v35 = sub_986520(a2);
      if ( v34 != (unsigned __int16)sub_2D5BAE0(a1[1], a1[3], *(__int64 **)(*(_QWORD *)v35 + 8LL), 0) )
        return 0;
      goto LABEL_31;
    case 49:
      v66 = sub_986520(a2);
      v38 = *(unsigned __int8 **)v66;
      v67 = *(_QWORD *)(*(_QWORD *)v66 + 8LL);
      if ( (*(_BYTE *)(v67 + 8) & 0xFD) != 0xC || v67 == *(_QWORD *)(v6 + 8) )
        return 0;
      v37 = a4;
      return sub_2D65BF0(v5, v38, v37);
    case 50:
      v68 = *(_QWORD *)(*(_QWORD *)sub_986520(a2) + 8LL);
      if ( (unsigned int)*(unsigned __int8 *)(v68 + 8) - 17 <= 1 )
        v68 = **(_QWORD **)(v68 + 16);
      v69 = *(_QWORD *)(a2 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v69 + 8) - 17 <= 1 )
        v69 = **(_QWORD **)(v69 + 16);
      v70 = *(_QWORD *)(a1[1] + 8);
      v71 = *(__int64 (**)())(*(_QWORD *)v70 + 80LL);
      if ( v71 == sub_23CE2F0
        || !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD))v71)(
              v70,
              *(_DWORD *)(v68 + 8) >> 8,
              *(_DWORD *)(v69 + 8) >> 8) )
      {
        return 0;
      }
      goto LABEL_31;
    case 56:
      if ( *(_BYTE *)a2 != 85 )
        return 0;
      v124 = *(_QWORD *)(a2 - 32);
      if ( !v124 )
        return 0;
      if ( *(_BYTE *)v124 )
        return 0;
      if ( *(_QWORD *)(v124 + 24) != *(_QWORD *)(a2 + 80) )
        return 0;
      if ( (*(_BYTE *)(v124 + 33) & 0x20) == 0 )
        return 0;
      if ( (unsigned int)sub_987FE0(a2) != 353 )
        return 0;
      v125 = a1[1];
      v126 = *(__int64 (**)())(*(_QWORD *)v125 + 1296LL);
      if ( v126 == sub_2D56630
        || !((unsigned __int8 (__fastcall *)(__int64, _QWORD))v126)(
              v125,
              *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) )
      {
        return 0;
      }
      goto LABEL_31;
    default:
      return 0;
  }
  while ( 1 )
  {
    v77 = v76 & 0xFFFFFFFFFFFFFFF8LL;
    v156 = v75 - 1;
    v78 = a1[3];
    v79 = (v76 >> 1) & 3;
    v80 = v76 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v76 )
    {
      if ( !v79 )
      {
        if ( v77 )
        {
          v81 = sub_AE4AC0(a1[3], v76 & 0xFFFFFFFFFFFFFFF8LL);
          v82 = *(_QWORD *)(sub_986520(a2) + 32LL * v156);
          v83 = *(_DWORD *)(v82 + 32) <= 0x40u;
          v84 = *(_QWORD **)(v82 + 24);
          if ( !v83 )
            v84 = (_QWORD *)*v84;
          v85 = v81 + 16LL * (unsigned int)v84 + 24;
          v86 = *(_QWORD *)v85;
          LOBYTE(v85) = *(_BYTE *)(v85 + 8);
          *(_QWORD *)&v161 = v86;
          BYTE8(v161) = v85;
          v140 += sub_CA1930(&v161);
LABEL_76:
          v80 = sub_BCBAE0(v76 & 0xFFFFFFFFFFFFFFF8LL, *v159, v87);
          goto LABEL_77;
        }
        goto LABEL_104;
      }
      if ( v79 == 2 )
      {
        v88 = v76 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v77 )
        {
LABEL_83:
          v89 = sub_BDB740(v78, v88);
          v91 = v90;
          v160[0] = v89;
          v87 = v89;
          v160[1] = v91;
          goto LABEL_84;
        }
LABEL_104:
        v148 = a1[3];
        v97 = sub_BCBAE0(v76 & 0xFFFFFFFFFFFFFFF8LL, *v159, v73);
        v78 = v148;
        v88 = v97;
        goto LABEL_83;
      }
      if ( v79 != 1 )
        goto LABEL_104;
      if ( v77 )
      {
        v88 = *(_QWORD *)(v77 + 24);
      }
      else
      {
        v150 = a1[3];
        v99 = sub_BCBAE0(0, *v159, v73);
        v78 = v150;
        v88 = v99;
      }
    }
    else
    {
      v149 = a1[3];
      v98 = sub_BCBAE0(v77, *v159, v73);
      v78 = v149;
      v88 = v98;
      if ( v79 != 1 )
        goto LABEL_83;
    }
    v95 = sub_9208B0(v78, v88);
    LOBYTE(v91) = v96;
    v161 = __PAIR128__(v96, v95);
    v87 = (unsigned __int64)(v95 + 7) >> 3;
LABEL_84:
    v146 = v87;
    if ( v87 )
    {
      if ( (_BYTE)v91 )
        return 0;
      v92 = sub_986520(a2);
      v87 = v146;
      v93 = *(_QWORD *)(v92 + 32LL * v156);
      if ( *(_BYTE *)v93 == 17
        && (v139 = v146,
            v147 = *(_QWORD *)(v92 + 32LL * v156),
            v141 = *(_DWORD *)(v93 + 32),
            v94 = sub_969260(v93 + 24),
            v87 = v139,
            v141 + 1 - v94 <= 0x40) )
      {
        v100 = *(_QWORD **)(v147 + 24);
        if ( v141 > 0x40 )
        {
          v87 = *v100 * v139;
          v140 += v87;
        }
        else if ( v141 )
        {
          v87 = ((__int64)((_QWORD)v100 << (64 - (unsigned __int8)v141)) >> (64 - (unsigned __int8)v141)) * v139;
          v140 += v87;
        }
      }
      else
      {
        if ( v142 != -1 )
          return 0;
        v138 = v87;
        v142 = v75 - 1;
      }
    }
    if ( !v76 )
      goto LABEL_76;
    if ( v79 == 2 )
    {
      if ( !v77 )
        goto LABEL_76;
    }
    else
    {
      if ( v79 != 1 || !v77 )
        goto LABEL_76;
      v80 = *(_QWORD *)(v77 + 24);
    }
LABEL_77:
    v73 = *(unsigned __int8 *)(v80 + 8);
    if ( (_BYTE)v73 == 16 )
    {
      v76 = *(_QWORD *)(v80 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
    }
    else if ( (unsigned int)(unsigned __int8)v73 - 17 > 1 )
    {
      v45 = (_BYTE)v73 == 15;
      v73 = 0;
      if ( v45 )
        v73 = v80 & 0xFFFFFFFFFFFFFFF9LL;
      v76 = v73;
    }
    else
    {
      v76 = v80 & 0xFFFFFFFFFFFFFFF9LL | 2;
    }
    v159 += 4;
    if ( v145 == v75 )
      break;
    ++v75;
  }
  v5 = (__int64)a1;
  v6 = a2;
  v101 = (const __m128i *)a1[12];
  v102 = v101->m128i_i64[1] + v140;
  if ( v142 == -1 )
    goto LABEL_142;
  v161 = (unsigned __int128)_mm_loadu_si128(v101);
  v162 = _mm_loadu_si128(v101 + 1);
  v163 = _mm_loadu_si128(v101 + 2);
  v164 = _mm_loadu_si128(v101 + 3);
  v165 = v101[4].m128i_i64[0];
  v103 = *(unsigned int *)(*a1 + 8);
  v101->m128i_i64[1] = v102;
  if ( (*(_BYTE *)(a2 + 1) & 2) == 0 )
    *(_BYTE *)(a1[12] + 64) = 0;
  v104 = (unsigned __int8 **)sub_986520(a2);
  v9 = sub_2D65BF0((__int64)a1, *v104, a4 + 1);
  if ( (_BYTE)v9 )
  {
LABEL_117:
    v110 = sub_986520(a2);
    if ( (unsigned __int8)sub_2D66030((__int64)a1, *(char **)(v110 + 32LL * v142), v138, a4) )
      return 1;
    v114 = (__m128i *)a1[12];
    *v114 = _mm_loadu_si128((const __m128i *)&v161);
    v114[1] = _mm_loadu_si128(&v162);
    v114[2] = _mm_loadu_si128(&v163);
    v114[3] = _mm_loadu_si128(&v164);
    v115 = (unsigned __int8)v165;
    v114[4].m128i_i8[0] = v165;
    sub_2D65B70(*a1, v103, v115, v111, v112, v113);
    v116 = a1[12];
    if ( !*(_BYTE *)(v116 + 16) )
    {
      *(_BYTE *)(v116 + 16) = 1;
      v117 = a1[12];
      *(_QWORD *)(v117 + 40) = *(_QWORD *)sub_986520(a2);
      *(_QWORD *)(a1[12] + 8) += v140;
      v118 = sub_986520(a2);
      v9 = sub_2D66030((__int64)a1, *(char **)(v118 + 32LL * v142), v138, a4);
      if ( !(_BYTE)v9 )
      {
        v108 = (__m128i *)a1[12];
        *v108 = _mm_loadu_si128((const __m128i *)&v161);
        v108[1] = _mm_loadu_si128(&v162);
        v108[2] = _mm_loadu_si128(&v163);
        v108[3] = _mm_loadu_si128(&v164);
        goto LABEL_121;
      }
      return 1;
    }
    return 0;
  }
  v108 = (__m128i *)a1[12];
  if ( !v108[1].m128i_i8[0] )
  {
    v108[1].m128i_i8[0] = 1;
    v109 = a1[12];
    *(_QWORD *)(v109 + 40) = *(_QWORD *)sub_986520(a2);
    goto LABEL_117;
  }
  *v108 = _mm_loadu_si128((const __m128i *)&v161);
  v108[1] = _mm_loadu_si128(&v162);
  v108[2] = _mm_loadu_si128(&v163);
  v108[3] = _mm_loadu_si128(&v164);
LABEL_121:
  v119 = (unsigned __int8)v165;
  v108[4].m128i_i8[0] = v165;
  sub_2D65B70(*a1, v103, v119, v105, v106, v107);
  return v9;
}
