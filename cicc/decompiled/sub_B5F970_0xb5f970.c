// Function: sub_B5F970
// Address: 0xb5f970
//
char __fastcall sub_B5F970(__int64 a1, int **a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  __int64 v5; // r15
  int *v6; // rbx
  __int64 v7; // r14
  __int64 v8; // r12
  int v9; // ecx
  __int64 v10; // r9
  char v11; // di
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // r9
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // r13
  unsigned __int64 v17; // rsi
  int v18; // edx
  __int64 *v19; // rcx
  __int64 v20; // r9
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // r13
  unsigned __int64 v23; // rsi
  unsigned __int64 v24; // rdx
  const __m128i *v25; // r12
  unsigned __int64 v26; // rax
  __m128i *v27; // r13
  __int64 v28; // r9
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rsi
  unsigned __int64 v31; // rdx
  const __m128i *v32; // r12
  __int64 v33; // r9
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // rsi
  __int64 v36; // r9
  unsigned __int64 v37; // rax
  unsigned __int64 v38; // rsi
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // rdx
  unsigned __int64 v41; // rcx
  __int64 *v42; // rdx
  __int64 v43; // r9
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // rsi
  unsigned int v46; // r15d
  __int64 v47; // rbx
  int v48; // edx
  __int64 v49; // r9
  unsigned __int64 v50; // rax
  unsigned __int64 v51; // rsi
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // r9
  unsigned __int64 v55; // rax
  unsigned __int64 v56; // r13
  unsigned __int64 v57; // rsi
  unsigned __int64 v58; // rdx
  const __m128i *v59; // r12
  __m128i *v60; // r13
  __int64 v61; // rbx
  __int64 v62; // rax
  int *v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rdi
  int v66; // eax
  __int64 v67; // rsi
  int v68; // eax
  __int64 v69; // rdx
  int v70; // ecx
  char v71; // al
  int v72; // r9d
  unsigned __int8 v73; // dl
  bool v74; // al
  int v75; // esi
  int v76; // r13d
  int v77; // r14d
  __int64 v78; // rdi
  __int64 v79; // rax
  __int64 v80; // rdi
  _QWORD *v81; // rbx
  unsigned __int8 v82; // al
  __int64 v83; // rax
  int *v84; // rdx
  unsigned int v85; // eax
  __int64 v86; // rdx
  __int64 v87; // rcx
  unsigned __int8 v88; // di
  int v89; // r9d
  unsigned int v90; // edx
  _QWORD *v91; // rbx
  int v92; // edx
  __int64 v93; // rdi
  unsigned __int8 v94; // al
  __int64 v95; // rdi
  __int64 v96; // rdx
  int v97; // ecx
  _QWORD *v98; // rbx
  int v99; // eax
  __int64 v100; // rax
  __int64 v101; // rdx
  int v102; // ecx
  unsigned __int64 v103; // rcx
  __int64 *v104; // rdx
  __int64 v105; // rax
  int *v106; // rdx
  unsigned int v107; // eax
  unsigned __int64 v108; // r13
  const __m128i *v109; // rbx
  __m128i *v110; // r13
  __m128i *v111; // r13
  __int64 v112; // rsi
  char *v113; // r12
  unsigned __int64 v114; // r12
  __int64 v115; // rsi
  __int64 v116; // rsi
  __int64 v117; // rsi
  char *v118; // r12
  char *v119; // r12
  __int64 v120; // rsi
  char *v121; // r12
  __int64 v123; // [rsp+0h] [rbp-90h]
  unsigned __int8 v124; // [rsp+8h] [rbp-88h]
  unsigned int v125; // [rsp+8h] [rbp-88h]
  unsigned __int8 v126; // [rsp+8h] [rbp-88h]
  int v127; // [rsp+8h] [rbp-88h]
  unsigned __int8 v128; // [rsp+8h] [rbp-88h]
  unsigned __int8 v129; // [rsp+8h] [rbp-88h]
  unsigned __int8 v130; // [rsp+8h] [rbp-88h]
  unsigned __int8 v131; // [rsp+8h] [rbp-88h]
  __int64 v132; // [rsp+10h] [rbp-80h]
  __int64 v133; // [rsp+18h] [rbp-78h]
  __int64 v134; // [rsp+20h] [rbp-70h]
  __int64 v135; // [rsp+28h] [rbp-68h]
  __int64 v136; // [rsp+30h] [rbp-60h]
  __int64 v137; // [rsp+38h] [rbp-58h]
  __int64 v138; // [rsp+40h] [rbp-50h] BYREF
  int *v139; // [rsp+48h] [rbp-48h]
  __int64 v140; // [rsp+50h] [rbp-40h]

  while ( 2 )
  {
    v5 = (__int64)a2[1];
    if ( !v5 )
      goto LABEL_4;
    v6 = *a2;
    v7 = a4;
    v8 = a1;
    v9 = **a2;
    v10 = (unsigned int)(*a2)[1];
    v11 = *((_BYTE *)*a2 + 8);
    *a2 += 3;
    a2[1] = (int *)(v5 - 1);
    v12 = a3;
    switch ( v9 )
    {
      case 0:
        LOBYTE(v13) = *(_BYTE *)(v8 + 8) != 7;
        return v13;
      case 1:
        goto LABEL_4;
      case 2:
        if ( *(_BYTE *)(v8 + 8) != 17 || *(_DWORD *)(v8 + 32) != 1 )
          goto LABEL_4;
        LODWORD(v13) = sub_BCAC40(*(_QWORD *)(v8 + 24), 64) ^ 1;
        return v13;
      case 3:
        LOBYTE(v13) = *(_BYTE *)(v8 + 8) != 11;
        return v13;
      case 4:
        LOBYTE(v13) = *(_BYTE *)(v8 + 8) != 9;
        return v13;
      case 5:
        LOBYTE(v13) = *(_BYTE *)(v8 + 8) != 0;
        return v13;
      case 6:
        LOBYTE(v13) = *(_BYTE *)(v8 + 8) != 1;
        return v13;
      case 7:
        LOBYTE(v13) = *(_BYTE *)(v8 + 8) != 2;
        return v13;
      case 8:
        LOBYTE(v13) = *(_BYTE *)(v8 + 8) != 3;
        return v13;
      case 9:
        LOBYTE(v13) = *(_BYTE *)(v8 + 8) != 5;
        return v13;
      case 10:
        LODWORD(v13) = sub_BCAC40(v8, (unsigned int)v10) ^ 1;
        return v13;
      case 11:
        v48 = *(unsigned __int8 *)(v8 + 8);
        if ( (unsigned int)(v48 - 17) > 1 || (_DWORD)v10 != *(_DWORD *)(v8 + 32) || ((_BYTE)v48 == 18) != v11 )
          goto LABEL_4;
        a1 = *(_QWORD *)(v8 + 24);
        a4 = v7;
        a3 = v12;
        continue;
      case 12:
        if ( *(_BYTE *)(v8 + 8) != 14 )
          goto LABEL_4;
        LOBYTE(v13) = *(_DWORD *)(v8 + 8) >> 8 != (_DWORD)v10;
        return v13;
      case 13:
        if ( *(_BYTE *)(v8 + 8) != 15
          || (*(_DWORD *)(v8 + 8) & 0x400) == 0
          || ((*(_DWORD *)(v8 + 8) >> 8) & 2) != 0
          || (_DWORD)v10 != *(_DWORD *)(v8 + 12) )
        {
          goto LABEL_4;
        }
        if ( !(_DWORD)v10 )
          goto LABEL_63;
        v46 = a5;
        v47 = 0;
        v123 = 8 * v10;
        while ( !(unsigned __int8)sub_B5F970(*(_QWORD *)(*(_QWORD *)(v8 + 16) + v47), a2, v12, v7, v46) )
        {
          v47 += 8;
          if ( v123 == v47 )
          {
LABEL_63:
            LOBYTE(v13) = 0;
            return v13;
          }
        }
        goto LABEL_4;
      case 14:
        v52 = *(unsigned int *)(a3 + 8);
        v13 = (unsigned int)v10 >> 3;
        if ( (unsigned int)v52 > (unsigned int)v13 )
        {
          LOBYTE(v13) = *(_QWORD *)(*(_QWORD *)v12 + 8 * v13) != v8;
          return v13;
        }
        if ( (unsigned int)v52 >= (unsigned int)v13 )
        {
          v72 = v10 & 7;
          if ( v72 != 7 )
          {
            if ( (unsigned int)(v52 + 1) > (unsigned __int64)*(unsigned int *)(v12 + 12) )
            {
              v127 = v72;
              sub_C8D5F0(v12, v12 + 16, (unsigned int)(v52 + 1), 8);
              v52 = *(unsigned int *)(v12 + 8);
              v72 = v127;
            }
            *(_QWORD *)(*(_QWORD *)v12 + 8 * v52) = v8;
            ++*(_DWORD *)(v12 + 8);
            switch ( v72 )
            {
              case 0:
                goto LABEL_63;
              case 1:
                if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
                  v8 = **(_QWORD **)(v8 + 16);
                LOBYTE(v13) = *(_BYTE *)(v8 + 8) != 12;
                return v13;
              case 2:
                if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
                  v8 = **(_QWORD **)(v8 + 16);
                v73 = *(_BYTE *)(v8 + 8);
                v74 = 1;
                if ( v73 > 3u && v73 != 5 )
                  v74 = (v73 & 0xFD) == 4;
                LOBYTE(v13) = !v74;
                return v13;
              case 3:
                LOBYTE(v13) = (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 > 1;
                return v13;
              case 4:
                LOBYTE(v13) = *(_BYTE *)(v8 + 8) != 14;
                return v13;
              default:
                goto LABEL_219;
            }
          }
        }
        if ( a5 )
          goto LABEL_4;
        v26 = *(unsigned int *)(v7 + 8);
        v22 = *(_QWORD *)v7;
        v103 = *(unsigned int *)(v7 + 12);
        v104 = (__int64 *)(*(_QWORD *)v7 + 24 * v26);
        if ( v26 < v103 )
        {
          if ( v104 )
          {
            *v104 = v8;
            v104[1] = (__int64)v6;
            v104[2] = v5;
          }
          goto LABEL_18;
        }
        v138 = v8;
        v25 = (const __m128i *)&v138;
        v139 = v6;
        v140 = v5;
        if ( v103 < v26 + 1 )
        {
          v126 = a5;
          v112 = v7 + 16;
          if ( v22 > (unsigned __int64)&v138 || v104 <= &v138 )
          {
            sub_C8D5F0(v7, v112, v26 + 1, 24);
            v26 = *(unsigned int *)(v7 + 8);
            v22 = *(_QWORD *)v7;
            a5 = v126;
          }
          else
          {
            v113 = (char *)&v138 - v22;
            sub_C8D5F0(v7, v112, v26 + 1, 24);
            v22 = *(_QWORD *)v7;
            v26 = *(unsigned int *)(v7 + 8);
            a5 = v126;
            v25 = (const __m128i *)&v113[*(_QWORD *)v7];
          }
        }
        goto LABEL_17;
      case 15:
        v49 = (unsigned int)v10 >> 3;
        if ( *(_DWORD *)(a3 + 8) > (unsigned int)v49 )
        {
          v96 = *(_QWORD *)(*(_QWORD *)a3 + 8 * v49);
          v97 = *(unsigned __int8 *)(v96 + 8);
          if ( (unsigned int)(v97 - 17) <= 1 )
          {
            v98 = *(_QWORD **)(v96 + 24);
            BYTE4(v132) = (_BYTE)v97 == 18;
            LODWORD(v132) = *(_DWORD *)(v96 + 32);
            v99 = sub_BCB060(v98);
            v100 = sub_BCD140(*v98, (unsigned int)(2 * v99));
            v13 = sub_BCE1B0(v100, v132);
            goto LABEL_148;
          }
          if ( (_BYTE)v97 == 12 )
          {
            v13 = sub_BCCE00(*(_QWORD *)v96, (unsigned int)(2 * (*(_DWORD *)(v96 + 8) >> 8)));
            goto LABEL_148;
          }
          goto LABEL_4;
        }
        if ( a5 )
          goto LABEL_4;
        v50 = *(unsigned int *)(v7 + 8);
        v16 = *(_QWORD *)v7;
        v51 = *(unsigned int *)(v7 + 12);
        v18 = *(_DWORD *)(v7 + 8);
        v19 = (__int64 *)(*(_QWORD *)v7 + 24 * v50);
        if ( v50 < v51 )
          goto LABEL_9;
        v31 = v50 + 1;
        v138 = v8;
        v32 = (const __m128i *)&v138;
        v139 = v6;
        v140 = v5;
        if ( v51 >= v50 + 1 )
          goto LABEL_178;
        v129 = a5;
        v116 = v7 + 16;
        if ( v16 <= (unsigned __int64)&v138 && v19 > &v138 )
          goto LABEL_204;
        goto LABEL_193;
      case 16:
        v36 = (unsigned int)v10 >> 3;
        if ( *(_DWORD *)(a3 + 8) <= (unsigned int)v36 )
        {
          if ( !a5 )
          {
            v37 = *(unsigned int *)(v7 + 8);
            v16 = *(_QWORD *)v7;
            v38 = *(unsigned int *)(v7 + 12);
            v18 = *(_DWORD *)(v7 + 8);
            v19 = (__int64 *)(*(_QWORD *)v7 + 24 * v37);
            if ( v37 >= v38 )
            {
              v31 = v37 + 1;
              v138 = v8;
              v32 = (const __m128i *)&v138;
              v139 = v6;
              v140 = v5;
              if ( v38 >= v37 + 1 )
                goto LABEL_178;
              v129 = a5;
              v116 = v7 + 16;
              if ( v16 <= (unsigned __int64)&v138 && v19 > &v138 )
                goto LABEL_204;
              goto LABEL_193;
            }
            goto LABEL_9;
          }
LABEL_4:
          LOBYTE(v13) = 1;
          return v13;
        }
        v91 = *(_QWORD **)(*(_QWORD *)a3 + 8 * v36);
        v92 = *((unsigned __int8 *)v91 + 8);
        if ( (unsigned int)(v92 - 17) > 1 )
        {
          if ( (_BYTE)v92 != 12 )
            goto LABEL_4;
          v13 = sub_BCCE00(*v91, *((_DWORD *)v91 + 2) >> 9);
LABEL_148:
          LOBYTE(v13) = v13 != v8;
          return v13;
        }
        v93 = v91[3];
        v94 = *(_BYTE *)(v93 + 8);
        if ( v94 > 3u )
        {
          if ( v94 != 5 && (v94 & 0xFD) != 4 )
          {
            v105 = sub_BCAE30(v93);
            v139 = v106;
            v138 = v105;
            v107 = sub_CA1930(&v138);
            v95 = sub_BCCE00(*v91, v107 >> 1);
            goto LABEL_147;
          }
        }
        else
        {
          if ( v94 == 2 )
          {
            v95 = sub_BCB140(*v91);
LABEL_147:
            BYTE4(v133) = *((_BYTE *)v91 + 8) == 18;
            LODWORD(v133) = *((_DWORD *)v91 + 8);
            v13 = sub_BCE1B0(v95, v133);
            goto LABEL_148;
          }
          if ( v94 == 3 )
          {
            v95 = sub_BCB160(*v91);
            goto LABEL_147;
          }
        }
LABEL_219:
        BUG();
      case 17:
        v33 = (unsigned int)v10 >> 3;
        if ( *(_DWORD *)(a3 + 8) > (unsigned int)v33 )
        {
          v101 = *(_QWORD *)(*(_QWORD *)a3 + 8 * v33);
          v102 = *(unsigned __int8 *)(v101 + 8);
          if ( (unsigned int)(v102 - 17) > 1 )
            goto LABEL_4;
          BYTE4(v134) = (_BYTE)v102 == 18;
          LODWORD(v134) = *(_DWORD *)(v101 + 32) >> 1;
          LOBYTE(v13) = v8 != sub_BCE1B0(*(_QWORD *)(v101 + 24), v134);
          return v13;
        }
        if ( a5 )
          goto LABEL_4;
        v34 = *(unsigned int *)(v7 + 8);
        v16 = *(_QWORD *)v7;
        v35 = *(unsigned int *)(v7 + 12);
        v18 = *(_DWORD *)(v7 + 8);
        v19 = (__int64 *)(*(_QWORD *)v7 + 24 * v34);
        if ( v34 < v35 )
          goto LABEL_9;
        v31 = v34 + 1;
        v138 = v8;
        v32 = (const __m128i *)&v138;
        v139 = v6;
        v140 = v5;
        if ( v35 >= v34 + 1 )
          goto LABEL_178;
        v129 = a5;
        v116 = v7 + 16;
        if ( v16 <= (unsigned __int64)&v138 && v19 > &v138 )
          goto LABEL_204;
        goto LABEL_193;
      case 18:
      case 19:
      case 20:
        v14 = (unsigned int)v10 >> 3;
        if ( *(_DWORD *)(a3 + 8) > (unsigned int)v14 )
        {
          v67 = *(_QWORD *)(*(_QWORD *)a3 + 8 * v14);
          v68 = *(unsigned __int8 *)(v67 + 8);
          if ( v68 == 17 )
          {
            BYTE4(v135) = 0;
            LODWORD(v135) = *(_DWORD *)(v67 + 32) / (unsigned int)(2 * v9 - 33);
          }
          else
          {
            if ( v68 != 18 )
              goto LABEL_4;
            BYTE4(v135) = 1;
            LODWORD(v135) = *(_DWORD *)(v67 + 32) / (unsigned int)(2 * v9 - 33);
          }
          LOBYTE(v13) = v8 != sub_BCE1B0(**(_QWORD **)(v67 + 16), v135);
          return v13;
        }
        if ( a5 )
          goto LABEL_4;
        v15 = *(unsigned int *)(v7 + 8);
        v16 = *(_QWORD *)v7;
        v17 = *(unsigned int *)(v7 + 12);
        v18 = *(_DWORD *)(v7 + 8);
        v19 = (__int64 *)(*(_QWORD *)v7 + 24 * v15);
        if ( v15 < v17 )
          goto LABEL_9;
        v31 = v15 + 1;
        v138 = v8;
        v32 = (const __m128i *)&v138;
        v139 = v6;
        v140 = v5;
        if ( v17 >= v15 + 1 )
          goto LABEL_178;
        v129 = a5;
        v116 = v7 + 16;
        if ( v16 <= (unsigned __int64)&v138 && v19 > &v138 )
          goto LABEL_204;
        goto LABEL_193;
      case 21:
        v43 = (unsigned int)v10 >> 3;
        if ( *(_DWORD *)(a3 + 8) > (unsigned int)v43 )
        {
          v87 = *(_QWORD *)(*(_QWORD *)a3 + 8 * v43);
          v88 = *(_BYTE *)(v87 + 8);
          v89 = *(unsigned __int8 *)(v8 + 8);
          v90 = v89 - 17;
          if ( (unsigned int)v88 - 17 > 1 )
          {
            if ( v90 <= 1 )
              goto LABEL_4;
          }
          else
          {
            if ( v90 > 1 || *(_DWORD *)(v87 + 32) != *(_DWORD *)(v8 + 32) || (v88 == 18) != ((_BYTE)v89 == 18) )
              goto LABEL_4;
            v8 = *(_QWORD *)(v8 + 24);
          }
          a4 = v7;
          a3 = v12;
          a1 = v8;
          continue;
        }
        *a2 = v6 + 6;
        a2[1] = (int *)(v5 - 2);
        if ( a5 )
          goto LABEL_4;
        v44 = *(unsigned int *)(v7 + 8);
        v16 = *(_QWORD *)v7;
        v45 = *(unsigned int *)(v7 + 12);
        v18 = *(_DWORD *)(v7 + 8);
        v19 = (__int64 *)(*(_QWORD *)v7 + 24 * v44);
        if ( v44 >= v45 )
        {
          v31 = v44 + 1;
          v138 = v8;
          v32 = (const __m128i *)&v138;
          v139 = v6;
          v140 = v5;
          if ( v45 >= v44 + 1 )
          {
LABEL_178:
            v111 = (__m128i *)(24LL * *(unsigned int *)(v7 + 8) + v16);
            *v111 = _mm_loadu_si128(v32);
            v111[1].m128i_i64[0] = v32[1].m128i_i64[0];
            ++*(_DWORD *)(v7 + 8);
            LOBYTE(v13) = a5;
            return v13;
          }
          v129 = a5;
          v116 = v7 + 16;
          if ( v16 <= (unsigned __int64)&v138 && v19 > &v138 )
          {
LABEL_204:
            v119 = (char *)&v138 - v16;
            sub_C8D5F0(v7, v116, v31, 24);
            v16 = *(_QWORD *)v7;
            a5 = v129;
            v32 = (const __m128i *)&v119[*(_QWORD *)v7];
            goto LABEL_178;
          }
LABEL_193:
          sub_C8D5F0(v7, v116, v31, 24);
          v16 = *(_QWORD *)v7;
          a5 = v129;
          goto LABEL_178;
        }
LABEL_9:
        if ( v19 )
        {
          *v19 = v8;
          v19[1] = (__int64)v6;
          v19[2] = v5;
          v18 = *(_DWORD *)(v7 + 8);
        }
        *(_DWORD *)(v7 + 8) = v18 + 1;
        LOBYTE(v13) = a5;
        return v13;
      case 22:
        v13 = *(unsigned int *)(a3 + 8);
        v10 = (unsigned __int16)v10;
        if ( (unsigned int)v13 <= (unsigned __int16)v10 )
        {
          if ( a5 )
            goto LABEL_4;
          v39 = (unsigned int)(v13 + 1);
          if ( v39 > *(unsigned int *)(v12 + 12) )
          {
            v124 = a5;
            sub_C8D5F0(v12, v12 + 16, v39, 8);
            v13 = *(unsigned int *)(v12 + 8);
            a5 = v124;
          }
          *(_QWORD *)(*(_QWORD *)v12 + 8 * v13) = v8;
          ++*(_DWORD *)(v12 + 8);
          v40 = *(unsigned int *)(v7 + 8);
          v41 = *(unsigned int *)(v7 + 12);
          LODWORD(v13) = *(_DWORD *)(v7 + 8);
          if ( v40 >= v41 )
          {
            v138 = v8;
            v139 = v6;
            v140 = v5;
            if ( v41 < v40 + 1 )
            {
              v114 = *(_QWORD *)v7;
              v109 = (const __m128i *)&v138;
              v115 = v7 + 16;
              v128 = a5;
              if ( *(_QWORD *)v7 > (unsigned __int64)&v138 || (unsigned __int64)&v138 >= v114 + 24 * v40 )
              {
                sub_C8D5F0(v7, v115, v40 + 1, 24);
                v108 = *(_QWORD *)v7;
                a5 = v128;
              }
              else
              {
                sub_C8D5F0(v7, v115, v40 + 1, 24);
                v108 = *(_QWORD *)v7;
                a5 = v128;
                v109 = (const __m128i *)((char *)&v138 + *(_QWORD *)v7 - v114);
              }
            }
            else
            {
              v108 = *(_QWORD *)v7;
              v109 = (const __m128i *)&v138;
            }
            v110 = (__m128i *)(24LL * *(unsigned int *)(v7 + 8) + v108);
            *v110 = _mm_loadu_si128(v109);
            v110[1].m128i_i64[0] = v109[1].m128i_i64[0];
            ++*(_DWORD *)(v7 + 8);
            LOBYTE(v13) = a5;
          }
          else
          {
            v42 = (__int64 *)(*(_QWORD *)v7 + 24 * v40);
            if ( v42 )
            {
              *v42 = v8;
              v42[1] = (__int64)v6;
              v42[2] = v5;
              LODWORD(v13) = *(_DWORD *)(v7 + 8);
            }
            *(_DWORD *)(v7 + 8) = v13 + 1;
            LOBYTE(v13) = a5;
          }
          return v13;
        }
        if ( !a5 )
        {
          if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
          {
            v125 = (unsigned __int16)v10;
            sub_C8D5F0(a3, a3 + 16, v13 + 1, 8);
            v13 = *(unsigned int *)(v12 + 8);
            v10 = v125;
          }
          *(_QWORD *)(*(_QWORD *)v12 + 8 * v13) = v8;
          ++*(_DWORD *)(v12 + 8);
        }
        v69 = *(_QWORD *)(*(_QWORD *)v12 + 8 * v10);
        v70 = *(unsigned __int8 *)(v69 + 8);
        if ( (unsigned int)(v70 - 17) <= 1 )
        {
          v71 = *(_BYTE *)(v8 + 8);
          if ( (unsigned __int8)(v71 - 17) <= 1u
            && *(_DWORD *)(v69 + 32) == *(_DWORD *)(v8 + 32)
            && ((_BYTE)v70 == 18) == (v71 == 18) )
          {
            LOBYTE(v13) = *(_BYTE *)(*(_QWORD *)(v8 + 24) + 8LL) != 14;
            return v13;
          }
        }
        goto LABEL_4;
      case 23:
        v28 = (unsigned int)v10 >> 3;
        if ( *(_DWORD *)(a3 + 8) > (unsigned int)v28 )
        {
          v86 = *(_QWORD *)(*(_QWORD *)a3 + 8 * v28);
          if ( (unsigned int)*(unsigned __int8 *)(v86 + 8) - 17 > 1 )
            goto LABEL_4;
          LOBYTE(v13) = *(_QWORD *)(v86 + 24) != v8;
          return v13;
        }
        if ( a5 )
          goto LABEL_4;
        v29 = *(unsigned int *)(v7 + 8);
        v16 = *(_QWORD *)v7;
        v30 = *(unsigned int *)(v7 + 12);
        v18 = *(_DWORD *)(v7 + 8);
        v19 = (__int64 *)(*(_QWORD *)v7 + 24 * v29);
        if ( v29 < v30 )
          goto LABEL_9;
        v31 = v29 + 1;
        v138 = v8;
        v32 = (const __m128i *)&v138;
        v139 = v6;
        v140 = v5;
        if ( v30 >= v29 + 1 )
          goto LABEL_178;
        v129 = a5;
        v116 = v7 + 16;
        if ( v16 <= (unsigned __int64)&v138 && v19 > &v138 )
          goto LABEL_204;
        goto LABEL_193;
      case 24:
      case 25:
        v20 = (unsigned int)v10 >> 3;
        if ( *(_DWORD *)(a3 + 8) > (unsigned int)v20 )
        {
          v13 = *(_QWORD *)(*(_QWORD *)a3 + 8 * v20);
          v75 = *(unsigned __int8 *)(v13 + 8);
          if ( (unsigned int)(v75 - 17) > 1 )
            goto LABEL_4;
          v76 = 0;
          v77 = (v9 != 24) + 1;
          while ( 1 )
          {
            BYTE4(v137) = (_BYTE)v75 == 18;
            LODWORD(v137) = 2 * *(_DWORD *)(v13 + 32);
            v79 = sub_BCE1B0(*(_QWORD *)(v13 + 24), v137);
            v80 = *(_QWORD *)(v79 + 24);
            v81 = (_QWORD *)v79;
            v82 = *(_BYTE *)(v80 + 8);
            if ( v82 <= 3u )
            {
              if ( v82 == 2 )
              {
                v78 = sub_BCB140(*v81);
              }
              else
              {
                if ( v82 != 3 )
                  goto LABEL_219;
                v78 = sub_BCB160(*v81);
              }
            }
            else
            {
              if ( v82 == 5 || (v82 & 0xFD) == 4 )
                goto LABEL_219;
              v83 = sub_BCAE30(v80);
              v139 = v84;
              v138 = v83;
              v85 = sub_CA1930(&v138);
              v78 = sub_BCCE00(*v81, v85 >> 1);
            }
            BYTE4(v136) = *((_BYTE *)v81 + 8) == 18;
            ++v76;
            LODWORD(v136) = *((_DWORD *)v81 + 8);
            v13 = sub_BCE1B0(v78, v136);
            if ( v77 == v76 )
              break;
            LOBYTE(v75) = *(_BYTE *)(v13 + 8);
          }
LABEL_96:
          LOBYTE(v13) = v8 != v13;
          return v13;
        }
        if ( a5 )
          goto LABEL_4;
        v21 = *(unsigned int *)(v7 + 8);
        v22 = *(_QWORD *)v7;
        v23 = *(unsigned int *)(v7 + 12);
        v18 = *(_DWORD *)(v7 + 8);
        v19 = (__int64 *)(*(_QWORD *)v7 + 24 * v21);
        if ( v21 >= v23 )
        {
          v24 = v21 + 1;
          v138 = v8;
          v25 = (const __m128i *)&v138;
          v139 = v6;
          v140 = v5;
          if ( v23 < v21 + 1 )
          {
            v131 = a5;
            v120 = v7 + 16;
            if ( v22 > (unsigned __int64)&v138 || v19 <= &v138 )
            {
              sub_C8D5F0(v7, v120, v24, 24);
              v22 = *(_QWORD *)v7;
              a5 = v131;
            }
            else
            {
              v121 = (char *)&v138 - v22;
              sub_C8D5F0(v7, v120, v24, 24);
              v22 = *(_QWORD *)v7;
              a5 = v131;
              v25 = (const __m128i *)&v121[*(_QWORD *)v7];
            }
          }
          v26 = *(unsigned int *)(v7 + 8);
LABEL_17:
          v27 = (__m128i *)(24 * v26 + v22);
          *v27 = _mm_loadu_si128(v25);
          v27[1].m128i_i64[0] = v25[1].m128i_i64[0];
LABEL_18:
          ++*(_DWORD *)(v7 + 8);
LABEL_19:
          LOBYTE(v13) = a5;
          return v13;
        }
        goto LABEL_9;
      case 26:
        v54 = (unsigned int)v10 >> 3;
        if ( *(_DWORD *)(a3 + 8) > (unsigned int)v54 )
        {
          v61 = *(_QWORD *)(*(_QWORD *)a3 + 8 * v54);
          if ( (unsigned int)*(unsigned __int8 *)(v61 + 8) - 17 <= 1 && (unsigned __int8)(*(_BYTE *)(v8 + 8) - 17) <= 1u )
          {
            v62 = sub_BCAE30(*(_QWORD *)(v61 + 24));
            v139 = v63;
            v138 = v62;
            v64 = sub_CA1930(&v138);
            v65 = sub_BCCE00(*(_QWORD *)v61, v64);
            v66 = *(_DWORD *)(v61 + 32);
            BYTE4(v138) = *(_BYTE *)(v61 + 8) == 18;
            LODWORD(v138) = v66;
            v13 = sub_BCE1B0(v65, v138);
            goto LABEL_96;
          }
          goto LABEL_4;
        }
        if ( a5 )
          goto LABEL_4;
        v55 = *(unsigned int *)(v7 + 8);
        v56 = *(_QWORD *)v7;
        v57 = *(unsigned int *)(v7 + 12);
        v18 = *(_DWORD *)(v7 + 8);
        v19 = (__int64 *)(*(_QWORD *)v7 + 24 * v55);
        if ( v55 >= v57 )
        {
          v58 = v55 + 1;
          v138 = v8;
          v59 = (const __m128i *)&v138;
          v139 = v6;
          v140 = v5;
          if ( v57 < v55 + 1 )
          {
            v130 = a5;
            v117 = v7 + 16;
            if ( v56 > (unsigned __int64)&v138 || v19 <= &v138 )
            {
              sub_C8D5F0(v7, v117, v58, 24);
              v56 = *(_QWORD *)v7;
              a5 = v130;
            }
            else
            {
              v118 = (char *)&v138 - v56;
              sub_C8D5F0(v7, v117, v58, 24);
              v56 = *(_QWORD *)v7;
              a5 = v130;
              v59 = (const __m128i *)&v118[*(_QWORD *)v7];
            }
          }
          v60 = (__m128i *)(24LL * *(unsigned int *)(v7 + 8) + v56);
          *v60 = _mm_loadu_si128(v59);
          v60[1].m128i_i64[0] = v59[1].m128i_i64[0];
          goto LABEL_18;
        }
        goto LABEL_9;
      case 27:
        LOBYTE(v13) = *(_BYTE *)(v8 + 8) != 10;
        return v13;
      case 28:
        LOBYTE(v13) = *(_BYTE *)(v8 + 8) != 6;
        return v13;
      case 29:
        if ( *(_BYTE *)(v8 + 8) != 20 || *(_QWORD *)(v8 + 32) != 15 )
          goto LABEL_4;
        v53 = *(_QWORD *)(v8 + 24);
        if ( *(_QWORD *)v53 != 0x2E34366863726161LL
          || *(_DWORD *)(v53 + 8) != 1868789363
          || *(_WORD *)(v53 + 12) != 28277
          || (a5 = 0, *(_BYTE *)(v53 + 14) != 116) )
        {
          a5 = 1;
        }
        goto LABEL_19;
      default:
        goto LABEL_219;
    }
  }
}
