// Function: sub_E74590
// Address: 0xe74590
//
__int64 __fastcall sub_E74590(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, _QWORD *a6, __int64 a7)
{
  _QWORD *v7; // r14
  __int64 *v8; // r13
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 *v11; // r12
  __int64 result; // rax
  __int64 *v13; // r14
  __int64 *v14; // r15
  _QWORD *v15; // r12
  _QWORD *v16; // rbx
  _QWORD *v17; // r13
  _QWORD *v18; // rdi
  __int64 v19; // rdi
  int v20; // eax
  __int64 v21; // r13
  __int64 v22; // r14
  __int64 v23; // r12
  __int64 v24; // rbx
  char v25; // al
  _QWORD *v26; // rdx
  _QWORD *v27; // r15
  _QWORD *v28; // r13
  _QWORD *v29; // rdi
  __int64 v30; // rdi
  __int64 *v31; // r14
  __int64 v32; // r12
  _QWORD *v33; // r15
  _QWORD *v34; // r13
  _QWORD *v35; // rbx
  _QWORD *v36; // rdi
  __int64 v37; // rdi
  int v38; // ecx
  _QWORD *v39; // r15
  __int64 v40; // r14
  _QWORD *v41; // r13
  _QWORD *v42; // rbx
  _QWORD *v43; // r12
  _QWORD *v44; // rdi
  __int64 v45; // rdi
  int v46; // eax
  __int64 v47; // rax
  __int64 v48; // r14
  __int64 *v49; // r13
  __int64 v50; // rbx
  char v51; // al
  _QWORD *v52; // rdx
  _QWORD *v53; // r12
  _QWORD *v54; // r15
  _QWORD *v55; // rdi
  __int64 v56; // rdi
  int v57; // eax
  _QWORD *v58; // r15
  _QWORD *v59; // rdi
  __int64 v60; // rdi
  int v61; // eax
  _QWORD *v62; // r12
  char *v63; // r14
  _QWORD *v64; // r15
  _QWORD *v65; // r13
  _QWORD *v66; // rbx
  _QWORD *v67; // rdi
  __int64 v68; // rdi
  int v69; // eax
  __int64 v70; // rax
  __int64 *v71; // r14
  char *v72; // r13
  _QWORD *v73; // r15
  _QWORD *v74; // rbx
  _QWORD *v75; // r12
  _QWORD *v76; // rdi
  __int64 v77; // rdi
  int v78; // eax
  char *v79; // r13
  _QWORD *v80; // r14
  __int64 v81; // rax
  _QWORD *v82; // r12
  _QWORD *v83; // rbx
  _QWORD *v84; // r15
  _QWORD *v85; // rdi
  __int64 v86; // rdi
  bool v87; // zf
  __int64 v88; // r12
  __int64 *v89; // r12
  __int64 *v90; // r14
  __int64 *v91; // r15
  _QWORD *v92; // r13
  _QWORD *v93; // rbx
  _QWORD *v94; // r12
  _QWORD *v95; // rdi
  __int64 v96; // rdi
  int v97; // eax
  __int64 v98; // rax
  __int64 *v99; // r14
  char *v100; // r13
  __int64 v101; // rax
  _QWORD *v102; // r15
  _QWORD *v103; // rbx
  _QWORD *v104; // r12
  _QWORD *v105; // rdi
  __int64 v106; // rdi
  _QWORD *v107; // r12
  char *v108; // r14
  _QWORD *v109; // r13
  _QWORD *v110; // r15
  _QWORD *v111; // rbx
  _QWORD *v112; // r12
  _QWORD *v113; // rdi
  __int64 v114; // rdi
  int v115; // eax
  __int64 v116; // r12
  _QWORD *v117; // r13
  _QWORD *v118; // rdi
  __int64 v119; // rdi
  __int64 v120; // r15
  __int64 v121; // rcx
  _QWORD *v122; // r14
  _QWORD *v123; // r13
  _QWORD *v124; // rbx
  _QWORD *v125; // rdi
  __int64 v126; // rdi
  __int64 v127; // r14
  __int64 v128; // rcx
  _QWORD *v129; // r15
  _QWORD *v130; // r12
  _QWORD *v131; // rbx
  _QWORD *v132; // rdi
  __int64 v133; // rdi
  _QWORD *v134; // [rsp+0h] [rbp-B0h]
  _QWORD *v135; // [rsp+0h] [rbp-B0h]
  _QWORD *v136; // [rsp+8h] [rbp-A8h]
  signed __int64 v137; // [rsp+10h] [rbp-A0h]
  __int64 v138; // [rsp+10h] [rbp-A0h]
  _QWORD *v139; // [rsp+10h] [rbp-A0h]
  __int64 v140; // [rsp+18h] [rbp-98h]
  _QWORD *v141; // [rsp+18h] [rbp-98h]
  signed __int64 v142; // [rsp+18h] [rbp-98h]
  unsigned __int64 v143; // [rsp+20h] [rbp-90h]
  __int64 v144; // [rsp+20h] [rbp-90h]
  _QWORD *v145; // [rsp+20h] [rbp-90h]
  unsigned __int64 v146; // [rsp+20h] [rbp-90h]
  _QWORD *v147; // [rsp+20h] [rbp-90h]
  unsigned __int64 v148; // [rsp+28h] [rbp-88h]
  signed __int64 v149; // [rsp+28h] [rbp-88h]
  unsigned __int64 v150; // [rsp+28h] [rbp-88h]
  __int64 v151; // [rsp+28h] [rbp-88h]
  signed __int64 v152; // [rsp+28h] [rbp-88h]
  __int64 v153; // [rsp+30h] [rbp-80h]
  __int64 v154; // [rsp+38h] [rbp-78h]
  __int64 v155; // [rsp+40h] [rbp-70h]
  __int64 v156; // [rsp+40h] [rbp-70h]
  char *v157; // [rsp+48h] [rbp-68h]
  _QWORD *v158; // [rsp+48h] [rbp-68h]
  __int64 v159; // [rsp+48h] [rbp-68h]
  char *v160; // [rsp+50h] [rbp-60h]
  __int64 v161; // [rsp+50h] [rbp-60h]
  __int64 v162; // [rsp+58h] [rbp-58h]
  __int64 v163; // [rsp+58h] [rbp-58h]
  __int64 v164; // [rsp+58h] [rbp-58h]
  __int64 v165; // [rsp+58h] [rbp-58h]
  __int64 v166; // [rsp+58h] [rbp-58h]
  __int64 v167; // [rsp+58h] [rbp-58h]
  __int64 v168; // [rsp+58h] [rbp-58h]
  unsigned __int64 v169; // [rsp+60h] [rbp-50h]
  unsigned __int64 v170; // [rsp+60h] [rbp-50h]
  __int64 v171; // [rsp+60h] [rbp-50h]
  __int64 *v172; // [rsp+68h] [rbp-48h]
  __int64 v173; // [rsp+68h] [rbp-48h]
  __int64 v174; // [rsp+70h] [rbp-40h]
  __int64 v175; // [rsp+70h] [rbp-40h]
  unsigned __int64 v176; // [rsp+70h] [rbp-40h]
  unsigned __int64 v177; // [rsp+70h] [rbp-40h]
  unsigned __int64 v178; // [rsp+70h] [rbp-40h]
  __int64 v179; // [rsp+78h] [rbp-38h]
  __int64 v180; // [rsp+78h] [rbp-38h]
  _QWORD *v181; // [rsp+78h] [rbp-38h]
  __int64 v182; // [rsp+78h] [rbp-38h]
  __int64 v183; // [rsp+78h] [rbp-38h]
  _QWORD *v184; // [rsp+78h] [rbp-38h]
  __int64 v185; // [rsp+78h] [rbp-38h]
  __int64 v186; // [rsp+78h] [rbp-38h]

  while ( 1 )
  {
    v7 = a6;
    v8 = a2;
    v9 = a7;
    v172 = (__int64 *)a1;
    if ( a5 <= a7 )
      v9 = a5;
    v174 = a3;
    v179 = a5;
    if ( a4 <= v9 )
      break;
    if ( a5 <= a7 )
    {
      result = 0xAAAAAAAAAAAAAAABLL;
      v156 = a3;
      v161 = a3 - (_QWORD)a2;
      v169 = 0xAAAAAAAAAAAAAAABLL * ((a3 - (__int64)a2) >> 5);
      if ( a3 - (__int64)a2 <= 0 )
        return result;
      v158 = a6;
      v13 = a2;
      v14 = a6;
      do
      {
        v15 = (_QWORD *)v14[4];
        v16 = (_QWORD *)v14[5];
        *v14 = *v13;
        v17 = v15;
        v14[1] = v13[1];
        v14[2] = v13[2];
        v14[3] = v13[3];
        v180 = v14[6];
        v14[4] = v13[4];
        v14[5] = v13[5];
        v14[6] = v13[6];
        v13[4] = 0;
        v13[5] = 0;
        for ( v13[6] = 0; v16 != v17; v17 += 13 )
        {
          v18 = (_QWORD *)v17[9];
          if ( v18 != v17 + 11 )
            j_j___libc_free_0(v18, v17[11] + 1LL);
          v19 = v17[6];
          if ( v19 )
            j_j___libc_free_0(v19, v17[8] - v19);
        }
        if ( v15 )
          j_j___libc_free_0(v15, v180 - (_QWORD)v15);
        v20 = *((_DWORD *)v13 + 14);
        v14 += 12;
        v13 += 12;
        *((_DWORD *)v14 - 10) = v20;
        *((_DWORD *)v14 - 9) = *((_DWORD *)v13 - 9);
        *((_DWORD *)v14 - 8) = *((_DWORD *)v13 - 8);
        *(v14 - 3) = *(v13 - 3);
        *((_BYTE *)v14 - 16) = *((_BYTE *)v13 - 16);
        *((_BYTE *)v14 - 15) = *((_BYTE *)v13 - 15);
        *((_DWORD *)v14 - 3) = *((_DWORD *)v13 - 3);
        *((_BYTE *)v14 - 8) = *((_BYTE *)v13 - 8);
        *((_BYTE *)v14 - 7) = *((_BYTE *)v13 - 7);
        --v169;
      }
      while ( v169 );
      result = 96;
      if ( v161 > 0 )
        result = v161;
      v21 = (__int64)v158 + result;
      if ( v172 != a2 )
      {
        if ( v158 == (_QWORD *)v21 )
          return result;
        v22 = v21 - 96;
        v23 = v174 - 96;
        v24 = (__int64)(a2 - 12);
        while ( 1 )
        {
          v25 = sub_E72550(v22, v24);
          v26 = *(_QWORD **)(v23 + 32);
          v27 = *(_QWORD **)(v23 + 40);
          v181 = v26;
          v175 = *(_QWORD *)(v23 + 48);
          if ( v25 )
          {
            v28 = *(_QWORD **)(v23 + 32);
            *(_QWORD *)v23 = *(_QWORD *)v24;
            *(_QWORD *)(v23 + 8) = *(_QWORD *)(v24 + 8);
            *(_QWORD *)(v23 + 16) = *(_QWORD *)(v24 + 16);
            *(_QWORD *)(v23 + 24) = *(_QWORD *)(v24 + 24);
            *(_QWORD *)(v23 + 32) = *(_QWORD *)(v24 + 32);
            *(_QWORD *)(v23 + 40) = *(_QWORD *)(v24 + 40);
            *(_QWORD *)(v23 + 48) = *(_QWORD *)(v24 + 48);
            *(_QWORD *)(v24 + 32) = 0;
            *(_QWORD *)(v24 + 40) = 0;
            *(_QWORD *)(v24 + 48) = 0;
            if ( v27 != v26 )
            {
              do
              {
                v29 = (_QWORD *)v28[9];
                if ( v29 != v28 + 11 )
                  j_j___libc_free_0(v29, v28[11] + 1LL);
                v30 = v28[6];
                if ( v30 )
                  j_j___libc_free_0(v30, v28[8] - v30);
                v28 += 13;
              }
              while ( v27 != v28 );
            }
            if ( v181 )
              j_j___libc_free_0(v181, v175 - (_QWORD)v181);
            *(_DWORD *)(v23 + 56) = *(_DWORD *)(v24 + 56);
            *(_DWORD *)(v23 + 60) = *(_DWORD *)(v24 + 60);
            *(_DWORD *)(v23 + 64) = *(_DWORD *)(v24 + 64);
            *(_QWORD *)(v23 + 72) = *(_QWORD *)(v24 + 72);
            *(_BYTE *)(v23 + 80) = *(_BYTE *)(v24 + 80);
            *(_BYTE *)(v23 + 81) = *(_BYTE *)(v24 + 81);
            *(_DWORD *)(v23 + 84) = *(_DWORD *)(v24 + 84);
            *(_BYTE *)(v23 + 88) = *(_BYTE *)(v24 + 88);
            *(_BYTE *)(v23 + 89) = *(_BYTE *)(v24 + 89);
            if ( (__int64 *)v24 == v172 )
            {
              v120 = v22 + 96;
              result = v22 + 96 - (_QWORD)v158;
              v177 = 0xAAAAAAAAAAAAAAABLL * (result >> 5);
              if ( result > 0 )
              {
                do
                {
                  v121 = *(_QWORD *)(v120 - 96);
                  v122 = *(_QWORD **)(v23 - 64);
                  v120 -= 96;
                  v23 -= 96;
                  result = *(_QWORD *)(v23 + 48);
                  v123 = *(_QWORD **)(v23 + 40);
                  *(_QWORD *)v23 = v121;
                  v124 = v122;
                  v185 = result;
                  *(_QWORD *)(v23 + 8) = *(_QWORD *)(v120 + 8);
                  *(_QWORD *)(v23 + 16) = *(_QWORD *)(v120 + 16);
                  *(_QWORD *)(v23 + 24) = *(_QWORD *)(v120 + 24);
                  *(_QWORD *)(v23 + 32) = *(_QWORD *)(v120 + 32);
                  *(_QWORD *)(v23 + 40) = *(_QWORD *)(v120 + 40);
                  *(_QWORD *)(v23 + 48) = *(_QWORD *)(v120 + 48);
                  *(_QWORD *)(v120 + 32) = 0;
                  *(_QWORD *)(v120 + 40) = 0;
                  for ( *(_QWORD *)(v120 + 48) = 0; v123 != v124; v124 += 13 )
                  {
                    v125 = (_QWORD *)v124[9];
                    if ( v125 != v124 + 11 )
                      result = j_j___libc_free_0(v125, v124[11] + 1LL);
                    v126 = v124[6];
                    if ( v126 )
                      result = j_j___libc_free_0(v126, v124[8] - v126);
                  }
                  if ( v122 )
                    result = j_j___libc_free_0(v122, v185 - (_QWORD)v122);
                  v87 = v177-- == 1;
                  *(_DWORD *)(v23 + 56) = *(_DWORD *)(v120 + 56);
                  *(_DWORD *)(v23 + 60) = *(_DWORD *)(v120 + 60);
                  *(_DWORD *)(v23 + 64) = *(_DWORD *)(v120 + 64);
                  *(_QWORD *)(v23 + 72) = *(_QWORD *)(v120 + 72);
                  *(_BYTE *)(v23 + 80) = *(_BYTE *)(v120 + 80);
                  *(_BYTE *)(v23 + 81) = *(_BYTE *)(v120 + 81);
                  *(_DWORD *)(v23 + 84) = *(_DWORD *)(v120 + 84);
                  *(_BYTE *)(v23 + 88) = *(_BYTE *)(v120 + 88);
                  *(_BYTE *)(v23 + 89) = *(_BYTE *)(v120 + 89);
                }
                while ( !v87 );
              }
              return result;
            }
            v24 -= 96;
          }
          else
          {
            *(_QWORD *)v23 = *(_QWORD *)v22;
            *(_QWORD *)(v23 + 8) = *(_QWORD *)(v22 + 8);
            *(_QWORD *)(v23 + 16) = *(_QWORD *)(v22 + 16);
            *(_QWORD *)(v23 + 24) = *(_QWORD *)(v22 + 24);
            *(_QWORD *)(v23 + 32) = *(_QWORD *)(v22 + 32);
            *(_QWORD *)(v23 + 40) = *(_QWORD *)(v22 + 40);
            *(_QWORD *)(v23 + 48) = *(_QWORD *)(v22 + 48);
            *(_QWORD *)(v22 + 32) = 0;
            *(_QWORD *)(v22 + 40) = 0;
            v117 = v26;
            for ( *(_QWORD *)(v22 + 48) = 0; v27 != v117; v117 += 13 )
            {
              v118 = (_QWORD *)v117[9];
              if ( v118 != v117 + 11 )
                j_j___libc_free_0(v118, v117[11] + 1LL);
              v119 = v117[6];
              if ( v119 )
                j_j___libc_free_0(v119, v117[8] - v119);
            }
            if ( v181 )
              j_j___libc_free_0(v181, v175 - (_QWORD)v181);
            *(_DWORD *)(v23 + 56) = *(_DWORD *)(v22 + 56);
            *(_DWORD *)(v23 + 60) = *(_DWORD *)(v22 + 60);
            *(_DWORD *)(v23 + 64) = *(_DWORD *)(v22 + 64);
            *(_QWORD *)(v23 + 72) = *(_QWORD *)(v22 + 72);
            *(_BYTE *)(v23 + 80) = *(_BYTE *)(v22 + 80);
            *(_BYTE *)(v23 + 81) = *(_BYTE *)(v22 + 81);
            *(_DWORD *)(v23 + 84) = *(_DWORD *)(v22 + 84);
            *(_BYTE *)(v23 + 88) = *(_BYTE *)(v22 + 88);
            result = *(unsigned __int8 *)(v22 + 89);
            *(_BYTE *)(v23 + 89) = result;
            if ( v158 == (_QWORD *)v22 )
              return result;
            v22 -= 96;
          }
          v23 -= 96;
        }
      }
      v178 = 0xAAAAAAAAAAAAAAABLL * (result >> 5);
      v127 = v156;
      do
      {
        v128 = *(_QWORD *)(v21 - 96);
        v129 = *(_QWORD **)(v127 - 64);
        v21 -= 96;
        v127 -= 96;
        result = *(_QWORD *)(v127 + 48);
        v130 = *(_QWORD **)(v127 + 40);
        *(_QWORD *)v127 = v128;
        v131 = v129;
        v186 = result;
        *(_QWORD *)(v127 + 8) = *(_QWORD *)(v21 + 8);
        *(_QWORD *)(v127 + 16) = *(_QWORD *)(v21 + 16);
        *(_QWORD *)(v127 + 24) = *(_QWORD *)(v21 + 24);
        *(_QWORD *)(v127 + 32) = *(_QWORD *)(v21 + 32);
        *(_QWORD *)(v127 + 40) = *(_QWORD *)(v21 + 40);
        *(_QWORD *)(v127 + 48) = *(_QWORD *)(v21 + 48);
        *(_QWORD *)(v21 + 32) = 0;
        *(_QWORD *)(v21 + 40) = 0;
        for ( *(_QWORD *)(v21 + 48) = 0; v130 != v131; v131 += 13 )
        {
          v132 = (_QWORD *)v131[9];
          if ( v132 != v131 + 11 )
            result = j_j___libc_free_0(v132, v131[11] + 1LL);
          v133 = v131[6];
          if ( v133 )
            result = j_j___libc_free_0(v133, v131[8] - v133);
        }
        if ( v129 )
          result = j_j___libc_free_0(v129, v186 - (_QWORD)v129);
        v87 = v178-- == 1;
        *(_DWORD *)(v127 + 56) = *(_DWORD *)(v21 + 56);
        *(_DWORD *)(v127 + 60) = *(_DWORD *)(v21 + 60);
        *(_DWORD *)(v127 + 64) = *(_DWORD *)(v21 + 64);
        *(_QWORD *)(v127 + 72) = *(_QWORD *)(v21 + 72);
        *(_BYTE *)(v127 + 80) = *(_BYTE *)(v21 + 80);
        *(_BYTE *)(v127 + 81) = *(_BYTE *)(v21 + 81);
        *(_DWORD *)(v127 + 84) = *(_DWORD *)(v21 + 84);
        *(_BYTE *)(v127 + 88) = *(_BYTE *)(v21 + 88);
        *(_BYTE *)(v127 + 89) = *(_BYTE *)(v21 + 89);
      }
      while ( !v87 );
      return result;
    }
    v10 = a4;
    if ( a4 <= a5 )
    {
      v154 = a5 / 2;
      v157 = (char *)&a2[4 * (a5 / 2) + 4 * ((a5 + ((unsigned __int64)a5 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
      v160 = (char *)sub_E727B0(a1, (__int64)a2, (__int64)v157);
      v155 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v160[-a1] >> 5);
    }
    else
    {
      v155 = a4 / 2;
      v160 = (char *)(a1 + 32 * (a4 / 2 + ((a4 + ((unsigned __int64)a4 >> 63)) & 0xFFFFFFFFFFFFFFFELL)));
      v157 = (char *)sub_E72840((__int64)a2, a3, (__int64)v160);
      v154 = 0xAAAAAAAAAAAAAAABLL * ((v157 - (char *)a2) >> 5);
    }
    v153 = v10 - v155;
    if ( v10 - v155 > v154 && v154 <= a7 )
    {
      v11 = (__int64 *)v160;
      if ( !v154 )
        goto LABEL_10;
      v150 = 0xAAAAAAAAAAAAAAABLL * ((v157 - (char *)a2) >> 5);
      v138 = (char *)a2 - v160;
      v146 = 0xAAAAAAAAAAAAAAABLL * (((char *)a2 - v160) >> 5);
      if ( v157 - (char *)a2 <= 0 )
      {
        if ( v138 <= 0 )
          goto LABEL_10;
        v142 = 0;
        v151 = 0;
LABEL_144:
        v139 = v7;
        v99 = v8;
        v100 = v157;
        do
        {
          v101 = *(v99 - 12);
          v102 = (_QWORD *)*((_QWORD *)v100 - 8);
          v99 -= 12;
          v100 -= 96;
          v103 = (_QWORD *)*((_QWORD *)v100 + 5);
          *(_QWORD *)v100 = v101;
          v104 = v102;
          *((_QWORD *)v100 + 1) = v99[1];
          *((_QWORD *)v100 + 2) = v99[2];
          *((_QWORD *)v100 + 3) = v99[3];
          v167 = *((_QWORD *)v100 + 6);
          *((_QWORD *)v100 + 4) = v99[4];
          *((_QWORD *)v100 + 5) = v99[5];
          *((_QWORD *)v100 + 6) = v99[6];
          v99[4] = 0;
          v99[5] = 0;
          for ( v99[6] = 0; v103 != v104; v104 += 13 )
          {
            v105 = (_QWORD *)v104[9];
            if ( v105 != v104 + 11 )
              j_j___libc_free_0(v105, v104[11] + 1LL);
            v106 = v104[6];
            if ( v106 )
              j_j___libc_free_0(v106, v104[8] - v106);
          }
          if ( v102 )
            j_j___libc_free_0(v102, v167 - (_QWORD)v102);
          v87 = v146-- == 1;
          *((_DWORD *)v100 + 14) = *((_DWORD *)v99 + 14);
          *((_DWORD *)v100 + 15) = *((_DWORD *)v99 + 15);
          *((_DWORD *)v100 + 16) = *((_DWORD *)v99 + 16);
          *((_QWORD *)v100 + 9) = v99[9];
          v100[80] = *((_BYTE *)v99 + 80);
          v100[81] = *((_BYTE *)v99 + 81);
          *((_DWORD *)v100 + 21) = *((_DWORD *)v99 + 21);
          v100[88] = *((_BYTE *)v99 + 88);
          v100[89] = *((_BYTE *)v99 + 89);
        }
        while ( !v87 );
        v7 = v139;
      }
      else
      {
        v89 = v7;
        v135 = v7;
        v90 = a2;
        v91 = v89;
        do
        {
          v92 = (_QWORD *)v91[4];
          v93 = (_QWORD *)v91[5];
          *v91 = *v90;
          v94 = v92;
          v91[1] = v90[1];
          v91[2] = v90[2];
          v91[3] = v90[3];
          v166 = v91[6];
          v91[4] = v90[4];
          v91[5] = v90[5];
          v91[6] = v90[6];
          v90[4] = 0;
          v90[5] = 0;
          for ( v90[6] = 0; v93 != v94; v94 += 13 )
          {
            v95 = (_QWORD *)v94[9];
            if ( v95 != v94 + 11 )
              j_j___libc_free_0(v95, v94[11] + 1LL);
            v96 = v94[6];
            if ( v96 )
              j_j___libc_free_0(v96, v94[8] - v96);
          }
          if ( v92 )
            j_j___libc_free_0(v92, v166 - (_QWORD)v92);
          v97 = *((_DWORD *)v90 + 14);
          v91 += 12;
          v90 += 12;
          *((_DWORD *)v91 - 10) = v97;
          *((_DWORD *)v91 - 9) = *((_DWORD *)v90 - 9);
          *((_DWORD *)v91 - 8) = *((_DWORD *)v90 - 8);
          *(v91 - 3) = *(v90 - 3);
          *((_BYTE *)v91 - 16) = *((_BYTE *)v90 - 16);
          *((_BYTE *)v91 - 15) = *((_BYTE *)v90 - 15);
          *((_DWORD *)v91 - 3) = *((_DWORD *)v90 - 3);
          *((_BYTE *)v91 - 8) = *((_BYTE *)v90 - 8);
          *((_BYTE *)v91 - 7) = *((_BYTE *)v90 - 7);
          --v150;
        }
        while ( v150 );
        v98 = 96;
        v8 = a2;
        v7 = v135;
        if ( v157 - (char *)a2 > 0 )
          v98 = v157 - (char *)a2;
        v151 = v98;
        v142 = 0xAAAAAAAAAAAAAAABLL * (v98 >> 5);
        if ( v138 > 0 )
          goto LABEL_144;
      }
      if ( v151 <= 0 )
      {
        v11 = (__int64 *)v160;
      }
      else
      {
        v107 = v7;
        v147 = v7;
        v152 = v142;
        v108 = v160;
        v109 = v107;
        do
        {
          v110 = (_QWORD *)*((_QWORD *)v108 + 4);
          v111 = (_QWORD *)*((_QWORD *)v108 + 5);
          *(_QWORD *)v108 = *v109;
          v112 = v110;
          *((_QWORD *)v108 + 1) = v109[1];
          *((_QWORD *)v108 + 2) = v109[2];
          *((_QWORD *)v108 + 3) = v109[3];
          v168 = *((_QWORD *)v108 + 6);
          *((_QWORD *)v108 + 4) = v109[4];
          *((_QWORD *)v108 + 5) = v109[5];
          *((_QWORD *)v108 + 6) = v109[6];
          v109[4] = 0;
          v109[5] = 0;
          for ( v109[6] = 0; v111 != v112; v112 += 13 )
          {
            v113 = (_QWORD *)v112[9];
            if ( v113 != v112 + 11 )
              j_j___libc_free_0(v113, v112[11] + 1LL);
            v114 = v112[6];
            if ( v114 )
              j_j___libc_free_0(v114, v112[8] - v114);
          }
          if ( v110 )
            j_j___libc_free_0(v110, v168 - (_QWORD)v110);
          v115 = *((_DWORD *)v109 + 14);
          v108 += 96;
          v109 += 12;
          *((_DWORD *)v108 - 10) = v115;
          *((_DWORD *)v108 - 9) = *((_DWORD *)v109 - 9);
          *((_DWORD *)v108 - 8) = *((_DWORD *)v109 - 8);
          *((_QWORD *)v108 - 3) = *(v109 - 3);
          *(v108 - 16) = *((_BYTE *)v109 - 16);
          *(v108 - 15) = *((_BYTE *)v109 - 15);
          *((_DWORD *)v108 - 3) = *((_DWORD *)v109 - 3);
          *(v108 - 8) = *((_BYTE *)v109 - 8);
          *(v108 - 7) = *((_BYTE *)v109 - 7);
          --v152;
        }
        while ( v152 );
        v7 = v147;
        v116 = 96 * v142;
        if ( v142 <= 0 )
          v116 = 96;
        v11 = (__int64 *)&v160[v116];
      }
      goto LABEL_10;
    }
    if ( v153 > a7 )
    {
      v11 = sub_E73DF0(v160, a2, v157);
      goto LABEL_10;
    }
    v11 = (__int64 *)v157;
    if ( !v153 )
      goto LABEL_10;
    v143 = 0xAAAAAAAAAAAAAAABLL * (((char *)a2 - v160) >> 5);
    v140 = v157 - (char *)a2;
    v148 = 0xAAAAAAAAAAAAAAABLL * ((v157 - (char *)a2) >> 5);
    if ( (char *)a2 - v160 <= 0 )
    {
      if ( v140 <= 0 )
        goto LABEL_10;
      v136 = v7;
      v137 = 0;
      v144 = 0;
    }
    else
    {
      v62 = v7;
      v134 = v7;
      v63 = v160;
      do
      {
        v64 = (_QWORD *)v62[4];
        v65 = (_QWORD *)v62[5];
        *v62 = *(_QWORD *)v63;
        v66 = v64;
        v62[1] = *((_QWORD *)v63 + 1);
        v62[2] = *((_QWORD *)v63 + 2);
        v62[3] = *((_QWORD *)v63 + 3);
        v163 = v62[6];
        v62[4] = *((_QWORD *)v63 + 4);
        v62[5] = *((_QWORD *)v63 + 5);
        v62[6] = *((_QWORD *)v63 + 6);
        *((_QWORD *)v63 + 4) = 0;
        *((_QWORD *)v63 + 5) = 0;
        for ( *((_QWORD *)v63 + 6) = 0; v65 != v66; v66 += 13 )
        {
          v67 = (_QWORD *)v66[9];
          if ( v67 != v66 + 11 )
            j_j___libc_free_0(v67, v66[11] + 1LL);
          v68 = v66[6];
          if ( v68 )
            j_j___libc_free_0(v68, v66[8] - v68);
        }
        if ( v64 )
          j_j___libc_free_0(v64, v163 - (_QWORD)v64);
        v69 = *((_DWORD *)v63 + 14);
        v62 += 12;
        v63 += 96;
        *((_DWORD *)v62 - 10) = v69;
        *((_DWORD *)v62 - 9) = *((_DWORD *)v63 - 9);
        *((_DWORD *)v62 - 8) = *((_DWORD *)v63 - 8);
        *(v62 - 3) = *((_QWORD *)v63 - 3);
        *((_BYTE *)v62 - 16) = *(v63 - 16);
        *((_BYTE *)v62 - 15) = *(v63 - 15);
        *((_DWORD *)v62 - 3) = *((_DWORD *)v63 - 3);
        *((_BYTE *)v62 - 8) = *(v63 - 8);
        *((_BYTE *)v62 - 7) = *(v63 - 7);
        --v143;
      }
      while ( v143 );
      v70 = 96;
      v7 = v134;
      v8 = a2;
      if ( (char *)a2 - v160 > 0 )
        v70 = (char *)a2 - v160;
      v144 = v70;
      v136 = (_QWORD *)((char *)v134 + v70);
      v137 = 0xAAAAAAAAAAAAAAABLL * (v70 >> 5);
      if ( v140 <= 0 )
        goto LABEL_115;
    }
    v141 = v7;
    v71 = v8;
    v72 = v160;
    do
    {
      v73 = (_QWORD *)*((_QWORD *)v72 + 4);
      v74 = (_QWORD *)*((_QWORD *)v72 + 5);
      *(_QWORD *)v72 = *v71;
      v75 = v73;
      *((_QWORD *)v72 + 1) = v71[1];
      *((_QWORD *)v72 + 2) = v71[2];
      *((_QWORD *)v72 + 3) = v71[3];
      v164 = *((_QWORD *)v72 + 6);
      *((_QWORD *)v72 + 4) = v71[4];
      *((_QWORD *)v72 + 5) = v71[5];
      *((_QWORD *)v72 + 6) = v71[6];
      v71[4] = 0;
      v71[5] = 0;
      for ( v71[6] = 0; v74 != v75; v75 += 13 )
      {
        v76 = (_QWORD *)v75[9];
        if ( v76 != v75 + 11 )
          j_j___libc_free_0(v76, v75[11] + 1LL);
        v77 = v75[6];
        if ( v77 )
          j_j___libc_free_0(v77, v75[8] - v77);
      }
      if ( v73 )
        j_j___libc_free_0(v73, v164 - (_QWORD)v73);
      v78 = *((_DWORD *)v71 + 14);
      v72 += 96;
      v71 += 12;
      *((_DWORD *)v72 - 10) = v78;
      *((_DWORD *)v72 - 9) = *((_DWORD *)v71 - 9);
      *((_DWORD *)v72 - 8) = *((_DWORD *)v71 - 8);
      *((_QWORD *)v72 - 3) = *(v71 - 3);
      *(v72 - 16) = *((_BYTE *)v71 - 16);
      *(v72 - 15) = *((_BYTE *)v71 - 15);
      *((_DWORD *)v72 - 3) = *((_DWORD *)v71 - 3);
      *(v72 - 8) = *((_BYTE *)v71 - 8);
      *(v72 - 7) = *((_BYTE *)v71 - 7);
      --v148;
    }
    while ( v148 );
    v7 = v141;
LABEL_115:
    if ( v144 <= 0 )
    {
      v11 = (__int64 *)v157;
    }
    else
    {
      v79 = v157;
      v145 = v7;
      v80 = v136;
      v149 = v137;
      do
      {
        v81 = *(v80 - 12);
        v82 = (_QWORD *)*((_QWORD *)v79 - 8);
        v80 -= 12;
        v79 -= 96;
        v83 = (_QWORD *)*((_QWORD *)v79 + 5);
        *(_QWORD *)v79 = v81;
        v84 = v82;
        *((_QWORD *)v79 + 1) = v80[1];
        *((_QWORD *)v79 + 2) = v80[2];
        *((_QWORD *)v79 + 3) = v80[3];
        v165 = *((_QWORD *)v79 + 6);
        *((_QWORD *)v79 + 4) = v80[4];
        *((_QWORD *)v79 + 5) = v80[5];
        *((_QWORD *)v79 + 6) = v80[6];
        v80[4] = 0;
        v80[5] = 0;
        for ( v80[6] = 0; v83 != v84; v84 += 13 )
        {
          v85 = (_QWORD *)v84[9];
          if ( v85 != v84 + 11 )
            j_j___libc_free_0(v85, v84[11] + 1LL);
          v86 = v84[6];
          if ( v86 )
            j_j___libc_free_0(v86, v84[8] - v86);
        }
        if ( v82 )
          j_j___libc_free_0(v82, v165 - (_QWORD)v82);
        v87 = v149-- == 1;
        *((_DWORD *)v79 + 14) = *((_DWORD *)v80 + 14);
        *((_DWORD *)v79 + 15) = *((_DWORD *)v80 + 15);
        *((_DWORD *)v79 + 16) = *((_DWORD *)v80 + 16);
        *((_QWORD *)v79 + 9) = v80[9];
        v79[80] = *((_BYTE *)v80 + 80);
        v79[81] = *((_BYTE *)v80 + 81);
        *((_DWORD *)v79 + 21) = *((_DWORD *)v80 + 21);
        v79[88] = *((_BYTE *)v80 + 88);
        v79[89] = *((_BYTE *)v80 + 89);
      }
      while ( !v87 );
      v7 = v145;
      v88 = -96 * v137;
      if ( v137 <= 0 )
        v88 = -96;
      v11 = (__int64 *)&v157[v88];
    }
LABEL_10:
    sub_E74590((_DWORD)v172, (_DWORD)v160, (_DWORD)v11, v155, v154, (_DWORD)v7, a7);
    a4 = v153;
    a2 = (__int64 *)v157;
    a6 = v7;
    a1 = (__int64)v11;
    a3 = v174;
    a5 = v179 - v154;
  }
  result = 0xAAAAAAAAAAAAAAABLL;
  v39 = a6;
  v162 = (__int64)a2 - a1;
  v170 = 0xAAAAAAAAAAAAAAABLL * (((__int64)a2 - a1) >> 5);
  if ( (__int64)a2 - a1 > 0 )
  {
    v40 = a1;
    v159 = (__int64)a6;
    do
    {
      v41 = (_QWORD *)v39[4];
      v42 = (_QWORD *)v39[5];
      *v39 = *(_QWORD *)v40;
      v43 = v41;
      v39[1] = *(_QWORD *)(v40 + 8);
      v39[2] = *(_QWORD *)(v40 + 16);
      v39[3] = *(_QWORD *)(v40 + 24);
      v183 = v39[6];
      v39[4] = *(_QWORD *)(v40 + 32);
      v39[5] = *(_QWORD *)(v40 + 40);
      v39[6] = *(_QWORD *)(v40 + 48);
      *(_QWORD *)(v40 + 32) = 0;
      *(_QWORD *)(v40 + 40) = 0;
      for ( *(_QWORD *)(v40 + 48) = 0; v42 != v43; v43 += 13 )
      {
        v44 = (_QWORD *)v43[9];
        if ( v44 != v43 + 11 )
          j_j___libc_free_0(v44, v43[11] + 1LL);
        v45 = v43[6];
        if ( v45 )
          j_j___libc_free_0(v45, v43[8] - v45);
      }
      if ( v41 )
        j_j___libc_free_0(v41, v183 - (_QWORD)v41);
      v46 = *(_DWORD *)(v40 + 56);
      v39 += 12;
      v40 += 96;
      *((_DWORD *)v39 - 10) = v46;
      *((_DWORD *)v39 - 9) = *(_DWORD *)(v40 - 36);
      *((_DWORD *)v39 - 8) = *(_DWORD *)(v40 - 32);
      *(v39 - 3) = *(_QWORD *)(v40 - 24);
      *((_BYTE *)v39 - 16) = *(_BYTE *)(v40 - 16);
      *((_BYTE *)v39 - 15) = *(_BYTE *)(v40 - 15);
      *((_DWORD *)v39 - 3) = *(_DWORD *)(v40 - 12);
      *((_BYTE *)v39 - 8) = *(_BYTE *)(v40 - 8);
      *((_BYTE *)v39 - 7) = *(_BYTE *)(v40 - 7);
      --v170;
    }
    while ( v170 );
    v47 = 96;
    if ( v162 > 0 )
      v47 = v162;
    result = v159 + v47;
    v171 = result;
    if ( v159 != result )
    {
      v48 = (__int64)a2;
      v49 = v172;
      v50 = v159;
      while ( v48 != v174 )
      {
        v51 = sub_E72550(v48, v50);
        v52 = (_QWORD *)v49[4];
        v53 = (_QWORD *)v49[5];
        v184 = v52;
        v173 = v49[6];
        if ( v51 )
        {
          v54 = (_QWORD *)v49[4];
          *v49 = *(_QWORD *)v48;
          v49[1] = *(_QWORD *)(v48 + 8);
          v49[2] = *(_QWORD *)(v48 + 16);
          v49[3] = *(_QWORD *)(v48 + 24);
          v49[4] = *(_QWORD *)(v48 + 32);
          v49[5] = *(_QWORD *)(v48 + 40);
          v49[6] = *(_QWORD *)(v48 + 48);
          *(_QWORD *)(v48 + 32) = 0;
          *(_QWORD *)(v48 + 40) = 0;
          *(_QWORD *)(v48 + 48) = 0;
          if ( v53 != v52 )
          {
            do
            {
              v55 = (_QWORD *)v54[9];
              if ( v55 != v54 + 11 )
                j_j___libc_free_0(v55, v54[11] + 1LL);
              v56 = v54[6];
              if ( v56 )
                j_j___libc_free_0(v56, v54[8] - v56);
              v54 += 13;
            }
            while ( v54 != v53 );
          }
          if ( v184 )
            j_j___libc_free_0(v184, v173 - (_QWORD)v184);
          v57 = *(_DWORD *)(v48 + 56);
          v48 += 96;
          *((_DWORD *)v49 + 14) = v57;
          *((_DWORD *)v49 + 15) = *(_DWORD *)(v48 - 36);
          *((_DWORD *)v49 + 16) = *(_DWORD *)(v48 - 32);
          v49[9] = *(_QWORD *)(v48 - 24);
          *((_BYTE *)v49 + 80) = *(_BYTE *)(v48 - 16);
          *((_BYTE *)v49 + 81) = *(_BYTE *)(v48 - 15);
          *((_DWORD *)v49 + 21) = *(_DWORD *)(v48 - 12);
          *((_BYTE *)v49 + 88) = *(_BYTE *)(v48 - 8);
          result = *(unsigned __int8 *)(v48 - 7);
          *((_BYTE *)v49 + 89) = result;
        }
        else
        {
          *v49 = *(_QWORD *)v50;
          v49[1] = *(_QWORD *)(v50 + 8);
          v49[2] = *(_QWORD *)(v50 + 16);
          v49[3] = *(_QWORD *)(v50 + 24);
          v49[4] = *(_QWORD *)(v50 + 32);
          v49[5] = *(_QWORD *)(v50 + 40);
          v49[6] = *(_QWORD *)(v50 + 48);
          *(_QWORD *)(v50 + 32) = 0;
          *(_QWORD *)(v50 + 40) = 0;
          v58 = v52;
          for ( *(_QWORD *)(v50 + 48) = 0; v53 != v58; v58 += 13 )
          {
            v59 = (_QWORD *)v58[9];
            if ( v59 != v58 + 11 )
              j_j___libc_free_0(v59, v58[11] + 1LL);
            v60 = v58[6];
            if ( v60 )
              j_j___libc_free_0(v60, v58[8] - v60);
          }
          if ( v184 )
            j_j___libc_free_0(v184, v173 - (_QWORD)v184);
          v61 = *(_DWORD *)(v50 + 56);
          v50 += 96;
          *((_DWORD *)v49 + 14) = v61;
          *((_DWORD *)v49 + 15) = *(_DWORD *)(v50 - 36);
          *((_DWORD *)v49 + 16) = *(_DWORD *)(v50 - 32);
          v49[9] = *(_QWORD *)(v50 - 24);
          *((_BYTE *)v49 + 80) = *(_BYTE *)(v50 - 16);
          *((_BYTE *)v49 + 81) = *(_BYTE *)(v50 - 15);
          *((_DWORD *)v49 + 21) = *(_DWORD *)(v50 - 12);
          *((_BYTE *)v49 + 88) = *(_BYTE *)(v50 - 8);
          result = *(unsigned __int8 *)(v50 - 7);
          *((_BYTE *)v49 + 89) = result;
        }
        v49 += 12;
        if ( v50 == v171 )
          return result;
      }
      result = v171 - v50;
      v176 = 0xAAAAAAAAAAAAAAABLL * ((v171 - v50) >> 5);
      if ( v171 - v50 > 0 )
      {
        v31 = v49;
        v32 = v50;
        do
        {
          v33 = (_QWORD *)v31[4];
          result = v31[6];
          v34 = (_QWORD *)v31[5];
          *v31 = *(_QWORD *)v32;
          v35 = v33;
          v182 = result;
          v31[1] = *(_QWORD *)(v32 + 8);
          v31[2] = *(_QWORD *)(v32 + 16);
          v31[3] = *(_QWORD *)(v32 + 24);
          v31[4] = *(_QWORD *)(v32 + 32);
          v31[5] = *(_QWORD *)(v32 + 40);
          v31[6] = *(_QWORD *)(v32 + 48);
          *(_QWORD *)(v32 + 32) = 0;
          *(_QWORD *)(v32 + 40) = 0;
          for ( *(_QWORD *)(v32 + 48) = 0; v34 != v35; v35 += 13 )
          {
            v36 = (_QWORD *)v35[9];
            if ( v36 != v35 + 11 )
              result = j_j___libc_free_0(v36, v35[11] + 1LL);
            v37 = v35[6];
            if ( v37 )
              result = j_j___libc_free_0(v37, v35[8] - v37);
          }
          if ( v33 )
            result = j_j___libc_free_0(v33, v182 - (_QWORD)v33);
          v38 = *(_DWORD *)(v32 + 56);
          v31 += 12;
          v32 += 96;
          *((_DWORD *)v31 - 10) = v38;
          *((_DWORD *)v31 - 9) = *(_DWORD *)(v32 - 36);
          *((_DWORD *)v31 - 8) = *(_DWORD *)(v32 - 32);
          *(v31 - 3) = *(_QWORD *)(v32 - 24);
          *((_BYTE *)v31 - 16) = *(_BYTE *)(v32 - 16);
          *((_BYTE *)v31 - 15) = *(_BYTE *)(v32 - 15);
          *((_DWORD *)v31 - 3) = *(_DWORD *)(v32 - 12);
          *((_BYTE *)v31 - 8) = *(_BYTE *)(v32 - 8);
          *((_BYTE *)v31 - 7) = *(_BYTE *)(v32 - 7);
          --v176;
        }
        while ( v176 );
      }
    }
  }
  return result;
}
