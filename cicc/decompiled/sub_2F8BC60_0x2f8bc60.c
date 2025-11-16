// Function: sub_2F8BC60
// Address: 0x2f8bc60
//
__int64 __fastcall sub_2F8BC60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v10; // r14
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rbx
  __int64 v14; // rax
  int v15; // ecx
  __int64 v16; // r10
  __int64 result; // rax
  __int64 v18; // rcx
  __int64 v19; // rbx
  __int64 v20; // r14
  __int64 v21; // r8
  char **v22; // r12
  __int64 v23; // rdx
  char **v24; // rsi
  __int64 v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r12
  __int64 v30; // r8
  __int64 v31; // rbx
  __int64 v32; // r13
  __int64 i; // rdi
  __int64 v34; // rax
  char **v35; // rsi
  __int64 v36; // rax
  char **v37; // rsi
  __int64 v38; // rcx
  __int64 v39; // rbx
  __int64 v40; // r14
  char **v41; // r13
  __int64 v42; // rax
  char **v43; // rsi
  __int64 v44; // rdi
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // rbx
  char **v48; // rsi
  __int64 v49; // r14
  __int64 v50; // r12
  __int64 v51; // rbx
  __int64 v52; // rdx
  __int64 v53; // r13
  __int64 v54; // rdi
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r13
  __int64 v58; // r15
  char **v59; // rbx
  __int64 v60; // rdx
  char **v61; // rsi
  __int64 v62; // rdi
  __int64 v63; // rdx
  __int64 v64; // rdx
  __int64 v65; // r15
  __int64 v66; // r13
  __int64 v67; // r15
  char **v68; // rbx
  __int64 v69; // r12
  __int64 v70; // rdx
  char **v71; // rsi
  __int64 v72; // rdi
  __int64 v73; // rdx
  __int64 v74; // r12
  __int64 v75; // r15
  __int64 v76; // r13
  __int64 v77; // rax
  char **v78; // rsi
  __int64 v79; // rdi
  __int64 v80; // rax
  __int64 v81; // rbx
  __int64 v82; // r13
  char **v83; // r15
  __int64 v84; // r12
  __int64 v85; // rax
  char **v86; // rsi
  __int64 v87; // rdi
  __int64 v88; // rdx
  __int64 v89; // rcx
  char **v90; // rbx
  __int64 v91; // r13
  __int64 v92; // r15
  __int64 v93; // rdx
  char **v94; // rsi
  __int64 v95; // rdi
  __int64 v96; // rdx
  __int64 v97; // rax
  __int64 v98; // r13
  __int64 v99; // r12
  __int64 v100; // r15
  __int64 v101; // rax
  char **v102; // rsi
  __int64 v103; // rdi
  __int64 v104; // r15
  __int64 v105; // r12
  char **v106; // r13
  __int64 v107; // rax
  char **v108; // rsi
  __int64 v109; // rdi
  __int64 v110; // rax
  __int64 v111; // rdx
  __int64 v112; // r13
  __int64 v113; // rbx
  __int64 v114; // r12
  __int64 v115; // rax
  char **v116; // rsi
  __int64 v117; // rdi
  __int64 v118; // rdx
  __int64 v119; // r13
  __int64 v120; // rbx
  __int64 v121; // r12
  __int64 v122; // rdi
  char **v123; // r14
  __int64 v124; // [rsp+0h] [rbp-90h]
  __int64 v125; // [rsp+10h] [rbp-80h]
  __int64 v126; // [rsp+10h] [rbp-80h]
  __int64 v127; // [rsp+10h] [rbp-80h]
  __int64 v128; // [rsp+10h] [rbp-80h]
  __int64 v129; // [rsp+18h] [rbp-78h]
  __int64 v130; // [rsp+18h] [rbp-78h]
  __int64 v131; // [rsp+18h] [rbp-78h]
  __int64 v132; // [rsp+20h] [rbp-70h]
  __int64 v133; // [rsp+20h] [rbp-70h]
  __int64 v135; // [rsp+30h] [rbp-60h]
  __int64 v136; // [rsp+30h] [rbp-60h]
  __int64 v137; // [rsp+38h] [rbp-58h]
  __int64 v138; // [rsp+40h] [rbp-50h]
  __int64 v139; // [rsp+40h] [rbp-50h]
  __int64 v140; // [rsp+48h] [rbp-48h]
  __int64 v141; // [rsp+50h] [rbp-40h]
  __int64 v142; // [rsp+50h] [rbp-40h]
  __int64 v143; // [rsp+50h] [rbp-40h]

  v7 = a5;
  v8 = a6;
  if ( a7 <= a5 )
    v7 = a7;
  v140 = a3;
  v141 = a4;
  if ( v7 < a4 )
  {
    v10 = a5;
    if ( a7 >= a5 )
    {
LABEL_32:
      result = 0x2E8BA2E8BA2E8BA3LL;
      v38 = v140 - a2;
      v143 = v140 - a2;
      v39 = 0x2E8BA2E8BA2E8BA3LL * ((v140 - a2) >> 3);
      if ( v140 - a2 <= 0 )
        return result;
      v40 = v8 + 16;
      v41 = (char **)(a2 + 16);
      do
      {
        v42 = (__int64)*(v41 - 2);
        v43 = v41;
        v44 = v40;
        v41 += 11;
        v40 += 88;
        *(_QWORD *)(v40 - 104) = v42;
        *(_DWORD *)(v40 - 96) = *((_DWORD *)v41 - 24);
        *(_BYTE *)(v40 - 92) = *((_BYTE *)v41 - 92);
        sub_2F8ABB0(v44, v43, a3, v38, a5, a6);
        *(_DWORD *)(v40 - 24) = *((_DWORD *)v41 - 6);
        --v39;
      }
      while ( v39 );
      result = 88;
      v45 = v143 - 72;
      if ( v143 > 0 )
        result = v140 - a2;
      v46 = 16;
      if ( v143 <= 0 )
        v45 = 16;
      v47 = v8 + result;
      v48 = (char **)(v8 + v45);
      if ( a2 == a1 )
      {
        v118 = 0x2E8BA2E8BA2E8BA3LL;
        v119 = 0x2E8BA2E8BA2E8BA3LL * (result >> 3);
        v120 = v47 - 160;
        v121 = v140 - 72;
        while ( 1 )
        {
          v122 = v121;
          v123 = (char **)v120;
          v121 -= 88;
          *(_QWORD *)(v121 + 72) = *(_QWORD *)(v120 + 72);
          *(_DWORD *)(v121 + 80) = *(_DWORD *)(v120 + 80);
          *(_BYTE *)(v121 + 84) = *(_BYTE *)(v120 + 84);
          sub_2F8ABB0(v122, v48, v118, v46, a5, a6);
          result = *(unsigned int *)(v120 + 152);
          *(_DWORD *)(v121 + 152) = result;
          if ( !--v119 )
            break;
          v120 -= 88;
          v48 = v123;
        }
        return result;
      }
      if ( v8 == v47 )
        return result;
      v49 = v140;
      v50 = a2 - 88;
      v51 = v47 - 88;
      while ( 1 )
      {
        v53 = v49 - 88;
        v54 = v49 - 72;
        if ( *(_DWORD *)(v51 + 8) > *(_DWORD *)(v50 + 8) )
        {
          *(_QWORD *)(v49 - 88) = *(_QWORD *)v50;
          *(_DWORD *)(v53 + 8) = *(_DWORD *)(v50 + 8);
          v52 = *(unsigned __int8 *)(v50 + 12);
          *(_BYTE *)(v53 + 12) = v52;
          sub_2F8ABB0(v54, (char **)(v50 + 16), v52, v46, a5, a6);
          *(_DWORD *)(v53 + 80) = *(_DWORD *)(v50 + 80);
          if ( v50 == a1 )
          {
            v111 = 0x2E8BA2E8BA2E8BA3LL;
            result = v51 + 88 - v8;
            v112 = 0x2E8BA2E8BA2E8BA3LL * (result >> 3);
            if ( result > 0 )
            {
              v113 = v51 + 16;
              v114 = v49 - 160;
              do
              {
                v115 = *(_QWORD *)(v113 - 16);
                v116 = (char **)v113;
                v117 = v114;
                v113 -= 88;
                v114 -= 88;
                *(_QWORD *)(v114 + 72) = v115;
                *(_DWORD *)(v114 + 80) = *(_DWORD *)(v113 + 80);
                *(_BYTE *)(v114 + 84) = *(_BYTE *)(v113 + 84);
                sub_2F8ABB0(v117, v116, v111, v46, a5, a6);
                result = *(unsigned int *)(v113 + 152);
                *(_DWORD *)(v114 + 152) = result;
                --v112;
              }
              while ( v112 );
            }
            return result;
          }
          v50 -= 88;
        }
        else
        {
          v55 = *(_QWORD *)v51;
          *(_QWORD *)(v49 - 88) = *(_QWORD *)v51;
          *(_DWORD *)(v53 + 8) = *(_DWORD *)(v51 + 8);
          *(_BYTE *)(v53 + 12) = *(_BYTE *)(v51 + 12);
          sub_2F8ABB0(v54, (char **)(v51 + 16), v55, v46, a5, a6);
          result = *(unsigned int *)(v51 + 80);
          *(_DWORD *)(v53 + 80) = result;
          if ( v8 == v51 )
            return result;
          v51 -= 88;
        }
        v49 -= 88;
      }
    }
    if ( a5 >= a4 )
      goto LABEL_15;
LABEL_6:
    v135 = a4 / 2;
    v138 = a1 + 88 * (a4 / 2);
    v137 = sub_2F8AD70(a2, v140, v138);
    v13 = 0x2E8BA2E8BA2E8BA3LL * ((v137 - a2) >> 3);
    while ( 1 )
    {
      v141 -= v135;
      if ( v141 > v13 && a7 >= v13 )
      {
        v14 = v138;
        if ( !v13 )
          goto LABEL_10;
        v88 = 0x2E8BA2E8BA2E8BA3LL;
        v131 = a2 - v138;
        v89 = v137 - a2;
        if ( v137 - a2 <= 0 )
        {
          if ( v131 <= 0 )
            goto LABEL_10;
          v133 = 0;
          v128 = 0;
LABEL_77:
          v98 = 0x2E8BA2E8BA2E8BA3LL * ((a2 - v138) >> 3);
          v99 = a2 - 72;
          v100 = v137 - 72;
          do
          {
            v101 = *(_QWORD *)(v99 - 16);
            v102 = (char **)v99;
            v103 = v100;
            v99 -= 88;
            v100 -= 88;
            *(_QWORD *)(v100 + 72) = v101;
            *(_DWORD *)(v100 + 80) = *(_DWORD *)(v99 + 80);
            *(_BYTE *)(v100 + 84) = *(_BYTE *)(v99 + 84);
            sub_2F8ABB0(v103, v102, v88, v89, v11, v12);
            *(_DWORD *)(v100 + 152) = *(_DWORD *)(v99 + 152);
            --v98;
          }
          while ( v98 );
        }
        else
        {
          v127 = v13;
          v90 = (char **)(a2 + 16);
          v91 = 0x2E8BA2E8BA2E8BA3LL * ((v137 - a2) >> 3);
          v92 = a6 + 16;
          do
          {
            v93 = (__int64)*(v90 - 2);
            v94 = v90;
            v95 = v92;
            v90 += 11;
            v92 += 88;
            *(_QWORD *)(v92 - 104) = v93;
            *(_DWORD *)(v92 - 96) = *((_DWORD *)v90 - 24);
            v96 = *((unsigned __int8 *)v90 - 92);
            *(_BYTE *)(v92 - 92) = v96;
            sub_2F8ABB0(v95, v94, v96, v89, v11, v12);
            v88 = *((unsigned int *)v90 - 6);
            *(_DWORD *)(v92 - 24) = v88;
            --v91;
          }
          while ( v91 );
          v89 = v137 - a2;
          v97 = 88;
          v13 = v127;
          if ( v137 - a2 > 0 )
            v97 = v137 - a2;
          v128 = v97;
          v133 = 0x2E8BA2E8BA2E8BA3LL * (v97 >> 3);
          if ( v131 > 0 )
            goto LABEL_77;
        }
        v14 = v138;
        if ( v128 > 0 )
        {
          v104 = v138 + 16;
          v105 = v133;
          v106 = (char **)(a6 + 16);
          do
          {
            v107 = (__int64)*(v106 - 2);
            v108 = v106;
            v109 = v104;
            v106 += 11;
            v104 += 88;
            *(_QWORD *)(v104 - 104) = v107;
            *(_DWORD *)(v104 - 96) = *((_DWORD *)v106 - 24);
            *(_BYTE *)(v104 - 92) = *((_BYTE *)v106 - 92);
            sub_2F8ABB0(v109, v108, v88, v89, v11, v12);
            *(_DWORD *)(v104 - 24) = *((_DWORD *)v106 - 6);
            --v105;
          }
          while ( v105 );
          v110 = 88 * v133;
          if ( v133 <= 0 )
            v110 = 88;
          v14 = v138 + v110;
        }
        goto LABEL_10;
      }
      if ( a7 < v141 )
      {
        v14 = sub_2F8B420(v138, a2, v137, v141, v11, v12);
        goto LABEL_10;
      }
      v14 = v137;
      if ( !v141 )
        goto LABEL_10;
      v129 = v137 - a2;
      v56 = a2 - v138;
      if ( a2 - v138 <= 0 )
      {
        if ( v129 <= 0 )
          goto LABEL_10;
        v132 = 0;
        v66 = 0;
        v126 = a6;
        v124 = v138 + 16;
      }
      else
      {
        v125 = v13;
        v57 = 0x2E8BA2E8BA2E8BA3LL * ((a2 - v138) >> 3);
        v58 = a6 + 16;
        v124 = v138 + 16;
        v59 = (char **)(v138 + 16);
        do
        {
          v60 = (__int64)*(v59 - 2);
          v61 = v59;
          v62 = v58;
          v59 += 11;
          v58 += 88;
          *(_QWORD *)(v58 - 104) = v60;
          *(_DWORD *)(v58 - 96) = *((_DWORD *)v59 - 24);
          v63 = *((unsigned __int8 *)v59 - 92);
          *(_BYTE *)(v58 - 92) = v63;
          sub_2F8ABB0(v62, v61, v63, v56, v11, v12);
          v64 = *((unsigned int *)v59 - 6);
          *(_DWORD *)(v58 - 24) = v64;
          --v57;
        }
        while ( v57 );
        v65 = 88;
        v13 = v125;
        v56 = 0x2E8BA2E8BA2E8BA3LL;
        if ( a2 - v138 > 0 )
          v65 = a2 - v138;
        v66 = v65;
        v126 = v65 + a6;
        v132 = 0x2E8BA2E8BA2E8BA3LL * (v65 >> 3);
        if ( v129 <= 0 )
          goto LABEL_59;
      }
      v130 = v13;
      v67 = v124;
      v68 = (char **)(a2 + 16);
      v69 = 0x2E8BA2E8BA2E8BA3LL * ((v137 - a2) >> 3);
      do
      {
        v70 = (__int64)*(v68 - 2);
        v71 = v68;
        v72 = v67;
        v68 += 11;
        v67 += 88;
        *(_QWORD *)(v67 - 104) = v70;
        *(_DWORD *)(v67 - 96) = *((_DWORD *)v68 - 24);
        v73 = *((unsigned __int8 *)v68 - 92);
        *(_BYTE *)(v67 - 92) = v73;
        sub_2F8ABB0(v72, v71, v73, v56, v11, v12);
        v64 = *((unsigned int *)v68 - 6);
        *(_DWORD *)(v67 - 24) = v64;
        --v69;
      }
      while ( v69 );
      v13 = v130;
LABEL_59:
      v14 = v137;
      if ( v66 > 0 )
      {
        v74 = v132;
        v75 = v137 - 72;
        v76 = v126 - 72;
        do
        {
          v77 = *(_QWORD *)(v76 - 16);
          v78 = (char **)v76;
          v79 = v75;
          v76 -= 88;
          v75 -= 88;
          *(_QWORD *)(v75 + 72) = v77;
          *(_DWORD *)(v75 + 80) = *(_DWORD *)(v76 + 80);
          *(_BYTE *)(v75 + 84) = *(_BYTE *)(v76 + 84);
          sub_2F8ABB0(v79, v78, v64, v56, v11, v12);
          *(_DWORD *)(v75 + 152) = *(_DWORD *)(v76 + 152);
          --v74;
        }
        while ( v74 );
        v80 = -88 * v132;
        if ( v132 <= 0 )
          v80 = -88;
        v14 = v137 + v80;
      }
LABEL_10:
      v15 = v135;
      v10 -= v13;
      v136 = v14;
      sub_2F8BC60(a1, v138, v14, v15, v13, a6, a7);
      a3 = v10;
      if ( a7 <= v10 )
        a3 = a7;
      if ( a3 >= v141 )
      {
        v8 = a6;
        a2 = v137;
        a1 = v136;
        break;
      }
      if ( a7 >= v10 )
      {
        v8 = a6;
        a2 = v137;
        a1 = v136;
        goto LABEL_32;
      }
      a4 = v141;
      a2 = v137;
      a1 = v136;
      if ( v10 < v141 )
        goto LABEL_6;
LABEL_15:
      v13 = v10 / 2;
      v137 = a2 + 88 * (v10 / 2);
      v138 = sub_2F8AD10(a1, a2, v137);
      v135 = 0x2E8BA2E8BA2E8BA3LL * ((v138 - v16) >> 3);
    }
  }
  result = 0x2E8BA2E8BA2E8BA3LL;
  v18 = a2 - a1;
  v142 = a2 - a1;
  v19 = 0x2E8BA2E8BA2E8BA3LL * ((a2 - a1) >> 3);
  if ( a2 - a1 > 0 )
  {
    v139 = a2;
    v20 = v8 + 16;
    v21 = a1 + 16;
    v22 = (char **)(a1 + 16);
    do
    {
      v23 = (__int64)*(v22 - 2);
      v24 = v22;
      v25 = v20;
      v22 += 11;
      v20 += 88;
      *(_QWORD *)(v20 - 104) = v23;
      *(_DWORD *)(v20 - 96) = *((_DWORD *)v22 - 24);
      v26 = *((unsigned __int8 *)v22 - 92);
      *(_BYTE *)(v20 - 92) = v26;
      sub_2F8ABB0(v25, v24, v26, v18, v21, a6);
      v27 = *((unsigned int *)v22 - 6);
      *(_DWORD *)(v20 - 24) = v27;
      --v19;
    }
    while ( v19 );
    v28 = v142;
    result = 88;
    v29 = v139;
    v30 = a1 + 16;
    if ( v142 > 0 )
      result = v142;
    v31 = v8 + result;
    if ( v8 != v8 + result )
    {
      if ( v140 != v139 )
      {
        v32 = a1;
        for ( i = a1 + 16; ; i = v32 + 16 )
        {
          if ( *(_DWORD *)(v29 + 8) > *(_DWORD *)(v8 + 8) )
          {
            v34 = *(_QWORD *)v29;
            v35 = (char **)(v29 + 16);
            v32 += 88;
            v29 += 88;
            *(_QWORD *)(v32 - 88) = v34;
            *(_DWORD *)(v32 - 80) = *(_DWORD *)(v29 - 80);
            *(_BYTE *)(v32 - 76) = *(_BYTE *)(v29 - 76);
            sub_2F8ABB0(i, v35, v27, v28, v30, a6);
            result = *(unsigned int *)(v29 - 8);
            *(_DWORD *)(v32 - 8) = result;
            if ( v8 == v31 )
              return result;
          }
          else
          {
            v36 = *(_QWORD *)v8;
            v37 = (char **)(v8 + 16);
            v8 += 88;
            v32 += 88;
            *(_QWORD *)(v32 - 88) = v36;
            *(_DWORD *)(v32 - 80) = *(_DWORD *)(v8 - 80);
            *(_BYTE *)(v32 - 76) = *(_BYTE *)(v8 - 76);
            sub_2F8ABB0(i, v37, v27, v28, v30, a6);
            result = *(unsigned int *)(v8 - 8);
            *(_DWORD *)(v32 - 8) = result;
            if ( v8 == v31 )
              return result;
          }
          if ( v140 == v29 )
            break;
        }
        a1 = v32;
      }
      if ( v8 != v31 )
      {
        result = 0x2E8BA2E8BA2E8BA3LL;
        v81 = v31 - v8;
        v82 = 0x2E8BA2E8BA2E8BA3LL * (v81 >> 3);
        if ( v81 > 0 )
        {
          v83 = (char **)(v8 + 16);
          v84 = a1 + 16;
          do
          {
            v85 = (__int64)*(v83 - 2);
            v86 = v83;
            v87 = v84;
            v83 += 11;
            v84 += 88;
            *(_QWORD *)(v84 - 104) = v85;
            *(_DWORD *)(v84 - 96) = *((_DWORD *)v83 - 24);
            *(_BYTE *)(v84 - 92) = *((_BYTE *)v83 - 92);
            sub_2F8ABB0(v87, v86, v27, v28, v30, a6);
            result = *((unsigned int *)v83 - 6);
            *(_DWORD *)(v84 - 24) = result;
            --v82;
          }
          while ( v82 );
        }
      }
    }
  }
  return result;
}
