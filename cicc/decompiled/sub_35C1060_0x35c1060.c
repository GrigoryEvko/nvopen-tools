// Function: sub_35C1060
// Address: 0x35c1060
//
_QWORD *__fastcall sub_35C1060(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rdi
  unsigned int v4; // r13d
  int *v5; // r12
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rcx
  _QWORD *v9; // rax
  float v10; // xmm0_4
  _QWORD *v11; // rdx
  float v12; // xmm1_4
  int *v13; // rax
  _BYTE *v14; // rsi
  unsigned int v15; // ecx
  _QWORD *v16; // r15
  __int64 v17; // rax
  unsigned int *v18; // r9
  unsigned int *v19; // r13
  __int64 v20; // rsi
  __int64 v21; // rbx
  __int64 v22; // rdx
  unsigned int v23; // edi
  unsigned int v24; // r12d
  _QWORD *v25; // r14
  __int64 v26; // r8
  __int64 v27; // rdi
  __int64 v28; // rsi
  __int64 v29; // rdx
  int v30; // eax
  __int64 v31; // r11
  __int64 v32; // rdx
  unsigned int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // r10
  __int64 v36; // r8
  __int64 v37; // rax
  __int64 v38; // r10
  __int64 v39; // rsi
  int *v40; // rdi
  _QWORD *v41; // rbx
  int *v42; // rax
  _BYTE *v43; // rsi
  unsigned int v44; // r8d
  _QWORD *v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r8
  __int64 v49; // rax
  __int64 v50; // rsi
  _DWORD *v51; // rdi
  __int64 v52; // rsi
  _DWORD *v53; // rax
  int v54; // eax
  __int64 v55; // r8
  __int64 v56; // rax
  int v57; // eax
  __int64 v58; // r8
  int *v59; // rdi
  int *v60; // rax
  _BYTE *v61; // rsi
  unsigned int v62; // ecx
  _QWORD *v63; // r15
  __int64 v64; // rax
  unsigned int *v65; // r11
  unsigned int *i; // r13
  __int64 v67; // rsi
  __int64 v68; // rbx
  __int64 v69; // rdx
  unsigned int v70; // edi
  unsigned int v71; // r12d
  _QWORD *v72; // r14
  __int64 v73; // r8
  __int64 v74; // rdi
  __int64 v75; // rsi
  __int64 v76; // rdx
  int v77; // eax
  __int64 v78; // r10
  unsigned int v79; // edx
  unsigned int j; // eax
  __int64 v81; // rdx
  __int64 v82; // r9
  __int64 v83; // r8
  __int64 v84; // rax
  __int64 v85; // r9
  __int64 v86; // rsi
  __int64 v87; // r8
  __int64 v88; // rax
  __int64 v89; // rsi
  _DWORD *v90; // rdi
  __int64 v91; // rsi
  _DWORD *v92; // rax
  int v93; // eax
  __int64 v94; // r8
  __int64 v95; // rax
  int v96; // eax
  __int64 v97; // r8
  unsigned int *v99; // [rsp+0h] [rbp-80h]
  unsigned int *v100; // [rsp+0h] [rbp-80h]
  unsigned int *v101; // [rsp+0h] [rbp-80h]
  unsigned int v102; // [rsp+0h] [rbp-80h]
  unsigned int v103; // [rsp+0h] [rbp-80h]
  unsigned int v104; // [rsp+0h] [rbp-80h]
  unsigned int *v105; // [rsp+0h] [rbp-80h]
  unsigned int *v106; // [rsp+0h] [rbp-80h]
  unsigned int *v107; // [rsp+0h] [rbp-80h]
  unsigned int v108; // [rsp+0h] [rbp-80h]
  unsigned int v109; // [rsp+0h] [rbp-80h]
  unsigned int v110; // [rsp+0h] [rbp-80h]
  unsigned int *v111; // [rsp+8h] [rbp-78h]
  unsigned int *v112; // [rsp+8h] [rbp-78h]
  unsigned int v113; // [rsp+8h] [rbp-78h]
  unsigned int v114; // [rsp+8h] [rbp-78h]
  unsigned int v115; // [rsp+8h] [rbp-78h]
  unsigned int v116; // [rsp+8h] [rbp-78h]
  unsigned int v117; // [rsp+8h] [rbp-78h]
  unsigned int *v118; // [rsp+8h] [rbp-78h]
  unsigned int *v119; // [rsp+8h] [rbp-78h]
  unsigned int *v120; // [rsp+8h] [rbp-78h]
  unsigned int v121; // [rsp+8h] [rbp-78h]
  unsigned int v122; // [rsp+8h] [rbp-78h]
  unsigned int v123; // [rsp+8h] [rbp-78h]
  unsigned int *v124; // [rsp+8h] [rbp-78h]
  unsigned int *v125; // [rsp+8h] [rbp-78h]
  unsigned int *v126; // [rsp+8h] [rbp-78h]
  unsigned int v127; // [rsp+10h] [rbp-70h]
  unsigned int v128; // [rsp+10h] [rbp-70h]
  unsigned int v129; // [rsp+10h] [rbp-70h]
  unsigned int v130; // [rsp+10h] [rbp-70h]
  unsigned int *v131; // [rsp+10h] [rbp-70h]
  unsigned int *v132; // [rsp+10h] [rbp-70h]
  int *v134; // [rsp+30h] [rbp-50h]
  int v136; // [rsp+44h] [rbp-3Ch] BYREF
  unsigned int v137; // [rsp+48h] [rbp-38h] BYREF
  int v138[13]; // [rsp+4Ch] [rbp-34h] BYREF

  v134 = (int *)(a2 + 14);
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
LABEL_2:
  while ( 1 )
  {
    v2 = a2;
    if ( !a2[6] )
      break;
LABEL_34:
    v40 = (int *)v2[4];
    v41 = v2;
    v138[0] = v40[8];
    v42 = sub_220F330(v40, v2 + 2);
    j_j___libc_free_0((unsigned __int64)v42);
    --v41[6];
    v43 = (_BYTE *)a1[1];
    if ( v43 == (_BYTE *)a1[2] )
    {
      sub_B8BBF0((__int64)a1, v43, v138);
      v44 = v138[0];
    }
    else
    {
      v44 = v138[0];
      if ( v43 )
      {
        *(_DWORD *)v43 = v138[0];
        v43 = (_BYTE *)a1[1];
        v44 = v138[0];
      }
      a1[1] = v43 + 4;
    }
    v45 = (_QWORD *)*a2;
    v46 = *(_QWORD *)(*a2 + 160LL) + 96LL * v44;
    v47 = *(_QWORD *)(v46 + 80) - *(_QWORD *)(v46 + 72);
    if ( v47 == 4 )
    {
      sub_35BF310(v45, v44);
    }
    else if ( v47 == 8 )
    {
      sub_35C0840(v45, v44);
    }
    else if ( v47 )
    {
      BUG();
    }
  }
  while ( 1 )
  {
    if ( a2[12] )
    {
      v59 = (int *)a2[10];
      v136 = v59[8];
      v60 = sub_220F330(v59, a2 + 8);
      j_j___libc_free_0((unsigned __int64)v60);
      --a2[12];
      v61 = (_BYTE *)a1[1];
      if ( v61 == (_BYTE *)a1[2] )
      {
        sub_B8BBF0((__int64)a1, v61, &v136);
        v62 = v136;
      }
      else
      {
        v62 = v136;
        if ( v61 )
        {
          *(_DWORD *)v61 = v136;
          v61 = (_BYTE *)a1[1];
          v62 = v136;
        }
        a1[1] = v61 + 4;
      }
      v63 = (_QWORD *)*a2;
      v64 = *(_QWORD *)(*a2 + 160LL) + 96LL * v62;
      v65 = *(unsigned int **)(v64 + 80);
      for ( i = *(unsigned int **)(v64 + 72); v65 != i; ++i )
      {
        v67 = v63[26];
        v68 = 48LL * *i;
        v69 = v67 + v68;
        v70 = *(_DWORD *)(v67 + v68 + 20);
        v71 = v70;
        if ( v70 == v62 )
          v71 = *(_DWORD *)(v69 + 24);
        v72 = (_QWORD *)v63[19];
        if ( v72 )
        {
          v73 = 96LL * v71;
          v74 = v73 + *(_QWORD *)(*v72 + 160LL);
          v75 = v68 + *(_QWORD *)(*v72 + 208LL);
          v76 = *(_QWORD *)v75;
          v77 = *(_DWORD *)(v74 + 24);
          if ( v71 == *(_DWORD *)(v75 + 24) )
          {
            *(_DWORD *)(v74 + 24) = v77 - *(_DWORD *)(v76 + 16);
            v78 = *(_QWORD *)(v76 + 32);
          }
          else
          {
            *(_DWORD *)(v74 + 24) = v77 - *(_DWORD *)(v76 + 20);
            v78 = *(_QWORD *)(v76 + 24);
          }
          v79 = *(_DWORD *)(v74 + 20);
          if ( v79 )
          {
            for ( j = 0; j < v79; ++j )
            {
              v81 = j;
              *(_DWORD *)(*(_QWORD *)(v74 + 32) + 4 * v81) -= *(unsigned __int8 *)(v78 + v81);
              v79 = *(_DWORD *)(v74 + 20);
            }
          }
          if ( *(_QWORD *)(v73 + *(_QWORD *)(*v72 + 160LL) + 80) - *(_QWORD *)(v73 + *(_QWORD *)(*v72 + 160LL) + 72) == 12 )
          {
            v137 = v71;
            v138[0] = v71;
            v96 = *(_DWORD *)(*(_QWORD *)(*v72 + 160LL) + v73 + 16);
            v97 = (__int64)(v72 + 1);
            switch ( v96 )
            {
              case 2:
                v104 = v62;
                v120 = v65;
                sub_35B9090(v72 + 7, (unsigned int *)v138);
                v97 = (__int64)(v72 + 1);
                v65 = v120;
                v62 = v104;
                break;
              case 3:
                v103 = v62;
                v119 = v65;
                sub_35B9090(v72 + 1, (unsigned int *)v138);
                v97 = (__int64)(v72 + 1);
                v65 = v119;
                v62 = v103;
                break;
              case 1:
                v102 = v62;
                v118 = v65;
                sub_35B9090(v72 + 13, (unsigned int *)v138);
                v62 = v102;
                v65 = v118;
                v97 = (__int64)(v72 + 1);
                break;
            }
            v114 = v62;
            v132 = v65;
            sub_B99820(v97, &v137);
            v65 = v132;
            v62 = v114;
            *(_DWORD *)(*(_QWORD *)(*v72 + 160LL) + 96LL * v137 + 16) = 3;
            v67 = v63[26];
            v69 = v67 + v68;
            v70 = *(_DWORD *)(v67 + v68 + 20);
          }
          else if ( *(_DWORD *)(v74 + 16) != 1
                 || v79 <= *(_DWORD *)(v74 + 24)
                 && (v138[0] = 0,
                     v90 = *(_DWORD **)(v74 + 32),
                     v130 = v62,
                     v91 = (__int64)&v90[v79],
                     v92 = sub_35B8490(v90, v91, v138),
                     v62 = v130,
                     (_DWORD *)v91 == v92) )
          {
            v67 = v63[26];
            v69 = v67 + v68;
            v70 = *(_DWORD *)(v67 + v68 + 20);
          }
          else
          {
            v137 = v71;
            v138[0] = v71;
            v93 = *(_DWORD *)(*(_QWORD *)(*v72 + 160LL) + v73 + 16);
            v94 = (__int64)(v72 + 7);
            switch ( v93 )
            {
              case 2:
                v108 = v62;
                v124 = v65;
                sub_35B9090(v72 + 7, (unsigned int *)v138);
                v94 = (__int64)(v72 + 7);
                v65 = v124;
                v62 = v108;
                break;
              case 3:
                v110 = v62;
                v126 = v65;
                sub_35B9090(v72 + 1, (unsigned int *)v138);
                v94 = (__int64)(v72 + 7);
                v65 = v126;
                v62 = v110;
                break;
              case 1:
                v109 = v62;
                v125 = v65;
                sub_35B9090(v72 + 13, (unsigned int *)v138);
                v62 = v109;
                v65 = v125;
                v94 = (__int64)(v72 + 7);
                break;
            }
            v113 = v62;
            v131 = v65;
            sub_B99820(v94, &v137);
            v62 = v113;
            v65 = v131;
            v95 = *(_QWORD *)(*v72 + 160LL) + 96LL * v137;
            *(_DWORD *)(v95 + 16) = 2;
            *(_BYTE *)(v95 + 64) = 1;
            v67 = v63[26];
            v69 = v67 + v68;
            v70 = *(_DWORD *)(v67 + v68 + 20);
          }
        }
        v82 = v63[20];
        if ( v71 == v70 )
        {
          v87 = *(_QWORD *)(v69 + 32);
          v88 = v82 + 96LL * v71;
          v89 = 48LL * *(unsigned int *)(*(_QWORD *)(v88 + 80) - 4LL) + v67;
          if ( *(_DWORD *)(v89 + 20) == v71 )
            *(_QWORD *)(v89 + 32) = v87;
          else
            *(_QWORD *)(v89 + 40) = v87;
          *(_DWORD *)(*(_QWORD *)(v88 + 72) + 4 * v87) = *(_DWORD *)(*(_QWORD *)(v88 + 80) - 4LL);
          *(_QWORD *)(v88 + 80) -= 4LL;
          *(_QWORD *)(v69 + 32) = -1;
        }
        else
        {
          v83 = *(unsigned int *)(v69 + 24);
          v84 = v82 + 96 * v83;
          v85 = *(_QWORD *)(v69 + 40);
          v86 = 48LL * *(unsigned int *)(*(_QWORD *)(v84 + 80) - 4LL) + v67;
          if ( (_DWORD)v83 == *(_DWORD *)(v86 + 20) )
            *(_QWORD *)(v86 + 32) = v85;
          else
            *(_QWORD *)(v86 + 40) = v85;
          *(_DWORD *)(*(_QWORD *)(v84 + 72) + 4 * v85) = *(_DWORD *)(*(_QWORD *)(v84 + 80) - 4LL);
          *(_QWORD *)(v84 + 80) -= 4LL;
          *(_QWORD *)(v69 + 40) = -1;
        }
      }
      goto LABEL_2;
    }
    if ( !a2[18] )
      return a1;
    v3 = a2[16];
    v4 = *(_DWORD *)(v3 + 32);
    v5 = v134;
    if ( v134 != (int *)v3 )
    {
      v6 = *a2;
      v5 = (int *)a2[16];
      while ( 1 )
      {
        v7 = sub_220EF30(v3);
        v3 = v7;
        if ( v134 == (int *)v7 )
          break;
        while ( 1 )
        {
          v8 = *(unsigned int *)(v7 + 32);
          v9 = (_QWORD *)(*(_QWORD *)(v6 + 160) + 96 * v8);
          v10 = **(float **)(*v9 + 8LL);
          v11 = (_QWORD *)(*(_QWORD *)(v6 + 160) + 96LL * v4);
          v12 = **(float **)(*v11 + 8LL);
          if ( v10 != v12 )
            break;
          if ( v11[10] - v11[9] > v9[10] - v9[9] )
          {
            v5 = (int *)v3;
            v4 = v8;
          }
          v7 = sub_220EF30(v3);
          v3 = v7;
          if ( v134 == (int *)v7 )
            goto LABEL_12;
        }
        if ( v12 > v10 )
        {
          v4 = v8;
          v5 = (int *)v3;
        }
      }
    }
LABEL_12:
    v136 = v4;
    v13 = sub_220F330(v5, v134);
    j_j___libc_free_0((unsigned __int64)v13);
    --a2[18];
    v14 = (_BYTE *)a1[1];
    if ( v14 == (_BYTE *)a1[2] )
    {
      sub_B8BBF0((__int64)a1, v14, &v136);
      v15 = v136;
    }
    else
    {
      v15 = v136;
      if ( v14 )
      {
        *(_DWORD *)v14 = v136;
        v14 = (_BYTE *)a1[1];
        v15 = v136;
      }
      a1[1] = v14 + 4;
    }
    v16 = (_QWORD *)*a2;
    v17 = *(_QWORD *)(*a2 + 160LL) + 96LL * v15;
    v18 = *(unsigned int **)(v17 + 80);
    v19 = *(unsigned int **)(v17 + 72);
    if ( v19 == v18 )
      goto LABEL_2;
    do
    {
      v20 = v16[26];
      v21 = 48LL * *v19;
      v22 = v20 + v21;
      v23 = *(_DWORD *)(v20 + v21 + 20);
      v24 = v23;
      if ( v15 == v23 )
        v24 = *(_DWORD *)(v22 + 24);
      v25 = (_QWORD *)v16[19];
      if ( v25 )
      {
        v26 = 96LL * v24;
        v27 = v26 + *(_QWORD *)(*v25 + 160LL);
        v28 = v21 + *(_QWORD *)(*v25 + 208LL);
        v29 = *(_QWORD *)v28;
        v30 = *(_DWORD *)(v27 + 24);
        if ( v24 == *(_DWORD *)(v28 + 24) )
        {
          *(_DWORD *)(v27 + 24) = v30 - *(_DWORD *)(v29 + 16);
          v31 = *(_QWORD *)(v29 + 32);
        }
        else
        {
          *(_DWORD *)(v27 + 24) = v30 - *(_DWORD *)(v29 + 20);
          v31 = *(_QWORD *)(v29 + 24);
        }
        v32 = *(unsigned int *)(v27 + 20);
        if ( (_DWORD)v32 )
        {
          v33 = 0;
          do
          {
            v34 = v33++;
            *(_DWORD *)(*(_QWORD *)(v27 + 32) + 4 * v34) -= *(unsigned __int8 *)(v31 + v34);
            v32 = *(unsigned int *)(v27 + 20);
          }
          while ( (unsigned int)v32 > v33 );
        }
        if ( *(_QWORD *)(v26 + *(_QWORD *)(*v25 + 160LL) + 80) - *(_QWORD *)(v26 + *(_QWORD *)(*v25 + 160LL) + 72) == 12 )
        {
          v137 = v24;
          v138[0] = v24;
          v57 = *(_DWORD *)(*(_QWORD *)(*v25 + 160LL) + v26 + 16);
          v58 = (__int64)(v25 + 1);
          switch ( v57 )
          {
            case 2:
              v101 = v18;
              v117 = v15;
              sub_35B9090(v25 + 7, (unsigned int *)v138);
              v58 = (__int64)(v25 + 1);
              v15 = v117;
              v18 = v101;
              break;
            case 3:
              v100 = v18;
              v116 = v15;
              sub_35B9090(v25 + 1, (unsigned int *)v138);
              v58 = (__int64)(v25 + 1);
              v15 = v116;
              v18 = v100;
              break;
            case 1:
              v99 = v18;
              v115 = v15;
              sub_35B9090(v25 + 13, (unsigned int *)v138);
              v18 = v99;
              v15 = v115;
              v58 = (__int64)(v25 + 1);
              break;
          }
          v112 = v18;
          v129 = v15;
          sub_B99820(v58, &v137);
          v15 = v129;
          v18 = v112;
          *(_DWORD *)(*(_QWORD *)(*v25 + 160LL) + 96LL * v137 + 16) = 3;
          v20 = v16[26];
          v22 = v20 + v21;
          v23 = *(_DWORD *)(v20 + v21 + 20);
        }
        else if ( *(_DWORD *)(v27 + 16) != 1
               || *(_DWORD *)(v27 + 24) >= (unsigned int)v32
               && (v138[0] = 0,
                   v51 = *(_DWORD **)(v27 + 32),
                   v127 = v15,
                   v52 = (__int64)&v51[v32],
                   v53 = sub_35B8490(v51, v52, v138),
                   v15 = v127,
                   (_DWORD *)v52 == v53) )
        {
          v20 = v16[26];
          v22 = v20 + v21;
          v23 = *(_DWORD *)(v20 + v21 + 20);
        }
        else
        {
          v137 = v24;
          v138[0] = v24;
          v54 = *(_DWORD *)(*(_QWORD *)(*v25 + 160LL) + v26 + 16);
          v55 = (__int64)(v25 + 7);
          switch ( v54 )
          {
            case 2:
              v107 = v18;
              v123 = v15;
              sub_35B9090(v25 + 7, (unsigned int *)v138);
              v55 = (__int64)(v25 + 7);
              v15 = v123;
              v18 = v107;
              break;
            case 3:
              v106 = v18;
              v122 = v15;
              sub_35B9090(v25 + 1, (unsigned int *)v138);
              v55 = (__int64)(v25 + 7);
              v15 = v122;
              v18 = v106;
              break;
            case 1:
              v105 = v18;
              v121 = v15;
              sub_35B9090(v25 + 13, (unsigned int *)v138);
              v18 = v105;
              v15 = v121;
              v55 = (__int64)(v25 + 7);
              break;
          }
          v111 = v18;
          v128 = v15;
          sub_B99820(v55, &v137);
          v18 = v111;
          v15 = v128;
          v56 = *(_QWORD *)(*v25 + 160LL) + 96LL * v137;
          *(_DWORD *)(v56 + 16) = 2;
          *(_BYTE *)(v56 + 64) = 1;
          v20 = v16[26];
          v22 = v20 + v21;
          v23 = *(_DWORD *)(v20 + v21 + 20);
        }
      }
      v35 = v16[20];
      if ( v24 == v23 )
      {
        v48 = *(_QWORD *)(v22 + 32);
        v49 = v35 + 96LL * v24;
        v50 = 48LL * *(unsigned int *)(*(_QWORD *)(v49 + 80) - 4LL) + v20;
        if ( v24 == *(_DWORD *)(v50 + 20) )
          *(_QWORD *)(v50 + 32) = v48;
        else
          *(_QWORD *)(v50 + 40) = v48;
        *(_DWORD *)(*(_QWORD *)(v49 + 72) + 4 * v48) = *(_DWORD *)(*(_QWORD *)(v49 + 80) - 4LL);
        *(_QWORD *)(v49 + 80) -= 4LL;
        *(_QWORD *)(v22 + 32) = -1;
      }
      else
      {
        v36 = *(unsigned int *)(v22 + 24);
        v37 = v35 + 96 * v36;
        v38 = *(_QWORD *)(v22 + 40);
        v39 = 48LL * *(unsigned int *)(*(_QWORD *)(v37 + 80) - 4LL) + v20;
        if ( (_DWORD)v36 == *(_DWORD *)(v39 + 20) )
          *(_QWORD *)(v39 + 32) = v38;
        else
          *(_QWORD *)(v39 + 40) = v38;
        *(_DWORD *)(*(_QWORD *)(v37 + 72) + 4 * v38) = *(_DWORD *)(*(_QWORD *)(v37 + 80) - 4LL);
        *(_QWORD *)(v37 + 80) -= 4LL;
        *(_QWORD *)(v22 + 40) = -1;
      }
      ++v19;
    }
    while ( v18 != v19 );
    v2 = a2;
    if ( a2[6] )
      goto LABEL_34;
  }
}
