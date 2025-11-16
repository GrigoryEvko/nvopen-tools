// Function: sub_391CCA0
// Address: 0x391cca0
//
__int64 __fastcall sub_391CCA0(
        __int64 a1,
        char *a2,
        unsigned __int64 a3,
        unsigned __int16 *a4,
        unsigned __int64 a5,
        _QWORD *a6)
{
  char *v8; // rbx
  __int64 v9; // rdi
  _BYTE *v10; // rax
  unsigned __int64 v12; // r14
  __int64 v13; // r15
  char v14; // al
  char v15; // dl
  char *v16; // rdx
  __int64 v17; // r15
  unsigned __int64 v18; // r14
  char v19; // si
  char v20; // al
  char *v21; // rax
  __int64 v22; // r15
  unsigned __int64 v23; // r14
  char v24; // si
  char v25; // al
  char *v26; // rax
  unsigned __int8 v27; // al
  __int64 v28; // r15
  unsigned __int64 v29; // r14
  char v30; // al
  char v31; // dl
  char *v32; // rdx
  __int64 v33; // r13
  unsigned __int64 v34; // rbx
  char v35; // al
  char v36; // dl
  char *v37; // rdx
  __int64 v38; // rbx
  unsigned __int64 v39; // r15
  size_t v40; // r14
  __int64 v41; // r15
  size_t v42; // r13
  char v43; // si
  char v44; // al
  char *v45; // rax
  __int64 v46; // r13
  unsigned __int64 v47; // rdx
  char *v48; // rdi
  unsigned __int64 v49; // r14
  char v50; // si
  char v51; // al
  __int64 v52; // r15
  unsigned __int64 v53; // r14
  char v54; // si
  char v55; // al
  char *v56; // rax
  __int64 v57; // r13
  unsigned __int64 v58; // rbx
  char v59; // al
  char v60; // dl
  char *v61; // rdx
  unsigned __int16 *v62; // rbx
  unsigned __int16 *v63; // r13
  __int64 v64; // r15
  unsigned __int64 v65; // r14
  char v66; // si
  char v67; // al
  char *v68; // rax
  __int64 v69; // r15
  unsigned __int64 v70; // r14
  char v71; // si
  char v72; // al
  char *v73; // rax
  __int64 v74; // r13
  unsigned __int64 v75; // rbx
  char v76; // al
  char v77; // dl
  char *v78; // rdx
  _QWORD *i; // r13
  size_t v80; // r14
  __int64 v81; // rbx
  size_t v82; // r15
  char v83; // si
  char v84; // al
  char *v85; // rax
  __int64 v86; // r8
  unsigned __int64 v87; // rax
  _BYTE *v88; // rdi
  __int64 v89; // r14
  unsigned __int64 v90; // rbx
  char v91; // si
  char v92; // al
  char *v93; // rax
  unsigned int *v94; // rbx
  __int64 v95; // r15
  unsigned __int64 v96; // r14
  char v97; // si
  char v98; // al
  char *v99; // rax
  __int64 v100; // r14
  unsigned __int64 v101; // r15
  char v102; // si
  char v103; // al
  char *v104; // rax
  unsigned __int64 v105; // r14
  char v106; // al
  char v107; // dl
  char *v108; // rdx
  size_t v109; // r14
  __int64 v110; // r15
  size_t v111; // r13
  char v112; // al
  char v113; // dl
  char *v114; // rdx
  __int64 v115; // r13
  void *v116; // rdi
  size_t v117; // r14
  __int64 v118; // r13
  size_t v119; // r15
  char v120; // al
  char v121; // dl
  char *v122; // rdx
  __int64 v123; // r8
  void *v124; // rdi
  __int64 v125; // r15
  unsigned __int64 v126; // r14
  char v127; // al
  char v128; // dl
  char *v129; // rdx
  __int64 v130; // r15
  unsigned __int64 v131; // r14
  char v132; // al
  char v133; // dl
  char *v134; // rdx
  __int64 v135; // r15
  unsigned __int64 v136; // r14
  char v137; // al
  char v138; // dl
  char *v139; // rdx
  __int64 v140; // [rsp+0h] [rbp-B0h]
  __int64 v144; // [rsp+28h] [rbp-88h]
  void *src; // [rsp+30h] [rbp-80h]
  char *srca; // [rsp+30h] [rbp-80h]
  char *srcb; // [rsp+30h] [rbp-80h]
  char *v148; // [rsp+38h] [rbp-78h]
  char *v149; // [rsp+38h] [rbp-78h]
  char *v150; // [rsp+38h] [rbp-78h]
  unsigned int *j; // [rsp+38h] [rbp-78h]
  _QWORD v152[4]; // [rsp+40h] [rbp-70h] BYREF
  _QWORD v153[10]; // [rsp+60h] [rbp-50h] BYREF

  v8 = a2;
  sub_391B490(a1, (__int64)v152, "linking", 7u);
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *(_BYTE **)(v9 + 24);
  if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 16) )
  {
    sub_16E7DE0(v9, 1);
    if ( !a3 )
    {
LABEL_3:
      if ( !*(_DWORD *)(a1 + 704) )
      {
LABEL_4:
        if ( !a5 )
          goto LABEL_5;
LABEL_68:
        sub_391B370(a1, (__int64)v153, 6);
        v57 = *(_QWORD *)(a1 + 8);
        v58 = a5;
        do
        {
          while ( 1 )
          {
            v59 = v58 & 0x7F;
            v60 = v58 & 0x7F | 0x80;
            v58 >>= 7;
            if ( v58 )
              v59 = v60;
            v61 = *(char **)(v57 + 24);
            if ( (unsigned __int64)v61 >= *(_QWORD *)(v57 + 16) )
              break;
            *(_QWORD *)(v57 + 24) = v61 + 1;
            *v61 = v59;
            if ( !v58 )
              goto LABEL_74;
          }
          sub_16E7DE0(v57, v59);
        }
        while ( v58 );
LABEL_74:
        v62 = a4;
        v63 = &a4[4 * a5];
        if ( a4 != v63 )
        {
          do
          {
            v64 = *(_QWORD *)(a1 + 8);
            v65 = *v62;
            do
            {
              while ( 1 )
              {
                v66 = v65 & 0x7F;
                v67 = v65 & 0x7F | 0x80;
                v65 >>= 7;
                if ( v65 )
                  v66 = v67;
                v68 = *(char **)(v64 + 24);
                if ( (unsigned __int64)v68 >= *(_QWORD *)(v64 + 16) )
                  break;
                *(_QWORD *)(v64 + 24) = v68 + 1;
                *v68 = v66;
                if ( !v65 )
                  goto LABEL_81;
              }
              sub_16E7DE0(v64, v66);
            }
            while ( v65 );
LABEL_81:
            v69 = *(_QWORD *)(a1 + 8);
            v70 = *((unsigned int *)v62 + 1);
            do
            {
              while ( 1 )
              {
                v71 = v70 & 0x7F;
                v72 = v70 & 0x7F | 0x80;
                v70 >>= 7;
                if ( v70 )
                  v71 = v72;
                v73 = *(char **)(v69 + 24);
                if ( (unsigned __int64)v73 >= *(_QWORD *)(v69 + 16) )
                  break;
                *(_QWORD *)(v69 + 24) = v73 + 1;
                *v73 = v71;
                if ( !v70 )
                  goto LABEL_87;
              }
              sub_16E7DE0(v69, v71);
            }
            while ( v70 );
LABEL_87:
            v62 += 4;
          }
          while ( v63 != v62 );
        }
        sub_3919EA0(a1, v153);
        if ( !a6[5] )
          return sub_3919EA0(a1, v152);
LABEL_89:
        sub_391B370(a1, (__int64)v153, 7);
        v74 = *(_QWORD *)(a1 + 8);
        v75 = a6[5];
        do
        {
          while ( 1 )
          {
            v76 = v75 & 0x7F;
            v77 = v75 & 0x7F | 0x80;
            v75 >>= 7;
            if ( v75 )
              v76 = v77;
            v78 = *(char **)(v74 + 24);
            if ( (unsigned __int64)v78 >= *(_QWORD *)(v74 + 16) )
              break;
            *(_QWORD *)(v74 + 24) = v78 + 1;
            *v78 = v76;
            if ( !v75 )
              goto LABEL_95;
          }
          sub_16E7DE0(v74, v76);
        }
        while ( v75 );
LABEL_95:
        for ( i = (_QWORD *)a6[3]; a6 + 1 != i; i = (_QWORD *)sub_220EF30((__int64)i) )
        {
          v80 = i[5];
          v81 = *(_QWORD *)(a1 + 8);
          v150 = (char *)i[4];
          v82 = v80;
          do
          {
            while ( 1 )
            {
              v83 = v82 & 0x7F;
              v84 = v82 & 0x7F | 0x80;
              v82 >>= 7;
              if ( v82 )
                v83 = v84;
              v85 = *(char **)(v81 + 24);
              if ( (unsigned __int64)v85 >= *(_QWORD *)(v81 + 16) )
                break;
              *(_QWORD *)(v81 + 24) = v85 + 1;
              *v85 = v83;
              if ( !v82 )
                goto LABEL_102;
            }
            sub_16E7DE0(v81, v83);
          }
          while ( v82 );
LABEL_102:
          v86 = *(_QWORD *)(a1 + 8);
          v87 = *(_QWORD *)(v86 + 16);
          v88 = *(_BYTE **)(v86 + 24);
          if ( v80 > v87 - (unsigned __int64)v88 )
          {
            sub_16E7EE0(*(_QWORD *)(a1 + 8), v150, v80);
            v86 = *(_QWORD *)(a1 + 8);
            v88 = *(_BYTE **)(v86 + 24);
            v87 = *(_QWORD *)(v86 + 16);
          }
          else if ( v80 )
          {
            v144 = *(_QWORD *)(a1 + 8);
            memcpy(v88, v150, v80);
            *(_QWORD *)(v144 + 24) += v80;
            v86 = *(_QWORD *)(a1 + 8);
            v88 = *(_BYTE **)(v86 + 24);
            v87 = *(_QWORD *)(v86 + 16);
          }
          if ( (unsigned __int64)v88 >= v87 )
          {
            sub_16E7DE0(v86, 0);
          }
          else
          {
            *(_QWORD *)(v86 + 24) = v88 + 1;
            *v88 = 0;
          }
          v89 = *(_QWORD *)(a1 + 8);
          v90 = (__int64)(i[7] - i[6]) >> 3;
          do
          {
            while ( 1 )
            {
              v91 = v90 & 0x7F;
              v92 = v90 & 0x7F | 0x80;
              v90 >>= 7;
              if ( v90 )
                v91 = v92;
              v93 = *(char **)(v89 + 24);
              if ( (unsigned __int64)v93 >= *(_QWORD *)(v89 + 16) )
                break;
              *(_QWORD *)(v89 + 24) = v93 + 1;
              *v93 = v91;
              if ( !v90 )
                goto LABEL_113;
            }
            sub_16E7DE0(v89, v91);
          }
          while ( v90 );
LABEL_113:
          v94 = (unsigned int *)i[6];
          for ( j = (unsigned int *)i[7]; j != v94; v94 += 2 )
          {
            v95 = *(_QWORD *)(a1 + 8);
            v96 = *v94;
            do
            {
              while ( 1 )
              {
                v97 = v96 & 0x7F;
                v98 = v96 & 0x7F | 0x80;
                v96 >>= 7;
                if ( v96 )
                  v97 = v98;
                v99 = *(char **)(v95 + 24);
                if ( (unsigned __int64)v99 >= *(_QWORD *)(v95 + 16) )
                  break;
                *(_QWORD *)(v95 + 24) = v99 + 1;
                *v99 = v97;
                if ( !v96 )
                  goto LABEL_120;
              }
              sub_16E7DE0(v95, v97);
            }
            while ( v96 );
LABEL_120:
            v100 = *(_QWORD *)(a1 + 8);
            v101 = v94[1];
            do
            {
              while ( 1 )
              {
                v102 = v101 & 0x7F;
                v103 = v101 & 0x7F | 0x80;
                v101 >>= 7;
                if ( v101 )
                  v102 = v103;
                v104 = *(char **)(v100 + 24);
                if ( (unsigned __int64)v104 >= *(_QWORD *)(v100 + 16) )
                  break;
                *(_QWORD *)(v100 + 24) = v104 + 1;
                *v104 = v102;
                if ( !v101 )
                  goto LABEL_126;
              }
              sub_16E7DE0(v100, v102);
            }
            while ( v101 );
LABEL_126:
            ;
          }
        }
        sub_3919EA0(a1, v153);
        return sub_3919EA0(a1, v152);
      }
      goto LABEL_37;
    }
  }
  else
  {
    *(_QWORD *)(v9 + 24) = v10 + 1;
    *v10 = 1;
    if ( !a3 )
      goto LABEL_3;
  }
  v12 = a3;
  sub_391B370(a1, (__int64)v153, 8);
  v13 = *(_QWORD *)(a1 + 8);
  do
  {
    while ( 1 )
    {
      v14 = v12 & 0x7F;
      v15 = v12 & 0x7F | 0x80;
      v12 >>= 7;
      if ( v12 )
        v14 = v15;
      v16 = *(char **)(v13 + 24);
      if ( (unsigned __int64)v16 >= *(_QWORD *)(v13 + 16) )
        break;
      *(_QWORD *)(v13 + 24) = v16 + 1;
      *v16 = v14;
      if ( !v12 )
        goto LABEL_14;
    }
    sub_16E7DE0(v13, v14);
  }
  while ( v12 );
LABEL_14:
  v148 = &a2[56 * a3];
  if ( a2 != v148 )
  {
    do
    {
      v17 = *(_QWORD *)(a1 + 8);
      v18 = (unsigned __int8)v8[16];
      do
      {
        while ( 1 )
        {
          v19 = v18 & 0x7F;
          v20 = v18 & 0x7F | 0x80;
          v18 >>= 7;
          if ( v18 )
            v19 = v20;
          v21 = *(char **)(v17 + 24);
          if ( (unsigned __int64)v21 >= *(_QWORD *)(v17 + 16) )
            break;
          *(_QWORD *)(v17 + 24) = v21 + 1;
          *v21 = v19;
          if ( !v18 )
            goto LABEL_21;
        }
        sub_16E7DE0(v17, v19);
      }
      while ( v18 );
LABEL_21:
      v22 = *(_QWORD *)(a1 + 8);
      v23 = *((unsigned int *)v8 + 5);
      do
      {
        while ( 1 )
        {
          v24 = v23 & 0x7F;
          v25 = v23 & 0x7F | 0x80;
          v23 >>= 7;
          if ( v23 )
            v24 = v25;
          v26 = *(char **)(v22 + 24);
          if ( (unsigned __int64)v26 >= *(_QWORD *)(v22 + 16) )
            break;
          *(_QWORD *)(v22 + 24) = v26 + 1;
          *v26 = v24;
          if ( !v23 )
            goto LABEL_27;
        }
        sub_16E7DE0(v22, v24);
      }
      while ( v23 );
LABEL_27:
      v27 = v8[16];
      if ( v27 == 1 )
      {
        v117 = *((_QWORD *)v8 + 1);
        v118 = *(_QWORD *)(a1 + 8);
        srcb = *(char **)v8;
        v119 = v117;
        do
        {
          while ( 1 )
          {
            v120 = v119 & 0x7F;
            v121 = v119 & 0x7F | 0x80;
            v119 >>= 7;
            if ( v119 )
              v120 = v121;
            v122 = *(char **)(v118 + 24);
            if ( (unsigned __int64)v122 >= *(_QWORD *)(v118 + 16) )
              break;
            *(_QWORD *)(v118 + 24) = v122 + 1;
            *v122 = v120;
            if ( !v119 )
              goto LABEL_151;
          }
          sub_16E7DE0(v118, v120);
        }
        while ( v119 );
LABEL_151:
        v123 = *(_QWORD *)(a1 + 8);
        v124 = *(void **)(v123 + 24);
        if ( v117 > *(_QWORD *)(v123 + 16) - (_QWORD)v124 )
        {
          sub_16E7EE0(*(_QWORD *)(a1 + 8), srcb, v117);
        }
        else if ( v117 )
        {
          v140 = *(_QWORD *)(a1 + 8);
          memcpy(v124, srcb, v117);
          *(_QWORD *)(v140 + 24) += v117;
        }
        if ( (v8[20] & 0x10) == 0 )
        {
          v125 = *(_QWORD *)(a1 + 8);
          v126 = *((unsigned int *)v8 + 10);
          do
          {
            while ( 1 )
            {
              v127 = v126 & 0x7F;
              v128 = v126 & 0x7F | 0x80;
              v126 >>= 7;
              if ( v126 )
                v127 = v128;
              v129 = *(char **)(v125 + 24);
              if ( (unsigned __int64)v129 >= *(_QWORD *)(v125 + 16) )
                break;
              *(_QWORD *)(v125 + 24) = v129 + 1;
              *v129 = v127;
              if ( !v126 )
                goto LABEL_161;
            }
            sub_16E7DE0(v125, v127);
          }
          while ( v126 );
LABEL_161:
          v130 = *(_QWORD *)(a1 + 8);
          v131 = *((unsigned int *)v8 + 11);
          do
          {
            while ( 1 )
            {
              v132 = v131 & 0x7F;
              v133 = v131 & 0x7F | 0x80;
              v131 >>= 7;
              if ( v131 )
                v132 = v133;
              v134 = *(char **)(v130 + 24);
              if ( (unsigned __int64)v134 >= *(_QWORD *)(v130 + 16) )
                break;
              *(_QWORD *)(v130 + 24) = v134 + 1;
              *v134 = v132;
              if ( !v131 )
                goto LABEL_167;
            }
            sub_16E7DE0(v130, v132);
          }
          while ( v131 );
LABEL_167:
          v135 = *(_QWORD *)(a1 + 8);
          v136 = *((unsigned int *)v8 + 12);
          do
          {
            v137 = v136 & 0x7F;
            v138 = v136 & 0x7F | 0x80;
            v136 >>= 7;
            if ( v136 )
              v137 = v138;
            v139 = *(char **)(v135 + 24);
            if ( (unsigned __int64)v139 < *(_QWORD *)(v135 + 16) )
            {
              *(_QWORD *)(v135 + 24) = v139 + 1;
              *v139 = v137;
            }
            else
            {
              sub_16E7DE0(v135, v137);
            }
          }
          while ( v136 );
        }
      }
      else
      {
        v28 = *(_QWORD *)(a1 + 8);
        if ( v27 <= 2u )
        {
          v105 = *((unsigned int *)v8 + 10);
          do
          {
            while ( 1 )
            {
              v106 = v105 & 0x7F;
              v107 = v105 & 0x7F | 0x80;
              v105 >>= 7;
              if ( v105 )
                v106 = v107;
              v108 = *(char **)(v28 + 24);
              if ( (unsigned __int64)v108 >= *(_QWORD *)(v28 + 16) )
                break;
              *(_QWORD *)(v28 + 24) = v108 + 1;
              *v108 = v106;
              if ( !v105 )
                goto LABEL_135;
            }
            sub_16E7DE0(v28, v106);
          }
          while ( v105 );
LABEL_135:
          if ( (v8[20] & 0x10) == 0 )
          {
            v109 = *((_QWORD *)v8 + 1);
            v110 = *(_QWORD *)(a1 + 8);
            srca = *(char **)v8;
            v111 = v109;
            do
            {
              while ( 1 )
              {
                v112 = v111 & 0x7F;
                v113 = v111 & 0x7F | 0x80;
                v111 >>= 7;
                if ( v111 )
                  v112 = v113;
                v114 = *(char **)(v110 + 24);
                if ( (unsigned __int64)v114 >= *(_QWORD *)(v110 + 16) )
                  break;
                *(_QWORD *)(v110 + 24) = v114 + 1;
                *v114 = v112;
                if ( !v111 )
                  goto LABEL_142;
              }
              sub_16E7DE0(v110, v112);
            }
            while ( v111 );
LABEL_142:
            v115 = *(_QWORD *)(a1 + 8);
            v116 = *(void **)(v115 + 24);
            if ( v109 > *(_QWORD *)(v115 + 16) - (_QWORD)v116 )
            {
              sub_16E7EE0(*(_QWORD *)(a1 + 8), srca, v109);
            }
            else if ( v109 )
            {
              memcpy(v116, srca, v109);
              *(_QWORD *)(v115 + 24) += v109;
            }
          }
        }
        else
        {
          v29 = *(unsigned int *)(*(_QWORD *)(a1 + 224) + 32LL * *((unsigned int *)v8 + 10) + 28);
          do
          {
            while ( 1 )
            {
              v30 = v29 & 0x7F;
              v31 = v29 & 0x7F | 0x80;
              v29 >>= 7;
              if ( v29 )
                v30 = v31;
              v32 = *(char **)(v28 + 24);
              if ( (unsigned __int64)v32 >= *(_QWORD *)(v28 + 16) )
                break;
              *(_QWORD *)(v28 + 24) = v32 + 1;
              *v32 = v30;
              if ( !v29 )
                goto LABEL_35;
            }
            sub_16E7DE0(v28, v30);
          }
          while ( v29 );
        }
      }
LABEL_35:
      v8 += 56;
    }
    while ( v148 != v8 );
  }
  sub_3919EA0(a1, v153);
  if ( !*(_DWORD *)(a1 + 704) )
    goto LABEL_4;
LABEL_37:
  sub_391B370(a1, (__int64)v153, 5);
  v33 = *(_QWORD *)(a1 + 8);
  v34 = *(unsigned int *)(a1 + 704);
  do
  {
    while ( 1 )
    {
      v35 = v34 & 0x7F;
      v36 = v34 & 0x7F | 0x80;
      v34 >>= 7;
      if ( v34 )
        v35 = v36;
      v37 = *(char **)(v33 + 24);
      if ( (unsigned __int64)v37 >= *(_QWORD *)(v33 + 16) )
        break;
      *(_QWORD *)(v33 + 24) = v37 + 1;
      *v37 = v35;
      if ( !v34 )
        goto LABEL_43;
    }
    sub_16E7DE0(v33, v35);
  }
  while ( v34 );
LABEL_43:
  v38 = *(_QWORD *)(a1 + 696);
  v39 = (unsigned __int64)*(unsigned int *)(a1 + 704) << 6;
  src = (void *)(v38 + v39);
  if ( v38 == v38 + v39 )
    goto LABEL_67;
  do
  {
    v40 = *(_QWORD *)(v38 + 16);
    v41 = *(_QWORD *)(a1 + 8);
    v149 = *(char **)(v38 + 8);
    v42 = v40;
    do
    {
      while ( 1 )
      {
        v43 = v42 & 0x7F;
        v44 = v42 & 0x7F | 0x80;
        v42 >>= 7;
        if ( v42 )
          v43 = v44;
        v45 = *(char **)(v41 + 24);
        if ( (unsigned __int64)v45 >= *(_QWORD *)(v41 + 16) )
          break;
        *(_QWORD *)(v41 + 24) = v45 + 1;
        *v45 = v43;
        if ( !v42 )
          goto LABEL_50;
      }
      sub_16E7DE0(v41, v43);
    }
    while ( v42 );
LABEL_50:
    v46 = *(_QWORD *)(a1 + 8);
    v47 = *(_QWORD *)(v46 + 16);
    v48 = *(char **)(v46 + 24);
    if ( v40 > v47 - (unsigned __int64)v48 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 8), v149, v40);
      v46 = *(_QWORD *)(a1 + 8);
      v48 = *(char **)(v46 + 24);
      v47 = *(_QWORD *)(v46 + 16);
    }
    else if ( v40 )
    {
      memcpy(v48, v149, v40);
      *(_QWORD *)(v46 + 24) += v40;
      v46 = *(_QWORD *)(a1 + 8);
      v48 = *(char **)(v46 + 24);
      v47 = *(_QWORD *)(v46 + 16);
    }
    v49 = *(unsigned int *)(v38 + 28);
    while ( 1 )
    {
      v50 = v49 & 0x7F;
      v51 = v49 & 0x7F | 0x80;
      v49 >>= 7;
      if ( v49 )
        v50 = v51;
      if ( v47 <= (unsigned __int64)v48 )
        break;
      *(_QWORD *)(v46 + 24) = v48 + 1;
      *v48 = v50;
      if ( !v49 )
        goto LABEL_60;
LABEL_55:
      v48 = *(char **)(v46 + 24);
      v47 = *(_QWORD *)(v46 + 16);
    }
    sub_16E7DE0(v46, v50);
    if ( v49 )
      goto LABEL_55;
LABEL_60:
    v52 = *(_QWORD *)(a1 + 8);
    v53 = *(unsigned int *)(v38 + 32);
    do
    {
      while ( 1 )
      {
        v54 = v53 & 0x7F;
        v55 = v53 & 0x7F | 0x80;
        v53 >>= 7;
        if ( v53 )
          v54 = v55;
        v56 = *(char **)(v52 + 24);
        if ( (unsigned __int64)v56 >= *(_QWORD *)(v52 + 16) )
          break;
        *(_QWORD *)(v52 + 24) = v56 + 1;
        *v56 = v54;
        if ( !v53 )
          goto LABEL_66;
      }
      sub_16E7DE0(v52, v54);
    }
    while ( v53 );
LABEL_66:
    v38 += 64;
  }
  while ( (void *)v38 != src );
LABEL_67:
  sub_3919EA0(a1, v153);
  if ( a5 )
    goto LABEL_68;
LABEL_5:
  if ( a6[5] )
    goto LABEL_89;
  return sub_3919EA0(a1, v152);
}
