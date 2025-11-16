// Function: sub_1C9D5C0
// Address: 0x1c9d5c0
//
__int64 __fastcall sub_1C9D5C0(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 i; // r15
  char v6; // r14
  char v7; // al
  __int64 v8; // rax
  int *v9; // rdx
  int *v10; // rax
  int *v11; // rcx
  __int64 v12; // rsi
  int *v13; // rax
  int *v14; // rsi
  __int64 v15; // rax
  unsigned int **v16; // rsi
  char v17; // al
  int *v18; // r9
  int *v19; // rax
  unsigned int v20; // edx
  _QWORD *v21; // rbx
  unsigned __int64 *v22; // r14
  unsigned __int64 *v23; // r15
  unsigned __int64 v24; // rsi
  _QWORD *v25; // r8
  _QWORD *v26; // rdi
  _QWORD *v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 *v30; // rax
  __int64 v31; // rcx
  unsigned int v32; // edi
  int *v33; // rdx
  int *v34; // rax
  int *v35; // r11
  __int64 v36; // r10
  __int64 v37; // r9
  char v38; // r9
  _QWORD *v39; // r10
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rax
  unsigned __int64 v43; // rcx
  _QWORD *v44; // rax
  _QWORD *v45; // rdx
  _BOOL8 v46; // rdi
  __int64 *v47; // r14
  __int64 *v48; // r13
  __int64 v49; // rbx
  __int64 v50; // rdi
  __int64 v51; // rax
  unsigned int v52; // edx
  int *v53; // rax
  int *v54; // rbx
  unsigned int v55; // edx
  __int64 v56; // rsi
  __int64 v57; // rcx
  __int64 v58; // rax
  int *v59; // rdx
  _BOOL4 v60; // r10d
  __int64 v61; // rax
  unsigned __int64 k; // rax
  unsigned __int8 v63; // cl
  int *v64; // r8
  __int64 v65; // rsi
  __int64 v66; // rax
  __int64 v67; // rax
  int *v68; // rax
  int *v69; // r8
  unsigned int v70; // edx
  __int64 v71; // rsi
  __int64 v72; // rcx
  __int64 v73; // rax
  int v74; // edx
  __int64 *v75; // rax
  int *v76; // rax
  int *v77; // r8
  __int64 v78; // rdi
  __int64 v79; // rsi
  int *v80; // r8
  __int64 v81; // rsi
  __int64 v82; // rax
  __int64 v83; // rax
  int *v84; // rax
  int *v85; // r8
  unsigned int v86; // edx
  __int64 v87; // rsi
  __int64 v88; // rcx
  _QWORD *v89; // rcx
  _QWORD *v90; // r9
  __int64 v91; // rsi
  __int64 v92; // rdx
  __int64 v93; // rax
  int *v94; // rdx
  _BOOL4 v95; // r11d
  __int64 v96; // rax
  __int64 *v97; // rsi
  __int64 v98; // rax
  int *v99; // rdx
  _BOOL4 v100; // r8d
  __int64 v101; // rax
  unsigned int **v102; // rbx
  unsigned int **j; // r12
  char v104; // [rsp+10h] [rbp-130h]
  unsigned int v105; // [rsp+10h] [rbp-130h]
  int *v106; // [rsp+10h] [rbp-130h]
  int v107; // [rsp+18h] [rbp-128h]
  int *v108; // [rsp+18h] [rbp-128h]
  unsigned __int64 v109; // [rsp+20h] [rbp-120h]
  _QWORD *v110; // [rsp+20h] [rbp-120h]
  _BOOL4 v111; // [rsp+20h] [rbp-120h]
  unsigned __int8 v113; // [rsp+30h] [rbp-110h]
  _BOOL4 v114; // [rsp+30h] [rbp-110h]
  char v115; // [rsp+37h] [rbp-109h]
  __int64 v116; // [rsp+38h] [rbp-108h]
  int *v117; // [rsp+38h] [rbp-108h]
  unsigned int v118; // [rsp+38h] [rbp-108h]
  int *v119; // [rsp+38h] [rbp-108h]
  _QWORD *v120; // [rsp+40h] [rbp-100h]
  _BOOL4 v121; // [rsp+40h] [rbp-100h]
  __int64 v122; // [rsp+40h] [rbp-100h]
  __int64 v123; // [rsp+40h] [rbp-100h]
  _QWORD *v124; // [rsp+40h] [rbp-100h]
  char v125; // [rsp+48h] [rbp-F8h]
  char v126; // [rsp+48h] [rbp-F8h]
  __int64 v127; // [rsp+48h] [rbp-F8h]
  int v128; // [rsp+48h] [rbp-F8h]
  unsigned int v129; // [rsp+54h] [rbp-ECh] BYREF
  __int64 v130; // [rsp+58h] [rbp-E8h] BYREF
  unsigned __int64 v131; // [rsp+60h] [rbp-E0h] BYREF
  unsigned int *v132; // [rsp+68h] [rbp-D8h] BYREF
  unsigned int **v133; // [rsp+70h] [rbp-D0h] BYREF
  unsigned int **v134; // [rsp+78h] [rbp-C8h]
  unsigned int **v135; // [rsp+80h] [rbp-C0h]
  __int64 *v136; // [rsp+90h] [rbp-B0h] BYREF
  __int64 *v137; // [rsp+98h] [rbp-A8h]
  __int64 *v138; // [rsp+A0h] [rbp-A0h]
  __int64 v139; // [rsp+B0h] [rbp-90h] BYREF
  int v140; // [rsp+B8h] [rbp-88h] BYREF
  int *v141; // [rsp+C0h] [rbp-80h]
  int *v142; // [rsp+C8h] [rbp-78h]
  int *v143; // [rsp+D0h] [rbp-70h]
  __int64 v144; // [rsp+D8h] [rbp-68h]
  char v145[8]; // [rsp+E0h] [rbp-60h] BYREF
  int v146; // [rsp+E8h] [rbp-58h] BYREF
  __int64 v147; // [rsp+F0h] [rbp-50h]
  int *v148; // [rsp+F8h] [rbp-48h]
  int *v149; // [rsp+100h] [rbp-40h]
  __int64 v150; // [rsp+108h] [rbp-38h]

  result = sub_1C2F070(a2);
  v115 = result;
  if ( !(_BYTE)result )
    return result;
  v3 = a2 + 72;
  v4 = *(_QWORD *)(a2 + 80);
  v133 = 0;
  v148 = &v146;
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v137 = 0;
  v138 = 0;
  v140 = 0;
  v141 = 0;
  v142 = &v140;
  v143 = &v140;
  v144 = 0;
  v146 = 0;
  v147 = 0;
  v149 = &v146;
  v150 = 0;
  if ( a2 + 72 == v4 )
    goto LABEL_9;
  if ( !v4 )
    BUG();
  while ( 1 )
  {
    i = *(_QWORD *)(v4 + 24);
    if ( i != v4 + 16 )
      break;
    v4 = *(_QWORD *)(v4 + 8);
    if ( v3 == v4 )
      goto LABEL_9;
    if ( !v4 )
      BUG();
  }
  if ( v4 == v3 )
  {
LABEL_9:
    if ( byte_4FBE1C0 )
      goto LABEL_10;
  }
  else
  {
    v125 = 0;
    v6 = 0;
    do
    {
      if ( !i )
        BUG();
      v7 = *(_BYTE *)(i - 8);
      switch ( v7 )
      {
        case 'E':
          v125 = v115;
          break;
        case 'N':
          v8 = *(_QWORD *)(i - 48);
          if ( *(_BYTE *)(v8 + 16) )
          {
            v6 = v115;
          }
          else if ( (*(_BYTE *)(v8 + 33) & 0x20) == 0 )
          {
            v6 = v115;
          }
          break;
        case '6':
          v130 = i - 24;
          v15 = *(_QWORD *)(i - 24);
          if ( *(_BYTE *)(v15 + 8) == 15 && !(*(_DWORD *)(v15 + 8) >> 8) )
          {
            v16 = v134;
            if ( v134 == v135 )
            {
              sub_14147F0((__int64)&v133, v134, &v130);
            }
            else
            {
              if ( v134 )
              {
                *v134 = (unsigned int *)(i - 24);
                v16 = v134;
              }
              v134 = v16 + 1;
            }
          }
          break;
        default:
          v130 = 0;
          if ( v7 == 55 )
          {
            v131 = i - 24;
            v116 = **(_QWORD **)(i - 72);
            v17 = *(_BYTE *)(v116 + 8);
            if ( v17 == 15 )
            {
              v18 = &v140;
              v19 = v141;
              v20 = *(_DWORD *)(**(_QWORD **)(i - 48) + 8LL) >> 8;
              v129 = v20;
              if ( !v141 )
                goto LABEL_212;
              do
              {
                if ( v20 > v19[8] )
                {
                  v19 = (int *)*((_QWORD *)v19 + 3);
                }
                else
                {
                  v18 = v19;
                  v19 = (int *)*((_QWORD *)v19 + 2);
                }
              }
              while ( v19 );
              if ( v18 == &v140 || v20 < v18[8] )
              {
LABEL_212:
                v132 = &v129;
                v18 = (int *)sub_1C9D250(&v139, (__int64)v18, &v132);
              }
              v108 = v18;
              LODWORD(v132) = *(_DWORD *)(v116 + 8) >> 8;
              v105 = (unsigned int)v132;
              v93 = sub_B996D0((__int64)(v18 + 10), (unsigned int *)&v132);
              if ( v94 )
              {
                v95 = 1;
                if ( !v93 && v108 + 12 != v94 )
                  v95 = v105 < v94[8];
                v114 = v95;
                v106 = v94;
                v96 = sub_22077B0(40);
                *(_DWORD *)(v96 + 32) = (_DWORD)v132;
                sub_220F040(v114, v96, v106, v108 + 12);
                ++*((_QWORD *)v108 + 10);
              }
              v97 = v137;
              if ( v137 == v138 )
              {
                sub_190D490((__int64)&v136, v137, &v131);
              }
              else
              {
                if ( v137 )
                {
                  *v137 = v131;
                  v97 = v137;
                }
                v137 = v97 + 1;
              }
              LODWORD(v132) = *(_DWORD *)(v116 + 8) >> 8;
              v118 = (unsigned int)v132;
              v98 = sub_B996D0((__int64)v145, (unsigned int *)&v132);
              if ( v99 )
              {
                v100 = v98 || v99 == &v146 || v118 < v99[8];
                v111 = v100;
                v119 = v99;
                v101 = sub_22077B0(40);
                *(_DWORD *)(v101 + 32) = (_DWORD)v132;
                sub_220F040(v111, v101, v119, &v146);
                ++v150;
              }
            }
            else if ( v17 == 13 && (unsigned __int8)sub_1C97B40(v116) )
            {
              goto LABEL_10;
            }
          }
          else if ( (unsigned __int8)(v7 - 58) <= 1u )
          {
            v6 = v115;
          }
          break;
      }
      for ( i = *(_QWORD *)(i + 8); i == v4 - 24 + 40; i = *(_QWORD *)(v4 + 24) )
      {
        v4 = *(_QWORD *)(v4 + 8);
        if ( v3 == v4 )
          goto LABEL_28;
        if ( !v4 )
          BUG();
      }
    }
    while ( v3 != v4 );
LABEL_28:
    if ( v6 || v125 )
      goto LABEL_10;
    if ( (!v150 || v150 == 1 && v148[8] == 1) && byte_4FBE1C0 )
    {
      v102 = v133;
      for ( j = v134; j != v102; ++v102 )
      {
        v132 = *v102;
        if ( *(_BYTE *)(*(_QWORD *)v132 + 8LL) == 15 && !(*(_DWORD *)(*(_QWORD *)v132 + 8LL) >> 8) )
          *(_DWORD *)sub_1C9D4C0(a1 + 62, (unsigned __int64 *)&v132) = 1;
      }
      goto LABEL_10;
    }
  }
  v9 = v141;
  if ( !v141 )
    goto LABEL_69;
  v10 = v141;
  do
  {
    v11 = v10;
    v10 = (int *)*((_QWORD *)v10 + 2);
  }
  while ( v10 );
  if ( v11 == &v140 || v11[8] )
  {
LABEL_69:
    v104 = 0;
    v107 = 0;
    goto LABEL_70;
  }
  LODWORD(v131) = 0;
  do
  {
    v12 = (__int64)v9;
    v9 = (int *)*((_QWORD *)v9 + 2);
  }
  while ( v9 );
  if ( (int *)v12 == &v140 || *(_DWORD *)(v12 + 32) )
  {
    v132 = (unsigned int *)&v131;
    v12 = sub_1C9D250(&v139, v12, &v132);
  }
  if ( *(_QWORD *)(v12 + 80) == 1 )
  {
    LODWORD(v131) = 0;
    v13 = v141;
    if ( v141 )
    {
      do
      {
        v14 = v13;
        v13 = (int *)*((_QWORD *)v13 + 2);
      }
      while ( v13 );
      if ( v14 != &v140 && !v14[8] )
        goto LABEL_48;
    }
    else
    {
      v14 = &v140;
    }
    v132 = (unsigned int *)&v131;
    v14 = (int *)sub_1C9D250(&v139, (__int64)v14, &v132);
LABEL_48:
    v107 = *(_DWORD *)(*((_QWORD *)v14 + 8) + 32LL);
    v104 = v115;
LABEL_70:
    v21 = a1 + 63;
    v113 = v104 ^ 1;
    do
    {
LABEL_71:
      v22 = (unsigned __int64 *)v133;
      v23 = (unsigned __int64 *)v134;
      if ( v134 == v133 )
        goto LABEL_10;
      v126 = 0;
      do
      {
        v24 = *v22;
        v25 = (_QWORD *)a1[64];
        v131 = *v22;
        if ( !v25 )
          goto LABEL_80;
        v26 = v21;
        v27 = v25;
        do
        {
          while ( 1 )
          {
            v28 = v27[2];
            v29 = v27[3];
            if ( v27[4] >= v24 )
              break;
            v27 = (_QWORD *)v27[3];
            if ( !v29 )
              goto LABEL_78;
          }
          v26 = v27;
          v27 = (_QWORD *)v27[2];
        }
        while ( v28 );
LABEL_78:
        if ( v26 == v21 || v26[4] > v24 )
        {
LABEL_80:
          v30 = *(__int64 **)(v24 - 24);
          v31 = *v30;
          v32 = *(_DWORD *)(*v30 + 8) >> 8;
          if ( v32 )
          {
            v33 = v141;
            if ( v32 == 1 )
            {
              if ( !v141 )
                goto LABEL_234;
              v76 = v141;
              v77 = &v140;
              do
              {
                while ( 1 )
                {
                  v78 = *((_QWORD *)v76 + 2);
                  v79 = *((_QWORD *)v76 + 3);
                  if ( v76[8] )
                    break;
                  v76 = (int *)*((_QWORD *)v76 + 3);
                  if ( !v79 )
                    goto LABEL_160;
                }
                v77 = v76;
                v76 = (int *)*((_QWORD *)v76 + 2);
              }
              while ( v78 );
LABEL_160:
              if ( v77 != &v140 && (unsigned int)v77[8] <= 1 )
              {
                LODWORD(v130) = 1;
                v80 = &v140;
                do
                {
                  while ( 1 )
                  {
                    v81 = *((_QWORD *)v33 + 2);
                    v82 = *((_QWORD *)v33 + 3);
                    if ( v33[8] )
                      break;
                    v33 = (int *)*((_QWORD *)v33 + 3);
                    if ( !v82 )
                      goto LABEL_168;
                  }
                  v80 = v33;
                  v33 = (int *)*((_QWORD *)v33 + 2);
                }
                while ( v81 );
LABEL_168:
                if ( v80 == &v140 || (unsigned int)v80[8] > 1 )
                {
                  v123 = v31;
                  v132 = (unsigned int *)&v130;
                  v83 = sub_1C9D250(&v139, (__int64)v80, &v132);
                  v31 = v123;
                  v80 = (int *)v83;
                }
                if ( *((_QWORD *)v80 + 10) == 1 )
                {
                  v84 = v141;
                  v85 = &v140;
                  v86 = *(_DWORD *)(v31 + 8) >> 8;
                  LODWORD(v130) = v86;
                  if ( !v141 )
                    goto LABEL_179;
                  do
                  {
                    while ( 1 )
                    {
                      v87 = *((_QWORD *)v84 + 2);
                      v88 = *((_QWORD *)v84 + 3);
                      if ( v86 <= v84[8] )
                        break;
                      v84 = (int *)*((_QWORD *)v84 + 3);
                      if ( !v88 )
                        goto LABEL_177;
                    }
                    v85 = v84;
                    v84 = (int *)*((_QWORD *)v84 + 2);
                  }
                  while ( v87 );
LABEL_177:
                  if ( v85 == &v140 || v86 < v85[8] )
                  {
LABEL_179:
                    v132 = (unsigned int *)&v130;
                    v85 = (int *)sub_1C9D250(&v139, (__int64)v85, &v132);
                  }
                  if ( *(_DWORD *)(*((_QWORD *)v85 + 8) + 32LL) == 1 && (v107 == 1 || !v104) )
                  {
                    *(_DWORD *)sub_1C9D4C0(a1 + 62, &v131) = 1;
                    v126 = v115;
                  }
                }
              }
              else
              {
LABEL_234:
                if ( v113 | (v107 == 1) )
                {
                  v126 = v113 | (v107 == 1);
                  *(_DWORD *)sub_1C9D4C0(a1 + 62, &v131) = 1;
                }
              }
            }
            else
            {
              if ( !v141 )
                goto LABEL_89;
              v34 = v141;
              v35 = &v140;
              do
              {
                while ( 1 )
                {
                  v36 = *((_QWORD *)v34 + 2);
                  v37 = *((_QWORD *)v34 + 3);
                  if ( v32 <= v34[8] )
                    break;
                  v34 = (int *)*((_QWORD *)v34 + 3);
                  if ( !v37 )
                    goto LABEL_87;
                }
                v35 = v34;
                v34 = (int *)*((_QWORD *)v34 + 2);
              }
              while ( v36 );
LABEL_87:
              if ( v35 != &v140 && v32 >= v35[8] )
              {
                LODWORD(v130) = v32;
                v64 = &v140;
                do
                {
                  while ( 1 )
                  {
                    v65 = *((_QWORD *)v33 + 2);
                    v66 = *((_QWORD *)v33 + 3);
                    if ( v32 <= v33[8] )
                      break;
                    v33 = (int *)*((_QWORD *)v33 + 3);
                    if ( !v66 )
                      goto LABEL_134;
                  }
                  v64 = v33;
                  v33 = (int *)*((_QWORD *)v33 + 2);
                }
                while ( v65 );
LABEL_134:
                if ( v64 == &v140 || v32 < v64[8] )
                {
                  v122 = v31;
                  v132 = (unsigned int *)&v130;
                  v67 = sub_1C9D250(&v139, (__int64)v64, &v132);
                  v31 = v122;
                  v64 = (int *)v67;
                }
                if ( *((_QWORD *)v64 + 10) == 1 )
                {
                  v68 = v141;
                  v69 = &v140;
                  v70 = *(_DWORD *)(v31 + 8) >> 8;
                  LODWORD(v130) = v70;
                  if ( !v141 )
                    goto LABEL_145;
                  do
                  {
                    while ( 1 )
                    {
                      v71 = *((_QWORD *)v68 + 2);
                      v72 = *((_QWORD *)v68 + 3);
                      if ( v70 <= v68[8] )
                        break;
                      v68 = (int *)*((_QWORD *)v68 + 3);
                      if ( !v72 )
                        goto LABEL_143;
                    }
                    v69 = v68;
                    v68 = (int *)*((_QWORD *)v68 + 2);
                  }
                  while ( v71 );
LABEL_143:
                  if ( v69 == &v140 || v70 < v69[8] )
                  {
LABEL_145:
                    v132 = (unsigned int *)&v130;
                    v69 = (int *)sub_1C9D250(&v139, (__int64)v69, &v132);
                  }
                  v73 = *((_QWORD *)v69 + 8);
                  v74 = *(_DWORD *)(v73 + 32);
                  if ( v74 && (v74 == v107 || !v104) )
                  {
                    v128 = *(_DWORD *)(v73 + 32);
                    *(_DWORD *)sub_1C9D4C0(a1 + 62, &v131) = v128;
                    v126 = v115;
                  }
                }
              }
              else
              {
LABEL_89:
                v38 = v104 & (v107 != 0);
                if ( v38 )
                {
                  v39 = v21;
                  if ( !v25 )
                    goto LABEL_97;
                  do
                  {
                    while ( 1 )
                    {
                      v40 = v25[2];
                      v41 = v25[3];
                      if ( v25[4] >= v24 )
                        break;
                      v25 = (_QWORD *)v25[3];
                      if ( !v41 )
                        goto LABEL_95;
                    }
                    v39 = v25;
                    v25 = (_QWORD *)v25[2];
                  }
                  while ( v40 );
LABEL_95:
                  if ( v39 == v21 || v39[4] > v24 )
                  {
LABEL_97:
                    v120 = v39;
                    v42 = sub_22077B0(48);
                    v43 = v131;
                    *(_DWORD *)(v42 + 40) = 0;
                    *(_QWORD *)(v42 + 32) = v43;
                    v109 = v43;
                    v127 = v42;
                    v44 = sub_1C9D3C0(a1 + 62, v120, (unsigned __int64 *)(v42 + 32));
                    if ( v45 )
                    {
                      v46 = v21 == v45 || v44 || v109 < v45[4];
                      sub_220F040(v46, v127, v45, v21);
                      ++a1[67];
                      v39 = (_QWORD *)v127;
                      v38 = v104 & (v107 != 0);
                    }
                    else
                    {
                      v124 = v44;
                      j_j___libc_free_0(v127, 48);
                      v38 = v104 & (v107 != 0);
                      v39 = v124;
                    }
                  }
                  v126 = v38;
                  *((_DWORD *)v39 + 10) = v107;
                }
              }
            }
          }
        }
        ++v22;
      }
      while ( v23 != v22 );
      if ( !v126 )
        goto LABEL_10;
      sub_1C96CB0(v141);
      v47 = v137;
      v141 = 0;
      v142 = &v140;
      v143 = &v140;
      v144 = 0;
    }
    while ( v137 == v136 );
    v110 = v21;
    v48 = v136;
LABEL_107:
    v49 = *v48;
    v50 = *(_QWORD *)(*v48 - 48);
    if ( *(_BYTE *)(*(_QWORD *)v50 + 8LL) != 15 )
      goto LABEL_121;
    LODWORD(v130) = *(_DWORD *)(*(_QWORD *)v50 + 8LL) >> 8;
    if ( (_DWORD)v130 )
      goto LABEL_109;
    for ( k = sub_1649C60(v50); ; k = sub_1649C60(*v75) )
    {
      v63 = *(_BYTE *)(k + 16);
      if ( v63 <= 0x17u )
      {
        if ( v63 != 5 || *(_WORD *)(k + 18) != 32 )
          goto LABEL_109;
      }
      else if ( v63 != 56 )
      {
        if ( v63 == 54 )
        {
          v132 = (unsigned int *)k;
          if ( *(_BYTE *)(*(_QWORD *)k + 8LL) == 15 )
          {
            if ( *(_DWORD *)(*(_QWORD *)k + 8LL) >> 8 )
            {
              LODWORD(v130) = *(_DWORD *)(*(_QWORD *)k + 8LL) >> 8;
            }
            else
            {
              v89 = (_QWORD *)a1[64];
              if ( v89 )
              {
                v90 = v110;
                do
                {
                  while ( 1 )
                  {
                    v91 = v89[2];
                    v92 = v89[3];
                    if ( v89[4] >= k )
                      break;
                    v89 = (_QWORD *)v89[3];
                    if ( !v92 )
                      goto LABEL_191;
                  }
                  v90 = v89;
                  v89 = (_QWORD *)v89[2];
                }
                while ( v91 );
LABEL_191:
                if ( v90 != v110 && v90[4] <= k )
                  LODWORD(v130) = *(_DWORD *)sub_1C9D4C0(a1 + 62, (unsigned __int64 *)&v132);
              }
            }
          }
        }
LABEL_109:
        v51 = **(_QWORD **)(v49 - 24);
        if ( *(_BYTE *)(v51 + 8) == 15 )
        {
          v52 = *(_DWORD *)(v51 + 8);
          v53 = v141;
          v54 = &v140;
          v55 = v52 >> 8;
          LODWORD(v131) = v55;
          if ( !v141 )
            goto LABEL_117;
          do
          {
            while ( 1 )
            {
              v56 = *((_QWORD *)v53 + 2);
              v57 = *((_QWORD *)v53 + 3);
              if ( v55 <= v53[8] )
                break;
              v53 = (int *)*((_QWORD *)v53 + 3);
              if ( !v57 )
                goto LABEL_115;
            }
            v54 = v53;
            v53 = (int *)*((_QWORD *)v53 + 2);
          }
          while ( v56 );
LABEL_115:
          if ( v54 == &v140 || v55 < v54[8] )
          {
LABEL_117:
            v132 = (unsigned int *)&v131;
            v54 = (int *)sub_1C9D250(&v139, (__int64)v54, &v132);
          }
          v58 = sub_B996D0((__int64)(v54 + 10), (unsigned int *)&v130);
          if ( v59 )
          {
            v60 = 1;
            if ( !v58 && v54 + 12 != v59 )
              v60 = (unsigned int)v130 < v59[8];
            v117 = v59;
            v121 = v60;
            v61 = sub_22077B0(40);
            *(_DWORD *)(v61 + 32) = v130;
            sub_220F040(v121, v61, v117, v54 + 12);
            ++*((_QWORD *)v54 + 10);
          }
        }
LABEL_121:
        if ( v47 == ++v48 )
        {
          v21 = v110;
          goto LABEL_71;
        }
        goto LABEL_107;
      }
      if ( (*(_BYTE *)(k + 23) & 0x40) != 0 )
        v75 = *(__int64 **)(k - 8);
      else
        v75 = (__int64 *)(k - 24LL * (*(_DWORD *)(k + 20) & 0xFFFFFFF));
    }
  }
LABEL_10:
  sub_1C96AE0(v147);
  result = sub_1C96CB0(v141);
  if ( v136 )
    result = j_j___libc_free_0(v136, (char *)v138 - (char *)v136);
  if ( v133 )
    return j_j___libc_free_0(v133, (char *)v135 - (char *)v133);
  return result;
}
