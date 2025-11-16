// Function: sub_38F4CC0
// Address: 0x38f4cc0
//
__int64 __fastcall sub_38F4CC0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        __m128 a5,
        double a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v10; // r12
  int *v11; // r14
  int v12; // edx
  void *v13; // rbx
  __int64 v14; // rax
  char *v15; // rsi
  __int64 v16; // rdx
  _DWORD *v17; // rax
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned __int64 v21; // rdx
  int *v22; // rbx
  int v23; // eax
  unsigned __int64 v24; // r15
  bool v25; // cc
  unsigned __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // rax
  int *v29; // rax
  unsigned __int64 v30; // rdi
  void *v31; // r13
  void *v32; // r13
  __int64 v33; // rdx
  __int64 v34; // r13
  void *v35; // r12
  __int64 v36; // rbx
  __int64 v37; // rdx
  __int64 v38; // r15
  __int64 v39; // r14
  __int64 v40; // rcx
  __int64 v41; // r12
  __int64 v42; // rsi
  __int64 v43; // rbx
  unsigned __int64 v44; // rdx
  int v45; // eax
  unsigned __int64 v46; // rbx
  unsigned __int64 v47; // rdi
  __int64 v48; // rax
  __int64 v49; // rax
  int *v50; // rax
  unsigned __int64 v51; // rdi
  void *v52; // rax
  __int64 v53; // r13
  __int64 v54; // r12
  __int64 v55; // rdx
  __int64 v56; // r15
  __int64 v57; // r14
  __int64 v58; // rsi
  __int64 v59; // rdx
  __int64 v60; // r13
  __int64 v61; // r12
  __int64 v62; // rdx
  __int64 v63; // r15
  __int64 v64; // r13
  __int64 v65; // rdx
  __int64 j; // r14
  void *v67; // rax
  __int64 v68; // rbx
  __int64 v69; // r14
  __int64 v70; // r12
  __int64 v71; // r15
  __int64 v72; // rax
  __int64 v73; // rbx
  __int64 v74; // r12
  __int64 i; // r14
  __int64 v76; // rax
  void *v77; // r12
  __int64 v78; // rbx
  __int64 v79; // r13
  __int64 v80; // rbx
  __int64 v81; // r12
  __int64 v82; // r14
  __int64 v83; // r12
  __int64 v84; // r15
  __int64 v85; // rbx
  void *v86; // r14
  __int64 v87; // r13
  __int64 v88; // r12
  __int64 v89; // rbx
  __int64 v90; // [rsp+8h] [rbp-F8h]
  __int64 v91; // [rsp+8h] [rbp-F8h]
  __int64 v92; // [rsp+10h] [rbp-F0h]
  __int64 v93; // [rsp+10h] [rbp-F0h]
  __int64 v94; // [rsp+20h] [rbp-E0h]
  __int64 v95; // [rsp+28h] [rbp-D8h]
  __int64 v96; // [rsp+28h] [rbp-D8h]
  void *v97; // [rsp+30h] [rbp-D0h]
  void *v98; // [rsp+30h] [rbp-D0h]
  __int64 v99; // [rsp+38h] [rbp-C8h]
  __int64 v100; // [rsp+38h] [rbp-C8h]
  __int64 v101; // [rsp+38h] [rbp-C8h]
  __int64 v102; // [rsp+38h] [rbp-C8h]
  __int64 v103; // [rsp+38h] [rbp-C8h]
  __int64 v104; // [rsp+40h] [rbp-C0h]
  __int64 v105; // [rsp+40h] [rbp-C0h]
  __int64 v106; // [rsp+40h] [rbp-C0h]
  __int64 v107; // [rsp+40h] [rbp-C0h]
  __int64 v108; // [rsp+48h] [rbp-B8h]
  __int64 v109; // [rsp+48h] [rbp-B8h]
  char v111; // [rsp+5Eh] [rbp-A2h]
  unsigned __int8 v112; // [rsp+5Fh] [rbp-A1h]
  _QWORD v113[2]; // [rsp+60h] [rbp-A0h] BYREF
  unsigned __int64 v114; // [rsp+70h] [rbp-90h] BYREF
  unsigned int v115; // [rsp+78h] [rbp-88h]
  char v116[8]; // [rsp+80h] [rbp-80h] BYREF
  void *v117; // [rsp+88h] [rbp-78h] BYREF
  __int64 v118; // [rsp+90h] [rbp-70h]
  const char *v119; // [rsp+A0h] [rbp-60h] BYREF
  void *v120; // [rsp+A8h] [rbp-58h] BYREF
  __int64 v121; // [rsp+B0h] [rbp-50h]
  unsigned __int64 v122; // [rsp+B8h] [rbp-48h]
  unsigned int v123; // [rsp+C0h] [rbp-40h]

  v10 = a1;
  v11 = *(int **)(a1 + 152);
  v12 = *v11;
  if ( *v11 != 13 )
  {
    v111 = 0;
    if ( v12 == 12 )
    {
      v44 = *(unsigned int *)(a1 + 160);
      *(_BYTE *)(a1 + 258) = 0;
      v45 = v44;
      v44 *= 40LL;
      v46 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v44 - 40) >> 3);
      if ( v44 > 0x28 )
      {
        do
        {
          a5 = (__m128)_mm_loadu_si128((const __m128i *)v11 + 3);
          v25 = (unsigned int)v11[8] <= 0x40;
          *v11 = v11[10];
          *(__m128 *)(v11 + 2) = a5;
          if ( !v25 )
          {
            v47 = *((_QWORD *)v11 + 3);
            if ( v47 )
              j_j___libc_free_0_0(v47);
          }
          v48 = *((_QWORD *)v11 + 8);
          v11 += 10;
          *((_QWORD *)v11 - 2) = v48;
          LODWORD(v48) = v11[8];
          v11[8] = 0;
          *(v11 - 2) = v48;
          --v46;
        }
        while ( v46 );
        v45 = *(_DWORD *)(v10 + 160);
        v11 = *(int **)(v10 + 152);
      }
      v49 = (unsigned int)(v45 - 1);
      *(_DWORD *)(v10 + 160) = v49;
      v50 = &v11[10 * v49];
      if ( (unsigned int)v50[8] > 0x40 )
      {
        v51 = *((_QWORD *)v50 + 3);
        if ( v51 )
          j_j___libc_free_0_0(v51);
      }
      if ( *(_DWORD *)(v10 + 160) )
      {
        v111 = 0;
        v12 = **(_DWORD **)(v10 + 152);
      }
      else
      {
        sub_392C2E0(&v119, v10 + 144);
        sub_38E90E0(v10 + 152, *(_QWORD *)(v10 + 152), (unsigned __int64)&v119);
        if ( v123 > 0x40 && v122 )
          j_j___libc_free_0_0(v122);
        v111 = 0;
        v12 = **(_DWORD **)(v10 + 152);
      }
    }
LABEL_3:
    if ( v12 != 1 )
      goto LABEL_4;
LABEL_35:
    LOWORD(v121) = 260;
    v119 = (const char *)(v10 + 216);
    return (unsigned __int8)sub_3909CF0(v10, &v119, 0, 0, a8, a9);
  }
  v21 = *(unsigned int *)(a1 + 160);
  *(_BYTE *)(a1 + 258) = 0;
  v22 = v11 + 10;
  v23 = v21;
  v21 *= 40LL;
  v24 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v21 - 40) >> 3);
  if ( v21 > 0x28 )
  {
    do
    {
      a4 = (__m128)_mm_loadu_si128((const __m128i *)(v22 + 2));
      v25 = (unsigned int)*(v22 - 2) <= 0x40;
      *(v22 - 10) = *v22;
      *((__m128 *)v22 - 2) = a4;
      if ( !v25 )
      {
        v26 = *((_QWORD *)v22 - 2);
        if ( v26 )
          j_j___libc_free_0_0(v26);
      }
      v27 = *((_QWORD *)v22 + 3);
      v22 += 10;
      *((_QWORD *)v22 - 7) = v27;
      LODWORD(v27) = *(v22 - 2);
      *(v22 - 2) = 0;
      *(v22 - 12) = v27;
      --v24;
    }
    while ( v24 );
    v23 = *(_DWORD *)(v10 + 160);
    v11 = *(int **)(v10 + 152);
  }
  v28 = (unsigned int)(v23 - 1);
  *(_DWORD *)(v10 + 160) = v28;
  v29 = &v11[10 * v28];
  if ( (unsigned int)v29[8] > 0x40 )
  {
    v30 = *((_QWORD *)v29 + 3);
    if ( v30 )
      j_j___libc_free_0_0(v30);
  }
  if ( *(_DWORD *)(v10 + 160) )
  {
    v111 = 1;
    v12 = **(_DWORD **)(v10 + 152);
    goto LABEL_3;
  }
  sub_392C2E0(&v119, v10 + 144);
  sub_38E90E0(v10 + 152, *(_QWORD *)(v10 + 152), (unsigned __int64)&v119);
  if ( v123 > 0x40 && v122 )
    j_j___libc_free_0_0(v122);
  v111 = 1;
  v12 = **(_DWORD **)(v10 + 152);
  if ( v12 == 1 )
    goto LABEL_35;
LABEL_4:
  v112 = v12 != 4 && (v12 & 0xFFFFFFFB) != 2;
  if ( v112 )
  {
    v119 = "unexpected token in directive";
    LOWORD(v121) = 259;
    return (unsigned __int8)sub_3909CF0(v10, &v119, 0, 0, a8, a9);
  }
  v13 = sub_16982C0();
  if ( (void *)a2 == v13 )
    sub_169C4E0(&v117, a2);
  else
    sub_1698360((__int64)&v117, a2);
  v14 = sub_3909460(v10);
  v15 = *(char **)(v14 + 8);
  v16 = *(_QWORD *)(v14 + 16);
  v17 = *(_DWORD **)(v10 + 152);
  v113[0] = v15;
  v113[1] = v16;
  if ( *v17 == 2 )
  {
    if ( (unsigned int)sub_16D1F70(v113, (__int64)"infinity", 8u) && (unsigned int)sub_16D1F70(v113, (__int64)"inf", 3u) )
    {
      if ( (unsigned int)sub_16D1F70(v113, (__int64)"nan", 3u) )
        goto LABEL_95;
      v115 = 64;
      v114 = 0xFFFFFFFFLL;
      if ( (void *)a2 == v13 )
        sub_169C580(&v120, (__int64)v13);
      else
        sub_1698390((__int64)&v120, a2);
      if ( v13 == v120 )
        sub_169CAA0((__int64)&v120, 0, 0, (__int64 *)&v114, a4.m128_f32[0]);
      else
        sub_16986F0(&v120, 0, 0, (__int64 *)&v114);
      if ( v115 > 0x40 && v114 )
        j_j___libc_free_0_0(v114);
      v32 = v120;
      if ( v13 == v117 )
      {
        v107 = v118;
        if ( v13 == v120 )
        {
          if ( v118 )
          {
            v103 = v118 + 32LL * *(_QWORD *)(v118 - 8);
            if ( v103 != v118 )
            {
              v98 = v13;
              v96 = v10;
              do
              {
                v103 -= 32;
                if ( v32 == *(void **)(v103 + 8) )
                {
                  v80 = *(_QWORD *)(v103 + 16);
                  if ( v80 )
                  {
                    v81 = v80 + 32LL * *(_QWORD *)(v80 - 8);
                    while ( v80 != v81 )
                    {
                      v81 -= 32;
                      if ( v32 == *(void **)(v81 + 8) )
                      {
                        v82 = *(_QWORD *)(v81 + 16);
                        if ( v82 )
                        {
                          if ( v82 != v82 + 32LL * *(_QWORD *)(v82 - 8) )
                          {
                            v93 = v81;
                            v83 = v82 + 32LL * *(_QWORD *)(v82 - 8);
                            v84 = v80;
                            do
                            {
                              v83 -= 32;
                              if ( v32 == *(void **)(v83 + 8) )
                              {
                                v85 = *(_QWORD *)(v83 + 16);
                                if ( v85 )
                                {
                                  if ( v85 != v85 + 32LL * *(_QWORD *)(v85 - 8) )
                                  {
                                    v91 = v82;
                                    v86 = v32;
                                    v87 = v83;
                                    v88 = *(_QWORD *)(v83 + 16);
                                    v89 = v85 + 32LL * *(_QWORD *)(v85 - 8);
                                    do
                                    {
                                      v89 -= 32;
                                      sub_127D120((_QWORD *)(v89 + 8));
                                    }
                                    while ( v88 != v89 );
                                    v85 = v88;
                                    v83 = v87;
                                    v32 = v86;
                                    v82 = v91;
                                  }
                                  j_j_j___libc_free_0_0(v85 - 8);
                                }
                              }
                              else
                              {
                                sub_1698460(v83 + 8);
                              }
                            }
                            while ( v82 != v83 );
                            v81 = v93;
                            v80 = v84;
                          }
                          j_j_j___libc_free_0_0(v82 - 8);
                        }
                      }
                      else
                      {
                        sub_1698460(v81 + 8);
                      }
                    }
                    j_j_j___libc_free_0_0(v80 - 8);
                  }
                }
                else
                {
                  sub_1698460(v103 + 8);
                }
              }
              while ( v103 != v107 );
              v13 = v98;
              v10 = v96;
            }
            j_j_j___libc_free_0_0(v107 - 8);
          }
          goto LABEL_178;
        }
        if ( !v118 )
          goto LABEL_151;
        if ( v118 + 32LL * *(_QWORD *)(v118 - 8) != v118 )
        {
          v76 = v10;
          v77 = v13;
          v78 = v118 + 32LL * *(_QWORD *)(v118 - 8);
          v79 = v76;
          do
          {
            v78 -= 32;
            sub_127D120((_QWORD *)(v78 + 8));
          }
          while ( v78 != v107 );
          v13 = v77;
          v10 = v79;
        }
        j_j_j___libc_free_0_0(v107 - 8);
        v67 = v120;
      }
      else
      {
        if ( v13 != v120 )
        {
          sub_16983E0((__int64)&v117, (__int64)&v120);
          goto LABEL_62;
        }
        sub_1698460((__int64)&v117);
        v67 = v120;
      }
      if ( v13 != v67 )
      {
LABEL_151:
        sub_1698450((__int64)&v117, (__int64)&v120);
        goto LABEL_62;
      }
LABEL_178:
      sub_169C7E0(&v117, &v120);
LABEL_62:
      if ( v13 == v120 )
      {
        v108 = v121;
        if ( v121 )
        {
          v33 = 32LL * *(_QWORD *)(v121 - 8);
          v34 = v121 + v33;
          if ( v121 != v121 + v33 )
          {
            v99 = v10;
            v35 = v13;
            do
            {
              v34 -= 32;
              if ( v35 == *(void **)(v34 + 8) )
              {
                v36 = *(_QWORD *)(v34 + 16);
                if ( v36 )
                {
                  v37 = 32LL * *(_QWORD *)(v36 - 8);
                  v38 = v36 + v37;
                  while ( v36 != v38 )
                  {
                    v38 -= 32;
                    if ( v35 == *(void **)(v38 + 8) )
                    {
                      v39 = *(_QWORD *)(v38 + 16);
                      if ( v39 )
                      {
                        v40 = v39 + 32LL * *(_QWORD *)(v39 - 8);
                        if ( v39 != v40 )
                        {
                          do
                          {
                            v104 = v40 - 32;
                            sub_127D120((_QWORD *)(v40 - 24));
                            v40 = v104;
                          }
                          while ( v39 != v104 );
                        }
                        j_j_j___libc_free_0_0(v39 - 8);
                      }
                    }
                    else
                    {
                      sub_1698460(v38 + 8);
                    }
                  }
                  j_j_j___libc_free_0_0(v36 - 8);
                }
              }
              else
              {
                sub_1698460(v34 + 8);
              }
            }
            while ( v108 != v34 );
            v13 = v35;
            v10 = v99;
          }
          j_j_j___libc_free_0_0(v108 - 8);
        }
        goto LABEL_9;
      }
      goto LABEL_45;
    }
    if ( (void *)a2 == v13 )
      sub_169C580(&v120, (__int64)v13);
    else
      sub_1698390((__int64)&v120, a2);
    if ( v13 == v120 )
      sub_169CA30((__int64)&v120, 0);
    else
      sub_169B4C0((__int64)&v120, 0);
    v31 = v120;
    if ( v13 == v117 )
    {
      v106 = v118;
      if ( v13 == v120 )
      {
        if ( v118 )
        {
          if ( v118 + 32LL * *(_QWORD *)(v118 - 8) != v118 )
          {
            v97 = v13;
            v68 = v118 + 32LL * *(_QWORD *)(v118 - 8);
            v95 = v10;
            do
            {
              v68 -= 32;
              if ( *(void **)(v68 + 8) == v31 )
              {
                v69 = *(_QWORD *)(v68 + 16);
                if ( v69 )
                {
                  v70 = v69 + 32LL * *(_QWORD *)(v69 - 8);
                  if ( v69 != v70 )
                  {
                    v92 = v68;
                    v102 = *(_QWORD *)(v68 + 16);
                    do
                    {
                      v70 -= 32;
                      if ( *(void **)(v70 + 8) == v31 )
                      {
                        v71 = *(_QWORD *)(v70 + 16);
                        if ( v71 )
                        {
                          v72 = 32LL * *(_QWORD *)(v71 - 8);
                          v73 = v71 + v72;
                          if ( v71 != v71 + v72 )
                          {
                            v90 = v70;
                            do
                            {
                              v73 -= 32;
                              if ( v31 == *(void **)(v73 + 8) )
                              {
                                v74 = *(_QWORD *)(v73 + 16);
                                if ( v74 )
                                {
                                  for ( i = v74 + 32LL * *(_QWORD *)(v74 - 8); v74 != i; sub_127D120((_QWORD *)(i + 8)) )
                                    i -= 32;
                                  j_j_j___libc_free_0_0(v74 - 8);
                                }
                              }
                              else
                              {
                                sub_1698460(v73 + 8);
                              }
                            }
                            while ( v71 != v73 );
                            v70 = v90;
                          }
                          j_j_j___libc_free_0_0(v71 - 8);
                        }
                      }
                      else
                      {
                        sub_1698460(v70 + 8);
                      }
                    }
                    while ( v102 != v70 );
                    v69 = v102;
                    v68 = v92;
                  }
                  j_j_j___libc_free_0_0(v69 - 8);
                }
              }
              else
              {
                sub_1698460(v68 + 8);
              }
            }
            while ( v68 != v106 );
            v13 = v97;
            v10 = v95;
          }
          j_j_j___libc_free_0_0(v106 - 8);
        }
        goto LABEL_124;
      }
      if ( !v118 )
        goto LABEL_102;
      v60 = v118 + 32LL * *(_QWORD *)(v118 - 8);
      if ( v60 != v118 )
      {
        v101 = v10;
        do
        {
          v60 -= 32;
          if ( v13 == *(void **)(v60 + 8) )
          {
            v61 = *(_QWORD *)(v60 + 16);
            if ( v61 )
            {
              v62 = 32LL * *(_QWORD *)(v61 - 8);
              v63 = v61 + v62;
              if ( v61 != v61 + v62 )
              {
                v94 = v60;
                do
                {
                  v63 -= 32;
                  if ( v13 == *(void **)(v63 + 8) )
                  {
                    v64 = *(_QWORD *)(v63 + 16);
                    if ( v64 )
                    {
                      v65 = 32LL * *(_QWORD *)(v64 - 8);
                      for ( j = v64 + v65; v64 != j; sub_127D120((_QWORD *)(j + 8)) )
                        j -= 32;
                      j_j_j___libc_free_0_0(v64 - 8);
                    }
                  }
                  else
                  {
                    sub_1698460(v63 + 8);
                  }
                }
                while ( v61 != v63 );
                v60 = v94;
              }
              j_j_j___libc_free_0_0(v61 - 8);
            }
          }
          else
          {
            sub_1698460(v60 + 8);
          }
        }
        while ( v60 != v106 );
        v10 = v101;
      }
      j_j_j___libc_free_0_0(v106 - 8);
      v52 = v120;
    }
    else
    {
      if ( v13 != v120 )
      {
        sub_16983E0((__int64)&v117, (__int64)&v120);
        goto LABEL_44;
      }
      sub_1698460((__int64)&v117);
      v52 = v120;
    }
    if ( v52 != v13 )
    {
LABEL_102:
      sub_1698450((__int64)&v117, (__int64)&v120);
      goto LABEL_44;
    }
LABEL_124:
    sub_169C7E0(&v117, &v120);
LABEL_44:
    if ( v13 == v120 )
    {
      v105 = v121;
      if ( v121 )
      {
        v53 = v121 + 32LL * *(_QWORD *)(v121 - 8);
        if ( v121 != v53 )
        {
          v100 = v10;
          do
          {
            v53 -= 32;
            if ( v13 == *(void **)(v53 + 8) )
            {
              v54 = *(_QWORD *)(v53 + 16);
              if ( v54 )
              {
                v55 = 32LL * *(_QWORD *)(v54 - 8);
                v56 = v54 + v55;
                while ( v54 != v56 )
                {
                  v56 -= 32;
                  if ( v13 == *(void **)(v56 + 8) )
                  {
                    v57 = *(_QWORD *)(v56 + 16);
                    if ( v57 )
                    {
                      v58 = 32LL * *(_QWORD *)(v57 - 8);
                      v59 = v57 + v58;
                      if ( v57 != v57 + v58 )
                      {
                        do
                        {
                          v109 = v59 - 32;
                          sub_127D120((_QWORD *)(v59 - 24));
                          v59 = v109;
                        }
                        while ( v57 != v109 );
                      }
                      j_j_j___libc_free_0_0(v57 - 8);
                    }
                  }
                  else
                  {
                    sub_1698460(v56 + 8);
                  }
                }
                j_j_j___libc_free_0_0(v54 - 8);
              }
            }
            else
            {
              sub_1698460(v53 + 8);
            }
          }
          while ( v105 != v53 );
          v10 = v100;
        }
        j_j_j___libc_free_0_0(v105 - 8);
      }
LABEL_9:
      if ( !v111 )
        goto LABEL_10;
      goto LABEL_46;
    }
LABEL_45:
    sub_1698460((__int64)&v120);
    if ( !v111 )
      goto LABEL_10;
LABEL_46:
    if ( v13 == v117 )
      sub_169C8D0((__int64)&v117, *(double *)a4.m128_u64, *(double *)a5.m128_u64, a6);
    else
      sub_1699490((__int64)&v117);
LABEL_10:
    sub_38EB180(v10);
    if ( v13 == v117 )
      sub_169D930((__int64)&v119, (__int64)&v117);
    else
      sub_169D7E0((__int64)&v119, (__int64 *)&v117);
    if ( *(_DWORD *)(a3 + 8) > 0x40u && *(_QWORD *)a3 )
      j_j___libc_free_0_0(*(_QWORD *)a3);
    *(_QWORD *)a3 = v119;
    *(_DWORD *)(a3 + 8) = (_DWORD)v120;
    goto LABEL_16;
  }
  if ( (unsigned int)sub_169E610((__int64)v116, v15, v16, 0) != 1 )
    goto LABEL_9;
LABEL_95:
  v119 = "invalid floating point literal";
  LOWORD(v121) = 259;
  v112 = sub_3909CF0(v10, &v119, 0, 0, v18, v19);
LABEL_16:
  if ( v13 == v117 )
  {
    v41 = v118;
    if ( v118 )
    {
      v42 = 32LL * *(_QWORD *)(v118 - 8);
      v43 = v118 + v42;
      if ( v118 != v118 + v42 )
      {
        do
        {
          v43 -= 32;
          sub_127D120((_QWORD *)(v43 + 8));
        }
        while ( v41 != v43 );
      }
      j_j_j___libc_free_0_0(v41 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v117);
  }
  return v112;
}
