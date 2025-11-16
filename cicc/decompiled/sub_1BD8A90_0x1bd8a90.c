// Function: sub_1BD8A90
// Address: 0x1bd8a90
//
__int64 __fastcall sub_1BD8A90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, int a6)
{
  __int64 v6; // rcx
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // rcx
  int v17; // r9d
  __int64 v18; // rdx
  __int64 v19; // rcx
  int v20; // r9d
  __int64 v21; // rdx
  __int64 v22; // rcx
  int v23; // r8d
  int v24; // r9d
  __int64 v25; // rdx
  __int64 v26; // rcx
  int v27; // r8d
  int v28; // r9d
  __int64 v29; // rdx
  __int64 v30; // rcx
  int v31; // r8d
  int v32; // r9d
  __int64 v33; // rdx
  __int64 v34; // rcx
  int v35; // r8d
  int v36; // r9d
  __int64 v37; // rax
  __int64 v38; // r13
  _QWORD *v39; // rdi
  _QWORD *v40; // rdi
  _QWORD *v41; // rdi
  _QWORD *v42; // rdi
  __int64 v43; // rdx
  __int64 v44; // rbx
  __int64 ***v45; // rax
  __int64 v46; // r13
  __int64 ***v47; // rdx
  int v48; // r12d
  char v49; // dl
  _QWORD *v50; // rdx
  _QWORD *v51; // rax
  __int64 v52; // rsi
  _QWORD *v53; // r15
  __int64 v54; // rax
  __int64 **v55; // rsi
  __int64 ***v56; // rdi
  __int64 ***v57; // rcx
  unsigned int *v58; // rax
  unsigned int *v59; // rax
  unsigned int *v60; // rax
  unsigned int *v61; // rax
  __int64 v62; // rsi
  __int64 v63; // rax
  __int64 v64; // rdi
  unsigned int v65; // ecx
  __int64 *v66; // rdx
  __int64 v67; // r9
  __int64 v68; // r15
  _QWORD *v69; // rax
  __int64 v70; // r13
  int v71; // ebx
  unsigned int v72; // r12d
  __int64 v73; // rax
  __int64 v74; // rax
  _WORD *v75; // rdx
  __int64 v76; // rdi
  __int64 v77; // rdx
  __m128i si128; // xmm0
  __int64 v79; // rax
  _WORD *v80; // rdx
  __int64 v81; // rdi
  __int64 v82; // rdx
  __m128i v83; // xmm0
  __int64 v84; // rax
  _WORD *v85; // rdx
  __int64 v87; // rdx
  __int64 v88; // rsi
  int v89; // edx
  char *v90; // rdi
  char *v91; // rax
  char *v92; // rdi
  char *v93; // rax
  __int64 v94; // rdi
  unsigned __int64 v95; // rdx
  __int64 v96; // rax
  __int64 v97; // rax
  char *v98; // rdi
  char *v99; // rax
  _QWORD *v100; // rdx
  int v101; // r11d
  unsigned int v102; // [rsp+28h] [rbp-488h]
  int v103; // [rsp+2Ch] [rbp-484h]
  __int64 v104; // [rsp+38h] [rbp-478h]
  __int64 v105; // [rsp+38h] [rbp-478h]
  __int64 v106; // [rsp+40h] [rbp-470h]
  __int64 v107; // [rsp+40h] [rbp-470h]
  __int64 *v108; // [rsp+40h] [rbp-470h]
  const char **v109; // [rsp+58h] [rbp-458h] BYREF
  unsigned __int64 v110[2]; // [rsp+60h] [rbp-450h] BYREF
  char v111; // [rsp+70h] [rbp-440h] BYREF
  char *v112; // [rsp+C0h] [rbp-3F0h]
  char v113; // [rsp+D0h] [rbp-3E0h] BYREF
  char *v114; // [rsp+F8h] [rbp-3B8h]
  char v115; // [rsp+108h] [rbp-3A8h] BYREF
  unsigned __int64 v116[2]; // [rsp+110h] [rbp-3A0h] BYREF
  __int16 v117; // [rsp+120h] [rbp-390h] BYREF
  char *v118; // [rsp+170h] [rbp-340h]
  char v119; // [rsp+180h] [rbp-330h] BYREF
  char *v120; // [rsp+1A8h] [rbp-308h]
  char v121; // [rsp+1B8h] [rbp-2F8h] BYREF
  unsigned __int64 v122[2]; // [rsp+1C0h] [rbp-2F0h] BYREF
  __int16 v123; // [rsp+1D0h] [rbp-2E0h] BYREF
  char *v124; // [rsp+220h] [rbp-290h]
  char v125; // [rsp+230h] [rbp-280h] BYREF
  char *v126; // [rsp+258h] [rbp-258h]
  char v127; // [rsp+268h] [rbp-248h] BYREF
  unsigned __int64 v128[2]; // [rsp+270h] [rbp-240h] BYREF
  _QWORD v129[10]; // [rsp+280h] [rbp-230h] BYREF
  char *v130; // [rsp+2D0h] [rbp-1E0h]
  char v131; // [rsp+2E0h] [rbp-1D0h] BYREF
  char *v132; // [rsp+308h] [rbp-1A8h]
  char v133; // [rsp+318h] [rbp-198h] BYREF
  __m128i v134; // [rsp+320h] [rbp-190h] BYREF
  _QWORD v135[2]; // [rsp+330h] [rbp-180h] BYREF
  int v136; // [rsp+340h] [rbp-170h]
  unsigned __int64 *v137; // [rsp+348h] [rbp-168h]
  char *v138; // [rsp+380h] [rbp-130h]
  char v139; // [rsp+390h] [rbp-120h] BYREF
  char *v140; // [rsp+3B8h] [rbp-F8h]
  char v141; // [rsp+3C8h] [rbp-E8h] BYREF
  void *s1; // [rsp+3D0h] [rbp-E0h] BYREF
  __int64 ***v143; // [rsp+3D8h] [rbp-D8h]
  __int64 ***v144; // [rsp+3E0h] [rbp-D0h] BYREF
  __int64 v145; // [rsp+3E8h] [rbp-C8h]
  int v146; // [rsp+3F0h] [rbp-C0h]
  _BYTE v147[56]; // [rsp+3F8h] [rbp-B8h] BYREF
  _BYTE *v148; // [rsp+430h] [rbp-80h]
  _BYTE v149[40]; // [rsp+440h] [rbp-70h] BYREF
  _BYTE *v150; // [rsp+468h] [rbp-48h]
  _BYTE v151[56]; // [rsp+478h] [rbp-38h] BYREF

  v6 = 0x2E8BA2E8BA2E8BA3LL;
  v8 = *(_QWORD *)a1;
  v102 = *(_DWORD *)(*(_QWORD *)a1 + 8LL);
  v9 = 0x2E8BA2E8BA2E8BA3LL * ((__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 4);
  if ( (_DWORD)v9 )
  {
    v103 = 0;
    v10 = (unsigned int)(v9 - 1);
    v11 = 0;
    v12 = 5 * v10;
    v104 = 176 * v10;
    while ( 1 )
    {
      v13 = v11 + v8;
      if ( !*(_BYTE *)(v13 + 88) )
      {
LABEL_3:
        v103 += sub_1BCC3C0(a1, (__int64 **)v13);
        if ( v11 == v104 )
          goto LABEL_67;
        goto LABEL_4;
      }
      sub_1BBD870((__int64)v110, v13, v12, v6, (int)a5, a6);
      v106 = *(_QWORD *)(a1 + 8);
      v14 = *(_QWORD *)a1 + v11 + 176;
      sub_1BBD870((__int64)v116, (__int64)v110, v15, v16, (int)v116, v17);
      sub_1BBD870((__int64)v122, (__int64)v116, v18, v19, (int)v116, v20);
      sub_1BBD870((__int64)v128, (__int64)v122, v21, v22, v23, v24);
      sub_1BBD9D0((__int64)&s1, (__int64)v128, v25, v26, v27, v28);
      sub_1BBD9D0((__int64)&v134, (__int64)&s1, v29, v30, v31, v32);
      if ( v150 != v151 )
        _libc_free((unsigned __int64)v150);
      if ( v148 != v149 )
        _libc_free((unsigned __int64)v148);
      if ( s1 != &v144 )
        _libc_free((unsigned __int64)s1);
      sub_1BBD870((__int64)&s1, (__int64)&v134, v33, v34, v35, v36);
      v6 = 0x2E8BA2E8BA2E8BA3LL;
      v37 = 0x2E8BA2E8BA2E8BA3LL * ((v106 - v14) >> 4);
      v12 = v37 >> 2;
      if ( v37 >> 2 > 0 )
      {
        v38 = v14 + 704 * v12;
        while ( 1 )
        {
          if ( *(_BYTE *)(v14 + 88) )
          {
            v12 = (unsigned int)v143;
            v42 = s1;
            if ( (unsigned int)v143 == (unsigned __int64)*(unsigned int *)(v14 + 8) )
            {
              v12 = 8LL * (unsigned int)v143;
              if ( !v12 || !memcmp(s1, *(const void **)v14, v12) )
                goto LABEL_29;
            }
            else if ( (unsigned int)v143 == (unsigned __int64)*(unsigned int *)(v14 + 104) )
            {
              v6 = (__int64)s1 + 8 * (unsigned int)v143;
              v58 = *(unsigned int **)(v14 + 96);
              if ( s1 == (void *)v6 )
                goto LABEL_29;
              while ( 1 )
              {
                v12 = *(_QWORD *)(*(_QWORD *)v14 + 8LL * *v58);
                if ( *v42 != v12 )
                  break;
                ++v42;
                ++v58;
                if ( (_QWORD *)v6 == v42 )
                  goto LABEL_29;
              }
            }
          }
          if ( *(_BYTE *)(v14 + 264) )
          {
            v12 = (unsigned int)v143;
            v39 = s1;
            if ( (unsigned int)v143 == (unsigned __int64)*(unsigned int *)(v14 + 184) )
            {
              v12 = 8LL * (unsigned int)v143;
              if ( !v12 || !memcmp(s1, *(const void **)(v14 + 176), v12) )
              {
LABEL_88:
                v14 += 176;
                goto LABEL_29;
              }
            }
            else if ( (unsigned int)v143 == (unsigned __int64)*(unsigned int *)(v14 + 280) )
            {
              v6 = (__int64)s1 + 8 * (unsigned int)v143;
              v59 = *(unsigned int **)(v14 + 272);
              if ( s1 == (void *)v6 )
                goto LABEL_88;
              while ( 1 )
              {
                v12 = *(_QWORD *)(*(_QWORD *)(v14 + 176) + 8LL * *v59);
                if ( *v39 != v12 )
                  break;
                ++v39;
                ++v59;
                if ( (_QWORD *)v6 == v39 )
                  goto LABEL_88;
              }
            }
          }
          if ( *(_BYTE *)(v14 + 440) )
          {
            v12 = (unsigned int)v143;
            v40 = s1;
            if ( (unsigned int)v143 == (unsigned __int64)*(unsigned int *)(v14 + 360) )
            {
              v12 = 8LL * (unsigned int)v143;
              if ( !v12 || !memcmp(s1, *(const void **)(v14 + 352), v12) )
              {
LABEL_91:
                v14 += 352;
                goto LABEL_29;
              }
            }
            else if ( (unsigned int)v143 == (unsigned __int64)*(unsigned int *)(v14 + 456) )
            {
              v6 = (__int64)s1 + 8 * (unsigned int)v143;
              v60 = *(unsigned int **)(v14 + 448);
              if ( s1 == (void *)v6 )
                goto LABEL_91;
              while ( 1 )
              {
                v12 = *(_QWORD *)(*(_QWORD *)(v14 + 352) + 8LL * *v60);
                if ( *v40 != v12 )
                  break;
                ++v40;
                ++v60;
                if ( (_QWORD *)v6 == v40 )
                  goto LABEL_91;
              }
            }
          }
          if ( !*(_BYTE *)(v14 + 616) )
            goto LABEL_24;
          v12 = (unsigned int)v143;
          v41 = s1;
          if ( (unsigned int)v143 == (unsigned __int64)*(unsigned int *)(v14 + 536) )
          {
            v12 = 8LL * (unsigned int)v143;
            if ( !v12 || !memcmp(s1, *(const void **)(v14 + 528), v12) )
            {
LABEL_94:
              v14 += 528;
              goto LABEL_29;
            }
LABEL_24:
            v14 += 704;
            if ( v14 == v38 )
              goto LABEL_115;
          }
          else
          {
            if ( (unsigned int)v143 != (unsigned __int64)*(unsigned int *)(v14 + 632) )
              goto LABEL_24;
            v6 = (__int64)s1 + 8 * (unsigned int)v143;
            v61 = *(unsigned int **)(v14 + 624);
            if ( s1 == (void *)v6 )
              goto LABEL_94;
            while ( 1 )
            {
              v12 = *(_QWORD *)(*(_QWORD *)(v14 + 528) + 8LL * *v61);
              if ( *v41 != v12 )
                break;
              ++v41;
              ++v61;
              if ( (_QWORD *)v6 == v41 )
                goto LABEL_94;
            }
            v14 += 704;
            if ( v14 == v38 )
            {
LABEL_115:
              v6 = 0x2E8BA2E8BA2E8BA3LL;
              v37 = 0x2E8BA2E8BA2E8BA3LL * ((v106 - v14) >> 4);
              break;
            }
          }
        }
      }
      if ( v37 == 2 )
        goto LABEL_167;
      if ( v37 == 3 )
        break;
      if ( v37 != 1 )
        goto LABEL_119;
LABEL_169:
      if ( !*(_BYTE *)(v14 + 88) )
        goto LABEL_119;
      v92 = (char *)s1;
      if ( (unsigned int)v143 == (unsigned __int64)*(unsigned int *)(v14 + 8) )
      {
        v12 = 8LL * (unsigned int)v143;
        if ( v12 && memcmp(s1, *(const void **)v14, v12) )
          goto LABEL_119;
      }
      else
      {
        v12 = *(unsigned int *)(v14 + 104);
        if ( (unsigned int)v143 != v12 )
          goto LABEL_119;
        v93 = (char *)s1 + 8 * (unsigned int)v143;
        v12 = *(_QWORD *)(v14 + 96);
        if ( s1 != v93 )
        {
          a5 = *(_QWORD **)v14;
          while ( 1 )
          {
            v6 = a5[*(unsigned int *)v12];
            if ( *(_QWORD *)v92 != v6 )
              break;
            v92 += 8;
            v12 += 4;
            if ( v93 == v92 )
              goto LABEL_29;
          }
LABEL_119:
          v14 = v106;
        }
      }
LABEL_29:
      if ( v150 != v151 )
        _libc_free((unsigned __int64)v150);
      if ( v148 != v149 )
        _libc_free((unsigned __int64)v148);
      if ( s1 != &v144 )
        _libc_free((unsigned __int64)s1);
      if ( v140 != &v141 )
        _libc_free((unsigned __int64)v140);
      if ( v138 != &v139 )
        _libc_free((unsigned __int64)v138);
      if ( (_QWORD *)v134.m128i_i64[0] != v135 )
        _libc_free(v134.m128i_u64[0]);
      if ( v132 != &v133 )
        _libc_free((unsigned __int64)v132);
      if ( v130 != &v131 )
        _libc_free((unsigned __int64)v130);
      if ( (_QWORD *)v128[0] != v129 )
        _libc_free(v128[0]);
      if ( v126 != &v127 )
        _libc_free((unsigned __int64)v126);
      if ( v124 != &v125 )
        _libc_free((unsigned __int64)v124);
      if ( (__int16 *)v122[0] != &v123 )
        _libc_free(v122[0]);
      if ( v120 != &v121 )
        _libc_free((unsigned __int64)v120);
      if ( v118 != &v119 )
        _libc_free((unsigned __int64)v118);
      if ( (__int16 *)v116[0] != &v117 )
        _libc_free(v116[0]);
      if ( v114 != &v115 )
        _libc_free((unsigned __int64)v114);
      if ( v112 != &v113 )
        _libc_free((unsigned __int64)v112);
      if ( (char *)v110[0] != &v111 )
        _libc_free(v110[0]);
      if ( v14 == v106 )
        goto LABEL_3;
      if ( v11 == v104 )
        goto LABEL_67;
LABEL_4:
      v8 = *(_QWORD *)a1;
      v11 += 176;
    }
    if ( *(_BYTE *)(v14 + 88) )
    {
      v90 = (char *)s1;
      if ( (unsigned int)v143 == (unsigned __int64)*(unsigned int *)(v14 + 8) )
      {
        v12 = 8LL * (unsigned int)v143;
        if ( !v12 || !memcmp(s1, *(const void **)v14, v12) )
          goto LABEL_29;
      }
      else
      {
        v12 = *(unsigned int *)(v14 + 104);
        if ( (unsigned int)v143 == v12 )
        {
          v91 = (char *)s1 + 8 * (unsigned int)v143;
          v12 = *(_QWORD *)(v14 + 96);
          if ( s1 == v91 )
            goto LABEL_29;
          while ( 1 )
          {
            v6 = *(_QWORD *)(*(_QWORD *)v14 + 8LL * *(unsigned int *)v12);
            if ( *(_QWORD *)v90 != v6 )
              break;
            v90 += 8;
            v12 += 4;
            if ( v91 == v90 )
              goto LABEL_29;
          }
        }
      }
    }
    v14 += 176;
LABEL_167:
    if ( *(_BYTE *)(v14 + 88) )
    {
      v98 = (char *)s1;
      if ( (unsigned int)v143 == (unsigned __int64)*(unsigned int *)(v14 + 8) )
      {
        v12 = 8LL * (unsigned int)v143;
        if ( !v12 || !memcmp(s1, *(const void **)v14, v12) )
          goto LABEL_29;
      }
      else
      {
        v12 = *(unsigned int *)(v14 + 104);
        if ( (unsigned int)v143 == v12 )
        {
          v99 = (char *)s1 + 8 * (unsigned int)v143;
          v12 = *(_QWORD *)(v14 + 96);
          if ( s1 == v99 )
            goto LABEL_29;
          a5 = *(_QWORD **)v14;
          while ( 1 )
          {
            v6 = a5[*(unsigned int *)v12];
            if ( *(_QWORD *)v98 != v6 )
              break;
            v98 += 8;
            v12 += 4;
            if ( v99 == v98 )
              goto LABEL_29;
          }
        }
      }
    }
    v14 += 176;
    goto LABEL_169;
  }
  v103 = 0;
LABEL_67:
  v43 = *(unsigned int *)(a1 + 392);
  v44 = *(_QWORD *)(a1 + 384);
  v45 = (__int64 ***)v147;
  s1 = 0;
  v143 = (__int64 ***)v147;
  v144 = (__int64 ***)v147;
  v145 = 16;
  v46 = v44 + 24 * v43;
  v146 = 0;
  if ( v44 != v46 )
  {
    v47 = (__int64 ***)v147;
    v48 = 0;
    v105 = a1 + 1472;
    while ( 1 )
    {
      v55 = *(__int64 ***)v44;
      if ( v47 != v45 )
        goto LABEL_69;
      v56 = &v45[HIDWORD(v145)];
      if ( v56 != v45 )
      {
        v57 = 0;
        while ( v55 != *v45 )
        {
          if ( *v45 == (__int64 **)-2LL )
            v57 = v45;
          if ( v56 == ++v45 )
          {
            if ( !v57 )
              goto LABEL_156;
            *v57 = v55;
            --v146;
            s1 = (char *)s1 + 1;
            goto LABEL_70;
          }
        }
        goto LABEL_75;
      }
LABEL_156:
      if ( HIDWORD(v145) < (unsigned int)v145 )
      {
        ++HIDWORD(v145);
        *v56 = v55;
        s1 = (char *)s1 + 1;
      }
      else
      {
LABEL_69:
        sub_16CCBA0((__int64)&s1, (__int64)v55);
        if ( !v49 )
          goto LABEL_75;
      }
LABEL_70:
      v50 = *(_QWORD **)(a1 + 800);
      v51 = *(_QWORD **)(a1 + 792);
      v52 = *(_QWORD *)(v44 + 8);
      if ( v50 == v51 )
      {
        v53 = &v51[*(unsigned int *)(a1 + 812)];
        if ( v51 == v53 )
        {
          v100 = *(_QWORD **)(a1 + 792);
        }
        else
        {
          do
          {
            if ( v52 == *v51 )
              break;
            ++v51;
          }
          while ( v53 != v51 );
          v100 = v53;
        }
      }
      else
      {
        v107 = *(_QWORD *)(v44 + 8);
        v53 = &v50[*(unsigned int *)(a1 + 808)];
        v51 = sub_16CC9F0(a1 + 784, v52);
        if ( v107 == *v51 )
        {
          v87 = *(_QWORD *)(a1 + 800);
          if ( v87 == *(_QWORD *)(a1 + 792) )
            v88 = *(unsigned int *)(a1 + 812);
          else
            v88 = *(unsigned int *)(a1 + 808);
          v100 = (_QWORD *)(v87 + 8 * v88);
        }
        else
        {
          v54 = *(_QWORD *)(a1 + 800);
          if ( v54 != *(_QWORD *)(a1 + 792) )
          {
            v51 = (_QWORD *)(v54 + 8LL * *(unsigned int *)(a1 + 808));
            goto LABEL_74;
          }
          v51 = (_QWORD *)(v54 + 8LL * *(unsigned int *)(a1 + 812));
          v100 = v51;
        }
      }
      while ( v100 != v51 && *v51 >= 0xFFFFFFFFFFFFFFFELL )
        ++v51;
LABEL_74:
      if ( v53 != v51 )
        goto LABEL_75;
      sub_16463B0(**(__int64 ***)v44, v102);
      v62 = ***(_QWORD ***)a1;
      v63 = *(unsigned int *)(a1 + 1496);
      v134.m128i_i64[0] = v62;
      if ( !(_DWORD)v63 )
        goto LABEL_155;
      v64 = *(_QWORD *)(a1 + 1480);
      v65 = (v63 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
      v66 = (__int64 *)(v64 + 16LL * v65);
      v67 = *v66;
      if ( v62 != *v66 )
      {
        v89 = 1;
        while ( v67 != -8 )
        {
          v101 = v89 + 1;
          v65 = (v63 - 1) & (v89 + v65);
          v66 = (__int64 *)(v64 + 16LL * v65);
          v67 = *v66;
          if ( v62 == *v66 )
            goto LABEL_122;
          v89 = v101;
        }
LABEL_155:
        v48 += sub_14A3470(*(_QWORD *)(a1 + 1320));
LABEL_75:
        v44 += 24;
        if ( v46 == v44 )
          goto LABEL_124;
        goto LABEL_76;
      }
LABEL_122:
      if ( v66 == (__int64 *)(v64 + 16 * v63) )
        goto LABEL_155;
      v68 = *(_QWORD *)sub_1BC4770(v105, (unsigned __int64 *)&v134);
      v69 = (_QWORD *)sub_15E0530(*(_QWORD *)(a1 + 1304));
      v108 = (__int64 *)sub_1644900(v69, v68);
      v44 += 24;
      sub_1BC4770(v105, (unsigned __int64 *)&v134);
      sub_16463B0(v108, v102);
      v48 += sub_14A33E0(*(_QWORD *)(a1 + 1320));
      if ( v46 == v44 )
      {
LABEL_124:
        v70 = v48;
        goto LABEL_125;
      }
LABEL_76:
      v47 = v144;
      v45 = v143;
    }
  }
  v70 = 0;
  v48 = 0;
LABEL_125:
  v71 = sub_1BBD340((__int64 ***)a1);
  v128[1] = 0;
  LOBYTE(v129[0]) = 0;
  v72 = v103 + v71 + v48;
  v128[0] = (unsigned __int64)v129;
  v136 = 1;
  v135[1] = 0;
  v134.m128i_i64[0] = (__int64)&unk_49EFBE0;
  v135[0] = 0;
  v134.m128i_i64[1] = 0;
  v137 = v128;
  v73 = sub_16E7EE0((__int64)&v134, "SLP: Spill Cost = ", 0x12u);
  v74 = sub_16E7AB0(v73, v71);
  v75 = *(_WORD **)(v74 + 24);
  v76 = v74;
  if ( *(_QWORD *)(v74 + 16) - (_QWORD)v75 <= 1u )
  {
    v97 = sub_16E7EE0(v74, ".\n", 2u);
    v77 = *(_QWORD *)(v97 + 24);
    v76 = v97;
  }
  else
  {
    *v75 = 2606;
    v77 = *(_QWORD *)(v74 + 24) + 2LL;
    *(_QWORD *)(v74 + 24) = v77;
  }
  if ( (unsigned __int64)(*(_QWORD *)(v76 + 16) - v77) <= 0x13 )
  {
    v76 = sub_16E7EE0(v76, "SLP: Extract Cost = ", 0x14u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_42CA9E0);
    *(_DWORD *)(v77 + 16) = 540876916;
    *(__m128i *)v77 = si128;
    *(_QWORD *)(v76 + 24) += 20LL;
  }
  v79 = sub_16E7AB0(v76, v70);
  v80 = *(_WORD **)(v79 + 24);
  v81 = v79;
  if ( *(_QWORD *)(v79 + 16) - (_QWORD)v80 <= 1u )
  {
    v96 = sub_16E7EE0(v79, ".\n", 2u);
    v82 = *(_QWORD *)(v96 + 24);
    v81 = v96;
  }
  else
  {
    *v80 = 2606;
    v82 = *(_QWORD *)(v79 + 24) + 2LL;
    *(_QWORD *)(v79 + 24) = v82;
  }
  if ( (unsigned __int64)(*(_QWORD *)(v81 + 16) - v82) <= 0x11 )
  {
    v81 = sub_16E7EE0(v81, "SLP: Total Cost = ", 0x12u);
  }
  else
  {
    v83 = _mm_load_si128((const __m128i *)&xmmword_42CA9F0);
    *(_WORD *)(v82 + 16) = 8253;
    *(__m128i *)v82 = v83;
    *(_QWORD *)(v81 + 24) += 18LL;
  }
  v84 = sub_16E7AB0(v81, (int)v72);
  v85 = *(_WORD **)(v84 + 24);
  if ( *(_QWORD *)(v84 + 16) - (_QWORD)v85 <= 1u )
  {
    sub_16E7EE0(v84, ".\n", 2u);
  }
  else
  {
    *v85 = 2606;
    *(_QWORD *)(v84 + 24) += 2LL;
  }
  sub_16E7BC0(v134.m128i_i64);
  if ( byte_4FB8F20 )
  {
    v94 = *(_QWORD *)(a1 + 1304);
    v123 = 260;
    v122[0] = (unsigned __int64)v128;
    v110[0] = (unsigned __int64)sub_1649960(v94);
    v110[1] = v95;
    v117 = 1283;
    v116[0] = (unsigned __int64)"SLP";
    v116[1] = (unsigned __int64)v110;
    v109 = (const char **)a1;
    sub_1BD8600(&v134, &v109, (__int64)v116, 0, (__int64)v122);
    if ( v134.m128i_i64[1] )
    {
      sub_16BED90(v134.m128i_i64[0], v134.m128i_i64[1], 0, 0);
      if ( (_QWORD *)v134.m128i_i64[0] != v135 )
        j_j___libc_free_0(v134.m128i_i64[0], v135[0] + 1LL);
    }
    else
    {
      sub_2240A30(&v134);
    }
  }
  if ( (_QWORD *)v128[0] != v129 )
    j_j___libc_free_0(v128[0], v129[0] + 1LL);
  if ( v144 != v143 )
    _libc_free((unsigned __int64)v144);
  return v72;
}
