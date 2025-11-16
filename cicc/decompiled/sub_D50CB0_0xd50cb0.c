// Function: sub_D50CB0
// Address: 0xd50cb0
//
__int64 __fastcall sub_D50CB0(__int64 a1, __int64 a2)
{
  __int64 *v4; // rsi
  __int64 v5; // r8
  __int64 v6; // r14
  __int64 v7; // r9
  __int64 v8; // rcx
  int v9; // eax
  _QWORD *v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdx
  _QWORD *v24; // rax
  __int64 v25; // r15
  _QWORD *v26; // r14
  __int64 v27; // rbx
  unsigned int v28; // eax
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // rdx
  unsigned __int64 v33; // rcx
  __int64 v34; // r14
  void *v35; // rbx
  size_t v36; // r15
  char *v37; // rax
  __int64 *v38; // r14
  __int64 *v39; // rax
  unsigned int v40; // esi
  __int64 v41; // rbx
  __int64 v42; // rax
  unsigned int v43; // ecx
  __int64 *v44; // rdx
  __int64 v45; // rdi
  unsigned __int64 *v46; // rdx
  unsigned __int64 *v47; // rax
  unsigned __int64 *v48; // r9
  __int64 *v49; // rcx
  _QWORD *v50; // rcx
  const void *v51; // r8
  unsigned __int64 v52; // r15
  __int64 v53; // r14
  __int64 v54; // rax
  char *v55; // rbx
  signed __int64 v56; // rdx
  bool v57; // zf
  int v58; // edx
  __int64 v59; // rcx
  unsigned int v60; // edx
  int v61; // r14d
  __int64 *v62; // rdx
  __int64 v63; // rdi
  __int64 *v64; // rcx
  __int64 v65; // r9
  unsigned __int64 *v66; // rdx
  __int64 v67; // r9
  __int64 v68; // rax
  __int64 v69; // rbx
  unsigned __int64 v70; // rbx
  unsigned __int64 v71; // rdx
  unsigned __int64 v72; // rax
  __int64 v73; // rbx
  __int64 v74; // rdx
  __int64 v75; // rcx
  int v76; // esi
  __int64 v77; // rdi
  int v78; // esi
  unsigned int v79; // edx
  __int64 *v80; // rax
  __int64 v81; // r10
  int v82; // eax
  int v83; // r8d
  size_t v85; // r10
  unsigned __int64 v86; // rdx
  signed __int64 v87; // rax
  bool v88; // cf
  unsigned __int64 v89; // rax
  __int64 v90; // rbx
  __int64 v91; // rbx
  __int64 v92; // rax
  __int64 *v93; // rsi
  char *v94; // rcx
  char *v95; // rax
  char *v96; // rbx
  char *v97; // rdx
  __int64 v98; // rsi
  int v99; // eax
  int v100; // eax
  __int64 v101; // rdi
  int v102; // r8d
  __int64 v103; // r15
  __int64 v104; // rdi
  __int64 *v105; // rax
  _QWORD *v106; // rax
  _BYTE *v107; // rdi
  int v108; // r14d
  __int64 v109; // [rsp+0h] [rbp-7E0h]
  __int64 *v110; // [rsp+0h] [rbp-7E0h]
  __int64 v111; // [rsp+8h] [rbp-7D8h]
  size_t v112; // [rsp+8h] [rbp-7D8h]
  __int64 *v113; // [rsp+8h] [rbp-7D8h]
  char *v114; // [rsp+8h] [rbp-7D8h]
  __int64 *v115; // [rsp+8h] [rbp-7D8h]
  __int64 *v116; // [rsp+28h] [rbp-7B8h]
  __int64 v117; // [rsp+48h] [rbp-798h]
  unsigned __int64 *v118; // [rsp+50h] [rbp-790h]
  void *v119; // [rsp+50h] [rbp-790h]
  unsigned int v120; // [rsp+60h] [rbp-780h]
  _QWORD *v121; // [rsp+60h] [rbp-780h]
  unsigned int v122; // [rsp+68h] [rbp-778h]
  _QWORD *v123; // [rsp+68h] [rbp-778h]
  __int64 v124; // [rsp+68h] [rbp-778h]
  const void *v125; // [rsp+68h] [rbp-778h]
  _QWORD *v126; // [rsp+68h] [rbp-778h]
  unsigned __int64 v127; // [rsp+70h] [rbp-770h]
  __int64 v128; // [rsp+70h] [rbp-770h]
  __int64 v129; // [rsp+70h] [rbp-770h]
  __int64 v130; // [rsp+70h] [rbp-770h]
  __int64 v131; // [rsp+70h] [rbp-770h]
  __int64 v132; // [rsp+80h] [rbp-760h] BYREF
  __int64 *v133; // [rsp+88h] [rbp-758h] BYREF
  void *v134[38]; // [rsp+90h] [rbp-750h] BYREF
  void *src; // [rsp+1C0h] [rbp-620h] BYREF
  __int64 v136; // [rsp+1C8h] [rbp-618h]
  __int64 v137; // [rsp+1D0h] [rbp-610h] BYREF
  int v138; // [rsp+1D8h] [rbp-608h]
  char v139; // [rsp+1DCh] [rbp-604h]
  char v140; // [rsp+1E0h] [rbp-600h] BYREF
  _BYTE *v141; // [rsp+220h] [rbp-5C0h] BYREF
  __int64 v142; // [rsp+228h] [rbp-5B8h]
  _BYTE v143[192]; // [rsp+230h] [rbp-5B0h] BYREF
  char v144[8]; // [rsp+2F0h] [rbp-4F0h] BYREF
  __int64 v145; // [rsp+2F8h] [rbp-4E8h]
  char v146; // [rsp+30Ch] [rbp-4D4h]
  char v147[64]; // [rsp+310h] [rbp-4D0h] BYREF
  _BYTE *v148; // [rsp+350h] [rbp-490h] BYREF
  __int64 v149; // [rsp+358h] [rbp-488h]
  _BYTE v150[192]; // [rsp+360h] [rbp-480h] BYREF
  char v151[8]; // [rsp+420h] [rbp-3C0h] BYREF
  __int64 v152; // [rsp+428h] [rbp-3B8h]
  char v153; // [rsp+43Ch] [rbp-3A4h]
  char v154[64]; // [rsp+440h] [rbp-3A0h] BYREF
  _BYTE *v155; // [rsp+480h] [rbp-360h] BYREF
  __int64 v156; // [rsp+488h] [rbp-358h]
  _BYTE v157[192]; // [rsp+490h] [rbp-350h] BYREF
  __int64 v158[3]; // [rsp+550h] [rbp-290h] BYREF
  char v159; // [rsp+56Ch] [rbp-274h]
  char *v160; // [rsp+5B0h] [rbp-230h] BYREF
  unsigned int v161; // [rsp+5B8h] [rbp-228h]
  char v162; // [rsp+5C0h] [rbp-220h] BYREF
  char v163[8]; // [rsp+680h] [rbp-160h] BYREF
  __int64 v164; // [rsp+688h] [rbp-158h]
  char v165; // [rsp+69Ch] [rbp-144h]
  char *v166; // [rsp+6E0h] [rbp-100h] BYREF
  unsigned int v167; // [rsp+6E8h] [rbp-F8h]
  char v168; // [rsp+6F0h] [rbp-F0h] BYREF

  v4 = *(__int64 **)(a2 + 96);
  memset(v134, 0, sizeof(v134));
  LODWORD(v134[2]) = 8;
  v134[1] = &v134[4];
  v134[12] = &v134[14];
  v136 = (__int64)&v140;
  v116 = v4;
  BYTE4(v134[3]) = 1;
  HIDWORD(v134[13]) = 8;
  src = 0;
  v137 = 8;
  v138 = 0;
  v139 = 1;
  v141 = v143;
  v142 = 0x800000000LL;
  sub_AE6EC0((__int64)&src, (__int64)v4);
  v6 = v4[3];
  v7 = v6 + 8LL * *((unsigned int *)v4 + 8);
  if ( HIDWORD(v142) <= (unsigned int)v142 )
  {
    v4 = (__int64 *)v143;
    v131 = v7;
    v103 = sub_C8D7D0((__int64)&v141, (__int64)v143, 0, 0x18u, (unsigned __int64 *)v158, v7);
    v104 = 24LL * (unsigned int)v142;
    v105 = (__int64 *)(v104 + v103);
    if ( v104 + v103 )
    {
      v7 = v131;
      v4 = v116;
      v105[1] = v6;
      *v105 = v131;
      v105[2] = (__int64)v116;
      v104 = 24LL * (unsigned int)v142;
    }
    v106 = v141;
    v107 = &v141[v104];
    if ( v141 != v107 )
    {
      v10 = (_QWORD *)v103;
      do
      {
        if ( v10 )
        {
          *v10 = *v106;
          v10[1] = v106[1];
          v8 = v106[2];
          v10[2] = v8;
        }
        v106 += 3;
        v10 += 3;
      }
      while ( v107 != (_BYTE *)v106 );
      v107 = v141;
    }
    v108 = v158[0];
    if ( v107 != v143 )
      _libc_free(v107, v4);
    LODWORD(v142) = v142 + 1;
    v141 = (_BYTE *)v103;
    HIDWORD(v142) = v108;
  }
  else
  {
    v8 = 3LL * (unsigned int)v142;
    v9 = v142;
    v10 = &v141[24 * (unsigned int)v142];
    if ( v10 )
    {
      *v10 = v7;
      v10[1] = v6;
      v10[2] = v4;
      v9 = v142;
    }
    LODWORD(v142) = v9 + 1;
  }
  sub_D4C9F0((__int64)&src, (__int64)v4, (__int64)v10, v8, v5, v7);
  sub_D4E7F0((__int64)v151, (__int64)v134);
  sub_D4E7F0((__int64)v144, (__int64)&src);
  sub_D4E7F0((__int64)v158, (__int64)v144);
  sub_D4E7F0((__int64)v163, (__int64)v151);
  if ( v148 != v150 )
    _libc_free(v148, v151);
  if ( !v146 )
    _libc_free(v145, v151);
  if ( v155 != v157 )
    _libc_free(v155, v151);
  if ( !v153 )
    _libc_free(v152, v151);
  if ( v141 != v143 )
    _libc_free(v141, v151);
  if ( !v139 )
    _libc_free(v136, v151);
  if ( v134[12] != &v134[14] )
    _libc_free(v134[12], v151);
  if ( !BYTE4(v134[3]) )
    _libc_free(v134[1], v151);
  sub_C8CD80((__int64)v144, (__int64)v147, (__int64)v158, v11, v12, v13);
  v148 = v150;
  v149 = 0x800000000LL;
  if ( v161 )
    sub_D4E910((__int64)&v148, (__int64 *)&v160, v161, v14, v15, v16);
  sub_C8CD80((__int64)v151, (__int64)v154, (__int64)v163, v14, v15, v16);
  v20 = v167;
  v155 = v157;
  v156 = 0x800000000LL;
  if ( v167 )
  {
    sub_D4E910((__int64)&v155, (__int64 *)&v166, v17, v167, v18, v19);
    v20 = (unsigned int)v156;
  }
LABEL_25:
  v21 = (unsigned int)v149;
  while ( 1 )
  {
    v22 = (__int64)v148;
    v23 = 24 * v21;
    if ( v21 == v20 )
      break;
LABEL_30:
    v25 = *(_QWORD *)&v148[v23 - 8];
    v26 = *(_QWORD **)v25;
    src = &v137;
    v136 = 0x400000000LL;
    v27 = v26[2];
    if ( !v27 )
      goto LABEL_74;
    while ( 1 )
    {
      v23 = *(_QWORD *)(v27 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v23 - 30) <= 0xAu )
        break;
      v27 = *(_QWORD *)(v27 + 8);
      if ( !v27 )
        goto LABEL_72;
    }
LABEL_34:
    v18 = *(_QWORD *)(v23 + 40);
    if ( v18 )
    {
      v23 = (unsigned int)(*(_DWORD *)(v18 + 44) + 1);
      v28 = *(_DWORD *)(v18 + 44) + 1;
    }
    else
    {
      v23 = 0;
      v28 = 0;
    }
    if ( v28 < *(_DWORD *)(a2 + 32) )
    {
      v29 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v23);
      if ( v29 )
      {
        if ( v25 != v29 && v25 != *(_QWORD *)(v29 + 8) )
        {
          if ( v29 == *(_QWORD *)(v25 + 8) )
            goto LABEL_48;
          v22 = *(unsigned int *)(v29 + 16);
          if ( *(_DWORD *)(v25 + 16) >= (unsigned int)v22 )
            goto LABEL_48;
          if ( *(_BYTE *)(a2 + 112) )
          {
            v22 = *(unsigned int *)(v25 + 72);
            if ( *(_DWORD *)(v29 + 72) < (unsigned int)v22 || *(_DWORD *)(v29 + 76) > *(_DWORD *)(v25 + 76) )
              goto LABEL_48;
          }
          else
          {
            v22 = *(unsigned int *)(a2 + 116);
            *(_DWORD *)(a2 + 116) = v22 + 1;
            if ( (unsigned int)(v22 + 1) > 0x20 )
            {
              v124 = v29;
              v129 = v18;
              sub_B19440(a2);
              v22 = *(unsigned int *)(v25 + 72);
              if ( *(_DWORD *)(v124 + 72) < (unsigned int)v22 )
                goto LABEL_48;
              v22 = *(unsigned int *)(v25 + 76);
              v18 = v129;
              if ( *(_DWORD *)(v124 + 76) > (unsigned int)v22 )
                goto LABEL_48;
            }
            else
            {
              v20 = *(unsigned int *)(v25 + 16);
              do
              {
                v23 = v29;
                v29 = *(_QWORD *)(v29 + 8);
              }
              while ( v29 && (unsigned int)v20 <= *(_DWORD *)(v29 + 16) );
              if ( v25 != v23 )
                goto LABEL_48;
            }
          }
        }
        v30 = (unsigned int)v136;
        v20 = HIDWORD(v136);
        v31 = (unsigned int)v136 + 1LL;
        if ( v31 > HIDWORD(v136) )
        {
          v22 = (__int64)&v137;
          v130 = v18;
          sub_C8D5F0((__int64)&src, &v137, v31, 8u, v18, v19);
          v30 = (unsigned int)v136;
          v18 = v130;
        }
        v23 = (__int64)src;
        *((_QWORD *)src + v30) = v18;
        LODWORD(v136) = v136 + 1;
      }
    }
LABEL_48:
    while ( 1 )
    {
      v27 = *(_QWORD *)(v27 + 8);
      if ( !v27 )
        break;
      v23 = *(_QWORD *)(v27 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v23 - 30) <= 0xAu )
        goto LABEL_34;
    }
    if ( !(_DWORD)v136 )
      goto LABEL_72;
    *(_QWORD *)(a1 + 136) += 160LL;
    v32 = *(_QWORD *)(a1 + 56);
    v33 = (v32 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( *(_QWORD *)(a1 + 64) >= v33 + 160 && v32 )
      *(_QWORD *)(a1 + 56) = v33 + 160;
    else
      v33 = sub_9D1E70(a1 + 56, 160, 160, 3);
    v134[0] = v26;
    *(_QWORD *)v33 = 0;
    v34 = v33 + 56;
    *(_QWORD *)(v33 + 8) = 0;
    *(_QWORD *)(v33 + 16) = 0;
    *(_QWORD *)(v33 + 24) = 0;
    *(_QWORD *)(v33 + 32) = 0;
    *(_QWORD *)(v33 + 40) = 0;
    *(_QWORD *)(v33 + 48) = 0;
    *(_QWORD *)(v33 + 56) = 0;
    *(_QWORD *)(v33 + 64) = v33 + 88;
    *(_QWORD *)(v33 + 72) = 8;
    *(_DWORD *)(v33 + 80) = 0;
    *(_BYTE *)(v33 + 84) = 1;
    *(_BYTE *)(v33 + 152) = 0;
    v117 = v33 + 32;
    v127 = v33;
    sub_9319A0(v33 + 32, 0, v134);
    sub_AE6EC0(v34, (__int64)v134[0]);
    memset(v134, 0, 24);
    v35 = src;
    v36 = 8LL * (unsigned int)v136;
    if ( !v36
      || (v37 = (char *)sub_22077B0(8LL * (unsigned int)v136),
          v38 = (__int64 *)&v37[v36],
          v134[0] = v37,
          v134[2] = &v37[v36],
          v39 = (__int64 *)memcpy(v37, v35, v36),
          v134[1] = v38,
          v39 == v38) )
    {
      v52 = 0;
      goto LABEL_70;
    }
    v120 = 0;
    v122 = 0;
    do
    {
      v40 = *(_DWORD *)(a1 + 24);
      v41 = *--v38;
      v134[1] = v38;
      v42 = *(_QWORD *)(a1 + 8);
      if ( v40 )
      {
        v43 = (v40 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
        v44 = (__int64 *)(v42 + 16LL * v43);
        v45 = *v44;
        if ( v41 == *v44 )
        {
LABEL_58:
          v46 = (unsigned __int64 *)v44[1];
          if ( v46 )
          {
            v47 = v46;
            do
            {
              v48 = v47;
              v47 = (unsigned __int64 *)*v47;
            }
            while ( v47 );
            if ( v48 != (unsigned __int64 *)v127 )
            {
              v71 = v48[4];
              v72 = v48[6];
              *v48 = v127;
              ++v120;
              v122 += (__int64)(v72 - v71) >> 3;
              v73 = *(_QWORD *)(*(_QWORD *)v71 + 16LL);
              if ( v73 )
              {
                while ( 1 )
                {
                  v74 = *(_QWORD *)(v73 + 24);
                  if ( (unsigned __int8)(*(_BYTE *)v74 - 30) <= 0xAu )
                    break;
                  v73 = *(_QWORD *)(v73 + 8);
                  if ( !v73 )
                    goto LABEL_112;
                }
                v38 = (__int64 *)v134[1];
LABEL_104:
                v75 = *(_QWORD *)(v74 + 40);
                v76 = *(_DWORD *)(a1 + 24);
                v77 = *(_QWORD *)(a1 + 8);
                v133 = (__int64 *)v75;
                if ( v76 )
                {
                  v78 = v76 - 1;
                  v79 = v78 & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
                  v80 = (__int64 *)(v77 + 16LL * v79);
                  v81 = *v80;
                  if ( v75 == *v80 )
                  {
LABEL_106:
                    if ( v48 == (unsigned __int64 *)v80[1] )
                      goto LABEL_102;
                  }
                  else
                  {
                    v82 = 1;
                    while ( v81 != -4096 )
                    {
                      v83 = v82 + 1;
                      v79 = v78 & (v82 + v79);
                      v80 = (__int64 *)(v77 + 16LL * v79);
                      v81 = *v80;
                      if ( v75 == *v80 )
                        goto LABEL_106;
                      v82 = v83;
                    }
                  }
                }
                if ( v38 == v134[2] )
                {
                  v118 = v48;
                  sub_9319A0((__int64)v134, v38, &v133);
                  v38 = (__int64 *)v134[1];
                  v48 = v118;
                }
                else
                {
                  if ( v38 )
                  {
                    *v38 = v75;
                    v38 = (__int64 *)v134[1];
                  }
                  v134[1] = ++v38;
                }
LABEL_102:
                while ( 1 )
                {
                  v73 = *(_QWORD *)(v73 + 8);
                  if ( !v73 )
                    break;
                  v74 = *(_QWORD *)(v73 + 24);
                  if ( (unsigned __int8)(*(_BYTE *)v74 - 30) <= 0xAu )
                    goto LABEL_104;
                }
              }
              else
              {
LABEL_112:
                v38 = (__int64 *)v134[1];
              }
            }
LABEL_62:
            v49 = (__int64 *)v134[0];
            continue;
          }
        }
        else
        {
          v58 = 1;
          while ( v45 != -4096 )
          {
            v102 = v58 + 1;
            v43 = (v40 - 1) & (v43 + v58);
            v44 = (__int64 *)(v42 + 16LL * v43);
            v45 = *v44;
            if ( v41 == *v44 )
              goto LABEL_58;
            v58 = v102;
          }
        }
      }
      if ( v41 )
      {
        v59 = (unsigned int)(*(_DWORD *)(v41 + 44) + 1);
        v60 = *(_DWORD *)(v41 + 44) + 1;
      }
      else
      {
        v59 = 0;
        v60 = 0;
      }
      if ( v60 >= *(_DWORD *)(a2 + 32) || !*(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v59) )
        goto LABEL_62;
      v132 = v41;
      if ( !v40 )
      {
        ++*(_QWORD *)a1;
        v133 = 0;
LABEL_190:
        v40 *= 2;
LABEL_191:
        sub_D4F150(a1, v40);
        sub_D4C730(a1, &v132, &v133);
        v101 = v132;
        v62 = v133;
        v100 = *(_DWORD *)(a1 + 16) + 1;
        goto LABEL_186;
      }
      v61 = 1;
      v62 = 0;
      LODWORD(v63) = (v40 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
      v64 = (__int64 *)(v42 + 16LL * (unsigned int)v63);
      v65 = *v64;
      if ( v41 == *v64 )
      {
LABEL_87:
        v66 = (unsigned __int64 *)(v64 + 1);
        goto LABEL_88;
      }
      while ( v65 != -4096 )
      {
        if ( !v62 && v65 == -8192 )
          v62 = v64;
        v63 = (v40 - 1) & ((_DWORD)v63 + v61);
        v64 = (__int64 *)(v42 + 16 * v63);
        v65 = *v64;
        if ( v41 == *v64 )
          goto LABEL_87;
        ++v61;
      }
      v99 = *(_DWORD *)(a1 + 16);
      if ( !v62 )
        v62 = v64;
      ++*(_QWORD *)a1;
      v100 = v99 + 1;
      v133 = v62;
      if ( 4 * v100 >= 3 * v40 )
        goto LABEL_190;
      v101 = v41;
      if ( v40 - *(_DWORD *)(a1 + 20) - v100 <= v40 >> 3 )
        goto LABEL_191;
LABEL_186:
      *(_DWORD *)(a1 + 16) = v100;
      if ( *v62 != -4096 )
        --*(_DWORD *)(a1 + 20);
      *v62 = v101;
      v66 = (unsigned __int64 *)(v62 + 1);
      *v66 = 0;
LABEL_88:
      *v66 = v127;
      ++v122;
      if ( v41 == **(_QWORD **)(v127 + 32) )
      {
        v38 = (__int64 *)v134[1];
        v49 = (__int64 *)v134[0];
        continue;
      }
      v133 = *(__int64 **)(v41 + 16);
      sub_D4B000((__int64 *)&v133);
      v67 = (__int64)v133;
      v38 = (__int64 *)v134[1];
      v49 = (__int64 *)v134[0];
      if ( !v133 )
        continue;
      v68 = (__int64)v133;
      v69 = 0;
      while ( 1 )
      {
        v68 = *(_QWORD *)(v68 + 8);
        if ( !v68 )
          break;
        while ( (unsigned __int8)(**(_BYTE **)(v68 + 24) - 30) <= 0xAu )
        {
          v68 = *(_QWORD *)(v68 + 8);
          ++v69;
          if ( !v68 )
            goto LABEL_94;
        }
      }
LABEL_94:
      v70 = v69 + 1;
      if ( v70 <= ((char *)v134[2] - (char *)v134[1]) >> 3 )
      {
        do
        {
          if ( v38 )
            *v38 = *(_QWORD *)(*(_QWORD *)(v67 + 24) + 40LL);
          ++v38;
          v133 = *(__int64 **)(v67 + 8);
          sub_D4B000((__int64 *)&v133);
          v67 = (__int64)v133;
        }
        while ( v133 );
        v49 = (__int64 *)v134[0];
        v38 = (__int64 *)((char *)v134[1] + 8 * v70);
        v134[1] = v38;
        continue;
      }
      v85 = (char *)v134[1] - (char *)v134[0];
      v86 = ((char *)v134[1] - (char *)v134[0]) >> 3;
      if ( v70 > 0xFFFFFFFFFFFFFFFLL - v86 )
        sub_4262D8((__int64)"vector::_M_range_insert");
      v87 = v70;
      if ( v70 < v86 )
        v87 = ((char *)v134[1] - (char *)v134[0]) >> 3;
      v88 = __CFADD__(v86, v87);
      v89 = v86 + v87;
      if ( v88 )
      {
        v91 = 0x7FFFFFFFFFFFFFF8LL;
      }
      else
      {
        if ( !v89 )
        {
          v119 = 0;
          v93 = (__int64 *)v134[0];
          v94 = 0;
          goto LABEL_159;
        }
        v90 = 0xFFFFFFFFFFFFFFFLL;
        if ( v89 <= 0xFFFFFFFFFFFFFFFLL )
          v90 = v89;
        v91 = 8 * v90;
      }
      v111 = (__int64)v133;
      v92 = sub_22077B0(v91);
      v93 = (__int64 *)v134[0];
      v67 = v111;
      v94 = (char *)v92;
      v119 = (void *)(v91 + v92);
      v85 = (char *)v38 - (char *)v134[0];
LABEL_159:
      if ( v93 != v38 )
      {
        v109 = v67;
        v112 = v85;
        v95 = (char *)memmove(v94, v93, v85);
        v67 = v109;
        v85 = v112;
        v94 = v95;
      }
      v113 = (__int64 *)v94;
      v96 = &v94[v85];
      do
      {
        if ( v96 )
          *(_QWORD *)v96 = *(_QWORD *)(*(_QWORD *)(v67 + 24) + 40LL);
        v96 += 8;
        v133 = *(__int64 **)(v67 + 8);
        sub_D4B000((__int64 *)&v133);
        v67 = (__int64)v133;
      }
      while ( v133 );
      v49 = v113;
      v97 = (char *)((char *)v134[1] - (char *)v38);
      if ( v38 != v134[1] )
      {
        v110 = v113;
        v114 = (char *)((char *)v134[1] - (char *)v38);
        memmove(v96, v38, (size_t)v97);
        v49 = v110;
        v97 = v114;
      }
      v38 = (__int64 *)&v97[(_QWORD)v96];
      if ( v134[0] )
      {
        v115 = v49;
        j_j___libc_free_0(v134[0], (char *)v134[2] - (char *)v134[0]);
        v49 = v115;
      }
      v134[0] = v49;
      v134[1] = v38;
      v134[2] = v119;
    }
    while ( v38 != v49 );
    v50 = (_QWORD *)v127;
    v51 = *(const void **)(v127 + 8);
    v52 = v122;
    if ( v120 > (unsigned __int64)((__int64)(*(_QWORD *)(v127 + 24) - (_QWORD)v51) >> 3) )
    {
      v128 = 8LL * v120;
      v53 = v50[2] - (_QWORD)v51;
      if ( v120 )
      {
        v123 = v50;
        v54 = sub_22077B0(8LL * v120);
        v50 = v123;
        v55 = (char *)v54;
        v51 = (const void *)v123[1];
        v56 = v123[2] - (_QWORD)v51;
      }
      else
      {
        v56 = v50[2] - (_QWORD)v51;
        v55 = 0;
      }
      if ( v56 > 0 )
      {
        v121 = v50;
        v125 = v51;
        memmove(v55, v51, v56);
        v50 = v121;
        v51 = v125;
        v98 = v121[3] - (_QWORD)v125;
      }
      else
      {
        if ( !v51 )
          goto LABEL_69;
        v98 = v50[3] - (_QWORD)v51;
      }
      v126 = v50;
      j_j___libc_free_0(v51, v98);
      v50 = v126;
LABEL_69:
      v50[1] = v55;
      v50[2] = &v55[v53];
      v50[3] = &v55[v128];
    }
LABEL_70:
    v22 = v52;
    sub_D4ACD0(v117, v52);
    if ( v134[0] )
    {
      v22 = (char *)v134[2] - (char *)v134[0];
      j_j___libc_free_0(v134[0], (char *)v134[2] - (char *)v134[0]);
    }
LABEL_72:
    if ( src != &v137 )
      _libc_free(src, v22);
LABEL_74:
    v57 = (_DWORD)v149 == 1;
    v21 = (unsigned int)(v149 - 1);
    LODWORD(v149) = v149 - 1;
    if ( !v57 )
    {
      sub_D4C9F0((__int64)v144, v22, v23, v20, v18, v19);
      v20 = (unsigned int)v156;
      goto LABEL_25;
    }
    v20 = (unsigned int)v156;
  }
  v18 = (__int64)&v148[v23];
  if ( &v148[v23] != v148 )
  {
    v20 = (__int64)v155;
    v24 = v148;
    while ( v24[2] == *(_QWORD *)(v20 + 16) && v24[1] == *(_QWORD *)(v20 + 8) && *v24 == *(_QWORD *)v20 )
    {
      v24 += 3;
      v20 += 24;
      if ( (_QWORD *)v18 == v24 )
        goto LABEL_129;
    }
    goto LABEL_30;
  }
LABEL_129:
  if ( v155 != v157 )
    _libc_free(v155, v148);
  if ( !v153 )
    _libc_free(v152, v22);
  if ( v148 != v150 )
    _libc_free(v148, v22);
  if ( !v146 )
    _libc_free(v145, v22);
  if ( v166 != &v168 )
    _libc_free(v166, v22);
  if ( !v165 )
    _libc_free(v164, v22);
  if ( v160 != &v162 )
    _libc_free(v160, v22);
  if ( !v159 )
    _libc_free(v158[1], v22);
  v158[0] = a1;
  return sub_D4D490(v158, *v116);
}
