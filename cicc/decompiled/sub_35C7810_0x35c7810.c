// Function: sub_35C7810
// Address: 0x35c7810
//
__int64 __fastcall sub_35C7810(_QWORD *a1, _BYTE *a2)
{
  __int64 *v2; // r12
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 (*v5)(); // rdx
  __int64 v6; // r9
  __int64 v7; // r15
  unsigned __int64 v8; // rax
  int v9; // edx
  char v10; // cl
  unsigned __int64 v11; // r8
  int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rdx
  __int64 v23; // r8
  _QWORD *v24; // r9
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 v27; // rdx
  _QWORD *v28; // rax
  __int64 v29; // rbx
  __int64 v30; // rsi
  __int64 v31; // rcx
  __int64 v32; // rdx
  unsigned __int64 v33; // r13
  __int64 i; // r13
  __int64 v35; // r14
  unsigned __int64 v36; // rbx
  __int64 j; // rbx
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rdx
  bool v40; // zf
  __int64 v41; // r12
  int v42; // edi
  _QWORD *v43; // rax
  char *v44; // r15
  unsigned __int64 v45; // rbx
  int v46; // eax
  unsigned int v47; // edx
  _QWORD *v48; // rcx
  __int64 v49; // r8
  int v50; // eax
  char *v51; // r13
  __int64 v52; // rdi
  unsigned int v53; // eax
  __int64 v54; // rax
  unsigned __int64 v55; // rdx
  __int64 v56; // rdx
  unsigned int v57; // r15d
  unsigned __int64 v58; // rbx
  char *v59; // r13
  char *v60; // r12
  __int64 v61; // rdi
  int v62; // eax
  char *v63; // rdx
  char v65; // al
  unsigned int v66; // r10d
  int v67; // edx
  __int64 v68; // r12
  int v69; // r11d
  _QWORD *v70; // r10
  int v71; // edx
  __int64 v72; // rax
  __int64 v73; // rdi
  int v74; // r11d
  __int64 *v75; // r13
  __int64 *v76; // r12
  __int64 v77; // rdx
  __int64 *v78; // r15
  __int64 *v79; // rax
  unsigned __int64 v80; // r13
  __int64 *v81; // rbx
  __int64 v82; // rdi
  int v83; // edx
  int v84; // r11d
  __int64 v85; // rax
  __int64 v86; // rdi
  _QWORD *v87; // r12
  __int64 v88; // rdx
  unsigned __int64 v89; // [rsp+8h] [rbp-8F8h]
  unsigned __int64 v90; // [rsp+10h] [rbp-8F0h]
  unsigned __int8 v91; // [rsp+47h] [rbp-8B9h]
  __int64 v92; // [rsp+50h] [rbp-8B0h]
  unsigned __int8 v93; // [rsp+58h] [rbp-8A8h]
  unsigned __int64 v94; // [rsp+58h] [rbp-8A8h]
  unsigned __int64 v95; // [rsp+60h] [rbp-8A0h]
  __int64 v96; // [rsp+70h] [rbp-890h]
  __int64 v97; // [rsp+78h] [rbp-888h]
  unsigned __int8 v98; // [rsp+78h] [rbp-888h]
  __int64 v99; // [rsp+80h] [rbp-880h]
  __int64 v100; // [rsp+88h] [rbp-878h]
  char *v101; // [rsp+90h] [rbp-870h] BYREF
  __int64 v102; // [rsp+98h] [rbp-868h]
  _BYTE v103[48]; // [rsp+A0h] [rbp-860h] BYREF
  __int64 v104; // [rsp+D0h] [rbp-830h] BYREF
  void *s; // [rsp+D8h] [rbp-828h] BYREF
  __int64 v106; // [rsp+E0h] [rbp-820h]
  _BYTE v107[48]; // [rsp+E8h] [rbp-818h] BYREF
  int v108; // [rsp+118h] [rbp-7E8h]
  char *v109; // [rsp+120h] [rbp-7E0h]
  __int64 v110; // [rsp+128h] [rbp-7D8h]
  char v111; // [rsp+130h] [rbp-7D0h] BYREF
  _QWORD v112[38]; // [rsp+1B0h] [rbp-750h] BYREF
  __int64 v113; // [rsp+2E0h] [rbp-620h] BYREF
  unsigned __int64 v114; // [rsp+2E8h] [rbp-618h]
  _DWORD v115[3]; // [rsp+2F0h] [rbp-610h] BYREF
  char v116; // [rsp+2FCh] [rbp-604h]
  __int64 v117; // [rsp+300h] [rbp-600h] BYREF
  __int64 *v118; // [rsp+340h] [rbp-5C0h]
  int v119; // [rsp+348h] [rbp-5B8h]
  int v120; // [rsp+34Ch] [rbp-5B4h]
  __int64 v121; // [rsp+350h] [rbp-5B0h] BYREF
  __int64 v122; // [rsp+358h] [rbp-5A8h]
  __int64 v123; // [rsp+360h] [rbp-5A0h]
  char v124[8]; // [rsp+410h] [rbp-4F0h] BYREF
  unsigned __int64 v125; // [rsp+418h] [rbp-4E8h]
  char v126; // [rsp+42Ch] [rbp-4D4h]
  char v127[64]; // [rsp+430h] [rbp-4D0h] BYREF
  _BYTE *v128; // [rsp+470h] [rbp-490h] BYREF
  __int64 v129; // [rsp+478h] [rbp-488h]
  _BYTE v130[192]; // [rsp+480h] [rbp-480h] BYREF
  char v131[8]; // [rsp+540h] [rbp-3C0h] BYREF
  unsigned __int64 v132; // [rsp+548h] [rbp-3B8h]
  char v133; // [rsp+55Ch] [rbp-3A4h]
  char v134[64]; // [rsp+560h] [rbp-3A0h] BYREF
  _BYTE *v135; // [rsp+5A0h] [rbp-360h] BYREF
  __int64 v136; // [rsp+5A8h] [rbp-358h]
  _BYTE v137[192]; // [rsp+5B0h] [rbp-350h] BYREF
  char v138[8]; // [rsp+670h] [rbp-290h] BYREF
  unsigned __int64 v139; // [rsp+678h] [rbp-288h]
  char v140; // [rsp+68Ch] [rbp-274h]
  char *v141; // [rsp+6D0h] [rbp-230h] BYREF
  int v142; // [rsp+6D8h] [rbp-228h]
  char v143; // [rsp+6E0h] [rbp-220h] BYREF
  char v144[8]; // [rsp+7A0h] [rbp-160h] BYREF
  unsigned __int64 v145; // [rsp+7A8h] [rbp-158h]
  char v146; // [rsp+7BCh] [rbp-144h]
  char *v147; // [rsp+800h] [rbp-100h] BYREF
  unsigned int v148; // [rsp+808h] [rbp-F8h]
  char v149; // [rsp+810h] [rbp-F0h] BYREF

  s = v107;
  v2 = (__int64 *)a1[2];
  v106 = 0x600000000LL;
  v3 = a1[4];
  v104 = 0;
  v108 = 0;
  v92 = v3;
  v4 = *v2;
  v96 = 0;
  v5 = *(__int64 (**)())(*v2 + 128);
  if ( v5 != sub_2DAC790 )
  {
    v96 = ((__int64 (__fastcall *)(__int64 *))v5)(v2);
    v4 = *v2;
  }
  v7 = (*(__int64 (__fastcall **)(__int64 *))(v4 + 200))(v2);
  v101 = v103;
  v102 = 0x600000000LL;
  v8 = (unsigned int)v106;
  v104 = v7;
  if ( 8LL * (unsigned int)v106 )
  {
    a2 = 0;
    memset(s, 0, 8LL * (unsigned int)v106);
    v8 = (unsigned int)v106;
  }
  v9 = *(_DWORD *)(v7 + 44);
  v10 = v108 & 0x3F;
  if ( (v108 & 0x3F) != 0 )
  {
    a2 = (_BYTE *)(-1LL << v10);
    *((_QWORD *)s + v8 - 1) &= ~(-1LL << v10);
    v8 = (unsigned int)v106;
  }
  v108 = v9;
  v11 = (unsigned int)(v9 + 63) >> 6;
  if ( v11 != v8 )
  {
    if ( v11 >= v8 )
    {
      v68 = v11 - v8;
      if ( v11 > HIDWORD(v106) )
      {
        a2 = v107;
        sub_C8D5F0((__int64)&s, v107, v11, 8u, v11, v6);
        v8 = (unsigned int)v106;
      }
      if ( 8 * v68 )
      {
        a2 = 0;
        memset((char *)s + 8 * v8, 0, 8 * v68);
        LODWORD(v8) = v106;
      }
      LOBYTE(v9) = v108;
      LODWORD(v106) = v68 + v8;
    }
    else
    {
      LODWORD(v106) = (unsigned int)(v9 + 63) >> 6;
    }
  }
  v12 = v9 & 0x3F;
  if ( v12 )
  {
    a2 = s;
    *((_QWORD *)s + (unsigned int)v106 - 1) &= ~(-1LL << v12);
  }
  v115[0] = 8;
  v109 = &v111;
  v110 = 0x1000000000LL;
  memset(v112, 0, sizeof(v112));
  LODWORD(v112[2]) = 8;
  v112[1] = &v112[4];
  v112[12] = &v112[14];
  v13 = a1[41];
  BYTE4(v112[3]) = 1;
  v117 = v13;
  HIDWORD(v112[13]) = 8;
  v114 = (unsigned __int64)&v117;
  v115[2] = 0;
  v116 = 1;
  v118 = &v121;
  v120 = 8;
  v115[1] = 1;
  v113 = 1;
  v14 = *(unsigned int *)(v13 + 120);
  v122 = *(_QWORD *)(v13 + 112);
  v123 = v13;
  v121 = v122 + 8 * v14;
  v119 = 1;
  sub_2DACB60((__int64)&v113, (__int64)a2, v122, v121, v11, v6);
  sub_2DACDE0((__int64)v131, (__int64)v112);
  sub_2DACDE0((__int64)v124, (__int64)&v113);
  sub_2DACDE0((__int64)v138, (__int64)v124);
  sub_2DACDE0((__int64)v144, (__int64)v131);
  if ( v128 != v130 )
    _libc_free((unsigned __int64)v128);
  if ( !v126 )
    _libc_free(v125);
  if ( v135 != v137 )
    _libc_free((unsigned __int64)v135);
  if ( !v133 )
    _libc_free(v132);
  if ( v118 != &v121 )
    _libc_free((unsigned __int64)v118);
  if ( !v116 )
    _libc_free(v114);
  if ( (_QWORD *)v112[12] != &v112[14] )
    _libc_free(v112[12]);
  if ( !BYTE4(v112[3]) )
    _libc_free(v112[1]);
  sub_C8CD80((__int64)v124, (__int64)v127, (__int64)v138, v15, v16, v17);
  v128 = v130;
  v129 = 0x800000000LL;
  if ( v142 )
    sub_35C72C0((__int64)&v128, (__int64 *)&v141, v18, v19, v20, v21);
  sub_C8CD80((__int64)v131, (__int64)v134, (__int64)v144, v19, v20, v21);
  v25 = v148;
  v135 = v137;
  v136 = 0x800000000LL;
  if ( v148 )
  {
    sub_35C72C0((__int64)&v135, (__int64 *)&v147, v22, v148, v23, (__int64)v24);
    v25 = (unsigned int)v136;
  }
  v91 = 0;
  v26 = (unsigned int)v129;
  while ( 1 )
  {
    v27 = 24 * v26;
    if ( v26 != v25 )
      goto LABEL_37;
    v23 = (__int64)&v128[v27];
    if ( v128 == &v128[v27] )
      break;
    v25 = (__int64)v135;
    v28 = v128;
    while ( v28[2] == *(_QWORD *)(v25 + 16) && v28[1] == *(_QWORD *)(v25 + 8) && *v28 == *(_QWORD *)v25 )
    {
      v28 += 3;
      v25 += 24;
      if ( (_QWORD *)v23 == v28 )
        goto LABEL_99;
    }
LABEL_37:
    v29 = *(_QWORD *)&v128[v27 - 8];
    LODWORD(v102) = 0;
    v30 = v29;
    sub_2E225E0(&v104, v29, v27, v25, v23, (__int64)v24);
    v99 = v29 + 48;
    v32 = *(_QWORD *)(v29 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v32 )
      BUG();
    v33 = *(_QWORD *)(v29 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( (*(_QWORD *)v33 & 4) == 0 && (*(_BYTE *)(v32 + 44) & 4) != 0 )
    {
      for ( i = *(_QWORD *)(*(_QWORD *)(v29 + 48) & 0xFFFFFFFFFFFFFFF8LL); ; i = *(_QWORD *)v33 )
      {
        v33 = i & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v33 + 44) & 4) == 0 )
          break;
      }
    }
    v35 = v7;
    if ( v99 != v33 )
    {
      while ( 1 )
      {
        v32 = *(_QWORD *)v33 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v32 )
          BUG();
        v36 = *(_QWORD *)v33 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_QWORD *)v32 & 4) == 0 && (*(_BYTE *)(v32 + 44) & 4) != 0 )
        {
          for ( j = *(_QWORD *)v32; ; j = *(_QWORD *)v36 )
          {
            v36 = j & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v36 + 44) & 4) == 0 )
              break;
          }
        }
        if ( *(_WORD *)(v33 + 68) == 43 )
        {
          if ( (*(_DWORD *)(v33 + 40) & 0xFFFFFF) != 0 && !**(_BYTE **)(v33 + 32) )
          {
            v54 = (unsigned int)v102;
            v31 = HIDWORD(v102);
            v55 = (unsigned int)v102 + 1LL;
            if ( v55 > HIDWORD(v102) )
            {
              v30 = (__int64)v103;
              sub_C8D5F0((__int64)&v101, v103, v55, 8u, v23, (__int64)v24);
              v54 = (unsigned int)v102;
            }
            v32 = (__int64)v101;
            *(_QWORD *)&v101[8 * v54] = v33;
            LODWORD(v102) = v102 + 1;
          }
          goto LABEL_53;
        }
        v38 = sub_2E8E4C0(v33, v96);
        v114 = v39;
        v23 = (unsigned __int8)v39;
        v113 = v38;
        if ( (_BYTE)v39 )
          break;
        if ( (_DWORD)v102 )
        {
          v56 = *(_QWORD *)(v33 + 32);
          v97 = v56 + 40LL * (*(_DWORD *)(v33 + 40) & 0xFFFFFF);
          if ( v56 != v97 )
          {
            v100 = *(_QWORD *)(v33 + 32);
            v95 = v36;
            v94 = v33;
            do
            {
              if ( !*(_BYTE *)v100 )
              {
                v57 = *(_DWORD *)(v100 + 8);
                v58 = (unsigned __int64)v101;
                v59 = &v101[8 * (unsigned int)v102];
                if ( v101 != v59 )
                {
                  do
                  {
                    v60 = v59;
                    v61 = *((_QWORD *)v59 - 1);
                    v59 -= 8;
                    if ( (unsigned int)sub_2E89C70(v61, v57, v35, 0) != -1 )
                    {
                      v62 = v102;
                      v63 = &v101[8 * (unsigned int)v102];
                      if ( v63 != v59 + 8 )
                      {
                        memmove(v59, v60, v63 - v60);
                        v62 = v102;
                      }
                      LODWORD(v102) = v62 - 1;
                    }
                  }
                  while ( (char *)v58 != v59 );
                }
              }
              v100 += 40;
            }
            while ( v97 != v100 );
            v36 = v95;
            v33 = v94;
          }
        }
        v30 = v33;
        sub_2E21F40(&v104, v33);
        if ( v99 == v36 )
        {
LABEL_58:
          v7 = v35;
          goto LABEL_59;
        }
LABEL_54:
        v33 = v36;
      }
      v30 = (__int64)s;
      v41 = *(unsigned int *)(*(_QWORD *)(v33 + 32) + 8LL);
      v31 = *(_DWORD *)(*(_QWORD *)(v104 + 8) + 24 * v41 + 16) & 0xFFF;
      v32 = *(_QWORD *)(v104 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v104 + 8) + 24 * v41 + 16) >> 12);
      do
      {
        if ( !v32 )
          break;
        if ( (*((_QWORD *)s + ((unsigned int)v31 >> 6)) & (1LL << v31)) != 0 )
          goto LABEL_53;
        v42 = *(__int16 *)v32;
        v32 += 2;
        v31 = (unsigned int)(v42 + v31);
      }
      while ( (_WORD)v42 );
      v31 = (unsigned int)v41;
      v32 = (unsigned int)v41 >> 6;
      v30 = *(_QWORD *)(v92 + 384);
      if ( (*(_QWORD *)(v30 + 8 * v32) & (1LL << v41)) == 0 )
      {
        v43 = &v112[2];
        v32 = (__int64)&v112[6];
        v112[0] = 0;
        v112[1] = 1;
        do
          *v43++ = -4096;
        while ( v43 != &v112[6] );
        v30 = (__int64)v101;
        v113 = (__int64)v115;
        v114 = 0x600000000LL;
        v44 = &v101[8 * (unsigned int)v102];
        if ( v101 != v44 )
        {
          v90 = v36;
          v45 = (unsigned __int64)v101;
          v93 = v23;
          v89 = v33;
          while ( 1 )
          {
            v51 = v44;
            v52 = *((_QWORD *)v44 - 1);
            v30 = (unsigned int)v41;
            v44 -= 8;
            if ( (unsigned int)sub_2E89C70(v52, v41, v35, 0) != -1 )
              break;
LABEL_76:
            if ( (char *)v45 == v44 )
            {
              v23 = v93;
              v36 = v90;
              v33 = v89;
              goto LABEL_121;
            }
          }
          v30 = v112[1] & 1;
          if ( (v112[1] & 1) != 0 )
          {
            v24 = &v112[2];
            v46 = 3;
            goto LABEL_72;
          }
          v53 = v112[3];
          v24 = (_QWORD *)v112[2];
          if ( LODWORD(v112[3]) )
          {
            v46 = LODWORD(v112[3]) - 1;
LABEL_72:
            v47 = v46 & (((unsigned int)*(_QWORD *)v44 >> 9) ^ ((unsigned int)*(_QWORD *)v44 >> 4));
            v48 = &v24[v47];
            v49 = *v48;
            if ( *v48 == *(_QWORD *)v44 )
            {
LABEL_73:
              v50 = v102;
              v32 = (__int64)&v101[8 * (unsigned int)v102];
              v31 = (__int64)(v44 + 8);
              if ( (char *)v32 != v44 + 8 )
              {
                v30 = (__int64)v51;
                memmove(v44, v51, v32 - (_QWORD)v51);
                v50 = v102;
              }
              LODWORD(v102) = v50 - 1;
              goto LABEL_76;
            }
            v69 = 1;
            v70 = 0;
            while ( v49 != -4096 )
            {
              if ( v49 == -8192 && !v70 )
                v70 = v48;
              v47 = v46 & (v69 + v47);
              v48 = &v24[v47];
              v49 = *v48;
              if ( *(_QWORD *)v44 == *v48 )
                goto LABEL_73;
              ++v69;
            }
            if ( v70 )
              v48 = v70;
            v66 = v112[1];
            ++v112[0];
            v67 = (LODWORD(v112[1]) >> 1) + 1;
            if ( (_BYTE)v30 )
            {
              v53 = 4;
              if ( (unsigned int)(4 * v67) >= 0xC )
              {
LABEL_142:
                v30 = 2 * v53;
                sub_35C7400((__int64)v112, v30);
                if ( (v112[1] & 1) != 0 )
                {
                  v24 = &v112[2];
                  v71 = 3;
                }
                else
                {
                  v24 = (_QWORD *)v112[2];
                  if ( !LODWORD(v112[3]) )
                    goto LABEL_201;
                  v71 = LODWORD(v112[3]) - 1;
                }
                v66 = v112[1];
                LODWORD(v72) = v71 & (((unsigned int)*(_QWORD *)v44 >> 9) ^ ((unsigned int)*(_QWORD *)v44 >> 4));
                v48 = &v24[(unsigned int)v72];
                v73 = *v48;
                if ( *(_QWORD *)v44 != *v48 )
                {
                  v74 = 1;
                  v30 = 0;
                  while ( v73 != -4096 )
                  {
                    if ( v73 == -8192 && !v30 )
                      v30 = (__int64)v48;
                    v72 = v71 & (unsigned int)(v72 + v74);
                    v48 = &v24[v72];
                    v73 = *v48;
                    if ( *(_QWORD *)v44 == *v48 )
                      goto LABEL_176;
                    ++v74;
                  }
                  goto LABEL_174;
                }
                goto LABEL_128;
              }
LABEL_127:
              v30 = v53 - HIDWORD(v112[1]) - v67;
              if ( (unsigned int)v30 <= v53 >> 3 )
              {
                sub_35C7400((__int64)v112, v53);
                if ( (v112[1] & 1) != 0 )
                {
                  v24 = &v112[2];
                  v83 = 3;
                }
                else
                {
                  v24 = (_QWORD *)v112[2];
                  if ( !LODWORD(v112[3]) )
                  {
LABEL_201:
                    LODWORD(v112[1]) = (2 * (LODWORD(v112[1]) >> 1) + 2) | v112[1] & 1;
                    BUG();
                  }
                  v83 = LODWORD(v112[3]) - 1;
                }
                v66 = v112[1];
                v84 = 1;
                v30 = 0;
                LODWORD(v85) = v83 & (((unsigned int)*(_QWORD *)v44 >> 9) ^ ((unsigned int)*(_QWORD *)v44 >> 4));
                v48 = &v24[(unsigned int)v85];
                v86 = *v48;
                if ( *v48 != *(_QWORD *)v44 )
                {
                  while ( v86 != -4096 )
                  {
                    if ( v86 == -8192 && !v30 )
                      v30 = (__int64)v48;
                    v85 = v83 & (unsigned int)(v85 + v84);
                    v48 = &v24[v85];
                    v86 = *v48;
                    if ( *(_QWORD *)v44 == *v48 )
                      goto LABEL_176;
                    ++v84;
                  }
LABEL_174:
                  if ( v30 )
                    v48 = (_QWORD *)v30;
LABEL_176:
                  v66 = v112[1];
                }
              }
LABEL_128:
              LODWORD(v112[1]) = (2 * (v66 >> 1) + 2) | v66 & 1;
              if ( *v48 != -4096 )
                --HIDWORD(v112[1]);
              *v48 = *(_QWORD *)v44;
              goto LABEL_73;
            }
            v53 = v112[3];
          }
          else
          {
            v66 = v112[1];
            ++v112[0];
            v48 = 0;
            v67 = (LODWORD(v112[1]) >> 1) + 1;
          }
          if ( 3 * v53 <= 4 * v67 )
            goto LABEL_142;
          goto LABEL_127;
        }
LABEL_121:
        if ( !(LODWORD(v112[1]) >> 1) )
        {
          v65 = v112[1] & 1;
          goto LABEL_123;
        }
        v98 = v23;
        sub_2E88E20(v33);
        v23 = v98;
        if ( !(LODWORD(v112[1]) >> 1) )
        {
          v65 = v112[1] & 1;
          if ( (v112[1] & 1) != 0 )
          {
            v87 = &v112[2];
            v88 = 4;
          }
          else
          {
            v87 = (_QWORD *)v112[2];
            v88 = LODWORD(v112[3]);
          }
          v76 = &v87[v88];
          v75 = v76;
          goto LABEL_158;
        }
        v65 = v112[1] & 1;
        if ( (v112[1] & 1) != 0 )
        {
          v75 = &v112[6];
          v76 = &v112[2];
          do
          {
LABEL_155:
            if ( *v76 != -4096 && *v76 != -8192 )
              break;
            ++v76;
          }
          while ( v76 != v75 );
LABEL_158:
          if ( !v65 )
          {
            v31 = v112[2];
            v77 = LODWORD(v112[3]);
            goto LABEL_160;
          }
          v31 = (__int64)&v112[2];
          v32 = 32;
        }
        else
        {
          v31 = v112[2];
          v77 = LODWORD(v112[3]);
          v75 = (__int64 *)(v112[2] + 8LL * LODWORD(v112[3]));
          v76 = (__int64 *)v112[2];
          if ( (__int64 *)v112[2] != v75 )
            goto LABEL_155;
LABEL_160:
          v32 = 8 * v77;
          v65 = 0;
        }
        v78 = (__int64 *)(v31 + v32);
        if ( (__int64 *)(v31 + v32) != v76 )
        {
          v79 = v75;
          v80 = v36;
          v81 = v79;
          do
          {
            v82 = *v76++;
            sub_2E88E20(v82);
            for ( ; v81 != v76; ++v76 )
            {
              if ( *v76 != -4096 && *v76 != -8192 )
                break;
            }
          }
          while ( v78 != v76 );
          v23 = v98;
          v36 = v80;
          v65 = v112[1] & 1;
        }
        v91 = v23;
LABEL_123:
        if ( !v65 )
        {
          v30 = 8LL * LODWORD(v112[3]);
          sub_C7D6A0(v112[2], v30, 8);
        }
      }
LABEL_53:
      if ( v99 == v36 )
        goto LABEL_58;
      goto LABEL_54;
    }
LABEL_59:
    v40 = (_DWORD)v129 == 1;
    v26 = (unsigned int)(v129 - 1);
    LODWORD(v129) = v129 - 1;
    if ( !v40 )
    {
      sub_2DACB60((__int64)v124, v30, v32, v31, v23, (__int64)v24);
      v26 = (unsigned int)v129;
    }
    v25 = (unsigned int)v136;
  }
LABEL_99:
  if ( v135 != v137 )
    _libc_free((unsigned __int64)v135);
  if ( !v133 )
    _libc_free(v132);
  if ( v128 != v130 )
    _libc_free((unsigned __int64)v128);
  if ( !v126 )
    _libc_free(v125);
  if ( v147 != &v149 )
    _libc_free((unsigned __int64)v147);
  if ( !v146 )
    _libc_free(v145);
  if ( v141 != &v143 )
    _libc_free((unsigned __int64)v141);
  if ( !v140 )
    _libc_free(v139);
  if ( v101 != v103 )
    _libc_free((unsigned __int64)v101);
  if ( s != v107 )
    _libc_free((unsigned __int64)s);
  return v91;
}
