// Function: sub_2F2B9E0
// Address: 0x2f2b9e0
//
__int64 __fastcall sub_2F2B9E0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  _QWORD *v6; // rdi
  unsigned int v8; // eax
  int *v9; // rdx
  int v10; // r13d
  int v11; // esi
  __int64 v12; // rbx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned __int64 v16; // r12
  __int64 v17; // rax
  unsigned int v18; // ebx
  int v19; // edx
  __int64 v20; // rax
  unsigned int v21; // eax
  __int64 v22; // r8
  int v23; // r15d
  unsigned int v24; // r14d
  __int64 v25; // rbx
  __int64 v26; // r13
  __int64 v27; // rdx
  __int64 v28; // rdx
  unsigned __int64 v29; // rax
  __int64 v30; // rdx
  char *v31; // rdi
  __int64 v33; // rdx
  char v34; // di
  __int64 v35; // r9
  int v36; // esi
  int v37; // r10d
  int *v38; // rcx
  unsigned int v39; // eax
  int *v40; // rdx
  int v41; // r8d
  unsigned int v42; // esi
  unsigned int v43; // eax
  __int64 v44; // rdx
  __int64 v45; // r8
  unsigned int v46; // edi
  __int64 v47; // rdx
  __int64 v48; // r8
  unsigned int v49; // ebx
  char *v50; // rax
  unsigned int v51; // edx
  int v52; // r14d
  unsigned int v53; // r13d
  __int64 v54; // rbx
  __int64 v55; // rax
  unsigned __int64 v56; // rdx
  unsigned int v57; // ecx
  __int64 v58; // rdx
  __int64 v59; // rdx
  int v60; // edx
  bool v61; // al
  unsigned int v62; // edi
  unsigned int v63; // eax
  __int64 v64; // rcx
  __int64 v65; // rsi
  __int64 v66; // rdx
  unsigned __int8 v67; // dl
  __int64 v68; // rax
  __int64 v69; // rax
  unsigned int *v70; // rax
  unsigned int v71; // edx
  __int64 v72; // rcx
  char *v73; // r8
  __int64 v74; // rdx
  char *v75; // rdx
  char *v76; // rax
  __int64 v77; // rdx
  int v78; // edx
  __int64 v79; // rsi
  int v80; // edx
  int v81; // r8d
  int *v82; // rdi
  unsigned int j; // eax
  unsigned int v84; // eax
  __int64 v85; // rax
  __int64 v86; // rsi
  int v87; // edx
  int v88; // r8d
  unsigned int k; // eax
  unsigned int v90; // eax
  unsigned int v91; // eax
  __int64 v92; // rax
  _QWORD *v93; // rdi
  int v94; // r15d
  char *v95; // r13
  unsigned int v96; // r14d
  char *v97; // rbx
  char **v98; // rax
  unsigned __int64 v99; // rdx
  char *v100; // r10
  __int64 v101; // rsi
  _QWORD *v102; // rax
  __int64 v103; // rsi
  unsigned int v104; // eax
  __int64 v105; // rdx
  unsigned __int64 v106; // rsi
  __int64 (__fastcall *v107)(__int64, __int64); // rax
  int v108; // edx
  __int64 v109; // rax
  __int64 v110; // rcx
  __int64 v111; // r8
  __int64 v112; // r9
  __int64 i; // rdi
  __int64 v114; // rcx
  __int64 v115; // rsi
  unsigned __int64 v116; // [rsp+10h] [rbp-170h]
  __int64 v117; // [rsp+10h] [rbp-170h]
  char *v118; // [rsp+10h] [rbp-170h]
  __int64 v119; // [rsp+18h] [rbp-168h]
  __int64 v120; // [rsp+18h] [rbp-168h]
  char *v121; // [rsp+18h] [rbp-168h]
  unsigned int v122; // [rsp+18h] [rbp-168h]
  int v123; // [rsp+20h] [rbp-160h]
  __int64 v124; // [rsp+30h] [rbp-150h]
  unsigned int v125; // [rsp+3Ch] [rbp-144h]
  _QWORD *v126; // [rsp+40h] [rbp-140h]
  int v127; // [rsp+48h] [rbp-138h]
  unsigned int v130; // [rsp+58h] [rbp-128h]
  __int64 v131; // [rsp+60h] [rbp-120h]
  int *v132; // [rsp+60h] [rbp-120h]
  __int64 v133; // [rsp+60h] [rbp-120h]
  __int64 v134; // [rsp+78h] [rbp-108h] BYREF
  char *v135; // [rsp+80h] [rbp-100h] BYREF
  __int128 v136; // [rsp+88h] [rbp-F8h] BYREF
  __int128 v137; // [rsp+98h] [rbp-E8h]
  _QWORD *v138; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v139; // [rsp+B8h] [rbp-C8h]
  _QWORD v140[4]; // [rsp+C0h] [rbp-C0h] BYREF
  char *v141; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v142; // [rsp+E8h] [rbp-98h]
  _BYTE v143[16]; // [rsp+F0h] [rbp-90h] BYREF
  __int64 v144; // [rsp+100h] [rbp-80h]

  v6 = v140;
  v123 = a4;
  v138 = v140;
  v140[0] = a4;
  v127 = 0;
  v139 = 0x400000001LL;
  v8 = 1;
  do
  {
    v9 = (int *)&v6[v8 - 1];
    v10 = *v9;
    v11 = v9[1];
    LODWORD(v139) = v8 - 1;
    v130 = v11;
    if ( (unsigned int)(v10 - 1) <= 0x3FFFFFFE )
      goto LABEL_32;
    v12 = *(_QWORD *)(a1 + 24);
    v124 = *(_QWORD *)(a1 + 8);
    v126 = (_QWORD *)v12;
    v16 = sub_2EBEE10(v12, v10);
    if ( v10 < 0 )
      v17 = *(_QWORD *)(*(_QWORD *)(v12 + 56) + 16LL * (v10 & 0x7FFFFFFF) + 8);
    else
      v17 = *(_QWORD *)(*(_QWORD *)(v12 + 304) + 8LL * (unsigned int)v10);
    if ( v17 )
    {
      if ( (*(_BYTE *)(v17 + 3) & 0x10) == 0 )
      {
        v17 = *(_QWORD *)(v17 + 32);
        if ( v17 )
        {
          if ( (*(_BYTE *)(v17 + 3) & 0x10) == 0 )
            BUG();
        }
      }
    }
    v18 = v11;
    v125 = -858993459 * ((v17 - *(_QWORD *)(*(_QWORD *)(v17 + 16) + 32LL)) >> 3);
    while ( 1 )
    {
      if ( !v16 )
      {
        v141 = v143;
        goto LABEL_28;
      }
      v19 = *(unsigned __int16 *)(v16 + 68);
      if ( (_WORD)v19 == 20 )
      {
        v33 = *(_QWORD *)(v16 + 32);
        if ( (*(_BYTE *)(v33 + 44) & 1) != 0 )
          goto LABEL_37;
        v57 = *(_DWORD *)(v33 + 40);
        v58 = *(unsigned int *)(v33 + 48);
        v135 = (char *)&v136 + 8;
        *(_QWORD *)&v136 = 0x200000001LL;
        *((_QWORD *)&v137 + 1) = 0;
        *((_QWORD *)&v136 + 1) = v58 | ((unsigned __int64)((v57 >> 8) & 0xFFF) << 32);
LABEL_83:
        v13 = (__int64)&v136 + 8;
        goto LABEL_84;
      }
      v20 = *(_QWORD *)(*(_QWORD *)(v16 + 16) + 24LL);
      if ( (v20 & 0x8000) != 0 )
      {
        v60 = *(_DWORD *)(v16 + 44);
        if ( (v60 & 4) == 0 && (v60 & 8) != 0 )
          v61 = sub_2E88A90(v16, 0x200000, 1);
        else
          v61 = (v20 & 0x200000) != 0;
        if ( v61 && (*(_BYTE *)(v16 + 45) & 0x40) == 0 )
          goto LABEL_37;
        if ( sub_2E8B090(v16) )
          goto LABEL_37;
        if ( *(_BYTE *)(*(_QWORD *)(v16 + 16) + 4LL) != 1 )
          goto LABEL_37;
        v62 = *(_DWORD *)(v16 + 40) & 0xFFFFFF;
        v63 = v125 + 1;
        if ( v62 == v125 + 1 )
          goto LABEL_37;
        v64 = *(_QWORD *)(v16 + 32);
        v65 = v62;
        do
        {
          v66 = v64 + 40LL * v63;
          if ( !*(_BYTE *)v66 )
          {
            if ( *(_DWORD *)(v66 + 8) )
            {
              v67 = *(_BYTE *)(v66 + 3);
              if ( (v67 & 0x20) == 0 || (((v67 & 0x10) != 0) & (v67 >> 6)) == 0 )
              {
                if ( v62 != (_DWORD)v65 )
                  goto LABEL_37;
                v65 = v63;
              }
            }
          }
          ++v63;
        }
        while ( v62 != v63 );
        if ( v62 <= (unsigned int)v65 )
          goto LABEL_37;
        v68 = *(unsigned int *)(v64 + 40LL * v125 + 8);
        if ( (int)v68 < 0 )
          v69 = *(_QWORD *)(v126[7] + 16 * (v68 & 0x7FFFFFFF) + 8);
        else
          v69 = *(_QWORD *)(v126[38] + 8 * v68);
        while ( v69 )
        {
          if ( (*(_BYTE *)(v69 + 3) & 0x10) == 0 && (*(_BYTE *)(v69 + 4) & 8) == 0 )
          {
            for ( i = *(_QWORD *)(v69 + 16); *(_WORD *)(i + 68) != 12; i = *(_QWORD *)(v69 + 16) )
            {
              do
              {
                v69 = *(_QWORD *)(v69 + 32);
                if ( !v69 )
                  goto LABEL_109;
              }
              while ( (*(_BYTE *)(v69 + 3) & 0x10) != 0 || (*(_BYTE *)(v69 + 4) & 8) != 0 || *(_QWORD *)(v69 + 16) == i );
            }
            goto LABEL_37;
          }
          v69 = *(_QWORD *)(v69 + 32);
        }
LABEL_109:
        v70 = (unsigned int *)(v64 + 40 * v65);
        if ( (v70[1] & 1) != 0 )
          goto LABEL_37;
        v71 = *v70;
        v72 = v70[2];
        v135 = (char *)&v136 + 8;
        *(_QWORD *)&v136 = 0x200000001LL;
        *((_QWORD *)&v137 + 1) = 0;
        *((_QWORD *)&v136 + 1) = v72 | ((unsigned __int64)((v71 >> 8) & 0xFFF) << 32);
        goto LABEL_83;
      }
      if ( (_BYTE)qword_5022A28 )
        goto LABEL_37;
      if ( (_WORD)v19 != 19 && (v20 & 0x200000000LL) == 0 )
      {
        if ( (_WORD)v19 != 9 && (v20 & 0x800000000LL) == 0 )
        {
          if ( (_WORD)v19 == 8 || (v20 & 0x400000000LL) != 0 )
          {
            if ( !v130 )
            {
              v141 = 0;
              LODWORD(v142) = 0;
              if ( (unsigned __int8)sub_2FE0C10(v124, v16, v125, &v141) )
              {
                if ( !HIDWORD(v141) )
                {
                  v135 = (char *)&v136 + 8;
                  *(_QWORD *)&v136 = 0x200000000LL;
                  *((_QWORD *)&v137 + 1) = 0;
                  sub_2F2B990(
                    (__int64)&v135,
                    ((unsigned __int64)(unsigned int)v142 << 32) | (unsigned int)v141,
                    (unsigned __int64)(unsigned int)v142 << 32,
                    v110,
                    v111,
                    v112);
                  v59 = (unsigned int)v136;
                  goto LABEL_118;
                }
              }
            }
          }
          else if ( (_WORD)v19 == 12 )
          {
            v114 = *(_QWORD *)(v16 + 32);
            if ( v130 == *(_QWORD *)(v114 + 144) && (*(_DWORD *)(v114 + 80) & 0xFFF00) == 0 )
            {
              v115 = *(unsigned int *)(v114 + 88);
              v135 = (char *)&v136 + 8;
              *(_QWORD *)&v136 = 0x200000000LL;
              *((_QWORD *)&v137 + 1) = 0;
              sub_2F2B990(
                (__int64)&v135,
                ((unsigned __int64)v130 << 32) | v115,
                (unsigned __int64)v130 << 32,
                v114,
                v14,
                v15);
              v59 = (unsigned int)v136;
              goto LABEL_118;
            }
          }
          else if ( !*(_WORD *)(v16 + 68) || v19 == 68 )
          {
            v144 = 0;
            v141 = v143;
            v142 = 0x200000000LL;
            v21 = *(_DWORD *)(v16 + 40) & 0xFFFFFF;
            if ( v21 <= 1 )
            {
              v135 = (char *)&v136 + 8;
              *(_QWORD *)&v136 = 0x200000000LL;
            }
            else
            {
              v22 = 40;
              v131 = a1;
              v23 = v10;
              v119 = a5;
              v24 = v18;
              v25 = 40;
              v15 = 80LL * ((v21 - 2) >> 1) + 120;
              v26 = v15;
              do
              {
                v28 = v25 + *(_QWORD *)(v16 + 32);
                if ( (*(_BYTE *)(v28 + 4) & 1) != 0 )
                {
                  v18 = v24;
                  v10 = v23;
                  v136 = 0;
                  a1 = v131;
                  a5 = v119;
                  v135 = (char *)&v136 + 8;
                  DWORD1(v136) = 2;
                  v137 = 0;
                  goto LABEL_216;
                }
                v29 = *(unsigned int *)(v28 + 8) | ((unsigned __int64)((*(_DWORD *)v28 >> 8) & 0xFFF) << 32);
                v30 = (unsigned int)v142;
                if ( (unsigned __int64)(unsigned int)v142 + 1 > HIDWORD(v142) )
                {
                  v116 = v29;
                  sub_C8D5F0((__int64)&v141, v143, (unsigned int)v142 + 1LL, 8u, v22, v15);
                  v30 = (unsigned int)v142;
                  v29 = v116;
                }
                v13 = (__int64)v141;
                v25 += 80;
                *(_QWORD *)&v141[8 * v30] = v29;
                v27 = (unsigned int)(v142 + 1);
                LODWORD(v142) = v142 + 1;
              }
              while ( v26 != v25 );
              v18 = v24;
              v10 = v23;
              a5 = v119;
              v135 = (char *)&v136 + 8;
              a1 = v131;
              *(_QWORD *)&v136 = 0x200000000LL;
              if ( (_DWORD)v27 )
                sub_2F29500((__int64)&v135, &v141, v27, v13, (__int64)&v141, v15);
            }
            *((_QWORD *)&v137 + 1) = v144;
LABEL_216:
            if ( v141 != v143 )
              _libc_free((unsigned __int64)v141);
            goto LABEL_117;
          }
          goto LABEL_37;
        }
        v134 = 0;
        v141 = 0;
        LODWORD(v142) = 0;
        if ( !(unsigned __int8)sub_2FE0C70(v124, v16, v125, &v134) )
        {
LABEL_37:
          v135 = (char *)&v136 + 8;
          v141 = v143;
          v142 = 0x200000000LL;
          v136 = 0;
          v137 = 0;
          goto LABEL_38;
        }
        if ( (_DWORD)v142 == v130 )
        {
          *((_QWORD *)&v137 + 1) = 0;
          v135 = (char *)&v136 + 8;
          *((_QWORD *)&v136 + 1) = v141;
          *(_QWORD *)&v136 = 0x200000001LL;
        }
        else
        {
          if ( (*(_QWORD *)(v126[7] + 16LL * (*(_DWORD *)(*(_QWORD *)(v16 + 32) + 40LL * v125 + 8) & 0x7FFFFFFF))
              & 0xFFFFFFFFFFFFFFF8LL) != (*(_QWORD *)(v126[7] + 16 * (v134 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL) )
            goto LABEL_37;
          if ( HIDWORD(v134) )
            goto LABEL_37;
          v85 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v126 + 16LL) + 200LL))(*(_QWORD *)(*v126 + 16LL));
          if ( (*(_OWORD *)(16LL * v130 + *(_QWORD *)(v85 + 272))
              & *(_OWORD *)(*(_QWORD *)(v85 + 272) + 16LL * (unsigned int)v142)) != 0 )
            goto LABEL_37;
          v135 = (char *)&v136 + 8;
          *(_QWORD *)&v136 = 0x200000001LL;
          *((_QWORD *)&v137 + 1) = 0;
          *((_QWORD *)&v136 + 1) = (unsigned int)v134 | ((unsigned __int64)v130 << 32);
        }
LABEL_120:
        v13 = (__int64)v135;
LABEL_84:
        v59 = *(unsigned int *)v13;
        *((_QWORD *)&v137 + 1) = v16;
        if ( (unsigned int)(v59 - 1) <= 0x3FFFFFFE )
          goto LABEL_85;
        if ( (int)v59 < 0 )
        {
          v74 = *(_QWORD *)(v126[7] + 16 * (v59 & 0x7FFFFFFF) + 8);
          if ( v74 )
            goto LABEL_129;
        }
        else
        {
          v74 = *(_QWORD *)(v126[38] + 8 * v59);
          if ( v74 )
          {
LABEL_129:
            if ( (*(_BYTE *)(v74 + 3) & 0x10) != 0
              || (v74 = *(_QWORD *)(v74 + 32)) != 0 && (*(_BYTE *)(v74 + 3) & 0x10) != 0 )
            {
              v16 = *(_QWORD *)(v74 + 16);
              v74 = (v74 - *(_QWORD *)(v16 + 32)) >> 3;
              v125 = -858993459 * v74;
              v130 = *(_DWORD *)(v13 + 4);
LABEL_131:
              v141 = v143;
              v142 = 0x200000000LL;
              sub_2F29500((__int64)&v141, &v135, v74, v13, (__int64)&v141, v15);
              v144 = *((_QWORD *)&v137 + 1);
              goto LABEL_39;
            }
          }
        }
        v16 = 0;
        goto LABEL_131;
      }
      v141 = v143;
      v142 = 0x800000000LL;
      if ( !(unsigned __int8)sub_2FE0AE0(v124, v16, v125, &v141) )
        goto LABEL_114;
      v73 = v141;
      v75 = &v141[12 * (unsigned int)v142];
      if ( v141 != v75 )
      {
        v13 = v130;
        v76 = v141;
        do
        {
          if ( *((_DWORD *)v76 + 2) == v130 )
          {
            v77 = *(_QWORD *)v76;
            *((_QWORD *)&v137 + 1) = 0;
            v135 = (char *)&v136 + 8;
            *((_QWORD *)&v136 + 1) = v77;
            *(_QWORD *)&v136 = 0x200000001LL;
            goto LABEL_115;
          }
          v76 += 12;
        }
        while ( v75 != v76 );
      }
      v92 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v126 + 16LL) + 200LL))(*(_QWORD *)(*v126 + 16LL));
      v13 = (__int64)v141;
      v93 = (_QWORD *)v92;
      v73 = &v141[12 * (unsigned int)v142];
      if ( v141 == v73 )
      {
LABEL_176:
        v136 = 0;
        v135 = (char *)&v136 + 8;
        DWORD1(v136) = 2;
        v137 = 0;
        goto LABEL_115;
      }
      v120 = a1;
      v94 = v10;
      v95 = &v141[12 * (unsigned int)v142];
      v117 = a5;
      v96 = v18;
      v97 = v141;
      while ( 1 )
      {
        v98 = (char **)(16LL * v130 + v93[34]);
        v99 = (unsigned __int64)*v98;
        v100 = v98[1];
        v101 = *((unsigned int *)v97 + 2);
        v102 = (_QWORD *)(v93[34] + 16 * v101);
        v13 = v99 & *v102;
        if ( v99 != v13 || v100 != (char *)((unsigned __int64)v100 & v102[1]) )
          goto LABEL_174;
        if ( (_DWORD)v101 )
          break;
        LODWORD(v101) = v130;
        if ( v130 )
          goto LABEL_178;
LABEL_174:
        v97 += 12;
        if ( v95 == v97 )
        {
          v18 = v96;
          v10 = v94;
          a1 = v120;
          a5 = v117;
          v73 = v141;
          goto LABEL_176;
        }
      }
      if ( v130 )
      {
        LODWORD(v101) = (*(__int64 (**)(void))(*v93 + 304LL))();
        if ( !(_DWORD)v101 )
          goto LABEL_174;
      }
LABEL_178:
      v13 = (__int64)v97;
      v15 = (unsigned int)v101;
      v18 = v96;
      v10 = v94;
      v103 = *(unsigned int *)(v13 + 4);
      a1 = v120;
      a5 = v117;
      if ( (_DWORD)v103 )
      {
        v121 = (char *)v13;
        v104 = (*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD))(*v93 + 296LL))(v93, v103, (unsigned int)v15);
        v13 = (__int64)v121;
        v15 = v104;
      }
      v105 = *(unsigned int *)v13;
      v106 = *(_QWORD *)(v126[7] + 16LL * (*(_DWORD *)v13 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
      v107 = *(__int64 (__fastcall **)(__int64, __int64))(*v93 + 272LL);
      if ( v107 == sub_2E85430 )
      {
LABEL_181:
        *((_QWORD *)&v137 + 1) = 0;
        v15 = v105 | (v15 << 32);
        v135 = (char *)&v136 + 8;
        v73 = v141;
        *((_QWORD *)&v136 + 1) = v15;
        *(_QWORD *)&v136 = 0x200000001LL;
        goto LABEL_115;
      }
      v118 = (char *)v13;
      v122 = v15;
      v109 = ((__int64 (__fastcall *)(_QWORD *, unsigned __int64, _QWORD))v107)(v93, v106, (unsigned int)v15);
      v15 = v122;
      v13 = (__int64)v118;
      if ( v106 == v109 )
      {
        v105 = *(unsigned int *)v118;
        goto LABEL_181;
      }
LABEL_114:
      v73 = v141;
      v136 = 0;
      v135 = (char *)&v136 + 8;
      DWORD1(v136) = 2;
      v137 = 0;
LABEL_115:
      if ( v73 != v143 )
        _libc_free((unsigned __int64)v73);
LABEL_117:
      v59 = (unsigned int)v136;
LABEL_118:
      if ( (int)v59 <= 0 )
      {
        v141 = v143;
        v142 = 0x200000000LL;
        if ( (_DWORD)v59 )
          goto LABEL_86;
      }
      else
      {
        if ( (_DWORD)v59 == 1 )
          goto LABEL_120;
        *((_QWORD *)&v137 + 1) = v16;
LABEL_85:
        v141 = v143;
        v142 = 0x200000000LL;
LABEL_86:
        sub_2F29500((__int64)&v141, &v135, v59, v13, (__int64)&v141, v15);
      }
LABEL_38:
      v16 = 0;
      v144 = *((_QWORD *)&v137 + 1);
LABEL_39:
      if ( v135 != (char *)&v136 + 8 )
        _libc_free((unsigned __int64)v135);
      if ( (int)v142 <= 0 )
      {
LABEL_28:
        v31 = v141;
LABEL_29:
        if ( v31 != v143 )
          _libc_free((unsigned __int64)v31);
        v6 = v138;
LABEL_32:
        LODWORD(v16) = 0;
        goto LABEL_33;
      }
      v34 = *(_BYTE *)(a5 + 8) & 1;
      if ( v34 )
      {
        v35 = a5 + 16;
        v36 = 3;
        goto LABEL_44;
      }
      v42 = *(_DWORD *)(a5 + 24);
      v35 = *(_QWORD *)(a5 + 16);
      if ( !v42 )
      {
        v43 = *(_DWORD *)(a5 + 8);
        ++*(_QWORD *)a5;
        v38 = 0;
        v44 = (v43 >> 1) + 1;
LABEL_61:
        v45 = 3 * v42;
        goto LABEL_62;
      }
      v36 = v42 - 1;
LABEL_44:
      v37 = 1;
      v38 = 0;
      v39 = v36
          & (((0xBF58476D1CE4E5B9LL * ((37 * v18) | ((unsigned __int64)(unsigned int)(37 * v10) << 32))) >> 31)
           ^ (756364221 * v18));
      while ( 2 )
      {
        v40 = (int *)(v35 + 48LL * v39);
        v41 = *v40;
        if ( *v40 == v10 && v18 == v40[1] )
        {
          if ( v40[4] <= 1 )
            goto LABEL_56;
          goto LABEL_28;
        }
        if ( v41 != -1 )
        {
          if ( v41 == -2 && v40[1] == -2 && !v38 )
            v38 = (int *)(v35 + 48LL * v39);
          goto LABEL_166;
        }
        if ( v40[1] != -1 )
        {
LABEL_166:
          v91 = v37 + v39;
          ++v37;
          v39 = v36 & v91;
          continue;
        }
        break;
      }
      v43 = *(_DWORD *)(a5 + 8);
      if ( !v38 )
        v38 = v40;
      ++*(_QWORD *)a5;
      v44 = (v43 >> 1) + 1;
      if ( !v34 )
      {
        v42 = *(_DWORD *)(a5 + 24);
        goto LABEL_61;
      }
      v45 = 12;
      v42 = 4;
LABEL_62:
      if ( (unsigned int)v45 <= 4 * (int)v44 )
      {
        sub_2F29C00(a5, 2 * v42, v44, (__int64)v38, v45, v35);
        if ( (*(_BYTE *)(a5 + 8) & 1) != 0 )
        {
          v79 = a5 + 16;
          v80 = 3;
          goto LABEL_143;
        }
        v78 = *(_DWORD *)(a5 + 24);
        v79 = *(_QWORD *)(a5 + 16);
        if ( v78 )
        {
          v80 = v78 - 1;
LABEL_143:
          v81 = 1;
          v82 = 0;
          for ( j = v80
                  & (((0xBF58476D1CE4E5B9LL * ((37 * v18) | ((unsigned __int64)(unsigned int)(37 * v10) << 32))) >> 31)
                   ^ (756364221 * v18)); ; j = v80 & v84 )
          {
            v38 = (int *)(v79 + 48LL * j);
            v35 = (unsigned int)*v38;
            if ( __PAIR64__(v18, v10) == *(_QWORD *)v38 )
              break;
            if ( (_DWORD)v35 == -1 )
            {
              if ( v38[1] == -1 )
              {
LABEL_225:
                if ( v82 )
                  v38 = v82;
                break;
              }
            }
            else if ( (_DWORD)v35 == -2 && v38[1] == -2 && !v82 )
            {
              v82 = (int *)(v79 + 48LL * j);
            }
            v84 = v81 + j;
            ++v81;
          }
LABEL_197:
          v43 = *(_DWORD *)(a5 + 8);
          goto LABEL_64;
        }
LABEL_233:
        *(_DWORD *)(a5 + 8) = (2 * (*(_DWORD *)(a5 + 8) >> 1) + 2) | *(_DWORD *)(a5 + 8) & 1;
        BUG();
      }
      v46 = v42 - *(_DWORD *)(a5 + 12) - v44;
      v47 = v42 >> 3;
      if ( v46 <= (unsigned int)v47 )
      {
        sub_2F29C00(a5, v42, v47, (__int64)v38, v45, v35);
        if ( (*(_BYTE *)(a5 + 8) & 1) != 0 )
        {
          v86 = a5 + 16;
          v87 = 3;
        }
        else
        {
          v108 = *(_DWORD *)(a5 + 24);
          v86 = *(_QWORD *)(a5 + 16);
          if ( !v108 )
            goto LABEL_233;
          v87 = v108 - 1;
        }
        v88 = 1;
        v82 = 0;
        for ( k = v87
                & (((0xBF58476D1CE4E5B9LL * ((37 * v18) | ((unsigned __int64)(unsigned int)(37 * v10) << 32))) >> 31)
                 ^ (756364221 * v18)); ; k = v87 & v90 )
        {
          v38 = (int *)(v86 + 48LL * k);
          v35 = (unsigned int)*v38;
          if ( __PAIR64__(v18, v10) == *(_QWORD *)v38 )
            break;
          if ( (_DWORD)v35 == -1 )
          {
            if ( v38[1] == -1 )
              goto LABEL_225;
          }
          else if ( (_DWORD)v35 == -2 && v38[1] == -2 && !v82 )
          {
            v82 = (int *)(v86 + 48LL * k);
          }
          v90 = v88 + k;
          ++v88;
        }
        goto LABEL_197;
      }
LABEL_64:
      *(_DWORD *)(a5 + 8) = (2 * (v43 >> 1) + 2) | v43 & 1;
      if ( *v38 != -1 || v38[1] != -1 )
        --*(_DWORD *)(a5 + 12);
      *v38 = v10;
      *((_QWORD *)v38 + 1) = v38 + 6;
      v38[1] = v18;
      *((_QWORD *)v38 + 2) = 0x200000000LL;
      if ( !(_DWORD)v142 )
      {
        *((_QWORD *)v38 + 5) = v144;
        v31 = v141;
        goto LABEL_68;
      }
      v132 = v38;
      sub_2F29420((__int64)(v38 + 2), (__int64)&v141, (unsigned int)v142, (__int64)v38, (__int64)&v141, v35);
      v49 = v142;
      *((_QWORD *)v132 + 5) = v144;
      v50 = v141;
      v31 = v141;
      if ( v49 > 1 )
        break;
LABEL_68:
      v10 = *(_DWORD *)v31;
      v18 = *((_DWORD *)v31 + 1);
      if ( (unsigned int)(*(_DWORD *)v31 - 1) <= 0x3FFFFFFE )
        goto LABEL_29;
      if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, _QWORD, unsigned __int64, _QWORD, __int64))(**(_QWORD **)(a1 + 16) + 264LL))(
             *(_QWORD *)(a1 + 16),
             a2,
             a3,
             *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 56LL) + 16LL * (v10 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
             v18,
             v35)
        && (!v127 || !v18) )
      {
        goto LABEL_56;
      }
      if ( v141 != v143 )
        _libc_free((unsigned __int64)v141);
    }
    if ( ++v127 >= (unsigned int)qword_5022868 )
      goto LABEL_29;
    v133 = a5;
    v51 = v139;
    v52 = v10;
    LODWORD(v16) = 0;
    v53 = v49;
    while ( 1 )
    {
      v54 = *(_QWORD *)&v50[8 * (int)v16];
      v55 = v51;
      v56 = v51 + 1LL;
      if ( v56 > HIDWORD(v139) )
      {
        sub_C8D5F0((__int64)&v138, v140, v56, 8u, v48, v35);
        v55 = (unsigned int)v139;
      }
      LODWORD(v16) = v16 + 1;
      v138[v55] = v54;
      v51 = v139 + 1;
      LODWORD(v139) = v139 + 1;
      if ( (unsigned int)v16 >= v53 )
        break;
      v50 = v141;
    }
    v10 = v52;
    a5 = v133;
LABEL_56:
    if ( v141 != v143 )
      _libc_free((unsigned __int64)v141);
    v8 = v139;
    v6 = v138;
  }
  while ( (_DWORD)v139 );
  LOBYTE(v16) = v10 != v123;
LABEL_33:
  if ( v6 != v140 )
    _libc_free((unsigned __int64)v6);
  return (unsigned int)v16;
}
