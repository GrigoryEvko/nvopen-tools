// Function: sub_2AFB7E0
// Address: 0x2afb7e0
//
void __fastcall sub_2AFB7E0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v8; // rbx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  char *v17; // rcx
  __int64 v18; // r8
  int v19; // eax
  char v20; // bl
  char v21; // al
  __int64 v22; // rdx
  __int64 v23; // rdx
  char v24; // cl
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  int v29; // eax
  __int64 v30; // r9
  __int64 v31; // rdx
  __int64 v32; // r13
  __int64 v33; // r12
  int v34; // ebx
  __int64 v35; // rax
  int v36; // r8d
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // r9
  __int64 v40; // rdx
  int v41; // eax
  char v42; // cl
  unsigned __int64 v43; // rax
  unsigned __int64 v44; // rax
  __int64 v45; // rax
  unsigned int v46; // edx
  __int64 v47; // r9
  __int64 v48; // r12
  __int64 v49; // rax
  __int64 v50; // r15
  int v51; // r8d
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // r9
  __int64 v55; // rdx
  char v56; // al
  __int64 v57; // rdx
  int v58; // eax
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 *v64; // r10
  __int64 *v65; // r11
  __int64 v66; // r9
  __int64 v67; // r8
  _QWORD *v68; // rdx
  __int64 v69; // rax
  __int64 v70; // r8
  char v71; // al
  __int64 v72; // rdi
  char *v73; // r10
  __int64 v74; // r11
  __int64 v75; // r13
  __int64 (__fastcall *v76)(__int64, _BYTE *, __int64, __int64); // rax
  __int64 v77; // rax
  __int64 v78; // r12
  char v79; // r13
  _QWORD *v80; // rax
  __int64 v81; // r9
  __int64 v82; // rbx
  unsigned int *v83; // r12
  __int64 v84; // r13
  __int64 v85; // rdx
  unsigned int v86; // esi
  _QWORD *v87; // rax
  __int64 v88; // r11
  unsigned int *v89; // rsi
  __int64 v90; // rdx
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // rax
  unsigned int *v94; // rbx
  __int64 v95; // r13
  __int64 v96; // rdx
  unsigned int v97; // esi
  __int64 v98; // rax
  __int64 v99; // rax
  __int64 v100; // r12
  __int64 v101; // rax
  __int64 v102; // rsi
  bool v103; // cf
  unsigned __int64 v104; // rax
  unsigned __int64 v105; // rsi
  __int64 v106; // rax
  _QWORD *v107; // r12
  __int64 v108; // r13
  unsigned __int64 *v109; // r12
  __int64 i; // rbx
  __int64 v111; // rax
  __int64 v112; // rbx
  unsigned __int64 *v113; // rbx
  unsigned __int64 v114; // rdi
  __int64 v115; // [rsp-10h] [rbp-1A0h]
  __int64 *v116; // [rsp+8h] [rbp-188h]
  __int64 *v117; // [rsp+8h] [rbp-188h]
  _QWORD *v118; // [rsp+10h] [rbp-180h]
  __int64 *v119; // [rsp+10h] [rbp-180h]
  __int64 v120; // [rsp+10h] [rbp-180h]
  __int64 *v121; // [rsp+10h] [rbp-180h]
  _QWORD *v122; // [rsp+10h] [rbp-180h]
  unsigned __int64 v123; // [rsp+18h] [rbp-178h]
  __int64 v124; // [rsp+20h] [rbp-170h]
  __int64 v125; // [rsp+20h] [rbp-170h]
  __int64 v126; // [rsp+28h] [rbp-168h]
  __int64 v127; // [rsp+28h] [rbp-168h]
  unsigned __int64 v128; // [rsp+28h] [rbp-168h]
  __int64 *v129; // [rsp+30h] [rbp-160h]
  __int64 v130; // [rsp+30h] [rbp-160h]
  __int64 *v131; // [rsp+30h] [rbp-160h]
  __int64 *v132; // [rsp+30h] [rbp-160h]
  _BOOL4 v133; // [rsp+38h] [rbp-158h]
  __int64 v134; // [rsp+40h] [rbp-150h]
  __int64 v135; // [rsp+48h] [rbp-148h]
  char v136; // [rsp+48h] [rbp-148h]
  __int64 v137; // [rsp+48h] [rbp-148h]
  char *v138; // [rsp+48h] [rbp-148h]
  __int64 v139; // [rsp+48h] [rbp-148h]
  unsigned int *v140; // [rsp+48h] [rbp-148h]
  char *v141; // [rsp+48h] [rbp-148h]
  __int64 *v142; // [rsp+50h] [rbp-140h]
  __int64 *v143; // [rsp+50h] [rbp-140h]
  __int64 v144; // [rsp+58h] [rbp-138h]
  __int64 v145; // [rsp+58h] [rbp-138h]
  __int64 v146; // [rsp+60h] [rbp-130h]
  int v147; // [rsp+60h] [rbp-130h]
  __int64 *v148; // [rsp+60h] [rbp-130h]
  __int64 *v149; // [rsp+60h] [rbp-130h]
  __int64 *v150; // [rsp+60h] [rbp-130h]
  __int64 v151; // [rsp+60h] [rbp-130h]
  __int64 *v152; // [rsp+60h] [rbp-130h]
  __int64 *v153; // [rsp+60h] [rbp-130h]
  _QWORD *v154; // [rsp+60h] [rbp-130h]
  _QWORD *v155; // [rsp+60h] [rbp-130h]
  _QWORD *v156; // [rsp+60h] [rbp-130h]
  __int64 v158; // [rsp+68h] [rbp-128h]
  unsigned int *v159; // [rsp+68h] [rbp-128h]
  __int64 v160; // [rsp+68h] [rbp-128h]
  char v161; // [rsp+68h] [rbp-128h]
  __int64 v162; // [rsp+68h] [rbp-128h]
  _QWORD v163[4]; // [rsp+70h] [rbp-120h] BYREF
  __int16 v164; // [rsp+90h] [rbp-100h]
  _QWORD v165[4]; // [rsp+A0h] [rbp-F0h] BYREF
  __int16 v166; // [rsp+C0h] [rbp-D0h]
  _BYTE v167[32]; // [rsp+D0h] [rbp-C0h] BYREF
  __int16 v168; // [rsp+F0h] [rbp-A0h]
  char *v169; // [rsp+100h] [rbp-90h] BYREF
  __int64 v170; // [rsp+108h] [rbp-88h]
  _BYTE v171[16]; // [rsp+110h] [rbp-80h] BYREF
  char *v172; // [rsp+120h] [rbp-70h] BYREF
  __int64 v173; // [rsp+128h] [rbp-68h]
  _BYTE v174[32]; // [rsp+130h] [rbp-60h] BYREF
  unsigned __int64 v175; // [rsp+150h] [rbp-40h]

  v5 = a1;
  v8 = a3;
  v9 = *(unsigned __int8 *)(a3 + 8);
  if ( (unsigned __int8)v9 > 3u && (_BYTE)v9 != 5 )
  {
    if ( (unsigned __int8)v9 > 0x14u )
      goto LABEL_152;
    v23 = 1463376;
    if ( !_bittest64(&v23, v9) )
    {
      if ( (_BYTE)v9 == 16 )
      {
        v24 = -1;
        v25 = *(unsigned int *)(a1 + 108) | (unsigned __int64)(1LL << *(_BYTE *)(a1 + 104));
        v136 = *(_BYTE *)(a1 + 104);
        v26 = v25 & -(__int64)v25;
        if ( v26 )
        {
          _BitScanReverse64(&v26, v26);
          v24 = 63 - (v26 ^ 0x3F);
        }
        *(_BYTE *)(a1 + 104) = v24;
        v27 = sub_9208B0(*(_QWORD *)a1, *(_QWORD *)(v8 + 24));
        v170 = v28;
        v169 = (char *)((unsigned __int64)(v27 + 7) >> 3);
        v29 = sub_CA1930(&v169);
        v31 = *(_QWORD *)(v8 + 32);
        v147 = v29;
        if ( (_DWORD)v31 )
        {
          v142 = a4;
          v144 = (unsigned int)v31;
          v32 = v8;
          v33 = 0;
          v34 = 0;
          v35 = *(unsigned int *)(a1 + 16);
          do
          {
            v36 = v33;
            if ( v35 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 20) )
            {
              sub_C8D5F0(a1 + 8, (const void *)(a1 + 24), v35 + 1, 4u, (unsigned int)v33, v30);
              v35 = *(unsigned int *)(a1 + 16);
              v36 = v33;
            }
            *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4 * v35) = v36;
            ++*(_DWORD *)(a1 + 16);
            v37 = sub_BCB2D0(*(_QWORD **)(a2 + 72));
            v38 = sub_ACD640(v37, v33, 0);
            v40 = *(unsigned int *)(a1 + 48);
            if ( v40 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
            {
              v126 = v38;
              sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v40 + 1, 8u, v40 + 1, v39);
              v40 = *(unsigned int *)(a1 + 48);
              v38 = v126;
            }
            ++v33;
            *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v40) = v38;
            ++*(_DWORD *)(a1 + 48);
            *(_DWORD *)(a1 + 108) = v34;
            sub_2AFB7E0(a1, a2, *(_QWORD *)(v32 + 24), v142, a5);
            v41 = *(_DWORD *)(a1 + 16);
            --*(_DWORD *)(a1 + 48);
            v34 += v147;
            v35 = (unsigned int)(v41 - 1);
            *(_DWORD *)(a1 + 16) = v35;
          }
          while ( v144 != v33 );
        }
LABEL_29:
        *(_BYTE *)(v5 + 104) = v136;
        return;
      }
      if ( (_BYTE)v9 == 15 )
      {
        v42 = -1;
        v43 = *(unsigned int *)(a1 + 108) | (unsigned __int64)(1LL << *(_BYTE *)(a1 + 104));
        v136 = *(_BYTE *)(a1 + 104);
        v44 = v43 & -(__int64)v43;
        if ( v44 )
        {
          _BitScanReverse64(&v44, v44);
          v42 = 63 - (v44 ^ 0x3F);
        }
        *(_BYTE *)(a1 + 104) = v42;
        v45 = sub_AE4AC0(*(_QWORD *)a1, v8);
        v46 = *(_DWORD *)(v8 + 12);
        if ( v46 )
        {
          v47 = v45 + 24;
          v48 = 0;
          v145 = v46;
          v49 = *(unsigned int *)(a1 + 16);
          v50 = v47;
          do
          {
            v51 = v48;
            if ( v49 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 20) )
            {
              sub_C8D5F0(a1 + 8, (const void *)(a1 + 24), v49 + 1, 4u, (unsigned int)v48, v47);
              v49 = *(unsigned int *)(a1 + 16);
              v51 = v48;
            }
            *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4 * v49) = v51;
            ++*(_DWORD *)(a1 + 16);
            v52 = sub_BCB2D0(*(_QWORD **)(a2 + 72));
            v53 = sub_ACD640(v52, v48, 0);
            v55 = *(unsigned int *)(a1 + 48);
            if ( v55 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
            {
              v127 = v53;
              sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v55 + 1, 8u, v55 + 1, v54);
              v55 = *(unsigned int *)(a1 + 48);
              v53 = v127;
            }
            v50 += 16;
            *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v55) = v53;
            ++*(_DWORD *)(a1 + 48);
            v56 = *(_BYTE *)(v50 - 8);
            v169 = *(char **)(v50 - 16);
            LOBYTE(v170) = v56;
            *(_DWORD *)(a1 + 108) = sub_CA1930(&v169);
            v57 = *(_QWORD *)(*(_QWORD *)(v8 + 16) + 8 * v48++);
            sub_2AFB7E0(a1, a2, v57, a4, a5);
            v58 = *(_DWORD *)(a1 + 16);
            --*(_DWORD *)(a1 + 48);
            v49 = (unsigned int)(v58 - 1);
            *(_DWORD *)(a1 + 16) = v49;
          }
          while ( v145 != v48 );
          v5 = a1;
        }
        goto LABEL_29;
      }
LABEL_152:
      BUG();
    }
  }
  v10 = *(unsigned int *)(a1 + 108) | (unsigned __int64)(1LL << *(_BYTE *)(a1 + 104));
  v11 = v10 & -(__int64)v10;
  if ( v11 )
  {
    _BitScanReverse64(&v12, v11);
    v11 = 0x8000000000000000LL >> ((unsigned __int8)v12 ^ 0x3Fu);
  }
  if ( !byte_500F108
    || v8 != sub_BCB2B0(*(_QWORD **)(a2 + 72)) && v8 != sub_BCB2C0(*(_QWORD **)(a2 + 72))
    || (v8 == sub_BCB2B0(*(_QWORD **)(a2 + 72))
      ? (v59 = 0x2E8BA2E8BA2E8BA3LL * ((__int64)(*(_QWORD *)(a1 + 120) - *(_QWORD *)(a1 + 112)) >> 3))
      : (v59 = 0x2E8BA2E8BA2E8BA3LL * ((__int64)(*(_QWORD *)(a1 + 144) - *(_QWORD *)(a1 + 136)) >> 3)),
        !v59 && (v11 & 3) != 0) )
  {
    sub_2AFB210(a1, a2, a4, a5, (__int64 *)(a1 + 112), 0);
    sub_2AFB210(a1, a2, a4, a5, (__int64 *)(a1 + 136), 1);
    v169 = v171;
    v170 = 0x400000000LL;
    if ( *(_DWORD *)(a1 + 16) )
      sub_2AF7210((__int64)&v169, a1 + 8, v13, v14, v15, v16);
    v17 = v174;
    v18 = 0;
    v173 = 0x400000000LL;
    v19 = *(_DWORD *)(a1 + 48);
    v172 = v174;
    if ( v19 )
    {
      sub_2AF7130((__int64)&v172, a1 + 40, v13, (__int64)v174, 0, v16);
      v17 = v172;
      v18 = (unsigned int)v173;
    }
    v175 = v11;
    v20 = -1;
    if ( v11 )
    {
      _BitScanReverse64(&v11, v11);
      v20 = 63 - (v11 ^ 0x3F);
    }
    v21 = *(_BYTE *)(a5 + 32);
    if ( v21 )
    {
      if ( v21 == 1 )
      {
        v163[0] = ".gep";
        v164 = 259;
      }
      else
      {
        if ( *(_BYTE *)(a5 + 33) == 1 )
        {
          v22 = *(_QWORD *)a5;
          v146 = *(_QWORD *)(a5 + 8);
        }
        else
        {
          v22 = a5;
          v21 = 2;
        }
        v163[0] = v22;
        LOBYTE(v164) = v21;
        v163[1] = v146;
        v163[2] = ".gep";
        HIBYTE(v164) = 3;
      }
    }
    else
    {
      v164 = 256;
    }
    v151 = sub_921130(
             (unsigned int **)a2,
             *(_QWORD *)(a1 + 96),
             *(_QWORD *)(a1 + 88),
             (_BYTE **)v17,
             v18,
             (__int64)v163,
             3u);
    v71 = *(_BYTE *)(a5 + 32);
    if ( v71 )
    {
      if ( v71 == 1 )
      {
        v165[0] = ".extract";
        v166 = 259;
      }
      else
      {
        if ( *(_BYTE *)(a5 + 33) == 1 )
        {
          v135 = *(_QWORD *)(a5 + 8);
          a5 = *(_QWORD *)a5;
        }
        else
        {
          v71 = 2;
        }
        LOBYTE(v166) = v71;
        HIBYTE(v166) = 3;
        v165[0] = a5;
        v165[1] = v135;
        v165[2] = ".extract";
      }
    }
    else
    {
      v70 = 256;
      v166 = 256;
    }
    v72 = *(_QWORD *)(a2 + 80);
    v73 = v169;
    v74 = (unsigned int)v170;
    v75 = *a4;
    v76 = *(__int64 (__fastcall **)(__int64, _BYTE *, __int64, __int64))(*(_QWORD *)v72 + 80LL);
    if ( v76 == sub_92FAE0 )
    {
      if ( *(_BYTE *)v75 > 0x15u )
        goto LABEL_94;
      v138 = v169;
      v158 = (unsigned int)v170;
      v77 = sub_AAADB0(v75, (unsigned int *)v169, (unsigned int)v170);
      v74 = v158;
      v73 = v138;
      v78 = v77;
    }
    else
    {
      v141 = v169;
      v162 = (unsigned int)v170;
      v98 = ((__int64 (__fastcall *)(__int64, __int64, char *, _QWORD, __int64, __int64))v76)(
              v72,
              v75,
              v169,
              (unsigned int)v170,
              v70,
              v115);
      v73 = v141;
      v74 = v162;
      v78 = v98;
    }
    if ( v78 )
    {
LABEL_83:
      v168 = 257;
      v79 = v20;
      v80 = sub_BD2C40(80, unk_3F10A10);
      v82 = (__int64)v80;
      if ( v80 )
        sub_B4D3C0((__int64)v80, v78, v151, 0, v79, v81, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v82,
        v167,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v83 = *(unsigned int **)a2;
      v84 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v84 )
      {
        do
        {
          v85 = *((_QWORD *)v83 + 1);
          v86 = *v83;
          v83 += 4;
          sub_B99FD0(v82, v86, v85);
        }
        while ( (unsigned int *)v84 != v83 );
      }
      if ( v172 != v174 )
        _libc_free((unsigned __int64)v172);
      if ( v169 != v171 )
        _libc_free((unsigned __int64)v169);
      return;
    }
LABEL_94:
    v139 = v74;
    v168 = 257;
    v159 = (unsigned int *)v73;
    v87 = sub_BD2C40(104, 1u);
    v88 = v139;
    v78 = (__int64)v87;
    if ( v87 )
    {
      v89 = v159;
      v90 = v139;
      v140 = v159;
      v160 = v88;
      v91 = sub_B501B0(*(_QWORD *)(v75 + 8), v89, v90);
      sub_B44260(v78, v91, 64, 1u, 0, 0);
      if ( *(_QWORD *)(v78 - 32) )
      {
        v92 = *(_QWORD *)(v78 - 24);
        **(_QWORD **)(v78 - 16) = v92;
        if ( v92 )
          *(_QWORD *)(v92 + 16) = *(_QWORD *)(v78 - 16);
      }
      *(_QWORD *)(v78 - 32) = v75;
      v93 = *(_QWORD *)(v75 + 16);
      *(_QWORD *)(v78 - 24) = v93;
      if ( v93 )
        *(_QWORD *)(v93 + 16) = v78 - 24;
      *(_QWORD *)(v78 - 16) = v75 + 16;
      *(_QWORD *)(v75 + 16) = v78 - 32;
      *(_QWORD *)(v78 + 72) = v78 + 88;
      *(_QWORD *)(v78 + 80) = 0x400000000LL;
      sub_B50030(v78, v140, v160, (__int64)v167);
    }
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v78,
      v165,
      *(_QWORD *)(a2 + 56),
      *(_QWORD *)(a2 + 64));
    if ( *(_QWORD *)a2 != *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8) )
    {
      v161 = v20;
      v94 = *(unsigned int **)a2;
      v95 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      do
      {
        v96 = *((_QWORD *)v94 + 1);
        v97 = *v94;
        v94 += 4;
        sub_B99FD0(v78, v97, v96);
      }
      while ( (unsigned int *)v95 != v94 );
      v20 = v161;
    }
    goto LABEL_83;
  }
  v134 = sub_BCB2B0(*(_QWORD **)(a2 + 72));
  v133 = v8 != v134;
  if ( v8 == v134 )
  {
    sub_2AFB210(a1, a2, a4, a5, (__int64 *)(a1 + 136), 1);
    v60 = sub_AE54E0(*(_QWORD *)a1, *(_QWORD *)(*a4 + 8), *(_QWORD **)(a1 + 40), *(unsigned int *)(a1 + 48));
    v64 = (__int64 *)(a1 + 112);
    v137 = v60;
    if ( !*(_BYTE *)(a1 + 184) )
    {
LABEL_52:
      v65 = v64;
      goto LABEL_53;
    }
    v65 = (__int64 *)(a1 + 112);
    if ( *(_QWORD *)(a1 + 176) + 1LL == v60 )
      goto LABEL_53;
LABEL_51:
    v148 = v64;
    sub_2AFB210(a1, a2, a4, a5, v64, v133);
    v64 = v148;
    goto LABEL_52;
  }
  sub_2AFB210(a1, a2, a4, a5, (__int64 *)(a1 + 112), 0);
  v99 = sub_AE54E0(*(_QWORD *)a1, *(_QWORD *)(*a4 + 8), *(_QWORD **)(a1 + 40), *(unsigned int *)(a1 + 48));
  v65 = (__int64 *)(a1 + 136);
  v137 = v99;
  if ( *(_BYTE *)(a1 + 184) )
  {
    v64 = (__int64 *)(a1 + 136);
    if ( v99 != *(_QWORD *)(a1 + 176) + 2LL )
      goto LABEL_51;
  }
LABEL_53:
  v66 = *(unsigned int *)(a1 + 16);
  v169 = v171;
  v170 = 0x400000000LL;
  if ( (_DWORD)v66 )
  {
    v143 = v65;
    sub_2AF7210((__int64)&v169, a1 + 8, v61, v62, v63, v66);
    v65 = v143;
  }
  v67 = *(unsigned int *)(a1 + 48);
  v172 = v174;
  v173 = 0x400000000LL;
  if ( (_DWORD)v67 )
  {
    v129 = v65;
    sub_2AF7130((__int64)&v172, a1 + 40, v61, v62, v67, v66);
    v65 = v129;
  }
  v175 = v11;
  v68 = (_QWORD *)v65[1];
  if ( v68 != (_QWORD *)v65[2] )
  {
    if ( v68 )
    {
      *v68 = v68 + 2;
      v68[1] = 0x400000000LL;
      if ( (_DWORD)v170 )
      {
        v132 = v65;
        v155 = v68;
        sub_2AF72F0((__int64)v68, &v169, (__int64)v68, v62, v67, v66);
        v65 = v132;
        v68 = v155;
      }
      v68[4] = v68 + 6;
      v68[5] = 0x400000000LL;
      if ( (_DWORD)v173 )
      {
        v131 = v65;
        v154 = v68;
        sub_2AF6CF0((__int64)(v68 + 4), &v172, (__int64)v68, v62, v67, v66);
        v65 = v131;
        v68 = v154;
      }
      v68[10] = v175;
      v68 = (_QWORD *)v65[1];
    }
    v65[1] = (__int64)(v68 + 11);
    goto LABEL_65;
  }
  v100 = (__int64)v68 - *v65;
  v128 = *v65;
  v101 = 0x2E8BA2E8BA2E8BA3LL * (v100 >> 3);
  if ( v101 == 0x1745D1745D1745DLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v102 = 1;
  if ( v101 )
    v102 = 0x2E8BA2E8BA2E8BA3LL * (v100 >> 3);
  v103 = __CFADD__(v102, v101);
  v104 = v102 + v101;
  if ( v103 )
  {
    v105 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v104 )
    {
      v124 = 88;
      v123 = 0;
      v130 = 0;
      goto LABEL_120;
    }
    if ( v104 > 0x1745D1745D1745DLL )
      v104 = 0x1745D1745D1745DLL;
    v105 = 88 * v104;
  }
  v116 = v65;
  v118 = (_QWORD *)v65[1];
  v106 = sub_22077B0(v105);
  v68 = v118;
  v130 = v106;
  v65 = v116;
  v123 = v106 + v105;
  v124 = v106 + 88;
LABEL_120:
  v107 = (_QWORD *)(v130 + v100);
  if ( v107 )
  {
    v62 = (unsigned int)v170;
    *v107 = v107 + 2;
    v107[1] = 0x400000000LL;
    if ( (_DWORD)v62 )
    {
      v117 = v65;
      v122 = v68;
      sub_2AF72F0((__int64)v107, &v169, (__int64)v68, v62, v67, v66);
      v65 = v117;
      v68 = v122;
    }
    v107[4] = v107 + 6;
    v107[5] = 0x400000000LL;
    if ( (_DWORD)v173 )
    {
      v121 = v65;
      v156 = v68;
      sub_2AF6CF0((__int64)(v107 + 4), &v172, (__int64)v68, v62, v67, v66);
      v65 = v121;
      v68 = v156;
    }
    v107[10] = v175;
  }
  if ( v68 != (_QWORD *)v128 )
  {
    v125 = v8;
    v119 = a4;
    v108 = v128;
    v109 = v68;
    v152 = v65;
    for ( i = v130; ; i += 88 )
    {
      if ( i )
      {
        *(_DWORD *)(i + 8) = 0;
        *(_QWORD *)i = i + 16;
        *(_DWORD *)(i + 12) = 4;
        if ( *(_DWORD *)(v108 + 8) )
          sub_2AF7210(i, v108, (__int64)v68, v62, v67, v66);
        *(_DWORD *)(i + 40) = 0;
        *(_QWORD *)(i + 32) = i + 48;
        *(_DWORD *)(i + 44) = 4;
        if ( *(_DWORD *)(v108 + 40) )
          sub_2AF7130(i + 32, v108 + 32, (__int64)v68, v62, v67, v66);
        *(_QWORD *)(i + 80) = *(_QWORD *)(v108 + 80);
      }
      v108 += 88;
      if ( v109 == (unsigned __int64 *)v108 )
        break;
    }
    v111 = i;
    v112 = v125;
    a4 = v119;
    v124 = v111 + 176;
    v120 = v112;
    v113 = (unsigned __int64 *)v128;
    do
    {
      v114 = v113[4];
      if ( (unsigned __int64 *)v114 != v113 + 6 )
        _libc_free(v114);
      if ( (unsigned __int64 *)*v113 != v113 + 2 )
        _libc_free(*v113);
      v113 += 11;
    }
    while ( v109 != v113 );
    v65 = v152;
    v8 = v120;
  }
  if ( v128 )
  {
    v153 = v65;
    j_j___libc_free_0(v128);
    v65 = v153;
  }
  *v65 = v130;
  v65[1] = v124;
  v65[2] = v123;
LABEL_65:
  if ( v172 != v174 )
  {
    v149 = v65;
    _libc_free((unsigned __int64)v172);
    v65 = v149;
  }
  if ( v169 != v171 )
  {
    v150 = v65;
    _libc_free((unsigned __int64)v169);
    v65 = v150;
  }
  *(_BYTE *)(v5 + 184) = 1;
  *(_QWORD *)(v5 + 176) = v137;
  v69 = 0x2E8BA2E8BA2E8BA3LL * ((v65[1] - *v65) >> 3);
  if ( v8 == v134 )
  {
    if ( v69 == *(_QWORD *)(v5 + 160) )
      goto LABEL_71;
  }
  else if ( v69 == *(_QWORD *)(v5 + 168) )
  {
LABEL_71:
    sub_2AFB210(v5, a2, a4, a5, v65, v133);
  }
}
