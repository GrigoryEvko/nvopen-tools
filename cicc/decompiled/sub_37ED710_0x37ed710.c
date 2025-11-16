// Function: sub_37ED710
// Address: 0x37ed710
//
void __fastcall sub_37ED710(size_t a1, __int64 a2)
{
  _DWORD *v2; // r13
  __int64 (*v3)(); // rax
  __int64 v4; // rax
  int v5; // ebx
  __int64 (*v6)(); // rax
  __int64 v7; // rax
  unsigned int v8; // eax
  int v9; // eax
  __int64 v10; // r9
  size_t v11; // r11
  int v12; // r15d
  __int64 (__fastcall *v13)(__int64); // rax
  int v14; // r13d
  __int64 v15; // r12
  __int64 v16; // r8
  int v17; // ecx
  int v18; // ecx
  unsigned __int64 v19; // rax
  int v20; // ecx
  int v21; // ecx
  __int64 v22; // rbx
  int v23; // ecx
  int v24; // ecx
  unsigned __int64 v25; // rax
  int v26; // ecx
  unsigned __int64 v27; // r11
  unsigned __int64 v28; // r11
  int v29; // eax
  unsigned int v30; // ecx
  __int64 v31; // rdx
  _DWORD *v32; // rax
  __int64 v33; // rdx
  _DWORD *i; // rdx
  __int64 v35; // r15
  __int64 v36; // r12
  _QWORD *v37; // rdi
  __int64 v38; // rax
  unsigned int v39; // eax
  __int64 v40; // rdx
  size_t v41; // r11
  __int64 v42; // r14
  __int64 v43; // rax
  __int64 v44; // r8
  __int64 v45; // r9
  size_t v46; // r11
  __int64 v47; // rdi
  __int64 (__fastcall *v48)(__int64); // rax
  int v49; // r12d
  unsigned int v50; // ebx
  size_t v51; // r13
  __int64 v52; // r9
  __int64 v53; // r13
  __int64 v54; // r12
  int v55; // ebx
  __int8 v56; // r8
  char v57; // r11
  char v58; // dl
  __int64 v59; // rdi
  int v60; // ecx
  unsigned int v61; // esi
  int *v62; // rax
  int v63; // r10d
  __int64 v64; // r10
  __int64 v65; // rsi
  int v66; // ecx
  __int64 v67; // rbx
  int v68; // eax
  int v69; // ecx
  unsigned __int64 v70; // rcx
  unsigned __int64 v71; // rdx
  int v72; // eax
  int v73; // r8d
  unsigned int m; // ecx
  __int64 *v75; // r13
  __int64 v76; // r8
  size_t v77; // rbx
  __int64 *v78; // r15
  __int64 v79; // r12
  __int64 v80; // r14
  __int64 v81; // rax
  unsigned __int64 v82; // rdx
  __int64 v83; // rcx
  __int8 v84; // r11
  __int32 v85; // ecx
  unsigned int v86; // edx
  int *v87; // rax
  int v88; // ecx
  unsigned int v89; // edx
  int v90; // ecx
  unsigned __int64 v91; // r12
  __int64 v92; // rax
  int k; // eax
  __int64 v94; // r8
  __int64 v95; // r9
  int v96; // eax
  int v97; // r10d
  __int64 v98; // r8
  int v99; // r10d
  unsigned int v100; // edi
  int v101; // ecx
  int v102; // esi
  int *v103; // rdx
  int v104; // r10d
  __int64 v105; // r8
  int v106; // r10d
  unsigned int v107; // edi
  int v108; // ecx
  int v109; // eax
  __int64 v110; // rsi
  bool v111; // zf
  _DWORD *v112; // rax
  __int64 v113; // rdx
  _DWORD *j; // rdx
  int v115; // esi
  unsigned int v116; // eax
  unsigned int v117; // ebx
  __int64 v118; // rdi
  char v119; // al
  __int64 v120; // rax
  _DWORD *v121; // rax
  __int64 v122; // rdx
  _DWORD *v123; // rdx
  int v124; // [rsp+10h] [rbp-1A0h]
  int v125; // [rsp+10h] [rbp-1A0h]
  __int64 v126; // [rsp+10h] [rbp-1A0h]
  __int64 v127; // [rsp+10h] [rbp-1A0h]
  __int64 v128; // [rsp+18h] [rbp-198h]
  size_t v129; // [rsp+28h] [rbp-188h]
  __int32 v130; // [rsp+34h] [rbp-17Ch]
  __int64 v132; // [rsp+48h] [rbp-168h]
  __int64 v133; // [rsp+48h] [rbp-168h]
  int v134; // [rsp+48h] [rbp-168h]
  __int64 v135; // [rsp+48h] [rbp-168h]
  __int64 v136; // [rsp+48h] [rbp-168h]
  int v137; // [rsp+50h] [rbp-160h]
  int v138; // [rsp+50h] [rbp-160h]
  __int64 v139; // [rsp+50h] [rbp-160h]
  int *v140; // [rsp+50h] [rbp-160h]
  unsigned __int64 v141; // [rsp+50h] [rbp-160h]
  unsigned __int64 v142; // [rsp+50h] [rbp-160h]
  __int8 v143; // [rsp+50h] [rbp-160h]
  __int8 v144; // [rsp+50h] [rbp-160h]
  unsigned int v145; // [rsp+58h] [rbp-158h]
  __int64 v146; // [rsp+58h] [rbp-158h]
  __int8 v147; // [rsp+60h] [rbp-150h]
  __int64 v148; // [rsp+60h] [rbp-150h]
  size_t v149; // [rsp+60h] [rbp-150h]
  size_t n; // [rsp+68h] [rbp-148h]
  size_t na; // [rsp+68h] [rbp-148h]
  size_t nd; // [rsp+68h] [rbp-148h]
  size_t ne; // [rsp+68h] [rbp-148h]
  unsigned int nb; // [rsp+68h] [rbp-148h]
  size_t nf; // [rsp+68h] [rbp-148h]
  size_t ng; // [rsp+68h] [rbp-148h]
  size_t nc; // [rsp+68h] [rbp-148h]
  size_t ni; // [rsp+68h] [rbp-148h]
  size_t nh; // [rsp+68h] [rbp-148h]
  size_t nj; // [rsp+68h] [rbp-148h]
  size_t nk; // [rsp+68h] [rbp-148h]
  size_t nl; // [rsp+68h] [rbp-148h]
  size_t nm; // [rsp+68h] [rbp-148h]
  __int64 v164; // [rsp+78h] [rbp-138h]
  __m128i v165; // [rsp+80h] [rbp-130h] BYREF
  _BYTE v166[20]; // [rsp+90h] [rbp-120h] BYREF
  _QWORD *v167; // [rsp+B0h] [rbp-100h] BYREF
  __int64 v168; // [rsp+B8h] [rbp-F8h]
  _QWORD v169[4]; // [rsp+C0h] [rbp-F0h] BYREF
  void *v170; // [rsp+E0h] [rbp-D0h] BYREF
  __int64 v171; // [rsp+E8h] [rbp-C8h]
  _BYTE v172[48]; // [rsp+F0h] [rbp-C0h] BYREF
  int v173; // [rsp+120h] [rbp-90h]
  void *v174; // [rsp+130h] [rbp-80h] BYREF
  __int64 v175; // [rsp+138h] [rbp-78h]
  _BYTE v176[48]; // [rsp+140h] [rbp-70h] BYREF
  int v177; // [rsp+170h] [rbp-40h]

  v2 = (_DWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  v3 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 136LL);
  if ( v3 == sub_2DD19D0 )
    BUG();
  v4 = v3();
  v5 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v4 + 352LL))(v4, a2);
  v6 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 136LL);
  if ( v6 == sub_2DD19D0 )
    BUG();
  v7 = v6();
  v8 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v7 + 360LL))(v7, a2);
  v9 = (*(__int64 (__fastcall **)(_DWORD *, _QWORD, __int64))(*(_QWORD *)v2 + 16LL))(v2, v8, 1);
  v11 = a1;
  v12 = v9;
  v13 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 24LL);
  if ( v13 == sub_2E241E0 )
  {
    v14 = v2[4];
  }
  else
  {
    v109 = ((__int64 (__fastcall *)(_DWORD *, __int64))v13)(v2, a2);
    v11 = a1;
    v14 = v109;
  }
  v15 = *(_QWORD *)(a2 + 328);
  if ( v15 != a2 + 320 )
  {
    n = v11;
    v16 = v5;
    v145 = (unsigned int)(v14 + 63) >> 6;
    do
    {
      v22 = *(_QWORD *)(n + 200) + 184LL * *(int *)(v15 + 24);
      v23 = *(_DWORD *)(v22 + 96);
      *(_QWORD *)v22 = v15;
      *(_QWORD *)(v22 + 8) = v16;
      *(_QWORD *)(v22 + 16) = v16;
      *(_DWORD *)(v22 + 24) = v12;
      *(_DWORD *)(v22 + 28) = v12;
      v24 = v23 & 0x3F;
      if ( v24 )
        *(_QWORD *)(*(_QWORD *)(v22 + 32) + 8LL * *(unsigned int *)(v22 + 40) - 8) &= ~(-1LL << v24);
      v25 = *(unsigned int *)(v22 + 40);
      *(_DWORD *)(v22 + 96) = v14;
      LOBYTE(v26) = v14;
      if ( v145 != v25 )
      {
        if ( v145 < v25 )
        {
          *(_DWORD *)(v22 + 40) = v145;
        }
        else
        {
          v27 = v145 - v25;
          if ( v145 > (unsigned __int64)*(unsigned int *)(v22 + 44) )
          {
            v135 = v16;
            v141 = v145 - v25;
            sub_C8D5F0(v22 + 32, (const void *)(v22 + 48), (unsigned int)(v14 + 63) >> 6, 8u, v16, v10);
            v25 = *(unsigned int *)(v22 + 40);
            v16 = v135;
            v27 = v141;
          }
          if ( 8 * v27 )
          {
            v132 = v16;
            v137 = v27;
            memset((void *)(*(_QWORD *)(v22 + 32) + 8 * v25), 0, 8 * v27);
            LODWORD(v25) = *(_DWORD *)(v22 + 40);
            v16 = v132;
            LODWORD(v27) = v137;
          }
          v26 = *(_DWORD *)(v22 + 96);
          *(_DWORD *)(v22 + 40) = v27 + v25;
        }
      }
      v17 = v26 & 0x3F;
      if ( v17 )
        *(_QWORD *)(*(_QWORD *)(v22 + 32) + 8LL * *(unsigned int *)(v22 + 40) - 8) &= ~(-1LL << v17);
      v18 = *(_DWORD *)(v22 + 168) & 0x3F;
      if ( v18 )
        *(_QWORD *)(*(_QWORD *)(v22 + 104) + 8LL * *(unsigned int *)(v22 + 112) - 8) &= ~(-1LL << v18);
      v19 = *(unsigned int *)(v22 + 112);
      *(_DWORD *)(v22 + 168) = v14;
      LOBYTE(v20) = v14;
      if ( v145 != v19 )
      {
        if ( v145 >= v19 )
        {
          v28 = v145 - v19;
          if ( v145 > (unsigned __int64)*(unsigned int *)(v22 + 116) )
          {
            v136 = v16;
            v142 = v145 - v19;
            sub_C8D5F0(v22 + 104, (const void *)(v22 + 120), (unsigned int)(v14 + 63) >> 6, 8u, v16, v10);
            v19 = *(unsigned int *)(v22 + 112);
            v16 = v136;
            v28 = v142;
          }
          if ( 8 * v28 )
          {
            v133 = v16;
            v138 = v28;
            memset((void *)(*(_QWORD *)(v22 + 104) + 8 * v19), 0, 8 * v28);
            LODWORD(v19) = *(_DWORD *)(v22 + 112);
            v16 = v133;
            LODWORD(v28) = v138;
          }
          v20 = *(_DWORD *)(v22 + 168);
          *(_DWORD *)(v22 + 112) = v28 + v19;
        }
        else
        {
          *(_DWORD *)(v22 + 112) = v145;
        }
      }
      v21 = v20 & 0x3F;
      if ( v21 )
        *(_QWORD *)(*(_QWORD *)(v22 + 104) + 8LL * *(unsigned int *)(v22 + 112) - 8) &= ~(-1LL << v21);
      v15 = *(_QWORD *)(v15 + 8);
    }
    while ( a2 + 320 != v15 );
    v11 = n;
  }
  ++*(_QWORD *)(v11 + 224);
  v128 = v11 + 224;
  v29 = *(_DWORD *)(v11 + 232) >> 1;
  if ( v29 )
  {
    if ( (*(_BYTE *)(v11 + 232) & 1) == 0 )
    {
      v30 = 4 * v29;
      goto LABEL_36;
    }
LABEL_148:
    v32 = (_DWORD *)(v11 + 240);
    v33 = 80;
    goto LABEL_39;
  }
  if ( *(_DWORD *)(v11 + 236) )
  {
    v30 = 0;
    if ( (*(_BYTE *)(v11 + 232) & 1) == 0 )
    {
LABEL_36:
      v31 = *(unsigned int *)(v11 + 248);
      if ( (unsigned int)v31 <= v30 || (unsigned int)v31 <= 0x40 )
      {
        v32 = *(_DWORD **)(v11 + 240);
        v33 = 5 * v31;
LABEL_39:
        for ( i = &v32[v33]; i != v32; v32 += 5 )
          *v32 = -1;
        *(_QWORD *)(v11 + 232) &= 1uLL;
        goto LABEL_42;
      }
      if ( v29 )
      {
        v116 = v29 - 1;
        if ( v116 )
        {
          _BitScanReverse(&v116, v116);
          v117 = 1 << (33 - (v116 ^ 0x1F));
          if ( v117 - 17 > 0x2E )
          {
            if ( (_DWORD)v31 == v117 )
            {
              v111 = (*(_QWORD *)(v11 + 232) & 1LL) == 0;
              *(_QWORD *)(v11 + 232) &= 1uLL;
              if ( v111 )
              {
                v121 = *(_DWORD **)(v11 + 240);
                v122 = 5 * v31;
              }
              else
              {
                v121 = (_DWORD *)(v11 + 240);
                v122 = 80;
              }
              v123 = &v121[v122];
              do
              {
                if ( v121 )
                  *v121 = -1;
                v121 += 5;
              }
              while ( v123 != v121 );
              goto LABEL_42;
            }
            nm = v11;
            sub_C7D6A0(*(_QWORD *)(v11 + 240), 20 * v31, 4);
            v11 = nm;
            v119 = *(_BYTE *)(nm + 232) | 1;
            *(_BYTE *)(nm + 232) = v119;
            if ( v117 <= 0x10 )
              goto LABEL_179;
            v118 = 20LL * v117;
          }
          else
          {
            nk = v11;
            v117 = 64;
            sub_C7D6A0(*(_QWORD *)(v11 + 240), 20 * v31, 4);
            v11 = nk;
            v118 = 1280;
            v119 = *(_BYTE *)(nk + 232);
          }
          nl = v11;
          *(_BYTE *)(v11 + 232) = v119 & 0xFE;
          v120 = sub_C7D670(v118, 4);
          v11 = nl;
          *(_QWORD *)(nl + 240) = v120;
          *(_DWORD *)(nl + 248) = v117;
LABEL_179:
          v111 = (*(_QWORD *)(v11 + 232) & 1LL) == 0;
          *(_QWORD *)(v11 + 232) &= 1uLL;
          if ( v111 )
          {
            v112 = *(_DWORD **)(v11 + 240);
            v113 = 5LL * *(unsigned int *)(v11 + 248);
          }
          else
          {
            v112 = (_DWORD *)(v11 + 240);
            v113 = 80;
          }
          for ( j = &v112[v113]; j != v112; v112 += 5 )
          {
            if ( v112 )
              *v112 = -1;
          }
          goto LABEL_42;
        }
        v110 = 20 * v31;
      }
      else
      {
        v110 = 20 * v31;
      }
      nj = v11;
      sub_C7D6A0(*(_QWORD *)(v11 + 240), v110, 4);
      v11 = nj;
      *(_BYTE *)(nj + 232) |= 1u;
      goto LABEL_179;
    }
    goto LABEL_148;
  }
LABEL_42:
  v35 = v11;
  v36 = *(_QWORD *)(v11 + 200);
  v37 = v169;
  v38 = *(_QWORD *)(v36 + 184LL * *(int *)(*(_QWORD *)(a2 + 328) + 24LL));
  v167 = v169;
  v169[0] = v38;
  v168 = 0x400000001LL;
  v39 = 1;
  while ( 1 )
  {
    v40 = v37[v39 - 1];
    LODWORD(v168) = v39 - 1;
    v41 = v36 + 184LL * *(int *)(v40 + 24);
    na = v41;
    v146 = *(_QWORD *)(v41 + 8);
    v134 = *(_DWORD *)(v41 + 24);
    v42 = *(_QWORD *)(*(_QWORD *)v41 + 32LL);
    v43 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v42 + 16) + 200LL))(*(_QWORD *)(v42 + 16));
    v46 = na;
    v47 = v43;
    v48 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v43 + 24LL);
    if ( v48 == sub_2E241E0 )
    {
      v49 = *(_DWORD *)(v47 + 16);
    }
    else
    {
      v96 = ((__int64 (__fastcall *)(__int64, __int64))v48)(v47, v42);
      v46 = na;
      v49 = v96;
    }
    v50 = (unsigned int)(v49 + 63) >> 6;
    v170 = v172;
    v171 = 0x600000000LL;
    if ( v50 > 6 )
    {
      v149 = v46;
      sub_C8D5F0((__int64)&v170, v172, v50, 8u, v44, v45);
      memset(v170, 0, 8LL * v50);
      LODWORD(v171) = (unsigned int)(v49 + 63) >> 6;
      v174 = v176;
      v175 = 0x600000000LL;
      v173 = v49;
      sub_C8D5F0((__int64)&v174, v176, v50, 8u, v94, v95);
      memset(v174, 0, 8LL * v50);
      LODWORD(v175) = (unsigned int)(v49 + 63) >> 6;
      v46 = v149;
    }
    else
    {
      if ( v50 )
      {
        v51 = 8LL * v50;
        if ( v51 )
        {
          nd = v46;
          memset(v172, 0, v51);
          v46 = nd;
        }
        LODWORD(v171) = (unsigned int)(v49 + 63) >> 6;
        v173 = v49;
        v174 = v176;
        HIDWORD(v175) = 6;
        if ( v51 )
        {
          ne = v46;
          memset(v176, 0, v51);
          v46 = ne;
        }
      }
      else
      {
        v173 = v49;
        LODWORD(v171) = 0;
        v174 = v176;
        HIDWORD(v175) = 6;
      }
      LODWORD(v175) = (unsigned int)(v49 + 63) >> 6;
    }
    v177 = v49;
    v52 = *(_QWORD *)(*(_QWORD *)v46 + 56LL);
    v53 = *(_QWORD *)v46 + 48LL;
    if ( v52 != v53 )
    {
      v129 = v46;
      while ( 1 )
      {
        if ( *(_WORD *)(v52 + 68) != 3 )
          goto LABEL_68;
        HIDWORD(v164) = 0;
        v54 = *(_QWORD *)(v42 + 360) + 104LL * *(unsigned int *)(*(_QWORD *)(v52 + 32) + 24LL);
        switch ( *(_BYTE *)(v54 + 32) )
        {
          case 3:
            v139 = *(_QWORD *)(v54 + 16);
            goto LABEL_104;
          case 5:
            v134 = *(_DWORD *)(v54 + 8);
            goto LABEL_68;
          case 6:
            goto LABEL_100;
          case 7:
            v134 = *(_DWORD *)(v54 + 8);
LABEL_100:
            v146 = *(_QWORD *)(v54 + 16);
            goto LABEL_68;
          case 8:
            v139 = *(_QWORD *)(v54 + 16) - v146;
LABEL_104:
            v147 = 0;
            v56 = 1;
            v57 = 0;
            nb = 0;
            v55 = *(_DWORD *)(v54 + 8);
            v58 = *(_BYTE *)(v35 + 232) & 1;
            if ( !v58 )
              goto LABEL_105;
            goto LABEL_57;
          case 9:
            v146 += *(_QWORD *)(v54 + 16);
            goto LABEL_68;
          case 0xB:
            *((_QWORD *)v174 + (*(_DWORD *)(v54 + 8) >> 6)) |= 1LL << *(_DWORD *)(v54 + 8);
            goto LABEL_68;
          case 0xD:
            v55 = *(_DWORD *)(v54 + 8);
            v56 = 0;
            v139 = 0;
            v147 = 1;
            v57 = 1;
            nb = *(_DWORD *)(v54 + 12);
            v58 = *(_BYTE *)(v35 + 232) & 1;
            if ( v58 )
            {
LABEL_57:
              v59 = v35 + 240;
              v60 = 15;
            }
            else
            {
LABEL_105:
              v83 = *(unsigned int *)(v35 + 248);
              v59 = *(_QWORD *)(v35 + 240);
              if ( !(_DWORD)v83 )
                goto LABEL_134;
              v60 = v83 - 1;
            }
            v61 = v60 & (37 * v55);
            v62 = (int *)(v59 + 20LL * v61);
            v63 = *v62;
            if ( v55 == *v62 )
              goto LABEL_59;
            for ( k = 1; ; k = v125 )
            {
              if ( v63 == -1 )
              {
                if ( v58 )
                {
                  v92 = 320;
                }
                else
                {
                  v83 = *(unsigned int *)(v35 + 248);
LABEL_134:
                  v92 = 20 * v83;
                }
                v62 = (int *)(v59 + v92);
                break;
              }
              v61 = v60 & (k + v61);
              v125 = k + 1;
              v62 = (int *)(v59 + 20LL * v61);
              v63 = *v62;
              if ( v55 == *v62 )
                break;
            }
LABEL_59:
            if ( v58 )
            {
              v64 = *(_QWORD *)(v35 + 224);
              LODWORD(v65) = 16;
              if ( v62 != (int *)(v59 + 320) )
              {
LABEL_61:
                if ( *((_BYTE *)v62 + 8) != v57
                  || v57 && v62[1] != nb
                  || *((_BYTE *)v62 + 16) != v56
                  || v56 && v139 != v62[3] )
                {
                  BUG();
                }
                goto LABEL_67;
              }
            }
            else
            {
              v64 = *(_QWORD *)(v35 + 224);
              v65 = *(unsigned int *)(v35 + 248);
              if ( v62 != (int *)(v59 + 20 * v65) )
                goto LABEL_61;
            }
            v84 = v56;
            v85 = v139;
            if ( !v56 )
              v85 = v130;
            v165.m128i_i8[12] = v56;
            LODWORD(v164) = nb;
            BYTE4(v164) = v147;
            v165.m128i_i32[2] = v85;
            *(_QWORD *)v166 = v164;
            v165.m128i_i64[0] = __PAIR64__(HIDWORD(v164), nb);
            v165.m128i_i8[4] = v147;
            v130 = v85;
            *(__m128i *)&v166[4] = _mm_loadu_si128(&v165);
            if ( v58 || *(_DWORD *)(v35 + 248) )
            {
              v86 = (v65 - 1) & (37 * v55);
              v87 = (int *)(v59 + 20LL * v86);
              v88 = *v87;
              if ( *v87 == v55 )
                goto LABEL_67;
              v124 = 1;
              v140 = 0;
              while ( v88 != -1 )
              {
                if ( v88 == -2 )
                {
                  if ( v140 )
                    v87 = v140;
                  v140 = v87;
                }
                v86 = (v65 - 1) & (v124 + v86);
                v87 = (int *)(v59 + 20LL * v86);
                v88 = *v87;
                if ( v55 == *v87 )
                  goto LABEL_67;
                ++v124;
              }
              v89 = *(_DWORD *)(v35 + 232);
              if ( v140 )
                v87 = v140;
              *(_QWORD *)(v35 + 224) = v64 + 1;
              v90 = (v89 >> 1) + 1;
            }
            else
            {
              v89 = *(_DWORD *)(v35 + 232);
              v87 = 0;
              *(_QWORD *)(v35 + 224) = v64 + 1;
              v90 = (v89 >> 1) + 1;
            }
            if ( 4 * v90 >= (unsigned int)(3 * v65) )
            {
              v127 = v52;
              v144 = v56;
              sub_37EBC90(v128, 2 * v65);
              v84 = v144;
              v52 = v127;
              if ( (*(_BYTE *)(v35 + 232) & 1) == 0 )
              {
                v104 = *(_DWORD *)(v35 + 248);
                v105 = *(_QWORD *)(v35 + 240);
                if ( v104 )
                {
                  v106 = v104 - 1;
                  goto LABEL_165;
                }
LABEL_214:
                *(_DWORD *)(v35 + 232) = (2 * (*(_DWORD *)(v35 + 232) >> 1) + 2) | *(_DWORD *)(v35 + 232) & 1;
                BUG();
              }
              v105 = v35 + 240;
              v106 = 15;
LABEL_165:
              v107 = v106 & (37 * v55);
              v87 = (int *)(v105 + 20LL * v107);
              v108 = *v87;
              if ( *v87 == v55 )
              {
LABEL_166:
                v89 = *(_DWORD *)(v35 + 232);
                goto LABEL_122;
              }
              v115 = 1;
              v103 = 0;
              while ( v108 != -1 )
              {
                if ( !v103 && v108 == -2 )
                  v103 = v87;
                v107 = v106 & (v115 + v107);
                v87 = (int *)(v105 + 20LL * v107);
                v108 = *v87;
                if ( v55 == *v87 )
                  goto LABEL_166;
                ++v115;
              }
LABEL_188:
              if ( v103 )
                v87 = v103;
              goto LABEL_166;
            }
            if ( (int)v65 - *(_DWORD *)(v35 + 236) - v90 <= (unsigned int)v65 >> 3 )
            {
              v126 = v52;
              v143 = v56;
              sub_37EBC90(v128, v65);
              v84 = v143;
              v52 = v126;
              if ( (*(_BYTE *)(v35 + 232) & 1) != 0 )
              {
                v98 = v35 + 240;
                v99 = 15;
              }
              else
              {
                v97 = *(_DWORD *)(v35 + 248);
                v98 = *(_QWORD *)(v35 + 240);
                if ( !v97 )
                  goto LABEL_214;
                v99 = v97 - 1;
              }
              v100 = v99 & (37 * v55);
              v87 = (int *)(v98 + 20LL * v100);
              v101 = *v87;
              if ( *v87 == v55 )
                goto LABEL_166;
              v102 = 1;
              v103 = 0;
              while ( v101 != -1 )
              {
                if ( v101 == -2 && !v103 )
                  v103 = v87;
                v100 = v99 & (v102 + v100);
                v87 = (int *)(v98 + 20LL * v100);
                v101 = *v87;
                if ( v55 == *v87 )
                  goto LABEL_166;
                ++v102;
              }
              goto LABEL_188;
            }
LABEL_122:
            *(_DWORD *)(v35 + 232) = (2 * (v89 >> 1) + 2) | v89 & 1;
            if ( *v87 != -1 )
              --*(_DWORD *)(v35 + 236);
            v166[16] = v84;
            *v87 = v55;
            *(_DWORD *)&v166[4] = nb;
            v166[8] = v147;
            *(_DWORD *)&v166[12] = v130;
            *(__m128i *)(v87 + 1) = _mm_loadu_si128((const __m128i *)&v166[4]);
LABEL_67:
            *((_QWORD *)v170 + (*(_DWORD *)(v54 + 8) >> 6)) |= 1LL << *(_DWORD *)(v54 + 8);
LABEL_68:
            if ( (*(_BYTE *)v52 & 4) == 0 )
            {
              while ( (*(_BYTE *)(v52 + 44) & 8) != 0 )
                v52 = *(_QWORD *)(v52 + 8);
            }
            v52 = *(_QWORD *)(v52 + 8);
            if ( v53 == v52 )
            {
              v46 = v129;
              goto LABEL_71;
            }
            break;
          default:
            goto LABEL_68;
        }
      }
    }
LABEL_71:
    v66 = *(_DWORD *)(v46 + 168);
    *(_BYTE *)(v46 + 176) = 1;
    v67 = v46 + 104;
    *(_QWORD *)(v46 + 16) = v146;
    *(_DWORD *)(v46 + 28) = v134;
    v68 = *(_DWORD *)(v46 + 96);
    v69 = v66 & 0x3F;
    if ( v69 )
      *(_QWORD *)(*(_QWORD *)(v46 + 104) + 8LL * *(unsigned int *)(v46 + 112) - 8) &= ~(-1LL << v69);
    v70 = *(unsigned int *)(v46 + 112);
    *(_DWORD *)(v46 + 168) = v68;
    v71 = (unsigned int)(v68 + 63) >> 6;
    if ( v71 != v70 )
    {
      if ( v71 >= v70 )
      {
        v91 = v71 - v70;
        if ( v71 > *(unsigned int *)(v46 + 116) )
        {
          nh = v46;
          sub_C8D5F0(v46 + 104, (const void *)(v46 + 120), v71, 8u, v71, v52);
          v46 = nh;
          v70 = *(unsigned int *)(nh + 112);
        }
        v71 = 8 * v91;
        if ( 8 * v91 )
        {
          ni = v46;
          memset((void *)(*(_QWORD *)(v46 + 104) + 8 * v70), 0, v71);
          v46 = ni;
          v70 = *(unsigned int *)(ni + 112);
        }
        v70 += v91;
        v68 = *(_DWORD *)(v46 + 168);
        *(_DWORD *)(v46 + 112) = v70;
      }
      else
      {
        *(_DWORD *)(v46 + 112) = v71;
      }
    }
    v72 = v68 & 0x3F;
    if ( v72 )
    {
      v71 = ~(-1LL << v72);
      *(_QWORD *)(*(_QWORD *)(v46 + 104) + 8LL * *(unsigned int *)(v46 + 112) - 8) &= v71;
      v73 = *(_DWORD *)(v46 + 40);
      if ( !v73 )
        goto LABEL_80;
    }
    else
    {
      v73 = *(_DWORD *)(v46 + 40);
      if ( !v73 )
        goto LABEL_82;
    }
    for ( m = 0; m != v73; ++m )
    {
      v71 = m;
      *(_QWORD *)(*(_QWORD *)(v46 + 104) + 8 * v71) = ~*((_QWORD *)v174 + v71)
                                                    & (*(_QWORD *)(*(_QWORD *)(v46 + 32) + 8 * v71)
                                                     | *((_QWORD *)v170 + v71));
    }
LABEL_80:
    v70 = *(_DWORD *)(v46 + 168) & 0x3F;
    if ( (*(_DWORD *)(v46 + 168) & 0x3F) != 0 )
    {
      v71 = *(_QWORD *)(v46 + 104);
      *(_QWORD *)(v71 + 8LL * *(unsigned int *)(v46 + 112) - 8) &= ~(-1LL << (*(_BYTE *)(v46 + 168) & 0x3F));
    }
LABEL_82:
    if ( v174 != v176 )
    {
      nf = v46;
      _libc_free((unsigned __int64)v174);
      v46 = nf;
    }
    if ( v170 != v172 )
    {
      ng = v46;
      _libc_free((unsigned __int64)v170);
      v46 = ng;
    }
    v75 = *(__int64 **)(*(_QWORD *)v46 + 112LL);
    if ( v75 != &v75[*(unsigned int *)(*(_QWORD *)v46 + 120LL)] )
    {
      v148 = v67;
      v76 = v35;
      v77 = v46;
      v78 = &v75[*(unsigned int *)(*(_QWORD *)v46 + 120LL)];
      do
      {
        while ( 1 )
        {
          v79 = *v75;
          v80 = *(_QWORD *)(v76 + 200) + 184LL * *(int *)(*v75 + 24);
          if ( !*(_BYTE *)(v80 + 176) )
            break;
          if ( v78 == ++v75 )
            goto LABEL_93;
        }
        nc = v76;
        *(_QWORD *)(v80 + 8) = *(_QWORD *)(v77 + 16);
        *(_DWORD *)(v80 + 24) = *(_DWORD *)(v77 + 28);
        sub_37EBA00(v80 + 32, v148, v71, v70, v76, v52);
        v70 = HIDWORD(v168);
        v76 = nc;
        *(_DWORD *)(v80 + 96) = *(_DWORD *)(v77 + 168);
        v81 = (unsigned int)v168;
        v82 = (unsigned int)v168 + 1LL;
        if ( v82 > v70 )
        {
          sub_C8D5F0((__int64)&v167, v169, v82, 8u, nc, v52);
          v81 = (unsigned int)v168;
          v76 = nc;
        }
        v71 = (unsigned __int64)v167;
        ++v75;
        v167[v81] = v79;
        LODWORD(v168) = v168 + 1;
      }
      while ( v78 != v75 );
LABEL_93:
      v35 = v76;
    }
    v39 = v168;
    v37 = v167;
    if ( !(_DWORD)v168 )
      break;
    v36 = *(_QWORD *)(v35 + 200);
  }
  if ( v167 != v169 )
    _libc_free((unsigned __int64)v167);
}
