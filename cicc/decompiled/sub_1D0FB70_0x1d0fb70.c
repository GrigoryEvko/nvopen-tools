// Function: sub_1D0FB70
// Address: 0x1d0fb70
//
void __fastcall sub_1D0FB70(__int64 a1, __int64 a2)
{
  unsigned int *v2; // rax
  __int64 v3; // r15
  int v4; // r14d
  char v5; // dl
  __int64 v6; // rdi
  __int64 (*v7)(); // rax
  char v8; // r12
  unsigned __int64 v9; // rbx
  unsigned __int64 *v10; // rax
  unsigned __int64 *v11; // rsi
  unsigned __int64 *v12; // rcx
  void *v13; // rdi
  signed __int64 v14; // rax
  unsigned __int64 v15; // r13
  unsigned int v16; // esi
  unsigned int v17; // edi
  __int64 v18; // rcx
  unsigned __int64 v19; // r8
  unsigned int v20; // r9d
  signed __int64 *v21; // rdx
  __int64 v22; // r10
  unsigned int v23; // r9d
  unsigned __int64 *v24; // rax
  unsigned __int64 v25; // r10
  __int64 v26; // rax
  __int64 *v27; // r12
  __int64 v28; // rbx
  __int64 *v29; // r13
  unsigned __int64 v30; // rax
  __int64 *v31; // rbx
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 *v34; // rax
  __int64 *v35; // rsi
  __int64 *v36; // rsi
  __int64 v37; // r10
  __int64 v38; // r8
  int v39; // ebx
  unsigned int v40; // edi
  _QWORD *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 *v44; // rax
  int v45; // r14d
  unsigned int v46; // r12d
  unsigned int v47; // r15d
  __int64 v48; // rbx
  unsigned int v49; // edi
  _QWORD *v50; // rax
  __int64 v51; // rcx
  __int64 v52; // r13
  __int64 v53; // rdi
  __int64 (*v54)(); // rax
  int v55; // r8d
  int v56; // r9d
  __int64 v57; // rax
  int v58; // r10d
  _QWORD *v59; // rdx
  int v60; // eax
  unsigned __int64 v61; // r12
  __int64 v62; // r14
  unsigned int v63; // r15d
  unsigned int v64; // ebx
  __int64 v65; // r13
  unsigned int v66; // esi
  __int64 v67; // rdi
  int v68; // r11d
  _QWORD *v69; // r8
  _QWORD *v70; // rdi
  unsigned int v71; // r13d
  int v72; // r10d
  __int64 v73; // rsi
  unsigned __int64 *v74; // rdx
  int v75; // eax
  int v76; // r11d
  signed __int64 *v77; // r14
  int v78; // edx
  __int64 v79; // rax
  __int64 v80; // rcx
  int v81; // edi
  unsigned __int64 *v82; // rsi
  __int64 v83; // rcx
  int v84; // edi
  signed __int64 *v85; // rsi
  int v86; // esi
  signed __int64 *v87; // rcx
  signed __int64 v88; // rdi
  unsigned __int64 *v89; // rcx
  int v90; // esi
  unsigned __int64 v91; // rdi
  int v92; // r10d
  _QWORD *v93; // r9
  int v94; // edx
  __int64 *v95; // r11
  int v96; // edi
  _QWORD *v97; // rsi
  __int64 v98; // rcx
  unsigned int v99; // ecx
  int v100; // edi
  int v101; // [rsp+Ch] [rbp-1A4h]
  char v102; // [rsp+20h] [rbp-190h]
  __int64 v103; // [rsp+20h] [rbp-190h]
  signed __int64 v104; // [rsp+20h] [rbp-190h]
  signed __int64 v105; // [rsp+20h] [rbp-190h]
  signed __int64 v106; // [rsp+28h] [rbp-188h]
  __int64 v108; // [rsp+30h] [rbp-180h]
  int v109; // [rsp+30h] [rbp-180h]
  signed __int64 v111; // [rsp+48h] [rbp-168h] BYREF
  __int64 v112; // [rsp+50h] [rbp-160h] BYREF
  __int64 v113; // [rsp+58h] [rbp-158h]
  __int64 v114; // [rsp+60h] [rbp-150h]
  unsigned int v115; // [rsp+68h] [rbp-148h]
  void *src; // [rsp+70h] [rbp-140h] BYREF
  __int64 v117; // [rsp+78h] [rbp-138h]
  _BYTE v118[32]; // [rsp+80h] [rbp-130h] BYREF
  __int64 *v119; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v120; // [rsp+A8h] [rbp-108h]
  _BYTE v121[32]; // [rsp+B0h] [rbp-100h] BYREF
  __int64 v122; // [rsp+D0h] [rbp-E0h] BYREF
  unsigned __int64 *v123; // [rsp+D8h] [rbp-D8h]
  unsigned __int64 *v124; // [rsp+E0h] [rbp-D0h]
  __int64 v125; // [rsp+E8h] [rbp-C8h]
  int v126; // [rsp+F0h] [rbp-C0h]
  _BYTE v127[184]; // [rsp+F8h] [rbp-B8h] BYREF

  v2 = (unsigned int *)(*(_QWORD *)(a2 + 32) + 40LL * (unsigned int)(*(_DWORD *)(a2 + 56) - 1));
  if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v2 + 40LL) + 16LL * v2[2]) != 1 )
    return;
  v3 = *(_QWORD *)(*(_QWORD *)v2 + 48LL);
  v122 = 0;
  v123 = (unsigned __int64 *)v127;
  v124 = (unsigned __int64 *)v127;
  src = v118;
  v117 = 0x400000000LL;
  v125 = 16;
  v126 = 0;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v106 = a2;
  v102 = 0;
  if ( !v3 )
    goto LABEL_21;
  v4 = 0;
  do
  {
    v9 = *(_QWORD *)(v3 + 16);
    if ( a2 == v9 )
    {
LABEL_6:
      v8 = (unsigned int)++v4 <= 0x63;
      goto LABEL_7;
    }
    v10 = v123;
    if ( v124 == v123 )
    {
      v11 = &v123[HIDWORD(v125)];
      if ( v123 != v11 )
      {
        v12 = 0;
        do
        {
          if ( v9 == *v10 )
            goto LABEL_6;
          if ( *v10 == -2 )
            v12 = v10;
          ++v10;
        }
        while ( v11 != v10 );
        if ( v12 )
        {
          *v12 = v9;
          --v126;
          ++v122;
          goto LABEL_5;
        }
      }
      if ( HIDWORD(v125) < (unsigned int)v125 )
      {
        ++HIDWORD(v125);
        *v11 = v9;
        ++v122;
        goto LABEL_5;
      }
    }
    sub_16CCBA0((__int64)&v122, *(_QWORD *)(v3 + 16));
    if ( !v5 )
      goto LABEL_6;
LABEL_5:
    v6 = *(_QWORD *)(a1 + 16);
    v7 = *(__int64 (**)())(*(_QWORD *)v6 + 576LL);
    if ( v7 == sub_1D0B160 )
      goto LABEL_6;
    v8 = ((__int64 (__fastcall *)(__int64, signed __int64, unsigned __int64, signed __int64 *, __int64 **))v7)(
           v6,
           v106,
           v9,
           &v111,
           &v119);
    if ( !v8 )
      goto LABEL_6;
    v14 = v111;
    v15 = (unsigned __int64)v119;
    if ( (__int64 *)v111 == v119 )
      goto LABEL_6;
    v16 = v115;
    if ( v115 )
    {
      v17 = v115 - 1;
      v18 = v113;
      LODWORD(v19) = 37 * v111;
      v20 = (v115 - 1) & (37 * v111);
      v21 = (signed __int64 *)(v113 + 16LL * v20);
      v22 = *v21;
      if ( v111 == *v21 )
        goto LABEL_30;
      v76 = 1;
      v77 = 0;
      while ( v22 != 0x7FFFFFFFFFFFFFFFLL )
      {
        if ( v22 == 0x8000000000000000LL && !v77 )
          v77 = v21;
        v20 = v17 & (v76 + v20);
        LODWORD(v19) = v76 + 1;
        v21 = (signed __int64 *)(v113 + 16LL * v20);
        v22 = *v21;
        if ( v111 == *v21 )
          goto LABEL_30;
        ++v76;
      }
      LODWORD(v19) = 37 * v111;
      if ( !v77 )
        v77 = v21;
      ++v112;
      v78 = v114 + 1;
      if ( 4 * ((int)v114 + 1) < 3 * v115 )
      {
        if ( v115 - HIDWORD(v114) - v78 <= v115 >> 3 )
        {
          v101 = 37 * v111;
          v105 = v111;
          sub_1D0F970((__int64)&v112, v115);
          if ( !v115 )
          {
LABEL_239:
            LODWORD(v114) = v114 + 1;
            BUG();
          }
          v86 = 1;
          v87 = 0;
          LODWORD(v19) = (v115 - 1) & v101;
          v77 = (signed __int64 *)(v113 + 16LL * (unsigned int)v19);
          v78 = v114 + 1;
          v14 = v105;
          v88 = *v77;
          if ( v105 != *v77 )
          {
            while ( 1 )
            {
              v20 = -1;
              if ( v88 == 0x7FFFFFFFFFFFFFFFLL )
                break;
              if ( v88 == 0x8000000000000000LL && !v87 )
                v87 = v77;
              v20 = v86 + 1;
              v19 = (v115 - 1) & ((_DWORD)v19 + v86);
              v77 = (signed __int64 *)(v113 + 16 * v19);
              v88 = *v77;
              if ( v105 == *v77 )
                goto LABEL_120;
              ++v86;
            }
            if ( v87 )
              v77 = v87;
          }
        }
        goto LABEL_120;
      }
    }
    else
    {
      ++v112;
    }
    v104 = v111;
    sub_1D0F970((__int64)&v112, 2 * v115);
    if ( !v115 )
      goto LABEL_239;
    v14 = v104;
    LODWORD(v83) = (v115 - 1) & (37 * v104);
    v77 = (signed __int64 *)(v113 + 16LL * (unsigned int)v83);
    v78 = v114 + 1;
    v19 = *v77;
    if ( v104 != *v77 )
    {
      v84 = 1;
      v85 = 0;
      while ( 1 )
      {
        v20 = -1;
        if ( v19 == 0x7FFFFFFFFFFFFFFFLL )
          break;
        if ( v19 == 0x8000000000000000LL && !v85 )
          v85 = v77;
        v20 = v84 + 1;
        v83 = (v115 - 1) & ((_DWORD)v83 + v84);
        v77 = (signed __int64 *)(v113 + 16 * v83);
        v19 = *v77;
        if ( v104 == *v77 )
          goto LABEL_120;
        ++v84;
      }
      if ( v85 )
        v77 = v85;
    }
LABEL_120:
    LODWORD(v114) = v78;
    if ( *v77 != 0x7FFFFFFFFFFFFFFFLL )
      --HIDWORD(v114);
    *v77 = v14;
    v77[1] = v106;
    v79 = (unsigned int)v117;
    if ( (unsigned int)v117 >= HIDWORD(v117) )
    {
      sub_16CD150((__int64)&src, v118, 0, 8, v19, v20);
      v79 = (unsigned int)v117;
    }
    *((_QWORD *)src + v79) = v111;
    v16 = v115;
    LODWORD(v117) = v117 + 1;
    v15 = (unsigned __int64)v119;
    v18 = v113;
    if ( !v115 )
    {
      ++v112;
      goto LABEL_126;
    }
    v17 = v115 - 1;
LABEL_30:
    v23 = v17 & (37 * v15);
    v24 = (unsigned __int64 *)(v18 + 16LL * v23);
    v25 = *v24;
    if ( v15 != *v24 )
    {
      LODWORD(v19) = 1;
      v74 = 0;
      while ( v25 != 0x7FFFFFFFFFFFFFFFLL )
      {
        if ( v25 == 0x8000000000000000LL && !v74 )
          v74 = v24;
        v23 = v17 & (v19 + v23);
        v24 = (unsigned __int64 *)(v18 + 16LL * v23);
        v25 = *v24;
        if ( v15 == *v24 )
          goto LABEL_31;
        LODWORD(v19) = v19 + 1;
      }
      if ( !v74 )
        v74 = v24;
      ++v112;
      v75 = v114 + 1;
      if ( 4 * ((int)v114 + 1) < 3 * v16 )
      {
        if ( v16 - (v75 + HIDWORD(v114)) <= v16 >> 3 )
        {
          sub_1D0F970((__int64)&v112, v16);
          if ( !v115 )
          {
LABEL_238:
            LODWORD(v114) = v114 + 1;
            BUG();
          }
          v89 = 0;
          LODWORD(v19) = (v115 - 1) & (37 * v15);
          v90 = 1;
          v75 = v114 + 1;
          v74 = (unsigned __int64 *)(v113 + 16LL * (unsigned int)v19);
          v91 = *v74;
          if ( *v74 != v15 )
          {
            while ( 1 )
            {
              v23 = -1;
              if ( v91 == 0x7FFFFFFFFFFFFFFFLL )
                break;
              if ( !v89 && v91 == 0x8000000000000000LL )
                v89 = v74;
              v23 = v90 + 1;
              v19 = (v115 - 1) & ((_DWORD)v19 + v90);
              v74 = (unsigned __int64 *)(v113 + 16 * v19);
              v91 = *v74;
              if ( v15 == *v74 )
                goto LABEL_111;
              ++v90;
            }
            if ( v89 )
              v74 = v89;
          }
        }
        goto LABEL_111;
      }
LABEL_126:
      sub_1D0F970((__int64)&v112, 2 * v16);
      if ( !v115 )
        goto LABEL_238;
      LODWORD(v80) = (v115 - 1) & (37 * v15);
      v75 = v114 + 1;
      v74 = (unsigned __int64 *)(v113 + 16LL * (unsigned int)v80);
      v19 = *v74;
      if ( *v74 != v15 )
      {
        v81 = 1;
        v82 = 0;
        while ( 1 )
        {
          v23 = -1;
          if ( v19 == 0x7FFFFFFFFFFFFFFFLL )
            break;
          if ( !v82 && v19 == 0x8000000000000000LL )
            v82 = v74;
          v23 = v81 + 1;
          v80 = (v115 - 1) & ((_DWORD)v80 + v81);
          v74 = (unsigned __int64 *)(v113 + 16 * v80);
          v19 = *v74;
          if ( *v74 == v15 )
            goto LABEL_111;
          ++v81;
        }
        if ( v82 )
          v74 = v82;
      }
LABEL_111:
      LODWORD(v114) = v75;
      if ( *v74 != 0x7FFFFFFFFFFFFFFFLL )
        --HIDWORD(v114);
      *v74 = v15;
      v74[1] = v9;
    }
LABEL_31:
    v26 = (unsigned int)v117;
    if ( (unsigned int)v117 >= HIDWORD(v117) )
    {
      sub_16CD150((__int64)&src, v118, 0, 8, v19, v23);
      v26 = (unsigned int)v117;
    }
    v102 = v8;
    v4 = 1;
    *((_QWORD *)src + v26) = v119;
    LODWORD(v117) = v117 + 1;
    if ( (__int64)v119 >= v111 )
      v9 = v106;
    v106 = v9;
LABEL_7:
    v3 = *(_QWORD *)(v3 + 32);
  }
  while ( v3 && v8 );
  if ( !v102 )
    goto LABEL_20;
  v27 = (__int64 *)src;
  v28 = 8LL * (unsigned int)v117;
  v29 = (__int64 *)((char *)src + v28);
  if ( (char *)src + v28 != src )
  {
    _BitScanReverse64(&v30, v28 >> 3);
    sub_1D0BED0((char *)src, (__int64 *)((char *)src + v28), 2LL * (int)(63 - (v30 ^ 0x3F)));
    if ( (unsigned __int64)v28 <= 0x80 )
    {
      sub_1D0BE20(v27, v29);
    }
    else
    {
      v31 = v27 + 16;
      sub_1D0BE20(v27, v27 + 16);
      if ( v29 != v27 + 16 )
      {
        v32 = *v31;
        v33 = v27[15];
        v34 = v27 + 15;
        if ( v33 <= *v31 )
          goto LABEL_43;
        while ( 1 )
        {
          do
          {
            v34[1] = v33;
            v35 = v34;
            v33 = *--v34;
          }
          while ( v32 < v33 );
          ++v31;
          *v35 = v32;
          if ( v29 == v31 )
            break;
          while ( 1 )
          {
            v32 = *v31;
            v33 = *(v31 - 1);
            v34 = v31 - 1;
            if ( v33 > *v31 )
              break;
LABEL_43:
            v36 = v31++;
            *v36 = v32;
            if ( v29 == v31 )
              goto LABEL_46;
          }
        }
      }
    }
LABEL_46:
    v27 = (__int64 *)src;
  }
  v119 = (__int64 *)v121;
  v120 = 0x400000000LL;
  v37 = *v27;
  v108 = *v27;
  if ( !v115 )
  {
    ++v112;
    goto LABEL_175;
  }
  LODWORD(v38) = v115 - 1;
  v39 = 37 * v37;
  v40 = (v115 - 1) & (37 * v37);
  v41 = (_QWORD *)(v113 + 16LL * v40);
  v42 = *v41;
  if ( v37 != *v41 )
  {
    v92 = 1;
    v93 = 0;
    while ( v42 != 0x7FFFFFFFFFFFFFFFLL )
    {
      if ( !v93 && v42 == 0x8000000000000000LL )
        v93 = v41;
      v40 = v38 & (v92 + v40);
      v41 = (_QWORD *)(v113 + 16LL * v40);
      v42 = *v41;
      if ( v108 == *v41 )
        goto LABEL_49;
      ++v92;
    }
    if ( v93 )
      v41 = v93;
    ++v112;
    v94 = v114 + 1;
    if ( 4 * ((int)v114 + 1) < 3 * v115 )
    {
      if ( v115 - HIDWORD(v114) - v94 > v115 >> 3 )
      {
LABEL_159:
        LODWORD(v114) = v94;
        if ( *v41 != 0x7FFFFFFFFFFFFFFFLL )
          --HIDWORD(v114);
        v41[1] = 0;
        *v41 = v108;
        if ( (unsigned int)v120 >= HIDWORD(v120) )
          sub_16CD150((__int64)&v119, v121, 0, 8, v38, (int)v93);
        v103 = 0;
        v44 = &v119[(unsigned int)v120];
        v43 = 0;
        goto LABEL_50;
      }
      sub_1D0F970((__int64)&v112, v115);
      if ( v115 )
      {
        LODWORD(v38) = v115 - 1;
        v96 = 1;
        v97 = 0;
        v98 = (v115 - 1) & v39;
        v41 = (_QWORD *)(v113 + 16 * v98);
        v94 = v114 + 1;
        v93 = (_QWORD *)*v41;
        if ( v108 == *v41 )
          goto LABEL_159;
        while ( v93 != (_QWORD *)0x7FFFFFFFFFFFFFFFLL )
        {
          if ( !v97 && v93 == (_QWORD *)0x8000000000000000LL )
            v97 = v41;
          LODWORD(v98) = v38 & (v96 + v98);
          v41 = (_QWORD *)(v113 + 16LL * (unsigned int)v98);
          v93 = (_QWORD *)*v41;
          if ( v108 == *v41 )
            goto LABEL_159;
          ++v96;
        }
LABEL_171:
        if ( v97 )
          v41 = v97;
        goto LABEL_159;
      }
      goto LABEL_237;
    }
LABEL_175:
    sub_1D0F970((__int64)&v112, 2 * v115);
    if ( v115 )
    {
      LODWORD(v93) = v115 - 1;
      v94 = v114 + 1;
      v99 = (v115 - 1) & (37 * v108);
      v41 = (_QWORD *)(v113 + 16LL * v99);
      v38 = *v41;
      if ( v108 == *v41 )
        goto LABEL_159;
      v100 = 1;
      v97 = 0;
      while ( v38 != 0x7FFFFFFFFFFFFFFFLL )
      {
        if ( !v97 && v38 == 0x8000000000000000LL )
          v97 = v41;
        v99 = (unsigned int)v93 & (v100 + v99);
        v41 = (_QWORD *)(v113 + 16LL * v99);
        v38 = *v41;
        if ( v108 == *v41 )
          goto LABEL_159;
        ++v100;
      }
      goto LABEL_171;
    }
LABEL_237:
    LODWORD(v114) = v114 + 1;
    BUG();
  }
LABEL_49:
  v103 = v41[1];
  v43 = v103;
  v44 = (__int64 *)v121;
LABEL_50:
  *v44 = v43;
  v45 = v117;
  v46 = 1;
  LODWORD(v120) = v120 + 1;
  if ( (_DWORD)v117 == 1 )
    goto LABEL_56;
  while ( 2 )
  {
    v47 = v46 - 1;
    v48 = *((_QWORD *)src + v46);
    if ( v115 )
    {
      v49 = (v115 - 1) & (37 * v48);
      v50 = (_QWORD *)(v113 + 16LL * v49);
      v51 = *v50;
      if ( v48 == *v50 )
      {
        v52 = v50[1];
        goto LABEL_54;
      }
      v58 = 1;
      v59 = 0;
      while ( v51 != 0x7FFFFFFFFFFFFFFFLL )
      {
        if ( v59 || v51 != 0x8000000000000000LL )
          v50 = v59;
        v49 = (v115 - 1) & (v58 + v49);
        v95 = (__int64 *)(v113 + 16LL * v49);
        v51 = *v95;
        if ( v48 == *v95 )
        {
          v52 = v95[1];
          goto LABEL_54;
        }
        ++v58;
        v59 = v50;
        v50 = (_QWORD *)(v113 + 16LL * v49);
      }
      if ( !v59 )
        v59 = v50;
      ++v112;
      v60 = v114 + 1;
      if ( 4 * ((int)v114 + 1) < 3 * v115 )
      {
        if ( v115 - HIDWORD(v114) - v60 <= v115 >> 3 )
        {
          sub_1D0F970((__int64)&v112, v115);
          if ( !v115 )
          {
LABEL_240:
            LODWORD(v114) = v114 + 1;
            BUG();
          }
          v70 = 0;
          v71 = (v115 - 1) & (37 * v48);
          v72 = 1;
          v60 = v114 + 1;
          v59 = (_QWORD *)(v113 + 16LL * v71);
          v73 = *v59;
          if ( v48 != *v59 )
          {
            while ( v73 != 0x7FFFFFFFFFFFFFFFLL )
            {
              if ( v73 == 0x8000000000000000LL && !v70 )
                v70 = v59;
              v71 = (v115 - 1) & (v71 + v72);
              v59 = (_QWORD *)(v113 + 16LL * v71);
              v73 = *v59;
              if ( v48 == *v59 )
                goto LABEL_71;
              ++v72;
            }
            if ( v70 )
              v59 = v70;
          }
        }
        goto LABEL_71;
      }
    }
    else
    {
      ++v112;
    }
    sub_1D0F970((__int64)&v112, 2 * v115);
    if ( !v115 )
      goto LABEL_240;
    v66 = (v115 - 1) & (37 * v48);
    v60 = v114 + 1;
    v59 = (_QWORD *)(v113 + 16LL * v66);
    v67 = *v59;
    if ( v48 != *v59 )
    {
      v68 = 1;
      v69 = 0;
      while ( v67 != 0x7FFFFFFFFFFFFFFFLL )
      {
        if ( !v69 && v67 == 0x8000000000000000LL )
          v69 = v59;
        v66 = (v115 - 1) & (v66 + v68);
        v59 = (_QWORD *)(v113 + 16LL * v66);
        v67 = *v59;
        if ( v48 == *v59 )
          goto LABEL_71;
        ++v68;
      }
      if ( v69 )
        v59 = v69;
    }
LABEL_71:
    LODWORD(v114) = v60;
    if ( *v59 != 0x7FFFFFFFFFFFFFFFLL )
      --HIDWORD(v114);
    *v59 = v48;
    v52 = 0;
    v59[1] = 0;
LABEL_54:
    v53 = *(_QWORD *)(a1 + 16);
    v54 = *(__int64 (**)())(*(_QWORD *)v53 + 584LL);
    if ( v54 == sub_1D0B170
      || !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, _QWORD))v54)(
            v53,
            v103,
            v52,
            v108,
            v48,
            v47) )
    {
      goto LABEL_55;
    }
    v57 = (unsigned int)v120;
    if ( (unsigned int)v120 >= HIDWORD(v120) )
    {
      sub_16CD150((__int64)&v119, v121, 0, 8, v55, v56);
      v57 = (unsigned int)v120;
    }
    v119[v57] = v52;
    LODWORD(v120) = v120 + 1;
    if ( v46 + 1 != v45 )
    {
      ++v46;
      continue;
    }
    break;
  }
  v47 = v46;
LABEL_55:
  if ( v47 )
  {
    v61 = 0;
    v62 = *v119;
    if ( (unsigned __int8)sub_1D0B420(*v119, 0, 0, 1, *(_QWORD *)(a1 + 624)) )
      v61 = (unsigned int)(*(_DWORD *)(v62 + 60) - 1);
    else
      v62 = 0;
    v63 = 1;
    v109 = v120;
    v64 = v120 - 1;
    if ( (_DWORD)v120 == 1 )
    {
LABEL_85:
      if ( v119 != (__int64 *)v121 )
        _libc_free((unsigned __int64)v119);
      j___libc_free_0(v113);
      v13 = src;
      if ( src != v118 )
        goto LABEL_22;
      goto LABEL_23;
    }
    while ( 1 )
    {
      v65 = v119[v63];
      if ( (unsigned __int8)sub_1D0B420(v65, v62, v61, v63 < v64, *(_QWORD *)(a1 + 624)) )
        break;
      if ( v63 >= v64 && v62 )
      {
        ++v63;
        sub_1D0B1B0(v62, *(_QWORD *)(a1 + 624), *(_QWORD *)(v62 + 40), (unsigned int)(*(_DWORD *)(v62 + 60) - 1), 0, 0);
        if ( v63 == v109 )
          goto LABEL_85;
      }
      else
      {
LABEL_80:
        if ( ++v63 == v109 )
          goto LABEL_85;
      }
    }
    if ( v63 < v64 )
    {
      v62 = v65;
      v61 = (unsigned int)(*(_DWORD *)(v65 + 60) - 1) | v61 & 0xFFFFFFFF00000000LL;
    }
    goto LABEL_80;
  }
LABEL_56:
  if ( v119 != (__int64 *)v121 )
    _libc_free((unsigned __int64)v119);
LABEL_20:
  v3 = v113;
LABEL_21:
  j___libc_free_0(v3);
  v13 = src;
  if ( src != v118 )
LABEL_22:
    _libc_free((unsigned __int64)v13);
LABEL_23:
  if ( v124 != v123 )
    _libc_free((unsigned __int64)v124);
}
