// Function: sub_37272A0
// Address: 0x37272a0
//
_QWORD *__fastcall sub_37272A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        __int64 a7,
        __int64 a8,
        __int128 a9)
{
  int v13; // eax
  int v14; // ecx
  int v15; // edi
  int v16; // esi
  __m128i v17; // xmm0
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // rax
  char *v32; // rbx
  __int64 v33; // rax
  __int64 v34; // rdx
  int v35; // ecx
  char v36; // al
  unsigned int v37; // r8d
  __int64 v38; // r15
  int v39; // eax
  __int64 *v40; // rdx
  int v41; // r11d
  unsigned int v42; // ecx
  __int64 *v43; // rax
  __int64 v44; // rdi
  unsigned int v45; // ecx
  int v46; // r14d
  __int64 v47; // r15
  int v48; // r14d
  __int64 *v49; // rsi
  unsigned int v50; // eax
  int v51; // r8d
  __int64 v52; // r10
  unsigned int v53; // eax
  __int64 v54; // rdx
  _QWORD *result; // rax
  __int64 v56; // r15
  _QWORD *v57; // r14
  _QWORD *v58; // rax
  __int64 v59; // r8
  unsigned __int64 *v60; // r9
  unsigned __int64 v61; // rdi
  __int64 v62; // rbx
  char v63; // r12
  __int128 v64; // rax
  __int64 v65; // r8
  __int64 v66; // r9
  char v67; // r13
  bool v68; // zf
  int v69; // eax
  __int64 v70; // rdx
  __int64 v71; // rcx
  _QWORD *v72; // rax
  unsigned __int64 v73; // r13
  __int64 v74; // rax
  unsigned __int64 v75; // r12
  int v76; // eax
  int v77; // ecx
  unsigned int j; // esi
  __int64 v79; // rax
  unsigned int v80; // esi
  __int64 v81; // rax
  unsigned __int64 v82; // r13
  unsigned __int64 v83; // r12
  void *v84; // rdi
  int v85; // eax
  __int64 v86; // rax
  int v87; // eax
  _QWORD *v88; // rsi
  size_t v89; // r10
  __int16 v90; // ax
  int v91; // eax
  int v92; // eax
  int v93; // r15d
  int v94; // r15d
  __int64 v95; // rcx
  __int64 *v96; // rdi
  unsigned int v97; // eax
  int v98; // r9d
  __int64 v99; // r11
  unsigned int v100; // eax
  _QWORD *v101; // [rsp+8h] [rbp-1B8h]
  const void *v102; // [rsp+10h] [rbp-1B0h]
  _QWORD *v103; // [rsp+18h] [rbp-1A8h]
  __int64 v104; // [rsp+20h] [rbp-1A0h]
  __int64 v105; // [rsp+28h] [rbp-198h]
  __int64 *v106; // [rsp+30h] [rbp-190h]
  __int64 v107; // [rsp+30h] [rbp-190h]
  __int64 *i; // [rsp+38h] [rbp-188h]
  int v109; // [rsp+38h] [rbp-188h]
  __int64 v110; // [rsp+40h] [rbp-180h]
  __int64 v111; // [rsp+40h] [rbp-180h]
  unsigned int v112; // [rsp+40h] [rbp-180h]
  unsigned int v113; // [rsp+40h] [rbp-180h]
  unsigned __int64 v114; // [rsp+48h] [rbp-178h]
  __int64 v115; // [rsp+58h] [rbp-168h]
  _QWORD *v116; // [rsp+58h] [rbp-168h]
  __int64 *v117; // [rsp+60h] [rbp-160h]
  __int64 v118; // [rsp+68h] [rbp-158h]
  unsigned int v119; // [rsp+78h] [rbp-148h]
  __int64 v120; // [rsp+78h] [rbp-148h]
  __int64 v121; // [rsp+80h] [rbp-140h]
  unsigned __int16 v122; // [rsp+88h] [rbp-138h]
  char v123; // [rsp+8Bh] [rbp-135h]
  int v124; // [rsp+8Ch] [rbp-134h]
  unsigned __int64 v125; // [rsp+90h] [rbp-130h]
  char *src; // [rsp+98h] [rbp-128h]
  __int64 *v128; // [rsp+A8h] [rbp-118h] BYREF
  __int128 v129; // [rsp+B0h] [rbp-110h]
  __int128 v130; // [rsp+C0h] [rbp-100h] BYREF
  __int64 v131; // [rsp+D0h] [rbp-F0h] BYREF
  int v132; // [rsp+D8h] [rbp-E8h]
  int v133; // [rsp+DCh] [rbp-E4h]
  _QWORD *v134; // [rsp+E0h] [rbp-E0h] BYREF
  __int64 v135; // [rsp+E8h] [rbp-D8h]
  _QWORD v136[2]; // [rsp+F0h] [rbp-D0h] BYREF
  __int128 v137; // [rsp+100h] [rbp-C0h] BYREF
  _BYTE v138[16]; // [rsp+110h] [rbp-B0h] BYREF
  char v139; // [rsp+120h] [rbp-A0h]
  char v140; // [rsp+121h] [rbp-9Fh]

  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  *(_BYTE *)(a1 + 16) = 0;
  v13 = a8;
  v14 = *(_DWORD *)(a3 + 136);
  v15 = *(_DWORD *)(a3 + 152);
  *(_DWORD *)(a1 + 24) = a5;
  if ( !a6 )
    v13 = 0;
  v16 = 0;
  *(_DWORD *)(a1 + 40) = v14;
  if ( !a6 )
    v16 = a8;
  *(_DWORD *)(a1 + 36) = v15;
  *(_DWORD *)(a1 + 32) = v13;
  *(_QWORD *)(a1 + 44) = 0x800000000LL;
  *(_DWORD *)(a1 + 28) = v16;
  *(_QWORD *)(a1 + 52) = 0x303037304D564C4CLL;
  *(_DWORD *)(a1 + 20) = 5;
  v117 = (__int64 *)(a1 + 64);
  sub_C656D0(a1 + 64, 6);
  v17 = _mm_loadu_si128((const __m128i *)&a9);
  v102 = (const void *)(a1 + 96);
  *(_QWORD *)(a1 + 80) = a1 + 96;
  v18 = *(_QWORD *)a1;
  *(_QWORD *)(a1 + 88) = 0x500000000LL;
  *(_QWORD *)(a1 + 152) = a1 + 168;
  *(_QWORD *)(a1 + 160) = 0x400000000LL;
  *(_QWORD *)(a1 + 240) = a5;
  *(_QWORD *)(a1 + 248) = a7;
  *(_QWORD *)(a1 + 200) = a1 + 216;
  *(_QWORD *)(a1 + 256) = a8;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 1;
  *(_QWORD *)(a1 + 232) = a4;
  *(_QWORD *)(a1 + 280) = 0;
  *(__m128i *)(a1 + 264) = v17;
  v140 = 1;
  *(_QWORD *)&v137 = "names_abbrev_start";
  v139 = 3;
  v21 = sub_31DCC50(v18, (__int64 *)&v137, v19, v20, a5);
  v22 = *(_QWORD *)a1;
  v140 = 1;
  *(_QWORD *)(a1 + 288) = v21;
  *(_QWORD *)&v137 = "names_abbrev_end";
  v139 = 3;
  v26 = sub_31DCC50(v22, (__int64 *)&v137, v23, v24, v25);
  v27 = *(_QWORD *)a1;
  v140 = 1;
  *(_QWORD *)(a1 + 296) = v26;
  *(_QWORD *)&v137 = "names_entries";
  v139 = 3;
  v31 = sub_31DCC50(v27, (__int64 *)&v137, v28, v29, v30);
  *(_BYTE *)(a1 + 312) = a6;
  *(_QWORD *)(a1 + 304) = v31;
  *(_QWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_DWORD *)(a1 + 344) = 0;
  v110 = a1 + 320;
  v106 = *(__int64 **)(a3 + 192);
  for ( i = *(__int64 **)(a3 + 184); v106 != i; i += 3 )
  {
    v115 = i[1];
    v118 = *i;
    if ( v115 == *i )
      continue;
    do
    {
      v32 = *(char **)(*(_QWORD *)v118 + 16LL);
      src = *(char **)(*(_QWORD *)v118 + 24LL);
      if ( src == v32 )
        goto LABEL_30;
      do
      {
        while ( 1 )
        {
          v33 = *(_QWORD *)v32;
          if ( *(_BYTE *)(*(_QWORD *)v32 + 16LL) != 1 )
            abort();
          v34 = *(_QWORD *)(v33 + 8);
          v35 = *(_DWORD *)(v33 + 44);
          v36 = *(_BYTE *)(v33 + 43) >> 7;
          v37 = *(_DWORD *)(a1 + 344);
          *(_QWORD *)&v137 = v34;
          DWORD2(v137) = v35;
          BYTE12(v137) = v36;
          if ( !v37 )
          {
            ++*(_QWORD *)(a1 + 320);
            goto LABEL_17;
          }
          v131 = v34;
          v119 = v37;
          v38 = *(_QWORD *)(a1 + 328);
          LODWORD(v130) = v35;
          v39 = sub_3723A60(&v131, &v130, (_BYTE *)&v137 + 12);
          v40 = 0;
          v41 = 1;
          v42 = (v119 - 1) & v39;
LABEL_11:
          v43 = (__int64 *)(v38 + 16LL * v42);
          v44 = *v43;
          if ( (_QWORD)v137 != *v43 || DWORD2(v137) != *((_DWORD *)v43 + 2) || BYTE12(v137) != *((_BYTE *)v43 + 12) )
            break;
          v32 += 8;
          if ( src == v32 )
            goto LABEL_30;
        }
        if ( v44 != -1 )
        {
          if ( v44 == -2 && *((_DWORD *)v43 + 2) == -2 && *((_BYTE *)v43 + 12) != 1 && !v40 )
            v40 = (__int64 *)(v38 + 16LL * v42);
LABEL_15:
          v45 = v41 + v42;
          ++v41;
          v42 = (v119 - 1) & v45;
          goto LABEL_11;
        }
        if ( *((_DWORD *)v43 + 2) != -1 || *((_BYTE *)v43 + 12) )
          goto LABEL_15;
        v37 = *(_DWORD *)(a1 + 344);
        if ( !v40 )
          v40 = (__int64 *)(v38 + 16LL * v42);
        v92 = *(_DWORD *)(a1 + 336);
        ++*(_QWORD *)(a1 + 320);
        v91 = v92 + 1;
        if ( 4 * v91 < 3 * v37 )
        {
          if ( v37 - (v91 + *(_DWORD *)(a1 + 340)) > v37 >> 3 )
            goto LABEL_83;
          sub_3727000(v110, v37);
          v93 = *(_DWORD *)(a1 + 344);
          v40 = 0;
          if ( !v93 )
            goto LABEL_82;
          v94 = v93 - 1;
          v95 = *(_QWORD *)(a1 + 328);
          LODWORD(v130) = DWORD2(v137);
          v120 = v95;
          v131 = v137;
          v96 = 0;
          v97 = v94 & sub_3723A60(&v131, &v130, (_BYTE *)&v137 + 12);
          v98 = 1;
          while ( 2 )
          {
            v40 = (__int64 *)(v120 + 16LL * v97);
            v99 = *v40;
            if ( (_QWORD)v137 == *v40 && DWORD2(v137) == *((_DWORD *)v40 + 2) )
            {
              if ( BYTE12(v137) == *((_BYTE *)v40 + 12) )
                goto LABEL_82;
              if ( v99 == -1 )
                goto LABEL_116;
LABEL_102:
              if ( v99 == -2 && *((_DWORD *)v40 + 2) == -2 && *((_BYTE *)v40 + 12) != 1 && !v96 )
                v96 = (__int64 *)(v120 + 16LL * v97);
            }
            else
            {
              if ( v99 != -1 )
                goto LABEL_102;
LABEL_116:
              if ( *((_DWORD *)v40 + 2) == -1 && !*((_BYTE *)v40 + 12) )
              {
                if ( v96 )
                  v40 = v96;
                goto LABEL_82;
              }
            }
            v100 = v98 + v97;
            ++v98;
            v97 = v94 & v100;
            continue;
          }
        }
LABEL_17:
        sub_3727000(v110, 2 * v37);
        v46 = *(_DWORD *)(a1 + 344);
        v40 = 0;
        if ( !v46 )
          goto LABEL_82;
        v47 = *(_QWORD *)(a1 + 328);
        v48 = v46 - 1;
        LODWORD(v130) = DWORD2(v137);
        v131 = v137;
        v49 = 0;
        v50 = v48 & sub_3723A60(&v131, &v130, (_BYTE *)&v137 + 12);
        v51 = 1;
        while ( 2 )
        {
          v40 = (__int64 *)(v47 + 16LL * v50);
          v52 = *v40;
          if ( (_QWORD)v137 == *v40 && DWORD2(v137) == *((_DWORD *)v40 + 2) && BYTE12(v137) == *((_BYTE *)v40 + 12) )
            goto LABEL_82;
          if ( v52 != -1 )
          {
            if ( v52 == -2 && *((_DWORD *)v40 + 2) == -2 && *((_BYTE *)v40 + 12) != 1 && !v49 )
              v49 = (__int64 *)(v47 + 16LL * v50);
            goto LABEL_26;
          }
          if ( *((_DWORD *)v40 + 2) != -1 || *((_BYTE *)v40 + 12) )
          {
LABEL_26:
            v53 = v51 + v50;
            ++v51;
            v50 = v48 & v53;
            continue;
          }
          break;
        }
        if ( v49 )
          v40 = v49;
LABEL_82:
        v91 = *(_DWORD *)(a1 + 336) + 1;
LABEL_83:
        *(_DWORD *)(a1 + 336) = v91;
        if ( *v40 != -1 || *((_DWORD *)v40 + 2) != -1 || *((_BYTE *)v40 + 12) )
          --*(_DWORD *)(a1 + 340);
        v32 += 8;
        *v40 = v137;
        *((_DWORD *)v40 + 2) = DWORD2(v137);
        *((_BYTE *)v40 + 12) = BYTE12(v137);
      }
      while ( src != v32 );
LABEL_30:
      v118 += 8;
    }
    while ( v115 != v118 );
  }
  v54 = *(_QWORD *)(a1 + 8);
  result = *(_QWORD **)(v54 + 184);
  v101 = *(_QWORD **)(v54 + 192);
  if ( result != v101 )
  {
    v103 = *(_QWORD **)(v54 + 184);
    v56 = a1;
    while ( 1 )
    {
      v104 = v103[1];
      if ( *v103 != v104 )
        break;
LABEL_69:
      v103 += 3;
      result = v103;
      if ( v101 == v103 )
        return result;
    }
    v105 = *v103;
    while ( 1 )
    {
      v116 = *(_QWORD **)(*(_QWORD *)v105 + 24LL);
      if ( v116 != *(_QWORD **)(*(_QWORD *)v105 + 16LL) )
        break;
LABEL_68:
      v105 += 8;
      if ( v104 == v105 )
        goto LABEL_69;
    }
    v57 = *(_QWORD **)(*(_QWORD *)v105 + 16LL);
    while ( 1 )
    {
      v62 = *v57;
      v63 = 0;
      *(_QWORD *)&v64 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(v56 + 264))(*(_QWORD *)(v56 + 272), *v57);
      v67 = BYTE12(v64);
      v130 = v64;
      v137 = v64;
      v68 = *(_BYTE *)(v62 + 32) == 0;
      v129 = v64;
      if ( !v68 )
      {
        v63 = 1;
        v124 = *(_DWORD *)(v62 + 44);
        v121 = *(_QWORD *)(v62 + 24);
        v123 = *(_BYTE *)(v62 + 43) >> 7;
      }
      v138[0] = v63;
      *(_QWORD *)&v137 = v121;
      DWORD2(v137) = v124;
      BYTE12(v137) = v123;
      if ( v63 )
      {
        v65 = *(_QWORD *)(v56 + 328);
        v66 = *(unsigned int *)(v56 + 344);
        v107 = v65;
        v109 = *(_DWORD *)(v56 + 344);
        v111 = v65 + 16 * v66;
        v122 = 25;
        if ( (_DWORD)v66 )
        {
          LODWORD(v128) = v124;
          v131 = v121;
          v76 = sub_3723A60(&v131, &v128, (_BYTE *)&v137 + 12);
          v77 = 1;
          v65 = v107;
          v66 = v137;
          for ( j = (v109 - 1) & v76; ; j = (v109 - 1) & v80 )
          {
            v79 = v107 + 16LL * j;
            if ( (_QWORD)v137 == *(_QWORD *)v79
              && DWORD2(v137) == *(_DWORD *)(v79 + 8)
              && BYTE12(v137) == *(_BYTE *)(v79 + 12) )
            {
              break;
            }
            if ( *(_QWORD *)v79 == -1 && *(_DWORD *)(v79 + 8) == -1 && !*(_BYTE *)(v79 + 12) )
            {
              v79 = *(_QWORD *)(v56 + 328) + 16LL * *(unsigned int *)(v56 + 344);
              break;
            }
            v80 = v77 + j;
            ++v77;
          }
          v68 = v79 == v111;
          v90 = 25;
          if ( !v68 )
            v90 = 19;
          v122 = v90;
        }
      }
      v69 = *(unsigned __int16 *)(v62 + 40);
      v131 = 0;
      v133 = 0;
      v132 = v69;
      v134 = v136;
      v135 = 0x100000000LL;
      if ( v67 )
      {
        LODWORD(v135) = 1;
        v136[0] = *(_QWORD *)((char *)&v129 + 4);
        v125 = v125 & 0xFFFF000000000000LL | 0x1300000003LL;
        v73 = v125;
        sub_C8D5F0((__int64)&v134, v136, 2u, 8u, v65, v66);
        v72 = v134;
        v70 = (unsigned int)v135;
      }
      else
      {
        v70 = 0;
        v71 = 0xFFFF0000FFFFFFFFLL;
        v72 = v136;
        v125 = v125 & 0xFFFF000000000000LL | 0x1300000003LL;
        v73 = v125;
      }
      v72[v70] = v73;
      v74 = (unsigned int)(v135 + 1);
      LODWORD(v135) = v135 + 1;
      if ( v63 )
      {
        v71 = HIDWORD(v135);
        v75 = ((unsigned __int64)v122 << 32) | v114 & 0xFFFF000000000000LL | 4;
        v114 = v75;
        if ( v74 + 1 > (unsigned __int64)HIDWORD(v135) )
        {
          sub_C8D5F0((__int64)&v134, v136, v74 + 1, 8u, v65, v66);
          v74 = (unsigned int)v135;
        }
        v70 = (__int64)v134;
        v134[v74] = v75;
        LODWORD(v135) = v135 + 1;
      }
      *((_QWORD *)&v137 + 1) = 0x2000000000LL;
      *(_QWORD *)&v137 = v138;
      sub_3723750((__int64)&v131, (__int64)&v137, v70, v71, v65, v66);
      v58 = sub_C65B40((__int64)v117, (__int64)&v137, (__int64 *)&v128, (__int64)off_4A3D240);
      if ( !v58 )
        break;
      *(_WORD *)(v62 + 42) = *((_DWORD *)v58 + 3) & 0x7FFF | *(_WORD *)(v62 + 42) & 0x8000;
      v61 = v137;
      if ( (_BYTE *)v137 != v138 )
        goto LABEL_40;
LABEL_41:
      if ( v134 != v136 )
        _libc_free((unsigned __int64)v134);
      if ( v116 == ++v57 )
        goto LABEL_68;
    }
    v81 = *(_QWORD *)(v56 + 136);
    *(_QWORD *)(v56 + 216) += 40LL;
    v82 = (v81 + 15) & 0xFFFFFFFFFFFFFFF0LL;
    if ( *(_QWORD *)(v56 + 144) >= v82 + 40 && v81 )
    {
      *(_QWORD *)(v56 + 136) = v82 + 40;
      v83 = (v81 + 15) & 0xFFFFFFFFFFFFFFF0LL;
      if ( !v82 )
        goto LABEL_64;
    }
    else
    {
      v83 = sub_9D1E70(v56 + 136, 40, 40, 4);
      v82 = v83;
    }
    v84 = (void *)(v83 + 32);
    v60 = (unsigned __int64 *)(v83 + 16);
    *(_QWORD *)v83 = v131;
    *(_DWORD *)(v83 + 8) = v132;
    v85 = v133;
    *(_QWORD *)(v83 + 16) = v83 + 32;
    *(_DWORD *)(v83 + 12) = v85;
    *(_QWORD *)(v83 + 24) = 0x100000000LL;
    v59 = (unsigned int)v135;
    if ( !(_DWORD)v135 || v60 == (unsigned __int64 *)&v134 )
    {
LABEL_64:
      v86 = *(unsigned int *)(v56 + 88);
      if ( v86 + 1 > (unsigned __int64)*(unsigned int *)(v56 + 92) )
      {
        sub_C8D5F0(v56 + 80, v102, v86 + 1, 8u, v59, (__int64)v60);
        v86 = *(unsigned int *)(v56 + 88);
      }
      *(_QWORD *)(*(_QWORD *)(v56 + 80) + 8 * v86) = v82;
      v87 = *(_DWORD *)(v56 + 88) + 1;
      *(_DWORD *)(v56 + 88) = v87;
      *(_DWORD *)(v83 + 12) = v87;
      sub_C657C0(v117, (__int64 *)v83, v128, (__int64)off_4A3D240);
      *(_WORD *)(v62 + 42) = *(_DWORD *)(v83 + 12) & 0x7FFF | *(_WORD *)(v62 + 42) & 0x8000;
      v61 = v137;
      if ( (_BYTE *)v137 == v138 )
        goto LABEL_41;
LABEL_40:
      _libc_free(v61);
      goto LABEL_41;
    }
    if ( v134 != v136 )
    {
      *(_QWORD *)(v83 + 16) = v134;
      *(_QWORD *)(v83 + 24) = v135;
      v135 = 0;
      v134 = v136;
      goto LABEL_64;
    }
    v88 = v136;
    v89 = 8;
    if ( (_DWORD)v135 != 1 )
    {
      v113 = v135;
      sub_C8D5F0(v83 + 16, (const void *)(v83 + 32), (unsigned int)v135, 8u, (unsigned int)v135, (__int64)v60);
      v59 = v113;
      v89 = 8LL * (unsigned int)v135;
      if ( !v89 )
        goto LABEL_74;
      v84 = *(void **)(v83 + 16);
      v88 = v134;
    }
    v112 = v59;
    memcpy(v84, v88, v89);
    v59 = v112;
LABEL_74:
    *(_DWORD *)(v83 + 24) = v59;
    LODWORD(v135) = 0;
    goto LABEL_64;
  }
  return result;
}
