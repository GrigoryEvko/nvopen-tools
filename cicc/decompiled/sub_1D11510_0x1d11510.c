// Function: sub_1D11510
// Address: 0x1d11510
//
__int64 __fastcall sub_1D11510(__int64 a1, __int64 **a2)
{
  int v2; // r8d
  int v3; // r9d
  __int64 v4; // rax
  __int64 v5; // rax
  int v6; // ecx
  __int64 v7; // rdi
  __int64 **v8; // rax
  __int64 *v9; // r14
  __int64 v10; // rsi
  int v11; // eax
  unsigned int *v12; // rax
  __int64 v13; // r13
  __int64 v14; // rax
  int v15; // edx
  unsigned int *v16; // rdx
  bool v17; // zf
  __int64 v18; // r12
  _BOOL8 v19; // rcx
  _BOOL8 v20; // rdx
  _BOOL8 v21; // rcx
  _BOOL8 v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // r12
  _QWORD *v26; // r15
  __int64 v27; // rax
  _QWORD *v28; // r12
  __int64 v29; // rdx
  unsigned __int64 v30; // rsi
  __int64 v31; // rax
  char *v32; // r12
  _QWORD *v33; // r13
  __int64 v34; // rax
  unsigned int *v35; // r15
  const __m128i *v36; // r13
  __int64 v37; // rbx
  __int64 v38; // r14
  __int64 v39; // rax
  int *v40; // r8
  __m128i *v41; // r12
  __int64 v42; // rdx
  __int64 v43; // rax
  __m128i v44; // xmm1
  __m128i v45; // xmm0
  __int64 v46; // rax
  int *v47; // r8
  __int64 v48; // rax
  char *v49; // r12
  __int64 v50; // rax
  char *v51; // r14
  __int64 v52; // rbx
  __int64 v53; // r15
  char *v54; // rax
  char *v55; // r8
  __int64 v56; // rax
  _QWORD *v57; // r15
  unsigned int v58; // r12d
  _QWORD *v59; // rbx
  __m128i *v60; // rax
  unsigned int v61; // r14d
  _QWORD *v62; // r15
  unsigned __int64 *v63; // rbx
  __int64 v64; // rdx
  unsigned __int64 v65; // rsi
  __int64 v66; // rsi
  unsigned int v67; // eax
  __int64 v68; // rax
  __int64 v69; // rdx
  unsigned __int64 v70; // rsi
  __int64 v71; // r12
  unsigned __int64 v72; // rdx
  __int64 v73; // r8
  __int64 *v74; // r13
  unsigned __int64 **v75; // rbx
  __int64 v76; // r14
  unsigned __int64 **v77; // r12
  unsigned __int64 *v78; // r15
  __int64 v79; // rdx
  unsigned __int64 v80; // rax
  __int64 v81; // rcx
  __int64 v82; // rax
  unsigned int v83; // r14d
  _QWORD *v84; // rbx
  unsigned __int64 *v85; // r12
  __int64 v86; // rcx
  unsigned __int64 v87; // rsi
  unsigned int v88; // eax
  __int64 v89; // rax
  __int64 v90; // rcx
  unsigned __int64 v91; // rsi
  _QWORD *v92; // r14
  __int64 v93; // rax
  int v94; // r8d
  int v95; // r9d
  __int64 v96; // rdx
  int v97; // [rsp+10h] [rbp-3B0h]
  unsigned __int64 v98; // [rsp+18h] [rbp-3A8h]
  unsigned __int64 v99; // [rsp+18h] [rbp-3A8h]
  __m128i *v100; // [rsp+18h] [rbp-3A8h]
  int v102; // [rsp+28h] [rbp-398h]
  unsigned __int64 v103; // [rsp+28h] [rbp-398h]
  unsigned __int64 v104; // [rsp+28h] [rbp-398h]
  unsigned __int64 v105; // [rsp+30h] [rbp-390h]
  unsigned int v106; // [rsp+38h] [rbp-388h]
  unsigned __int64 v107; // [rsp+38h] [rbp-388h]
  int v108; // [rsp+38h] [rbp-388h]
  __int64 v109; // [rsp+38h] [rbp-388h]
  char v111; // [rsp+48h] [rbp-378h]
  _QWORD *i; // [rsp+48h] [rbp-378h]
  _QWORD *v113; // [rsp+48h] [rbp-378h]
  int *v114; // [rsp+48h] [rbp-378h]
  char *v115; // [rsp+48h] [rbp-378h]
  _QWORD *v116; // [rsp+48h] [rbp-378h]
  _QWORD *v117; // [rsp+48h] [rbp-378h]
  unsigned int v118; // [rsp+48h] [rbp-378h]
  __int64 v119; // [rsp+48h] [rbp-378h]
  __int64 v120; // [rsp+50h] [rbp-370h] BYREF
  __int64 v121; // [rsp+58h] [rbp-368h]
  __int64 v122; // [rsp+60h] [rbp-360h]
  int v123; // [rsp+68h] [rbp-358h]
  __int64 v124; // [rsp+70h] [rbp-350h] BYREF
  __int64 v125; // [rsp+78h] [rbp-348h]
  __int64 v126; // [rsp+80h] [rbp-340h]
  int v127; // [rsp+88h] [rbp-338h]
  _BYTE v128[40]; // [rsp+90h] [rbp-330h] BYREF
  __int64 v129; // [rsp+B8h] [rbp-308h]
  __int64 *v130; // [rsp+C0h] [rbp-300h]
  unsigned __int64 **v131; // [rsp+D0h] [rbp-2F0h] BYREF
  __int64 j; // [rsp+D8h] [rbp-2E8h]
  _BYTE v133[64]; // [rsp+E0h] [rbp-2E0h] BYREF
  unsigned __int64 v134[2]; // [rsp+120h] [rbp-2A0h] BYREF
  _BYTE v135[40]; // [rsp+130h] [rbp-290h] BYREF
  int v136; // [rsp+158h] [rbp-268h] BYREF
  __int64 v137; // [rsp+160h] [rbp-260h]
  int *v138; // [rsp+168h] [rbp-258h]
  int *v139; // [rsp+170h] [rbp-250h]
  __int64 v140; // [rsp+178h] [rbp-248h]
  __m128i *v141; // [rsp+180h] [rbp-240h] BYREF
  __int64 v142; // [rsp+188h] [rbp-238h]
  _BYTE v143[560]; // [rsp+190h] [rbp-230h] BYREF

  sub_1FE79E0(v128, *(_QWORD *)(a1 + 616), *a2);
  v120 = 0;
  v141 = (__m128i *)v143;
  v142 = 0x2000000000LL;
  v134[0] = (unsigned __int64)v135;
  v134[1] = 0x800000000LL;
  v138 = &v136;
  v139 = &v136;
  v4 = *(_QWORD *)(a1 + 624);
  v121 = 0;
  v5 = *(_QWORD *)(v4 + 648);
  v122 = 0;
  v123 = 0;
  v124 = 0;
  v6 = *(_DWORD *)(v5 + 112);
  v125 = 0;
  v126 = 0;
  v127 = 0;
  v136 = 0;
  v137 = 0;
  v140 = 0;
  if ( v6 || *(_DWORD *)(v5 + 384) )
  {
    v7 = *(_QWORD *)(a1 + 616);
    if ( v7 == *(_QWORD *)(*(_QWORD *)(v7 + 56) + 328LL) )
    {
      v26 = *(_QWORD **)(v5 + 376);
      for ( i = &v26[*(unsigned int *)(v5 + 384)]; i != v26; ++v26 )
      {
        v27 = sub_1FE7480(v128, *v26, &v120);
        if ( v27 )
        {
          v107 = v27;
          v28 = *a2;
          sub_1DD5BA0(*(_QWORD *)(a1 + 616) + 16LL, v27);
          v29 = *(_QWORD *)v107;
          v30 = *v28 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v107 + 8) = v28;
          *(_QWORD *)v107 = v30 | v29 & 7;
          *(_QWORD *)(v30 + 8) = v107;
          *v28 = *v28 & 7LL | v107;
        }
      }
      goto LABEL_49;
    }
  }
  else
  {
    if ( !*(_DWORD *)(v5 + 656) )
    {
      v8 = *(__int64 ***)(a1 + 640);
      v102 = (__int64)(*(_QWORD *)(a1 + 648) - (_QWORD)v8) >> 3;
      if ( !v102 )
      {
        v23 = 0;
        goto LABEL_40;
      }
      v111 = 0;
      goto LABEL_5;
    }
    v7 = *(_QWORD *)(a1 + 616);
    if ( *(_QWORD *)(*(_QWORD *)(v7 + 56) + 328LL) == v7 )
    {
LABEL_49:
      v8 = *(__int64 ***)(a1 + 640);
      v111 = 1;
      v102 = (__int64)(*(_QWORD *)(a1 + 648) - (_QWORD)v8) >> 3;
      if ( !v102 )
      {
LABEL_50:
        v7 = *(_QWORD *)(a1 + 616);
        goto LABEL_51;
      }
LABEL_5:
      v106 = 0;
      v9 = *v8;
      if ( !*v8 )
        goto LABEL_32;
LABEL_6:
      if ( !*v9 )
      {
        sub_1D10F80((_QWORD *)a1, (__int64)v9, (__int64)&v124, *a2);
        goto LABEL_30;
      }
      v131 = (unsigned __int64 **)v133;
      j = 0x400000000LL;
      v10 = *v9;
      v11 = *(_DWORD *)(*v9 + 56);
      if ( v11 )
      {
        v12 = (unsigned int *)(*(_QWORD *)(v10 + 32) + 40LL * (unsigned int)(v11 - 1));
        v13 = *(_QWORD *)v12;
        if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v12 + 40LL) + 16LL * v12[2]) == 111 )
        {
          v14 = 0;
          while ( 1 )
          {
            v131[v14] = (unsigned __int64 *)v13;
            v14 = (unsigned int)(j + 1);
            LODWORD(j) = j + 1;
            v15 = *(_DWORD *)(v13 + 56);
            if ( !v15 )
              break;
            v16 = (unsigned int *)(*(_QWORD *)(v13 + 32) + 40LL * (unsigned int)(v15 - 1));
            v13 = *(_QWORD *)v16;
            if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v16 + 40LL) + 16LL * v16[2]) != 111 )
              break;
            if ( (unsigned int)v14 >= HIDWORD(j) )
            {
              sub_16CD150((__int64)&v131, v133, 0, 8, v2, v3);
              v14 = (unsigned int)j;
            }
          }
          if ( (_DWORD)v14 )
          {
            while ( 1 )
            {
              v18 = (__int64)v131[v14 - 1];
              v19 = (*((_BYTE *)v9 + 229) & 0x20) != 0;
              v20 = v9[2] != (_QWORD)v9;
              if ( *(__int16 *)(v18 + 24) >= 0 )
              {
                sub_1FEA180(v128, v18, v20, v19, &v120);
                if ( v111 )
                  goto LABEL_18;
              }
              else
              {
                sub_1FEABF0(v128, v18, v20, v19, &v120);
                if ( v111 )
LABEL_18:
                  sub_1D0CB80(v18, *(_QWORD *)(a1 + 624), (__int64)v128, (__int64)&v120, (__int64)&v141, (__int64)v134);
              }
              v17 = (_DWORD)j == 1;
              v14 = (unsigned int)(j - 1);
              LODWORD(j) = j - 1;
              if ( v17 )
              {
                v10 = *v9;
                goto LABEL_26;
              }
            }
          }
          v10 = *v9;
        }
      }
LABEL_26:
      v21 = (*((_BYTE *)v9 + 229) & 0x20) != 0;
      v22 = v9[2] != (_QWORD)v9;
      if ( *(__int16 *)(v10 + 24) < 0 )
      {
        sub_1FEABF0(v128, v10, v22, v21, &v120);
        if ( !v111 )
          goto LABEL_28;
      }
      else
      {
        sub_1FEA180(v128, v10, v22, v21, &v120);
        if ( !v111 )
        {
LABEL_28:
          if ( v131 != (unsigned __int64 **)v133 )
            _libc_free((unsigned __int64)v131);
LABEL_30:
          while ( ++v106 != v102 )
          {
            v9 = *(__int64 **)(*(_QWORD *)(a1 + 640) + 8LL * v106);
            if ( v9 )
              goto LABEL_6;
LABEL_32:
            (*(void (__fastcall **)(_QWORD, __int64, __int64 *))(**(_QWORD **)(a1 + 16) + 632LL))(
              *(_QWORD *)(a1 + 16),
              v129,
              *a2);
          }
          if ( !v111 )
            goto LABEL_39;
          goto LABEL_50;
        }
      }
      sub_1D0CB80(*v9, *(_QWORD *)(a1 + 624), (__int64)v128, (__int64)&v120, (__int64)&v141, (__int64)v134);
      goto LABEL_28;
    }
  }
  v8 = *(__int64 ***)(a1 + 640);
  v102 = (__int64)(*(_QWORD *)(a1 + 648) - (_QWORD)v8) >> 3;
  if ( v102 )
  {
    v111 = 1;
    goto LABEL_5;
  }
LABEL_51:
  v31 = sub_1DD5D10(v7);
  v32 = (char *)v141;
  v33 = (_QWORD *)v31;
  v34 = 16LL * (unsigned int)v142;
  v35 = (unsigned int *)&v141[(unsigned __int64)v34 / 0x10];
  if ( v34 )
  {
    v113 = v33;
    v36 = v141;
    v37 = v34 >> 4;
    while ( 1 )
    {
      v38 = 4 * v37;
      v39 = sub_2207800(16 * v37, &unk_435FF63);
      v40 = (int *)v39;
      if ( v39 )
        break;
      v37 >>= 1;
      if ( !v37 )
      {
        v32 = (char *)v36;
        v33 = v113;
        goto LABEL_115;
      }
    }
    v41 = (__m128i *)v36;
    v42 = v39 + v38 * 4;
    v43 = v39 + 16;
    v44 = _mm_loadu_si128(v36);
    v33 = v113;
    *(__m128i *)(v43 - 16) = v44;
    if ( v42 == v43 )
    {
      v46 = (__int64)v40;
    }
    else
    {
      do
      {
        v45 = _mm_loadu_si128((const __m128i *)(v43 - 16));
        v43 += 16;
        *(__m128i *)(v43 - 16) = v45;
      }
      while ( v42 != v43 );
      v46 = (__int64)&v40[v38 - 4];
    }
    v114 = v40;
    v41->m128i_i32[0] = *(_DWORD *)v46;
    v41->m128i_i64[1] = *(_QWORD *)(v46 + 8);
    sub_1D0F8A0(v41->m128i_i32, v35, v40, v37);
    v47 = v114;
  }
  else
  {
LABEL_115:
    v38 = 0;
    sub_1D0D470(v32, v35);
    v47 = 0;
  }
  j_j___libc_free_0(v47, v38 * 4);
  v48 = *(_QWORD *)(*(_QWORD *)(a1 + 624) + 648LL);
  v49 = *(char **)(v48 + 104);
  v50 = 8LL * *(unsigned int *)(v48 + 112);
  v51 = &v49[v50];
  if ( v50 )
  {
    v52 = v50 >> 3;
    while ( 1 )
    {
      v53 = 8 * v52;
      v54 = (char *)sub_2207800(8 * v52, &unk_435FF63);
      if ( v54 )
        break;
      v52 >>= 1;
      if ( !v52 )
        goto LABEL_112;
    }
    v115 = v54;
    sub_1D0D960(v49, v51, v54, (char *)v52);
    v55 = v115;
  }
  else
  {
LABEL_112:
    v53 = 0;
    sub_1D0D290(v49, v51);
    v55 = 0;
  }
  j_j___libc_free_0(v55, v53);
  v56 = *(_QWORD *)(*(_QWORD *)(a1 + 624) + 648LL);
  v57 = *(_QWORD **)(v56 + 104);
  v116 = &v57[*(unsigned int *)(v56 + 112)];
  v97 = v142;
  if ( !(_DWORD)v142 || &v57[*(unsigned int *)(v56 + 112)] == v57 )
    goto LABEL_100;
  v58 = 0;
  v59 = *(_QWORD **)(v56 + 104);
  v108 = 0;
  do
  {
    v60 = &v141[v108];
    if ( v60->m128i_i64[1] )
    {
      v61 = v60->m128i_i32[0];
      v62 = v59;
      v63 = (unsigned __int64 *)v60->m128i_i64[1];
      while ( 1 )
      {
        while ( 1 )
        {
          v66 = *v62;
          v67 = *(_DWORD *)(*v62 + 40LL);
          if ( v67 < v58 || v61 <= v67 )
          {
            v59 = v62;
            v58 = v61;
            goto LABEL_97;
          }
          if ( !*(_BYTE *)(v66 + 49) )
          {
            v68 = sub_1FE7480(v128, v66, &v120);
            if ( v68 )
              break;
          }
LABEL_68:
          if ( ++v62 == v116 )
            goto LABEL_75;
        }
        if ( !v58 )
        {
          v98 = v68;
          sub_1DD5BA0(*(_QWORD *)(a1 + 616) + 16LL, v68);
          v64 = *(_QWORD *)v98;
          v65 = *v33 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v98 + 8) = v33;
          *(_QWORD *)v98 = v65 | v64 & 7;
          *(_QWORD *)(v65 + 8) = v98;
          *v33 = *v33 & 7LL | v98;
          goto LABEL_68;
        }
        v99 = v68;
        ++v62;
        sub_1DD5BA0(v63[3] + 16, v68);
        v69 = *(_QWORD *)v99;
        v70 = *v63 & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v99 + 8) = v63;
        *(_QWORD *)v99 = v70 | v69 & 7;
        *(_QWORD *)(v70 + 8) = v99;
        *v63 = *v63 & 7 | v99;
        if ( v62 == v116 )
        {
LABEL_75:
          v131 = (unsigned __int64 **)v133;
          j = 0x800000000LL;
          goto LABEL_76;
        }
      }
    }
LABEL_97:
    ++v108;
  }
  while ( v108 != v97 && v59 != v116 );
  v57 = v59;
LABEL_100:
  v92 = v116;
  v131 = (unsigned __int64 **)v133;
  for ( j = 0x800000000LL; v57 != v92; ++v57 )
  {
    if ( !*(_BYTE *)(*v57 + 49LL) )
    {
      v93 = sub_1FE7480(v128, *v57, &v120);
      if ( v93 )
      {
        v96 = (unsigned int)j;
        if ( (unsigned int)j >= HIDWORD(j) )
        {
          v119 = v93;
          sub_16CD150((__int64)&v131, v133, 0, 8, v94, v95);
          v96 = (unsigned int)j;
          v93 = v119;
        }
        v131[v96] = (unsigned __int64 *)v93;
        LODWORD(j) = j + 1;
      }
    }
  }
LABEL_76:
  v71 = v129;
  v73 = sub_1DD5EE0(v129);
  if ( &v131[(unsigned int)j] != v131 )
  {
    v117 = v33;
    v74 = (__int64 *)v73;
    v75 = v131;
    v76 = v71 + 16;
    v77 = &v131[(unsigned int)j];
    do
    {
      v78 = *v75++;
      sub_1DD5BA0(v76, v78);
      v79 = *v74;
      v80 = *v78;
      v78[1] = (unsigned __int64)v74;
      v72 = v79 & 0xFFFFFFFFFFFFFFF8LL;
      *v78 = v72 | v80 & 7;
      *(_QWORD *)(v72 + 8) = v78;
      *v74 = *v74 & 7 | (unsigned __int64)v78;
    }
    while ( v77 != v75 );
    v33 = v117;
  }
  v81 = (__int64)v141;
  v82 = *(_QWORD *)(*(_QWORD *)(a1 + 624) + 648LL);
  v109 = *(_QWORD *)(v82 + 648) + 8LL * *(unsigned int *)(v82 + 656);
  v100 = &v141[(unsigned int)v142];
  if ( v141 == v100 )
    goto LABEL_92;
  v105 = (unsigned __int64)v141;
  v83 = 0;
  v84 = *(_QWORD **)(v82 + 648);
LABEL_82:
  v85 = *(unsigned __int64 **)(v105 + 8);
  if ( !v85 )
    goto LABEL_109;
  if ( (_QWORD *)v109 != v84 )
  {
    v118 = *(_DWORD *)v105;
    do
    {
      while ( 1 )
      {
        v88 = *(_DWORD *)(*v84 + 16LL);
        if ( v118 <= v88 || v88 < v83 )
        {
          v83 = v118;
LABEL_109:
          v105 += 16LL;
          if ( v100 == (__m128i *)v105 )
            goto LABEL_92;
          goto LABEL_82;
        }
        v89 = sub_1FE7940(v128, *v84, v72, v81, v73);
        if ( v89 )
          break;
LABEL_86:
        if ( ++v84 == (_QWORD *)v109 )
          goto LABEL_92;
      }
      if ( !v83 )
      {
        v103 = v89;
        sub_1DD5BA0(*(_QWORD *)(a1 + 616) + 16LL, v89);
        v86 = *(_QWORD *)v103;
        v87 = *v33 & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v103 + 8) = v33;
        *(_QWORD *)v103 = v87 | v86 & 7;
        *(_QWORD *)(v87 + 8) = v103;
        v81 = *v33 & 7LL;
        *v33 = v81 | v103;
        goto LABEL_86;
      }
      v104 = v89;
      ++v84;
      sub_1DD5BA0(v85[3] + 16, v89);
      v90 = *(_QWORD *)v104;
      v91 = *v85 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v104 + 8) = v85;
      *(_QWORD *)v104 = v91 | v90 & 7;
      *(_QWORD *)(v91 + 8) = v104;
      v81 = *v85 & 7;
      *v85 = v81 | v104;
    }
    while ( v84 != (_QWORD *)v109 );
  }
LABEL_92:
  if ( v131 != (unsigned __int64 **)v133 )
    _libc_free((unsigned __int64)v131);
LABEL_39:
  v23 = v137;
LABEL_40:
  v24 = v129;
  *a2 = v130;
  sub_1D0C7F0(v23);
  if ( (_BYTE *)v134[0] != v135 )
    _libc_free(v134[0]);
  if ( v141 != (__m128i *)v143 )
    _libc_free((unsigned __int64)v141);
  j___libc_free_0(v125);
  j___libc_free_0(v121);
  return v24;
}
