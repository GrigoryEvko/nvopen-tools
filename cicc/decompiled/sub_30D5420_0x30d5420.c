// Function: sub_30D5420
// Address: 0x30d5420
//
const char *__fastcall sub_30D5420(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // rax
  unsigned int v3; // ecx
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // rbx
  char v9; // al
  __int64 v10; // rcx
  int v11; // edx
  __int64 v12; // rsi
  int v13; // edx
  unsigned int v14; // edi
  __int64 *v15; // rax
  __int64 v16; // r8
  _BYTE *v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // r13
  __int64 v20; // rdi
  __int64 v21; // r14
  __int64 *v22; // r13
  int v23; // eax
  __int64 v24; // rdx
  int v25; // r13d
  __int64 v26; // rax
  unsigned int v27; // edx
  unsigned __int64 *v28; // r14
  unsigned __int64 *v29; // rax
  __int64 v30; // rax
  unsigned __int64 v31; // rsi
  char v32; // r13
  const char *result; // rax
  int v34; // edx
  bool v35; // cc
  int v36; // edx
  int v37; // edx
  unsigned int v38; // eax
  __int64 v39; // rsi
  int v40; // r8d
  int v41; // ecx
  __int64 v42; // rsi
  __int64 v43; // rdi
  int v44; // ecx
  unsigned int v45; // edx
  __int64 v46; // r8
  int v47; // eax
  int v48; // r9d
  int v49; // eax
  int v50; // r9d
  __int64 v51; // rax
  unsigned __int64 v52; // rbx
  unsigned __int64 v53; // r13
  int v54; // r12d
  __int64 v55; // rsi
  _QWORD *v56; // rax
  _QWORD *v57; // rdx
  __int64 v58; // rsi
  unsigned int v59; // ecx
  __int64 v60; // rsi
  __int64 v61; // rdx
  _QWORD *v62; // rax
  _QWORD *v63; // rdx
  unsigned __int64 v64; // r12
  unsigned __int64 v65; // r14
  __int64 v66; // r15
  __int64 *v67; // rbx
  __int64 *v68; // r13
  __int64 v69; // rdi
  __int64 v70; // rax
  __int64 v71; // rax
  unsigned int v72; // eax
  __int64 v73; // rdx
  char v74; // al
  unsigned __int64 v75; // rdi
  unsigned __int64 v76; // rdi
  __int64 *v77; // rbx
  __int64 *v78; // r12
  __int64 v79; // rsi
  __int64 v80; // rdi
  __int64 *v81; // rax
  __int64 *v82; // rbx
  __int64 *v83; // r13
  __int64 v84; // rdi
  unsigned int v85; // ecx
  __int64 v86; // rsi
  __int64 *v87; // rbx
  __int64 *v88; // r12
  __int64 v89; // rsi
  __int64 v90; // rdi
  _BYTE *v91; // rbx
  _BYTE *v92; // r12
  unsigned __int64 v93; // r13
  unsigned __int64 v94; // rdi
  unsigned __int64 v95; // rdi
  unsigned __int64 v96; // rdi
  unsigned __int64 *v97; // rax
  int v98; // eax
  __int64 v99; // rax
  unsigned __int64 v100; // rsi
  _QWORD *v101; // rdi
  __int64 v102; // r8
  unsigned int v103; // eax
  int v104; // r12d
  unsigned int v105; // eax
  _QWORD *v106; // rax
  _QWORD *i; // rdx
  _QWORD *v108; // r8
  unsigned int v109; // [rsp+Ch] [rbp-1A4h]
  __int64 v110; // [rsp+10h] [rbp-1A0h]
  __int64 *v111; // [rsp+18h] [rbp-198h]
  __int64 v112; // [rsp+18h] [rbp-198h]
  unsigned int v113; // [rsp+28h] [rbp-188h]
  unsigned __int64 *v114; // [rsp+30h] [rbp-180h] BYREF
  unsigned int v115; // [rsp+38h] [rbp-178h]
  __int64 v116; // [rsp+40h] [rbp-170h] BYREF
  int v117; // [rsp+48h] [rbp-168h]
  unsigned __int64 *v118; // [rsp+50h] [rbp-160h] BYREF
  unsigned int v119; // [rsp+58h] [rbp-158h]
  unsigned __int64 *v120; // [rsp+60h] [rbp-150h] BYREF
  __int64 v121; // [rsp+68h] [rbp-148h]
  char v122; // [rsp+70h] [rbp-140h] BYREF
  _BYTE *v123; // [rsp+78h] [rbp-138h]
  __int64 v124; // [rsp+80h] [rbp-130h]
  _BYTE v125[56]; // [rsp+88h] [rbp-128h] BYREF
  __int64 v126; // [rsp+C0h] [rbp-F0h]
  __int64 v127; // [rsp+C8h] [rbp-E8h]
  char v128; // [rsp+D0h] [rbp-E0h]
  __int64 v129; // [rsp+D4h] [rbp-DCh]
  unsigned __int64 v130; // [rsp+E0h] [rbp-D0h] BYREF
  _QWORD *v131; // [rsp+E8h] [rbp-C8h]
  __int64 v132; // [rsp+F0h] [rbp-C0h]
  unsigned int v133; // [rsp+F8h] [rbp-B8h]
  unsigned __int64 v134; // [rsp+100h] [rbp-B0h]
  unsigned __int64 v135; // [rsp+108h] [rbp-A8h]
  __int64 v136; // [rsp+118h] [rbp-98h]
  __int64 j; // [rsp+120h] [rbp-90h]
  __int64 *v138; // [rsp+128h] [rbp-88h]
  unsigned int v139; // [rsp+130h] [rbp-80h]
  char v140; // [rsp+138h] [rbp-78h] BYREF
  __int64 *v141; // [rsp+158h] [rbp-58h]
  unsigned int v142; // [rsp+160h] [rbp-50h]
  __int64 v143; // [rsp+168h] [rbp-48h] BYREF

  v1 = a1;
  v2 = sub_B43CB0(*(_QWORD *)(a1 + 96));
  if ( !(unsigned __int8)sub_B2D610(v2, 18) )
    goto LABEL_2;
  v51 = *(_QWORD *)(a1 + 72);
  v129 = 0;
  v120 = (unsigned __int64 *)&v122;
  v121 = 0x100000000LL;
  v123 = v125;
  v124 = 0x600000000LL;
  v126 = 0;
  v128 = 0;
  v127 = v51;
  HIDWORD(v129) = *(_DWORD *)(v51 + 92);
  sub_B1F440((__int64)&v120);
  sub_D51D90((__int64)&v130, (__int64)&v120);
  v52 = v134;
  v53 = v135;
  if ( v134 != v135 )
  {
    v54 = 0;
    while ( 1 )
    {
      v55 = **(_QWORD **)(*(_QWORD *)v52 + 32LL);
      if ( *(_BYTE *)(a1 + 292) )
      {
        v56 = *(_QWORD **)(a1 + 272);
        v57 = &v56[*(unsigned int *)(a1 + 284)];
        if ( v56 == v57 )
          goto LABEL_147;
        while ( v55 != *v56 )
        {
          if ( v57 == ++v56 )
            goto LABEL_147;
        }
LABEL_88:
        v52 += 8LL;
        if ( v53 == v52 )
          goto LABEL_89;
      }
      else
      {
        if ( sub_C8CA60(a1 + 264, v55) )
          goto LABEL_88;
LABEL_147:
        v52 += 8LL;
        ++v54;
        if ( v53 == v52 )
        {
LABEL_89:
          v58 = 25 * v54;
          goto LABEL_90;
        }
      }
    }
  }
  v58 = 0;
LABEL_90:
  sub_30D0F50(a1, v58);
  ++v130;
  if ( !(_DWORD)v132 )
  {
    v60 = HIDWORD(v132);
    if ( !HIDWORD(v132) )
      goto LABEL_97;
    v61 = v133;
    if ( v133 <= 0x40 )
    {
LABEL_94:
      v62 = v131;
      v63 = &v131[2 * v61];
      if ( v131 != v63 )
      {
        do
        {
          *v62 = -4096;
          v62 += 2;
        }
        while ( v63 != v62 );
      }
      goto LABEL_96;
    }
    v60 = 16LL * v133;
    sub_C7D6A0((__int64)v131, v60, 8);
    v133 = 0;
    goto LABEL_152;
  }
  v59 = 4 * v132;
  v60 = 64;
  v61 = v133;
  if ( (unsigned int)(4 * v132) < 0x40 )
    v59 = 64;
  if ( v133 <= v59 )
    goto LABEL_94;
  v101 = v131;
  v102 = 2LL * v133;
  if ( (_DWORD)v132 == 1 )
  {
    v104 = 64;
    goto LABEL_184;
  }
  _BitScanReverse(&v103, v132 - 1);
  v104 = 1 << (33 - (v103 ^ 0x1F));
  if ( v104 < 64 )
    v104 = 64;
  if ( v104 != v133 )
  {
LABEL_184:
    v60 = 16LL * v133;
    sub_C7D6A0((__int64)v131, v102 * 8, 8);
    v105 = sub_30D1A00(v104);
    v133 = v105;
    if ( v105 )
    {
      v60 = 8;
      v106 = (_QWORD *)sub_C7D670(16LL * v105, 8);
      v132 = 0;
      v131 = v106;
      for ( i = &v106[2 * v133]; i != v106; v106 += 2 )
      {
        if ( v106 )
          *v106 = -4096;
      }
      goto LABEL_97;
    }
LABEL_152:
    v131 = 0;
LABEL_96:
    v132 = 0;
    goto LABEL_97;
  }
  v132 = 0;
  v108 = &v131[v102];
  do
  {
    if ( v101 )
      *v101 = -4096;
    v101 += 2;
  }
  while ( v108 != v101 );
LABEL_97:
  v64 = v134;
  if ( v134 != v135 )
  {
    v112 = v1;
    v65 = v135;
    do
    {
      v66 = *(_QWORD *)v64;
      v67 = *(__int64 **)(*(_QWORD *)v64 + 8LL);
      v68 = *(__int64 **)(*(_QWORD *)v64 + 16LL);
      if ( v67 == v68 )
      {
        *(_BYTE *)(v66 + 152) = 1;
      }
      else
      {
        do
        {
          v69 = *v67++;
          sub_D47BB0(v69, v60);
        }
        while ( v68 != v67 );
        *(_BYTE *)(v66 + 152) = 1;
        v70 = *(_QWORD *)(v66 + 8);
        if ( *(_QWORD *)(v66 + 16) != v70 )
          *(_QWORD *)(v66 + 16) = v70;
      }
      v71 = *(_QWORD *)(v66 + 32);
      if ( v71 != *(_QWORD *)(v66 + 40) )
        *(_QWORD *)(v66 + 40) = v71;
      ++*(_QWORD *)(v66 + 56);
      if ( *(_BYTE *)(v66 + 84) )
      {
        *(_QWORD *)v66 = 0;
      }
      else
      {
        v72 = 4 * (*(_DWORD *)(v66 + 76) - *(_DWORD *)(v66 + 80));
        v73 = *(unsigned int *)(v66 + 72);
        if ( v72 < 0x20 )
          v72 = 32;
        if ( (unsigned int)v73 > v72 )
        {
          sub_C8C990(v66 + 56, v60);
        }
        else
        {
          v60 = 0xFFFFFFFFLL;
          memset(*(void **)(v66 + 64), -1, 8 * v73);
        }
        v74 = *(_BYTE *)(v66 + 84);
        *(_QWORD *)v66 = 0;
        if ( !v74 )
          _libc_free(*(_QWORD *)(v66 + 64));
      }
      v75 = *(_QWORD *)(v66 + 32);
      if ( v75 )
      {
        v60 = *(_QWORD *)(v66 + 48) - v75;
        j_j___libc_free_0(v75);
      }
      v76 = *(_QWORD *)(v66 + 8);
      if ( v76 )
      {
        v60 = *(_QWORD *)(v66 + 24) - v76;
        j_j___libc_free_0(v76);
      }
      v64 += 8LL;
    }
    while ( v65 != v64 );
    v1 = v112;
    if ( v134 != v135 )
      v135 = v134;
  }
  v77 = v141;
  v78 = &v141[2 * v142];
  if ( v141 != v78 )
  {
    do
    {
      v79 = v77[1];
      v80 = *v77;
      v77 += 2;
      sub_C7D6A0(v80, v79, 16);
    }
    while ( v78 != v77 );
  }
  v142 = 0;
  if ( v139 )
  {
    v81 = v138;
    v143 = 0;
    v82 = &v138[v139];
    v83 = v138 + 1;
    v136 = *v138;
    for ( j = v136 + 4096; v82 != v83; v81 = v138 )
    {
      v84 = *v83;
      v85 = (unsigned int)(v83 - v81) >> 7;
      v86 = 4096LL << v85;
      if ( v85 >= 0x1E )
        v86 = 0x40000000000LL;
      ++v83;
      sub_C7D6A0(v84, v86, 16);
    }
    v139 = 1;
    sub_C7D6A0(*v81, 4096, 16);
    v87 = v141;
    v88 = &v141[2 * v142];
    if ( v141 == v88 )
      goto LABEL_129;
    do
    {
      v89 = v87[1];
      v90 = *v87;
      v87 += 2;
      sub_C7D6A0(v90, v89, 16);
    }
    while ( v88 != v87 );
  }
  v88 = v141;
LABEL_129:
  if ( v88 != &v143 )
    _libc_free((unsigned __int64)v88);
  if ( v138 != (__int64 *)&v140 )
    _libc_free((unsigned __int64)v138);
  if ( v134 )
    j_j___libc_free_0(v134);
  sub_C7D6A0((__int64)v131, 16LL * v133, 8);
  v91 = v123;
  v92 = &v123[8 * (unsigned int)v124];
  if ( v123 != v92 )
  {
    do
    {
      v93 = *((_QWORD *)v92 - 1);
      v92 -= 8;
      if ( v93 )
      {
        v94 = *(_QWORD *)(v93 + 24);
        if ( v94 != v93 + 40 )
          _libc_free(v94);
        j_j___libc_free_0(v93);
      }
    }
    while ( v91 != v92 );
    v92 = v123;
  }
  if ( v92 != v125 )
    _libc_free((unsigned __int64)v92);
  if ( v120 != (unsigned __int64 *)&v122 )
    _libc_free((unsigned __int64)v120);
LABEL_2:
  v3 = *(_DWORD *)(v1 + 132);
  if ( v3 > *(_DWORD *)(v1 + 128) / 0xAu )
  {
    if ( v3 <= *(_DWORD *)(v1 + 128) >> 1 )
      *(_DWORD *)(v1 + 704) -= *(_DWORD *)(v1 + 656) / 2;
  }
  else
  {
    *(_DWORD *)(v1 + 704) -= *(_DWORD *)(v1 + 656);
  }
  v130 = sub_30D4EB0(*(_QWORD *)(v1 + 96), "function-inline-cost", 0x14u);
  if ( BYTE4(v130) )
    *(_DWORD *)(v1 + 716) = v130;
  v130 = sub_30D4EB0(*(_QWORD *)(v1 + 96), "function-inline-cost-multiplier", 0x1Fu);
  if ( BYTE4(v130) )
    *(_DWORD *)(v1 + 716) *= (_DWORD)v130;
  v130 = sub_30D4EB0(*(_QWORD *)(v1 + 96), "function-inline-threshold", 0x19u);
  if ( BYTE4(v130) )
    *(_DWORD *)(v1 + 704) = v130;
  LOBYTE(v109) = *(_BYTE *)(v1 + 714);
  if ( !(_BYTE)v109 || !*(_DWORD *)(v1 + 704) )
    goto LABEL_57;
  v4 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(v1 + 32))(*(_QWORD *)(v1 + 40), *(_QWORD *)(v1 + 72));
  v115 = 128;
  v111 = (__int64 *)v4;
  sub_C43690((__int64)&v114, 0, 0);
  v5 = *(_QWORD *)(v1 + 72);
  v6 = *(_QWORD *)(v5 + 80);
  v110 = v5 + 72;
  if ( v6 != v5 + 72 )
  {
    while ( 1 )
    {
      if ( !v6 )
      {
        LODWORD(v121) = 128;
        sub_C43690((__int64)&v120, 0, 0);
        BUG();
      }
      v7 = v6 + 24;
      LODWORD(v121) = 128;
      sub_C43690((__int64)&v120, 0, 0);
      v8 = *(_QWORD *)(v6 + 32);
      if ( v8 != v6 + 24 )
        break;
LABEL_26:
      v130 = sub_FDD2C0(v111, v6 - 24, 0);
      v131 = (_QWORD *)v18;
      sub_C47170((__int64)&v120, v130);
      sub_C45EE0((__int64)&v114, (__int64 *)&v120);
      if ( (unsigned int)v121 > 0x40 && v120 )
        j_j___libc_free_0_0((unsigned __int64)v120);
      v6 = *(_QWORD *)(v6 + 8);
      if ( v110 == v6 )
      {
        v5 = *(_QWORD *)(v1 + 72);
        goto LABEL_31;
      }
    }
    while ( 1 )
    {
      if ( !v8 )
        BUG();
      v9 = *(_BYTE *)(v8 - 24);
      if ( v9 == 31 )
        break;
      v10 = *(_QWORD *)(v1 + 144);
      v11 = *(_DWORD *)(v1 + 160);
      if ( v9 == 32 )
      {
        v12 = **(_QWORD **)(v8 - 32);
        if ( !v11 )
          goto LABEL_17;
        v13 = v11 - 1;
        v14 = v13 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v15 = (__int64 *)(v10 + 16LL * v14);
        v16 = *v15;
        if ( *v15 != v12 )
        {
          v49 = 1;
          while ( v16 != -4096 )
          {
            v50 = v49 + 1;
            v14 = v13 & (v49 + v14);
            v15 = (__int64 *)(v10 + 16LL * v14);
            v16 = *v15;
            if ( v12 == *v15 )
              goto LABEL_23;
            v49 = v50;
          }
          goto LABEL_17;
        }
LABEL_23:
        v17 = (_BYTE *)v15[1];
        if ( v17 && *v17 == 17 )
          goto LABEL_25;
LABEL_17:
        v8 = *(_QWORD *)(v8 + 8);
        if ( v7 == v8 )
          goto LABEL_26;
      }
      else
      {
        if ( !v11 )
          goto LABEL_17;
        v37 = v11 - 1;
        v38 = v37 & (((unsigned int)(v8 - 24) >> 9) ^ ((unsigned int)(v8 - 24) >> 4));
        v39 = *(_QWORD *)(v10 + 16LL * v38);
        if ( v39 != v8 - 24 )
        {
          v40 = 1;
          while ( v39 != -4096 )
          {
            v38 = v37 & (v40 + v38);
            v39 = *(_QWORD *)(v10 + 16LL * v38);
            if ( v8 - 24 == v39 )
              goto LABEL_25;
            ++v40;
          }
          goto LABEL_17;
        }
LABEL_25:
        sub_C46A40((__int64)&v120, (int)qword_5030168);
        v8 = *(_QWORD *)(v8 + 8);
        if ( v7 == v8 )
          goto LABEL_26;
      }
    }
    if ( (*(_DWORD *)(v8 - 20) & 0x7FFFFFF) != 3 )
      goto LABEL_17;
    v41 = *(_DWORD *)(v1 + 160);
    v42 = *(_QWORD *)(v8 - 120);
    v43 = *(_QWORD *)(v1 + 144);
    if ( !v41 )
      goto LABEL_17;
    v44 = v41 - 1;
    v45 = v44 & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
    v15 = (__int64 *)(v43 + 16LL * v45);
    v46 = *v15;
    if ( *v15 != v42 )
    {
      v47 = 1;
      while ( v46 != -4096 )
      {
        v48 = v47 + 1;
        v45 = v44 & (v47 + v45);
        v15 = (__int64 *)(v43 + 16LL * v45);
        v46 = *v15;
        if ( v42 == *v15 )
          goto LABEL_23;
        v47 = v48;
      }
      goto LABEL_17;
    }
    goto LABEL_23;
  }
LABEL_31:
  sub_B2EE70((__int64)&v130, v5, 0);
  v19 = v130;
  sub_C46A40((__int64)&v114, v130 >> 1);
  sub_C45850((__int64)&v120, &v114, v19);
  if ( v115 > 0x40 && v114 )
    j_j___libc_free_0_0((unsigned __int64)v114);
  v20 = *(_QWORD *)(v1 + 40);
  v114 = v120;
  v115 = v121;
  v21 = *(_QWORD *)(*(_QWORD *)(v1 + 96) + 40LL);
  v22 = (__int64 *)(*(__int64 (__fastcall **)(__int64, _QWORD))(v1 + 32))(v20, *(_QWORD *)(v21 + 72));
  v23 = sub_30D4FE0(*(__int64 **)(v1 + 8), *(unsigned __int8 **)(v1 + 96), *(_QWORD *)(v1 + 80));
  sub_C46A40((__int64)&v114, v23);
  v120 = (unsigned __int64 *)sub_FDD2C0(v22, v21, 0);
  v121 = v24;
  sub_C47170((__int64)&v114, (unsigned __int64)v120);
  v25 = *(_DWORD *)(v1 + 716) - *(_DWORD *)(v1 + 724) - dword_50306A8;
  if ( *(_DWORD *)(v1 + 716) - *(_DWORD *)(v1 + 724) <= dword_50306A8 )
    v25 = 1;
  v120 = (unsigned __int64 *)sub_30D4EB0(*(_QWORD *)(v1 + 96), "inline-cycle-savings-for-test", 0x1Du);
  if ( BYTE4(v120) )
  {
    if ( v115 > 0x40 )
    {
      *v114 = (int)v120;
      memset(v114 + 1, 0, 8 * (unsigned int)(((unsigned __int64)v115 + 63) >> 6) - 8);
    }
    else
    {
      v97 = (unsigned __int64 *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v115) & (int)v120);
      if ( !v115 )
        v97 = 0;
      v114 = v97;
    }
  }
  v26 = sub_30D4EB0(*(_QWORD *)(v1 + 96), "inline-runtime-cost-for-test", 0x1Cu);
  v119 = 128;
  v120 = (unsigned __int64 *)v26;
  if ( BYTE4(v26) )
    v25 = (int)v120;
  sub_C43690((__int64)&v118, v25, 0);
  if ( *(_BYTE *)(v1 + 768) )
  {
    v35 = *(_DWORD *)(v1 + 760) <= 0x40u;
    *(_BYTE *)(v1 + 768) = 0;
    if ( !v35 )
    {
      v95 = *(_QWORD *)(v1 + 752);
      if ( v95 )
        j_j___libc_free_0_0(v95);
    }
    if ( *(_DWORD *)(v1 + 744) > 0x40u )
    {
      v96 = *(_QWORD *)(v1 + 736);
      if ( v96 )
        j_j___libc_free_0_0(v96);
    }
  }
  v27 = v119;
  v119 = 0;
  v28 = v118;
  LODWORD(v121) = v115;
  if ( v115 > 0x40 )
  {
    v113 = v27;
    sub_C43780((__int64)&v120, (const void **)&v114);
    v98 = v121;
    *(_QWORD *)(v1 + 736) = v28;
    v35 = v119 <= 0x40;
    *(_BYTE *)(v1 + 768) = 1;
    *(_DWORD *)(v1 + 760) = v98;
    v99 = (__int64)v120;
    *(_DWORD *)(v1 + 744) = v113;
    *(_QWORD *)(v1 + 752) = v99;
    if ( !v35 && v118 )
      j_j___libc_free_0_0((unsigned __int64)v118);
  }
  else
  {
    *(_DWORD *)(v1 + 760) = v115;
    v29 = v114;
    *(_DWORD *)(v1 + 744) = v27;
    *(_QWORD *)(v1 + 736) = v28;
    *(_QWORD *)(v1 + 752) = v29;
    *(_BYTE *)(v1 + 768) = 1;
  }
  v30 = sub_D844E0(*(_QWORD *)(v1 + 64));
  v117 = 128;
  sub_C43690((__int64)&v116, v30, 0);
  sub_C47170((__int64)&v116, v25);
  v119 = v115;
  if ( v115 > 0x40 )
    sub_C43780((__int64)&v118, (const void **)&v114);
  else
    v118 = v114;
  if ( (unsigned int)sub_23DF0D0(&dword_50307E8) )
    v31 = (unsigned int)qword_5030868;
  else
    v31 = (unsigned int)sub_DF93E0(*(_QWORD *)(v1 + 8));
  sub_C47170((__int64)&v118, v31);
  v32 = 1;
  if ( (int)sub_C49970((__int64)&v118, (unsigned __int64 *)&v116) < 0 )
  {
    LODWORD(v121) = v115;
    if ( v115 > 0x40 )
      sub_C43780((__int64)&v120, (const void **)&v114);
    else
      v120 = v114;
    if ( (unsigned int)sub_23DF0D0(&dword_5030708) )
      v100 = (unsigned int)qword_5030788;
    else
      v100 = (unsigned int)sub_DF9410(*(_QWORD *)(v1 + 8));
    v32 = 0;
    sub_C47170((__int64)&v120, v100);
    v109 = (unsigned int)sub_C49970((__int64)&v120, (unsigned __int64 *)&v116) >> 31;
    sub_969240((__int64 *)&v120);
  }
  if ( v119 > 0x40 && v118 )
    j_j___libc_free_0_0((unsigned __int64)v118);
  sub_969240(&v116);
  if ( v115 > 0x40 && v114 )
    j_j___libc_free_0_0((unsigned __int64)v114);
  if ( (_BYTE)v109 )
  {
    result = "Cost over threshold.";
    *(_BYTE *)(v1 + 729) = 1;
    if ( v32 )
      return 0;
    return result;
  }
LABEL_57:
  result = 0;
  if ( !*(_BYTE *)(v1 + 713) )
  {
    v34 = *(_DWORD *)(v1 + 704);
    *(_BYTE *)(v1 + 728) = 1;
    v35 = v34 <= 0;
    v36 = 1;
    if ( !v35 )
      v36 = *(_DWORD *)(v1 + 704);
    if ( *(_DWORD *)(v1 + 716) >= v36 )
      return "Cost over threshold.";
  }
  return result;
}
