// Function: sub_2815B30
// Address: 0x2815b30
//
__int64 __fastcall sub_2815B30(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rbx
  char v8; // cl
  __int64 v9; // rdi
  int v10; // r10d
  unsigned int v11; // esi
  void **v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // rdx
  __int64 result; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  int v18; // r10d
  unsigned int v19; // esi
  _QWORD *v20; // rdx
  _QWORD *v21; // r8
  __int64 v22; // rax
  unsigned int v23; // esi
  int v24; // eax
  _QWORD *v25; // r15
  _QWORD *v26; // r13
  char v27; // r12
  __int64 v28; // rbx
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rdx
  __int64 v33; // rcx
  _BYTE *v34; // rdi
  _QWORD *v35; // r15
  _QWORD *v36; // r13
  char v37; // r12
  __int64 v38; // rbx
  __int64 v39; // rax
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rdx
  __int64 v43; // rcx
  _QWORD *v44; // r15
  _QWORD *v45; // r13
  char v46; // r12
  __int64 v47; // rbx
  __int64 v48; // rax
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rdx
  __int64 v52; // rcx
  _QWORD *v53; // r15
  _QWORD *v54; // r13
  char v55; // r12
  __int64 v56; // rbx
  __int64 v57; // rax
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // rsi
  __int64 v63; // rsi
  __int64 v64; // r12
  __int64 v65; // rax
  _QWORD *v66; // r15
  _QWORD *v67; // r13
  char v68; // r12
  __int64 v69; // rbx
  __int64 v70; // rax
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // rdx
  __int64 v74; // rsi
  _QWORD *v75; // r15
  _QWORD *v76; // r13
  char v77; // r12
  __int64 v78; // rbx
  __int64 v79; // rax
  __int64 v80; // r8
  __int64 v81; // r9
  __int64 v82; // rdx
  __int64 v83; // rcx
  __int64 v84; // r12
  _QWORD *v85; // rdx
  _QWORD *v86; // r10
  __int64 v87; // rcx
  _QWORD *v88; // rax
  __int64 v89; // r15
  _QWORD *v90; // r12
  char v91; // al
  _QWORD *v92; // r15
  _QWORD *v93; // r13
  char v94; // r12
  __int64 v95; // rbx
  __int64 v96; // rax
  __int64 v97; // r8
  __int64 v98; // r9
  __int64 v99; // rdx
  __int64 v100; // rsi
  unsigned int v101; // edx
  _QWORD *v102; // r11
  int v103; // edi
  unsigned int v104; // r10d
  int v105; // r12d
  _QWORD *v106; // r13
  _QWORD *i; // r15
  __int64 v108; // rax
  __int64 v109; // r9
  __int64 v110; // rdx
  unsigned __int64 v111; // r8
  __int64 v112; // rdi
  int v113; // ecx
  unsigned int v114; // edx
  __int64 v115; // rsi
  __int64 v116; // rdi
  int v117; // ecx
  unsigned int v118; // edx
  __int64 v119; // rsi
  int v120; // r8d
  _QWORD *v121; // r10
  int v122; // ecx
  int v123; // ecx
  size_t v124; // r12
  __int64 v125; // r13
  __int64 v126; // rdx
  __int64 v127; // rdi
  unsigned int v128; // ecx
  signed __int64 v129; // r11
  int v130; // r10d
  const void *v131; // r8
  __int64 v132; // r9
  _BYTE *v133; // rdi
  _BYTE *v134; // rdi
  int v135; // r8d
  _BYTE *v136; // rdi
  size_t v137; // [rsp+0h] [rbp-E0h]
  void *src; // [rsp+8h] [rbp-D8h]
  __int64 v139; // [rsp+10h] [rbp-D0h]
  __int64 v140; // [rsp+10h] [rbp-D0h]
  __int64 v141; // [rsp+10h] [rbp-D0h]
  __int64 v142; // [rsp+10h] [rbp-D0h]
  __int64 v143; // [rsp+10h] [rbp-D0h]
  __int64 v144; // [rsp+10h] [rbp-D0h]
  __int64 v145; // [rsp+10h] [rbp-D0h]
  _QWORD *v146; // [rsp+10h] [rbp-D0h]
  int v147; // [rsp+10h] [rbp-D0h]
  size_t n; // [rsp+18h] [rbp-C8h]
  signed __int64 na; // [rsp+18h] [rbp-C8h]
  int nb; // [rsp+18h] [rbp-C8h]
  size_t nc; // [rsp+18h] [rbp-C8h]
  _QWORD *dest; // [rsp+20h] [rbp-C0h]
  _QWORD *desta; // [rsp+20h] [rbp-C0h]
  _QWORD *destb; // [rsp+20h] [rbp-C0h]
  _QWORD *destc; // [rsp+20h] [rbp-C0h]
  _QWORD *destd; // [rsp+20h] [rbp-C0h]
  _QWORD *deste; // [rsp+20h] [rbp-C0h]
  _QWORD *destf; // [rsp+20h] [rbp-C0h]
  __int64 v159; // [rsp+28h] [rbp-B8h]
  __int64 v160; // [rsp+28h] [rbp-B8h]
  __int64 v161; // [rsp+28h] [rbp-B8h]
  __int64 v162; // [rsp+28h] [rbp-B8h]
  __int64 v163; // [rsp+28h] [rbp-B8h]
  _BYTE *v164; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v165; // [rsp+38h] [rbp-A8h]
  _BYTE v166[16]; // [rsp+40h] [rbp-A0h] BYREF
  _BYTE *v167; // [rsp+50h] [rbp-90h] BYREF
  __int64 v168; // [rsp+58h] [rbp-88h]
  _BYTE v169[32]; // [rsp+60h] [rbp-80h] BYREF
  _BYTE *v170; // [rsp+80h] [rbp-60h] BYREF
  __int64 v171; // [rsp+88h] [rbp-58h]
  _BYTE v172[80]; // [rsp+90h] [rbp-50h] BYREF

  v7 = a2;
  v8 = *(_BYTE *)(a1 + 16) & 1;
  if ( v8 )
  {
    v9 = a1 + 24;
    v10 = 3;
  }
  else
  {
    v16 = *(unsigned int *)(a1 + 32);
    v9 = *(_QWORD *)(a1 + 24);
    if ( !(_DWORD)v16 )
      goto LABEL_17;
    v10 = v16 - 1;
  }
  v11 = v10 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = (void **)(v9 + 16LL * (v10 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4))));
  v13 = *v12;
  if ( *v12 == v7 )
    goto LABEL_4;
  v24 = 1;
  while ( v13 != (_QWORD *)-4096LL )
  {
    a5 = (unsigned int)(v24 + 1);
    v11 = v10 & (v24 + v11);
    v12 = (void **)(v9 + 16LL * v11);
    v13 = *v12;
    if ( *v12 == v7 )
      goto LABEL_4;
    v24 = a5;
  }
  if ( v8 )
  {
    v22 = 64;
    goto LABEL_18;
  }
  v16 = *(unsigned int *)(a1 + 32);
LABEL_17:
  v22 = 16 * v16;
LABEL_18:
  v12 = (void **)(v9 + v22);
LABEL_4:
  v14 = 64;
  if ( !v8 )
    v14 = 16LL * *(unsigned int *)(a1 + 32);
  if ( v12 != (void **)(v9 + v14) )
    return (__int64)v12[1];
  switch ( *((_WORD *)v7 + 12) )
  {
    case 0:
    case 1:
    case 0xF:
    case 0x10:
      goto LABEL_11;
    case 2:
      v63 = sub_2815B30(a1, v7[4]);
      if ( v63 == v7[4] )
        goto LABEL_108;
      result = (__int64)sub_DC5200(*(_QWORD *)a1, v63, v7[5], 0);
      v8 = *(_BYTE *)(a1 + 16) & 1;
      goto LABEL_12;
    case 3:
      v62 = sub_2815B30(a1, v7[4]);
      if ( v62 == v7[4] )
        goto LABEL_108;
      result = (__int64)sub_DC2B70(*(_QWORD *)a1, v62, v7[5], 0);
      v8 = *(_BYTE *)(a1 + 16) & 1;
      goto LABEL_12;
    case 4:
      v100 = sub_2815B30(a1, v7[4]);
      if ( v100 == v7[4] )
        goto LABEL_108;
      result = (__int64)sub_DC5000(*(_QWORD *)a1, v100, v7[5], 0);
      v8 = *(_BYTE *)(a1 + 16) & 1;
      goto LABEL_12;
    case 5:
      v92 = (_QWORD *)v7[4];
      v170 = v172;
      v171 = 0x200000000LL;
      v93 = &v92[v7[5]];
      if ( v92 == v93 )
        goto LABEL_11;
      destf = v7;
      v94 = 0;
      do
      {
        v95 = *v92;
        v96 = sub_2815B30(a1, *v92);
        v99 = (unsigned int)v171;
        if ( (unsigned __int64)(unsigned int)v171 + 1 > HIDWORD(v171) )
        {
          v139 = v96;
          sub_C8D5F0((__int64)&v170, v172, (unsigned int)v171 + 1LL, 8u, v97, v98);
          v99 = (unsigned int)v171;
          v96 = v139;
        }
        *(_QWORD *)&v170[8 * v99] = v96;
        v34 = v170;
        LODWORD(v171) = v171 + 1;
        ++v92;
        v94 |= *(_QWORD *)&v170[8 * (unsigned int)v171 - 8] != v95;
      }
      while ( v93 != v92 );
      v7 = destf;
      result = (__int64)destf;
      if ( v94 )
      {
        result = (__int64)sub_DC7EB0(*(__int64 **)a1, (__int64)&v170, 0, 0);
        v34 = v170;
      }
      goto LABEL_46;
    case 6:
      v66 = (_QWORD *)v7[4];
      v170 = v172;
      v171 = 0x200000000LL;
      v67 = &v66[v7[5]];
      if ( v66 == v67 )
        goto LABEL_11;
      destd = v7;
      v68 = 0;
      do
      {
        v69 = *v66;
        v70 = sub_2815B30(a1, *v66);
        v73 = (unsigned int)v171;
        if ( (unsigned __int64)(unsigned int)v171 + 1 > HIDWORD(v171) )
        {
          v141 = v70;
          sub_C8D5F0((__int64)&v170, v172, (unsigned int)v171 + 1LL, 8u, v71, v72);
          v73 = (unsigned int)v171;
          v70 = v141;
        }
        *(_QWORD *)&v170[8 * v73] = v70;
        v34 = v170;
        LODWORD(v171) = v171 + 1;
        ++v66;
        v68 |= *(_QWORD *)&v170[8 * (unsigned int)v171 - 8] != v69;
      }
      while ( v67 != v66 );
      v7 = destd;
      result = (__int64)destd;
      if ( v68 )
      {
        result = (__int64)sub_DC8BD0(*(__int64 **)a1, (__int64)&v170, 0, 0);
        v34 = v170;
      }
      goto LABEL_46;
    case 7:
      v64 = sub_2815B30(a1, v7[4]);
      v65 = sub_2815B30(a1, v7[5]);
      if ( v64 == v7[4] && v65 == v7[5] )
        goto LABEL_108;
      result = sub_DCB270(*(_QWORD *)a1, v64, v65);
      goto LABEL_61;
    case 8:
      v84 = v7[6];
      v85 = *(_QWORD **)(a1 + 96);
      v86 = (_QWORD *)v7[4];
      v87 = v7[5];
      v164 = v166;
      v165 = 0x200000000LL;
      if ( (_QWORD *)v84 == v85 )
      {
        v124 = 8 * v87;
        v125 = (8 * v87) >> 3;
        if ( (unsigned __int64)(8 * v87) > 0x10 )
        {
          nc = (size_t)v86;
          sub_C8D5F0((__int64)&v164, v166, (8 * v87) >> 3, 8u, a5, a6);
          v86 = (_QWORD *)nc;
          v136 = &v164[8 * (unsigned int)v165];
        }
        else
        {
          if ( !v124 )
          {
LABEL_139:
            v126 = *(_QWORD *)(a1 + 104);
            v127 = *(_QWORD *)a1;
            v128 = *((_WORD *)v7 + 14) & 7;
            LODWORD(v165) = v125 + v165;
            result = (__int64)sub_DBFF60(v127, (unsigned int *)&v164, v126, v128);
            goto LABEL_89;
          }
          v136 = v166;
        }
        memcpy(v136, v86, v124);
        goto LABEL_139;
      }
      if ( v84 )
      {
        v88 = (_QWORD *)v84;
        while ( 1 )
        {
          v88 = (_QWORD *)*v88;
          if ( v85 == v88 )
            break;
          if ( !v88 )
            goto LABEL_117;
        }
        v89 = *(_QWORD *)a1;
        if ( v87 == 2 )
        {
          v90 = (_QWORD *)v86[1];
LABEL_85:
          v91 = sub_DBEDC0(v89, (__int64)v90);
          if ( *(_BYTE *)(a1 + 89) && v91 && v7[5] == 2 )
          {
            result = sub_2815B30(a1, *(_QWORD *)v7[4]);
          }
          else
          {
            *(_BYTE *)(a1 + 88) = 0;
            result = (__int64)v7;
          }
          goto LABEL_89;
        }
        v129 = 8 * v87 - 8;
        v168 = 0x300000000LL;
        v167 = v169;
        v163 = v129 >> 3;
        if ( (unsigned __int64)v129 > 0x18 )
        {
          v146 = v86;
          na = 8 * v87 - 8;
          sub_C8D5F0((__int64)&v167, v169, na >> 3, 8u, a5, a6);
          v129 = na;
          v86 = v146;
          v133 = &v167[8 * (unsigned int)v168];
        }
        else
        {
          if ( 8 * v87 == 8 )
            goto LABEL_142;
          v133 = v169;
        }
        memcpy(v133, v86 + 1, v129);
LABEL_142:
        LODWORD(v168) = v163 + v168;
        v130 = v168;
        v131 = v167;
        v132 = 8LL * (unsigned int)v168;
        v170 = v172;
        v171 = 0x400000000LL;
        if ( (unsigned int)v168 > 4uLL )
        {
          v137 = 8LL * (unsigned int)v168;
          src = v167;
          v147 = v168;
          sub_C8D5F0((__int64)&v170, v172, (unsigned int)v168, 8u, (__int64)v167, v132);
          v130 = v147;
          v131 = src;
          v132 = v137;
          v134 = &v170[8 * (unsigned int)v171];
        }
        else
        {
          if ( !v132 )
          {
LABEL_144:
            LODWORD(v171) = v130 + v171;
            v90 = sub_DBFF60(v89, (unsigned int *)&v170, v84, 0);
            if ( v170 != v172 )
              _libc_free((unsigned __int64)v170);
            if ( v167 != v169 )
              _libc_free((unsigned __int64)v167);
            goto LABEL_85;
          }
          v134 = v172;
        }
        nb = v130;
        memcpy(v134, v131, v132);
        v130 = nb;
        goto LABEL_144;
      }
LABEL_117:
      v106 = &v86[v87];
      for ( i = v86; v106 != i; LODWORD(v165) = v165 + 1 )
      {
        v108 = sub_2815B30(a1, *i);
        v110 = (unsigned int)v165;
        v111 = (unsigned int)v165 + 1LL;
        if ( v111 > HIDWORD(v165) )
        {
          n = v108;
          sub_C8D5F0((__int64)&v164, v166, (unsigned int)v165 + 1LL, 8u, v111, v109);
          v110 = (unsigned int)v165;
          v108 = n;
        }
        ++i;
        *(_QWORD *)&v164[8 * v110] = v108;
      }
      result = (__int64)sub_DBFF60(*(_QWORD *)a1, (unsigned int *)&v164, v84, *((_WORD *)v7 + 14) & 7);
LABEL_89:
      if ( v164 != v166 )
      {
        v160 = result;
        _libc_free((unsigned __int64)v164);
        result = v160;
      }
LABEL_61:
      v8 = *(_BYTE *)(a1 + 16) & 1;
LABEL_12:
      if ( v8 )
      {
        v17 = a1 + 24;
        v18 = 3;
      }
      else
      {
        v23 = *(_DWORD *)(a1 + 32);
        v17 = *(_QWORD *)(a1 + 24);
        if ( !v23 )
        {
          v101 = *(_DWORD *)(a1 + 16);
          ++*(_QWORD *)(a1 + 8);
          v102 = 0;
          v103 = (v101 >> 1) + 1;
LABEL_101:
          v104 = 3 * v23;
          goto LABEL_102;
        }
        v18 = v23 - 1;
      }
      v19 = v18 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v20 = (_QWORD *)(v17 + 16LL * v19);
      v21 = (_QWORD *)*v20;
      if ( (_QWORD *)*v20 == v7 )
        return v20[1];
      v105 = 1;
      v102 = 0;
      while ( v21 != (_QWORD *)-4096LL )
      {
        if ( !v102 && v21 == (_QWORD *)-8192LL )
          v102 = v20;
        v19 = v18 & (v105 + v19);
        v20 = (_QWORD *)(v17 + 16LL * v19);
        v21 = (_QWORD *)*v20;
        if ( (_QWORD *)*v20 == v7 )
          return v20[1];
        ++v105;
      }
      if ( !v102 )
        v102 = v20;
      v101 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)(a1 + 8);
      v103 = (v101 >> 1) + 1;
      if ( !v8 )
      {
        v23 = *(_DWORD *)(a1 + 32);
        goto LABEL_101;
      }
      v104 = 12;
      v23 = 4;
LABEL_102:
      if ( v104 <= 4 * v103 )
      {
        v161 = result;
        sub_DB0DD0(a1 + 8, 2 * v23);
        result = v161;
        if ( (*(_BYTE *)(a1 + 16) & 1) != 0 )
        {
          v112 = a1 + 24;
          v113 = 3;
        }
        else
        {
          v122 = *(_DWORD *)(a1 + 32);
          v112 = *(_QWORD *)(a1 + 24);
          if ( !v122 )
            goto LABEL_177;
          v113 = v122 - 1;
        }
        v114 = v113 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v102 = (_QWORD *)(v112 + 16LL * v114);
        v115 = *v102;
        if ( (_QWORD *)*v102 != v7 )
        {
          v135 = 1;
          v121 = 0;
          while ( v115 != -4096 )
          {
            if ( !v121 && v115 == -8192 )
              v121 = v102;
            v114 = v113 & (v135 + v114);
            v102 = (_QWORD *)(v112 + 16LL * v114);
            v115 = *v102;
            if ( (_QWORD *)*v102 == v7 )
              goto LABEL_125;
            ++v135;
          }
          goto LABEL_131;
        }
LABEL_125:
        v101 = *(_DWORD *)(a1 + 16);
        goto LABEL_104;
      }
      if ( v23 - *(_DWORD *)(a1 + 20) - v103 <= v23 >> 3 )
      {
        v162 = result;
        sub_DB0DD0(a1 + 8, v23);
        result = v162;
        if ( (*(_BYTE *)(a1 + 16) & 1) != 0 )
        {
          v116 = a1 + 24;
          v117 = 3;
          goto LABEL_128;
        }
        v123 = *(_DWORD *)(a1 + 32);
        v116 = *(_QWORD *)(a1 + 24);
        if ( v123 )
        {
          v117 = v123 - 1;
LABEL_128:
          v118 = v117 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
          v102 = (_QWORD *)(v116 + 16LL * v118);
          v119 = *v102;
          if ( (_QWORD *)*v102 != v7 )
          {
            v120 = 1;
            v121 = 0;
            while ( v119 != -4096 )
            {
              if ( v119 == -8192 && !v121 )
                v121 = v102;
              v118 = v117 & (v120 + v118);
              v102 = (_QWORD *)(v116 + 16LL * v118);
              v119 = *v102;
              if ( (_QWORD *)*v102 == v7 )
                goto LABEL_125;
              ++v120;
            }
LABEL_131:
            if ( v121 )
              v102 = v121;
            goto LABEL_125;
          }
          goto LABEL_125;
        }
LABEL_177:
        *(_DWORD *)(a1 + 16) = (2 * (*(_DWORD *)(a1 + 16) >> 1) + 2) | *(_DWORD *)(a1 + 16) & 1;
        BUG();
      }
LABEL_104:
      *(_DWORD *)(a1 + 16) = (2 * (v101 >> 1) + 2) | v101 & 1;
      if ( *v102 != -4096 )
        --*(_DWORD *)(a1 + 20);
      *v102 = v7;
      v102[1] = result;
      return result;
    case 9:
      v75 = (_QWORD *)v7[4];
      v170 = v172;
      v171 = 0x200000000LL;
      v76 = &v75[v7[5]];
      if ( v75 == v76 )
        goto LABEL_11;
      deste = v7;
      v77 = 0;
      do
      {
        v78 = *v75;
        v79 = sub_2815B30(a1, *v75);
        v82 = (unsigned int)v171;
        if ( (unsigned __int64)(unsigned int)v171 + 1 > HIDWORD(v171) )
        {
          v145 = v79;
          sub_C8D5F0((__int64)&v170, v172, (unsigned int)v171 + 1LL, 8u, v80, v81);
          v82 = (unsigned int)v171;
          v79 = v145;
        }
        v83 = (__int64)v170;
        *(_QWORD *)&v170[8 * v82] = v79;
        v34 = v170;
        LODWORD(v171) = v171 + 1;
        ++v75;
        v77 |= *(_QWORD *)&v170[8 * (unsigned int)v171 - 8] != v78;
      }
      while ( v76 != v75 );
      v7 = deste;
      result = (__int64)deste;
      if ( v77 )
      {
        result = sub_DCE040(*(__int64 **)a1, (__int64)&v170, v82, v83, v80);
        v34 = v170;
      }
      goto LABEL_46;
    case 0xA:
      v53 = (_QWORD *)v7[4];
      v170 = v172;
      v171 = 0x200000000LL;
      v54 = &v53[v7[5]];
      if ( v53 == v54 )
        goto LABEL_11;
      destc = v7;
      v55 = 0;
      do
      {
        v56 = *v53;
        v57 = sub_2815B30(a1, *v53);
        v60 = (unsigned int)v171;
        if ( (unsigned __int64)(unsigned int)v171 + 1 > HIDWORD(v171) )
        {
          v140 = v57;
          sub_C8D5F0((__int64)&v170, v172, (unsigned int)v171 + 1LL, 8u, v58, v59);
          v60 = (unsigned int)v171;
          v57 = v140;
        }
        v61 = (__int64)v170;
        *(_QWORD *)&v170[8 * v60] = v57;
        v34 = v170;
        LODWORD(v171) = v171 + 1;
        ++v53;
        v55 |= *(_QWORD *)&v170[8 * (unsigned int)v171 - 8] != v56;
      }
      while ( v54 != v53 );
      v7 = destc;
      result = (__int64)destc;
      if ( v55 )
      {
        result = sub_DCDF90(*(__int64 **)a1, (__int64)&v170, v60, v61, v58);
        v34 = v170;
      }
      goto LABEL_46;
    case 0xB:
      v44 = (_QWORD *)v7[4];
      v170 = v172;
      v171 = 0x200000000LL;
      v45 = &v44[v7[5]];
      if ( v44 == v45 )
        goto LABEL_11;
      destb = v7;
      v46 = 0;
      do
      {
        v47 = *v44;
        v48 = sub_2815B30(a1, *v44);
        v51 = (unsigned int)v171;
        if ( (unsigned __int64)(unsigned int)v171 + 1 > HIDWORD(v171) )
        {
          v142 = v48;
          sub_C8D5F0((__int64)&v170, v172, (unsigned int)v171 + 1LL, 8u, v49, v50);
          v51 = (unsigned int)v171;
          v48 = v142;
        }
        v52 = (__int64)v170;
        *(_QWORD *)&v170[8 * v51] = v48;
        v34 = v170;
        LODWORD(v171) = v171 + 1;
        ++v44;
        v46 |= *(_QWORD *)&v170[8 * (unsigned int)v171 - 8] != v47;
      }
      while ( v45 != v44 );
      v7 = destb;
      result = (__int64)destb;
      if ( v46 )
      {
        result = (__int64)sub_DCEE50(*(__int64 **)a1, (__int64)&v170, 0, v52, v49);
        v34 = v170;
      }
      goto LABEL_46;
    case 0xC:
      v35 = (_QWORD *)v7[4];
      v170 = v172;
      v171 = 0x200000000LL;
      v36 = &v35[v7[5]];
      if ( v35 == v36 )
        goto LABEL_11;
      desta = v7;
      v37 = 0;
      do
      {
        v38 = *v35;
        v39 = sub_2815B30(a1, *v35);
        v42 = (unsigned int)v171;
        if ( (unsigned __int64)(unsigned int)v171 + 1 > HIDWORD(v171) )
        {
          v144 = v39;
          sub_C8D5F0((__int64)&v170, v172, (unsigned int)v171 + 1LL, 8u, v40, v41);
          v42 = (unsigned int)v171;
          v39 = v144;
        }
        v43 = (__int64)v170;
        *(_QWORD *)&v170[8 * v42] = v39;
        v34 = v170;
        LODWORD(v171) = v171 + 1;
        ++v35;
        v37 |= *(_QWORD *)&v170[8 * (unsigned int)v171 - 8] != v38;
      }
      while ( v36 != v35 );
      v7 = desta;
      result = (__int64)desta;
      if ( v37 )
      {
        result = sub_DCE150(*(__int64 **)a1, (__int64)&v170, v42, v43, v40);
        v34 = v170;
      }
      goto LABEL_46;
    case 0xD:
      v25 = (_QWORD *)v7[4];
      v170 = v172;
      v171 = 0x200000000LL;
      v26 = &v25[v7[5]];
      if ( v25 == v26 )
      {
LABEL_11:
        result = (__int64)v7;
        goto LABEL_12;
      }
      dest = v7;
      v27 = 0;
      do
      {
        v28 = *v25;
        v29 = sub_2815B30(a1, *v25);
        v32 = (unsigned int)v171;
        if ( (unsigned __int64)(unsigned int)v171 + 1 > HIDWORD(v171) )
        {
          v143 = v29;
          sub_C8D5F0((__int64)&v170, v172, (unsigned int)v171 + 1LL, 8u, v30, v31);
          v32 = (unsigned int)v171;
          v29 = v143;
        }
        v33 = (__int64)v170;
        *(_QWORD *)&v170[8 * v32] = v29;
        v34 = v170;
        LODWORD(v171) = v171 + 1;
        ++v25;
        v27 |= *(_QWORD *)&v170[8 * (unsigned int)v171 - 8] != v28;
      }
      while ( v26 != v25 );
      v7 = dest;
      result = (__int64)dest;
      if ( v27 )
      {
        result = (__int64)sub_DCEE50(*(__int64 **)a1, (__int64)&v170, 1, v33, v30);
        v34 = v170;
      }
LABEL_46:
      if ( v34 == v172 )
        goto LABEL_61;
      v159 = result;
      _libc_free((unsigned __int64)v34);
      result = v159;
      v8 = *(_BYTE *)(a1 + 16) & 1;
      goto LABEL_12;
    case 0xE:
      v74 = sub_2815B30(a1, v7[4]);
      if ( v74 == v7[4] )
      {
LABEL_108:
        result = (__int64)v7;
        v8 = *(_BYTE *)(a1 + 16) & 1;
      }
      else
      {
        result = (__int64)sub_DD3A70(*(_QWORD *)a1, v74, v7[5]);
        v8 = *(_BYTE *)(a1 + 16) & 1;
      }
      goto LABEL_12;
    default:
      BUG();
  }
}
