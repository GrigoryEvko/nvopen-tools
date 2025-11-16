// Function: sub_19241A0
// Address: 0x19241a0
//
unsigned __int64 __fastcall sub_19241A0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // r12
  __int64 *v12; // r9
  __int64 v13; // r13
  __int64 v14; // rcx
  __int64 *v15; // rax
  __int64 v16; // rdx
  int v17; // esi
  int v18; // esi
  __int64 v19; // r10
  unsigned int v20; // r8d
  __int64 *v21; // rdi
  __int64 v22; // r14
  unsigned int v23; // r11d
  unsigned int v24; // r8d
  __int64 *v25; // rdi
  __int64 v26; // r15
  __int64 v27; // r14
  double v28; // xmm4_8
  double v29; // xmm5_8
  __int64 *v30; // r8
  __int64 *v31; // r13
  unsigned int v32; // esi
  double v33; // xmm4_8
  double v34; // xmm5_8
  double v35; // xmm4_8
  double v36; // xmm5_8
  __int64 v37; // r15
  char v38; // al
  unsigned int v39; // esi
  char v40; // al
  unsigned int v41; // esi
  __int64 v42; // rax
  __int64 v43; // r15
  __int64 v44; // r14
  unsigned __int64 v45; // r14
  unsigned int v46; // esi
  __int64 v47; // r10
  __int64 v48; // rdi
  unsigned int v49; // edx
  unsigned __int64 *v50; // rax
  unsigned __int64 v51; // rcx
  int v52; // r14d
  unsigned int v53; // esi
  __int64 v54; // rdi
  unsigned int v55; // edx
  __int64 *v56; // rax
  __int64 v57; // rcx
  __int64 v58; // rax
  char v59; // al
  __int64 v60; // rax
  int v62; // edi
  int v63; // r14d
  int v64; // edi
  unsigned __int8 v65; // al
  __int64 v66; // rax
  __int64 v67; // rcx
  __int64 v68; // r15
  __int64 v69; // r12
  __int64 *v70; // r14
  __int64 v71; // rax
  __int64 *v72; // rbx
  __int64 *v73; // rdx
  __int64 v74; // r15
  __int64 v75; // r12
  __int64 *v76; // r13
  __int64 v77; // r14
  int v78; // r9d
  unsigned __int64 *v79; // r11
  int v80; // edx
  int v81; // edx
  int v82; // r11d
  int v83; // r11d
  __int64 v84; // r10
  int v85; // edx
  __int64 v86; // rcx
  __int64 v87; // r8
  int v88; // edi
  __int64 *v89; // rsi
  int v90; // r9d
  __int64 *v91; // r11
  int v92; // edx
  int v93; // r11d
  int v94; // r15d
  int v95; // r15d
  __int64 v96; // r11
  unsigned int v97; // ecx
  unsigned __int64 v98; // rdi
  int v99; // r9d
  unsigned __int64 *v100; // rsi
  int v101; // r11d
  int v102; // r11d
  unsigned __int64 *v103; // rcx
  unsigned int v104; // r15d
  __int64 v105; // r9
  int v106; // r8d
  unsigned __int64 v107; // rsi
  int v108; // r10d
  int v109; // r10d
  __int64 v110; // r9
  __int64 *v111; // rcx
  __int64 v112; // r15
  int v113; // esi
  __int64 v114; // rdi
  __int64 v115; // [rsp+8h] [rbp-78h]
  __int64 v116; // [rsp+10h] [rbp-70h]
  __int64 v117; // [rsp+18h] [rbp-68h]
  int v118; // [rsp+20h] [rbp-60h]
  int v119; // [rsp+24h] [rbp-5Ch]
  unsigned int v120; // [rsp+28h] [rbp-58h]
  int v121; // [rsp+2Ch] [rbp-54h]
  __int64 v122; // [rsp+30h] [rbp-50h]
  __int64 v123; // [rsp+38h] [rbp-48h]
  __int64 v124; // [rsp+40h] [rbp-40h]
  __int64 v125; // [rsp+40h] [rbp-40h]
  __int64 v126; // [rsp+48h] [rbp-38h]
  __int64 v127; // [rsp+48h] [rbp-38h]

  v122 = *(_QWORD *)a2 + 56LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 == v122 )
  {
    v120 = 0;
    v60 = 0;
    return v120 | (unsigned __int64)(v60 << 32);
  }
  v118 = 0;
  v123 = *(_QWORD *)a2 + 8LL;
  v119 = 0;
  v121 = 0;
  v120 = 0;
  while ( 2 )
  {
    v11 = 0;
    v12 = *(__int64 **)v123;
    v13 = *(_QWORD *)(v123 - 8);
    v14 = *(_QWORD *)v123 + 8LL * *(unsigned int *)(v123 + 8);
    v15 = *(__int64 **)v123;
    if ( *(_QWORD *)v123 == v14 )
    {
LABEL_44:
      v11 = *v12;
      v42 = 24LL * (*(_DWORD *)(*v12 + 20) & 0xFFFFFFF);
      if ( (*(_BYTE *)(*v12 + 23) & 0x40) != 0 )
      {
        v43 = *(_QWORD *)(v11 - 8);
        v44 = v43 + v42;
      }
      else
      {
        v44 = *v12;
        v43 = v11 - v42;
      }
      if ( v43 == v44 )
        goto LABEL_50;
      while ( *(_BYTE *)(*(_QWORD *)v43 + 16LL) <= 0x17u
           || sub_15CC8F0(*(_QWORD *)(a1 + 216), *(_QWORD *)(*(_QWORD *)v43 + 40LL), v13) )
      {
        v43 += 24;
        if ( v44 == v43 )
          goto LABEL_50;
      }
      if ( *(_BYTE *)(a1 + 636) )
        goto LABEL_38;
      v59 = *(_BYTE *)(v11 + 16);
      if ( v59 == 54 )
      {
        v125 = *(_QWORD *)(v11 - 24);
        if ( !v125 )
          BUG();
        if ( *(_BYTE *)(v125 + 16) != 56 )
          goto LABEL_38;
        goto LABEL_62;
      }
      if ( v59 != 55 )
        goto LABEL_38;
      v117 = *(_QWORD *)(v11 - 48);
      v125 = *(_QWORD *)(v11 - 24);
      v65 = *(_BYTE *)(v117 + 16);
      if ( *(_BYTE *)(v125 + 16) == 56 )
      {
        if ( v65 <= 0x17u )
        {
LABEL_62:
          v117 = 0;
          goto LABEL_82;
        }
      }
      else
      {
        if ( v65 <= 0x17u )
          goto LABEL_38;
        v125 = 0;
      }
      if ( v65 == 56 )
      {
        if ( !sub_1921D00(a1, (__int64 *)v117, v13) )
          goto LABEL_38;
      }
      else if ( !sub_15CC8F0(*(_QWORD *)(a1 + 216), *(_QWORD *)(v117 + 40), v13) )
      {
        goto LABEL_38;
      }
      if ( !v125 )
        goto LABEL_38;
LABEL_82:
      v66 = 24LL * (*(_DWORD *)(v125 + 20) & 0xFFFFFFF);
      if ( (*(_BYTE *)(v125 + 23) & 0x40) != 0 )
      {
        v67 = *(_QWORD *)(v125 - 8);
        v127 = v67 + v66;
      }
      else
      {
        v127 = v125;
        v67 = v125 - v66;
      }
      v68 = v67;
      if ( v127 == v67 )
        goto LABEL_101;
      v116 = v11;
      v69 = a1;
      do
      {
        v70 = *(__int64 **)v68;
        if ( *(_BYTE *)(*(_QWORD *)v68 + 16LL) > 0x17u && !sub_15CC8F0(*(_QWORD *)(v69 + 216), v70[5], v13) )
        {
          if ( *((_BYTE *)v70 + 16) != 56 )
          {
            a1 = v69;
            goto LABEL_38;
          }
          v71 = 24LL * (*((_DWORD *)v70 + 5) & 0xFFFFFFF);
          if ( (*((_BYTE *)v70 + 23) & 0x40) != 0 )
          {
            v72 = (__int64 *)*(v70 - 1);
            v73 = &v72[(unsigned __int64)v71 / 8];
          }
          else
          {
            v73 = v70;
            v72 = &v70[v71 / 0xFFFFFFFFFFFFFFF8LL];
          }
          if ( v72 != v73 )
          {
            v115 = v68;
            v74 = v69;
            v75 = v13;
            v76 = v73;
            while ( 1 )
            {
              v77 = *v72;
              if ( *(_BYTE *)(*v72 + 16) > 0x17u
                && !sub_15CC8F0(*(_QWORD *)(v74 + 216), *(_QWORD *)(v77 + 40), v75)
                && (*(_BYTE *)(v77 + 16) != 56 || !sub_1921D00(v74, (__int64 *)v77, v75)) )
              {
                break;
              }
              v72 += 3;
              if ( v76 == v72 )
              {
                v13 = v75;
                v69 = v74;
                v68 = v115;
                goto LABEL_99;
              }
            }
            a1 = v74;
            goto LABEL_38;
          }
        }
LABEL_99:
        v68 += 24;
      }
      while ( v127 != v68 );
      a1 = v69;
      v11 = v116;
LABEL_101:
      sub_19222E0(a1, v11, v13, v123, v125);
      if ( v117 && *(_BYTE *)(v117 + 16) == 56 )
        sub_19222E0(a1, v11, v13, v123, v117);
LABEL_50:
      v45 = sub_157EBA0(v13);
      sub_14191F0(*(_QWORD *)(a1 + 240), v11);
      sub_15F22F0((_QWORD *)v11, v45);
      v46 = *(_DWORD *)(a1 + 288);
      v47 = a1 + 264;
      if ( v46 )
      {
        v48 = *(_QWORD *)(a1 + 272);
        v49 = (v46 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
        v50 = (unsigned __int64 *)(v48 + 16LL * v49);
        v51 = *v50;
        if ( v45 == *v50 )
        {
LABEL_52:
          v52 = *((_DWORD *)v50 + 2);
          *((_DWORD *)v50 + 2) = v52 + 1;
          v53 = *(_DWORD *)(a1 + 288);
          if ( v53 )
            goto LABEL_53;
LABEL_118:
          ++*(_QWORD *)(a1 + 264);
LABEL_119:
          sub_1542080(v47, 2 * v53);
          v82 = *(_DWORD *)(a1 + 288);
          if ( v82 )
          {
            v83 = v82 - 1;
            v84 = *(_QWORD *)(a1 + 272);
            v85 = *(_DWORD *)(a1 + 280) + 1;
            LODWORD(v86) = v83 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
            v56 = (__int64 *)(v84 + 16LL * (unsigned int)v86);
            v87 = *v56;
            if ( v11 != *v56 )
            {
              v88 = 1;
              v89 = 0;
              while ( v87 != -8 )
              {
                if ( !v89 && v87 == -16 )
                  v89 = v56;
                v86 = v83 & (unsigned int)(v86 + v88);
                v56 = (__int64 *)(v84 + 16 * v86);
                v87 = *v56;
                if ( v11 == *v56 )
                  goto LABEL_132;
                ++v88;
              }
              if ( v89 )
                v56 = v89;
            }
            goto LABEL_132;
          }
LABEL_194:
          ++*(_DWORD *)(a1 + 280);
          BUG();
        }
        v78 = 1;
        v79 = 0;
        while ( v51 != -8 )
        {
          if ( !v79 && v51 == -16 )
            v79 = v50;
          v49 = (v46 - 1) & (v78 + v49);
          v50 = (unsigned __int64 *)(v48 + 16LL * v49);
          v51 = *v50;
          if ( v45 == *v50 )
            goto LABEL_52;
          ++v78;
        }
        v80 = *(_DWORD *)(a1 + 280);
        if ( v79 )
          v50 = v79;
        ++*(_QWORD *)(a1 + 264);
        v81 = v80 + 1;
        if ( 4 * v81 < 3 * v46 )
        {
          if ( v46 - *(_DWORD *)(a1 + 284) - v81 <= v46 >> 3 )
          {
            sub_1542080(a1 + 264, v46);
            v101 = *(_DWORD *)(a1 + 288);
            if ( !v101 )
              goto LABEL_194;
            v102 = v101 - 1;
            v103 = 0;
            v47 = a1 + 264;
            v104 = v102 & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
            v105 = *(_QWORD *)(a1 + 272);
            v106 = 1;
            v81 = *(_DWORD *)(a1 + 280) + 1;
            v50 = (unsigned __int64 *)(v105 + 16LL * v104);
            v107 = *v50;
            if ( v45 != *v50 )
            {
              while ( v107 != -8 )
              {
                if ( v107 == -16 && !v103 )
                  v103 = v50;
                v104 = v102 & (v106 + v104);
                v50 = (unsigned __int64 *)(v105 + 16LL * v104);
                v107 = *v50;
                if ( v45 == *v50 )
                  goto LABEL_115;
                ++v106;
              }
              if ( v103 )
                v50 = v103;
            }
          }
          goto LABEL_115;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 264);
      }
      sub_1542080(a1 + 264, 2 * v46);
      v94 = *(_DWORD *)(a1 + 288);
      if ( !v94 )
        goto LABEL_194;
      v95 = v94 - 1;
      v96 = *(_QWORD *)(a1 + 272);
      v47 = a1 + 264;
      v81 = *(_DWORD *)(a1 + 280) + 1;
      v97 = v95 & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
      v50 = (unsigned __int64 *)(v96 + 16LL * v97);
      v98 = *v50;
      if ( v45 != *v50 )
      {
        v99 = 1;
        v100 = 0;
        while ( v98 != -8 )
        {
          if ( v98 == -16 && !v100 )
            v100 = v50;
          v97 = v95 & (v99 + v97);
          v50 = (unsigned __int64 *)(v96 + 16LL * v97);
          v98 = *v50;
          if ( v45 == *v50 )
            goto LABEL_115;
          ++v99;
        }
        if ( v100 )
          v50 = v100;
      }
LABEL_115:
      *(_DWORD *)(a1 + 280) = v81;
      if ( *v50 != -8 )
        --*(_DWORD *)(a1 + 284);
      *v50 = v45;
      v52 = 0;
      *((_DWORD *)v50 + 2) = 0;
      *((_DWORD *)v50 + 2) = 1;
      v53 = *(_DWORD *)(a1 + 288);
      if ( !v53 )
        goto LABEL_118;
LABEL_53:
      v54 = *(_QWORD *)(a1 + 272);
      v55 = (v53 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v56 = (__int64 *)(v54 + 16LL * v55);
      v57 = *v56;
      if ( v11 != *v56 )
      {
        v90 = 1;
        v91 = 0;
        while ( v57 != -8 )
        {
          if ( v57 == -16 && !v91 )
            v91 = v56;
          v55 = (v53 - 1) & (v90 + v55);
          v56 = (__int64 *)(v54 + 16LL * v55);
          v57 = *v56;
          if ( v11 == *v56 )
            goto LABEL_54;
          ++v90;
        }
        v92 = *(_DWORD *)(a1 + 280);
        if ( v91 )
          v56 = v91;
        ++*(_QWORD *)(a1 + 264);
        v85 = v92 + 1;
        if ( 4 * v85 >= 3 * v53 )
          goto LABEL_119;
        if ( v53 - *(_DWORD *)(a1 + 284) - v85 <= v53 >> 3 )
        {
          sub_1542080(v47, v53);
          v108 = *(_DWORD *)(a1 + 288);
          if ( v108 )
          {
            v109 = v108 - 1;
            v110 = *(_QWORD *)(a1 + 272);
            v111 = 0;
            LODWORD(v112) = v109 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
            v113 = 1;
            v85 = *(_DWORD *)(a1 + 280) + 1;
            v56 = (__int64 *)(v110 + 16LL * (unsigned int)v112);
            v114 = *v56;
            if ( v11 != *v56 )
            {
              while ( v114 != -8 )
              {
                if ( v114 == -16 && !v111 )
                  v111 = v56;
                v112 = v109 & (unsigned int)(v112 + v113);
                v56 = (__int64 *)(v110 + 16 * v112);
                v114 = *v56;
                if ( v11 == *v56 )
                  goto LABEL_132;
                ++v113;
              }
              if ( v111 )
                v56 = v111;
            }
            goto LABEL_132;
          }
          goto LABEL_194;
        }
LABEL_132:
        *(_DWORD *)(a1 + 280) = v85;
        if ( *v56 != -8 )
          --*(_DWORD *)(a1 + 284);
        *v56 = v11;
        *((_DWORD *)v56 + 2) = 0;
      }
LABEL_54:
      *((_DWORD *)v56 + 2) = v52;
      v58 = sub_1422850(*(_QWORD *)(a1 + 248), v11);
      v27 = v58;
      if ( v58 )
      {
        sub_386E700(*(_QWORD *)(a1 + 256), v58, v13, 1);
        v30 = *(__int64 **)v123;
        v126 = *(_QWORD *)v123 + 8LL * *(unsigned int *)(v123 + 8);
        if ( v126 != *(_QWORD *)v123 )
          goto LABEL_17;
        goto LABEL_33;
      }
      v30 = *(__int64 **)v123;
      v126 = *(_QWORD *)v123 + 8LL * *(unsigned int *)(v123 + 8);
      if ( *(_QWORD *)v123 != v126 )
        goto LABEL_17;
      goto LABEL_34;
    }
    do
    {
      while ( 1 )
      {
        v16 = *v15;
        if ( v13 == *(_QWORD *)(*v15 + 40) )
        {
          if ( !v11 )
          {
            v11 = *v15;
            goto LABEL_5;
          }
          v17 = *(_DWORD *)(a1 + 288);
          if ( v17 )
            break;
        }
LABEL_5:
        if ( (__int64 *)v14 == ++v15 )
          goto LABEL_15;
      }
      v18 = v17 - 1;
      v19 = *(_QWORD *)(a1 + 272);
      v20 = v18 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v21 = (__int64 *)(v19 + 16LL * v20);
      v22 = *v21;
      if ( v16 == *v21 )
      {
LABEL_10:
        v23 = *((_DWORD *)v21 + 2);
      }
      else
      {
        v64 = 1;
        while ( v22 != -8 )
        {
          v93 = v64 + 1;
          v20 = v18 & (v64 + v20);
          v21 = (__int64 *)(v19 + 16LL * v20);
          v22 = *v21;
          if ( v16 == *v21 )
            goto LABEL_10;
          v64 = v93;
        }
        v23 = 0;
      }
      v24 = v18 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v25 = (__int64 *)(v19 + 16LL * v24);
      v26 = *v25;
      if ( *v25 != v11 )
      {
        v62 = 1;
        while ( v26 != -8 )
        {
          v63 = v62 + 1;
          v24 = v18 & (v62 + v24);
          v25 = (__int64 *)(v19 + 16LL * v24);
          v26 = *v25;
          if ( *v25 == v11 )
            goto LABEL_12;
          v62 = v63;
        }
        goto LABEL_5;
      }
LABEL_12:
      if ( *((_DWORD *)v25 + 2) > v23 )
        v11 = *v15;
      ++v15;
    }
    while ( (__int64 *)v14 != v15 );
LABEL_15:
    if ( !v11 )
      goto LABEL_44;
    v27 = sub_1422850(*(_QWORD *)(a1 + 248), v11);
    v30 = *(__int64 **)v123;
    v126 = *(_QWORD *)v123 + 8LL * *(unsigned int *)(v123 + 8);
    if ( *(_QWORD *)v123 != v126 )
    {
LABEL_17:
      v31 = v30;
      do
      {
        v37 = *v31;
        if ( *v31 != v11 )
        {
          v38 = *(_BYTE *)(v11 + 16);
          switch ( v38 )
          {
            case '6':
              v32 = 1 << (*(unsigned __int16 *)(v37 + 18) >> 1) >> 1;
              if ( v32 > 1 << (*(unsigned __int16 *)(v11 + 18) >> 1) >> 1 )
                v32 = 1 << (*(unsigned __int16 *)(v11 + 18) >> 1) >> 1;
              sub_15F8F50(v11, v32);
              break;
            case '7':
              v39 = 1 << (*(unsigned __int16 *)(v37 + 18) >> 1) >> 1;
              if ( v39 > 1 << (*(unsigned __int16 *)(v11 + 18) >> 1) >> 1 )
                v39 = 1 << (*(unsigned __int16 *)(v11 + 18) >> 1) >> 1;
              sub_15F9450(v11, v39);
              break;
            case '5':
              v41 = (unsigned int)(1 << *(_WORD *)(v37 + 18)) >> 1;
              if ( v41 < (unsigned int)(1 << *(_WORD *)(v11 + 18)) >> 1 )
                v41 = (unsigned int)(1 << *(_WORD *)(v11 + 18)) >> 1;
              sub_15F8A20(v11, v41);
              break;
          }
          if ( v27 )
          {
            v124 = sub_1422850(*(_QWORD *)(a1 + 248), v37);
            sub_164D160(v124, v27, a3, a4, a5, a6, v33, v34, a9, a10);
            sub_386B550(*(_QWORD *)(a1 + 256), v124);
          }
          sub_15F2780((unsigned __int8 *)v11, v37);
          sub_1AEC0C0(v11, v37, &unk_42BE300, 7);
          sub_164D160(v37, v11, a3, a4, a5, a6, v35, v36, a9, a10);
          sub_14191F0(*(_QWORD *)(a1 + 240), v37);
          sub_15F20C0((_QWORD *)v37);
        }
        ++v31;
      }
      while ( (__int64 *)v126 != v31 );
    }
    if ( v27 )
LABEL_33:
      sub_1922AA0(a1, v27, a3, a4, a5, a6, v28, v29, a9, a10);
LABEL_34:
    v40 = *(_BYTE *)(v11 + 16);
    switch ( v40 )
    {
      case '6':
        ++v121;
        break;
      case '7':
        ++v119;
        break;
      case 'N':
        ++v118;
        break;
      default:
        ++v120;
        break;
    }
LABEL_38:
    if ( v122 != v123 + 48 )
    {
      v123 += 56;
      continue;
    }
    break;
  }
  v60 = (unsigned int)(v121 + v118 + v119);
  return v120 | (unsigned __int64)(v60 << 32);
}
