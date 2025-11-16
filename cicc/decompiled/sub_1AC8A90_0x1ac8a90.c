// Function: sub_1AC8A90
// Address: 0x1ac8a90
//
__int64 __fastcall sub_1AC8A90(
        __int64 a1,
        __int64 a2,
        _QWORD *a3,
        _QWORD *a4,
        __int64 a5,
        int a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  _DWORD *v14; // r10
  _QWORD *v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // rdi
  _QWORD *v21; // rdx
  unsigned int v22; // r12d
  __int64 v24; // r13
  __int64 v25; // rdx
  _QWORD *v26; // r15
  unsigned int v27; // ebx
  __int64 v28; // rdi
  _QWORD *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r12
  __int64 v32; // r10
  unsigned int v33; // esi
  int v34; // eax
  __int64 v35; // r9
  unsigned int v36; // edx
  __int64 v37; // rsi
  int v38; // r14d
  _QWORD *v39; // rdi
  __int64 v40; // rax
  __int64 v41; // r12
  __int64 v42; // r13
  __int64 v43; // r15
  unsigned int v44; // eax
  unsigned __int64 v45; // r14
  _QWORD *v46; // rbx
  unsigned int v47; // edx
  __int64 v48; // r14
  __int64 v49; // rax
  char v50; // di
  unsigned int v51; // esi
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rcx
  __int64 v55; // rdx
  __int64 v56; // rdx
  __int64 v57; // rbx
  __int64 v58; // rax
  __int64 v59; // rdi
  unsigned int v60; // esi
  __int64 v61; // r9
  __int64 *v62; // rax
  __int64 v63; // rdi
  _QWORD *v64; // rdx
  unsigned int v65; // ecx
  __int64 v66; // r9
  unsigned int v67; // esi
  unsigned int v68; // r8d
  __int64 *v69; // rcx
  __int64 v70; // r11
  __int64 v71; // rbx
  int v72; // edi
  int v73; // edi
  __int64 v74; // r9
  unsigned int v75; // esi
  int v76; // r11d
  __int64 *v77; // r10
  __int64 v78; // r8
  int v79; // edi
  int v80; // edi
  int v81; // edi
  int v82; // edi
  __int64 v83; // r9
  int v84; // r11d
  unsigned int v85; // esi
  int v86; // r9d
  int v87; // r9d
  int v88; // edi
  unsigned int v89; // r14d
  _QWORD *v90; // rsi
  __int64 v91; // rdx
  unsigned __int64 v92; // rdx
  unsigned int v93; // eax
  __int64 v94; // rdx
  int v95; // ecx
  __int64 v96; // rax
  int v97; // esi
  __int64 v98; // rdi
  int v99; // esi
  unsigned int v100; // ecx
  __int64 *v101; // rax
  __int64 v102; // r9
  int v103; // r10d
  int v104; // eax
  int v105; // r8d
  _QWORD *v106; // [rsp+8h] [rbp-198h]
  __int64 v107; // [rsp+8h] [rbp-198h]
  int v108; // [rsp+8h] [rbp-198h]
  _QWORD *v109; // [rsp+8h] [rbp-198h]
  __int64 v110; // [rsp+10h] [rbp-190h]
  int v111; // [rsp+10h] [rbp-190h]
  _QWORD *v112; // [rsp+10h] [rbp-190h]
  unsigned int v113; // [rsp+10h] [rbp-190h]
  __int64 v114; // [rsp+10h] [rbp-190h]
  __int64 v115; // [rsp+18h] [rbp-188h]
  __int64 v116; // [rsp+18h] [rbp-188h]
  __int64 *v117; // [rsp+18h] [rbp-188h]
  __int64 v118; // [rsp+18h] [rbp-188h]
  _QWORD *v119; // [rsp+18h] [rbp-188h]
  _QWORD *v120; // [rsp+18h] [rbp-188h]
  __int64 v122; // [rsp+28h] [rbp-178h]
  _DWORD *v123; // [rsp+28h] [rbp-178h]
  _DWORD *v124; // [rsp+28h] [rbp-178h]
  __int64 v125; // [rsp+38h] [rbp-168h] BYREF
  __int64 v126; // [rsp+40h] [rbp-160h] BYREF
  _BYTE *v127; // [rsp+48h] [rbp-158h]
  _BYTE *v128; // [rsp+50h] [rbp-150h]
  __int64 v129; // [rsp+58h] [rbp-148h]
  int v130; // [rsp+60h] [rbp-140h]
  _BYTE v131[312]; // [rsp+68h] [rbp-138h] BYREF

  v14 = (_DWORD *)a1;
  v17 = *(_QWORD **)(a1 + 80);
  v18 = *(unsigned int *)(a1 + 88);
  v19 = (__int64)&v17[v18];
  v20 = (8 * v18) >> 3;
  if ( (8 * v18) >> 5 )
  {
    v21 = &v17[4 * ((8 * v18) >> 5)];
    while ( *v17 != a2 )
    {
      if ( v17[1] == a2 )
      {
        ++v17;
        goto LABEL_8;
      }
      if ( v17[2] == a2 )
      {
        v17 += 2;
        goto LABEL_8;
      }
      if ( v17[3] == a2 )
      {
        v17 += 3;
        goto LABEL_8;
      }
      v17 += 4;
      if ( v17 == v21 )
      {
        v20 = (v19 - (__int64)v17) >> 3;
        goto LABEL_11;
      }
    }
    goto LABEL_8;
  }
LABEL_11:
  switch ( v20 )
  {
    case 2LL:
LABEL_117:
      if ( *v17 == a2 )
        goto LABEL_8;
      ++v17;
LABEL_119:
      if ( *v17 != a2 )
        break;
LABEL_8:
      v22 = 0;
      if ( (_QWORD *)v19 != v17 )
        return v22;
      break;
    case 3LL:
      if ( *v17 == a2 )
        goto LABEL_8;
      ++v17;
      goto LABEL_117;
    case 1LL:
      goto LABEL_119;
  }
  if ( (unsigned int)v18 >= v14[23] )
  {
    v18 = (__int64)(v14 + 24);
    v120 = a4;
    v124 = v14;
    sub_16CD150((__int64)(v14 + 20), v14 + 24, 0, 8, a5, a6);
    v14 = v124;
    a4 = v120;
    v19 = *((_QWORD *)v124 + 10) + 8LL * (unsigned int)v124[22];
  }
  *(_QWORD *)v19 = a2;
  ++v14[22];
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    v119 = a4;
    v123 = v14;
    sub_15E08E0(a2, v18);
    v24 = *(_QWORD *)(a2 + 88);
    v14 = v123;
    a4 = v119;
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
    {
      sub_15E08E0(a2, v18);
      v25 = *(_QWORD *)(a2 + 88);
      v14 = v123;
      a4 = v119;
    }
    else
    {
      v25 = *(_QWORD *)(a2 + 88);
    }
  }
  else
  {
    v24 = *(_QWORD *)(a2 + 88);
    v25 = v24;
  }
  v122 = v25 + 40LL * *(_QWORD *)(a2 + 96);
  if ( v122 == v24 )
    goto LABEL_34;
  v115 = a2;
  v26 = v14;
  v27 = 0;
  do
  {
    v31 = v26[6];
    v32 = *(_QWORD *)(*a4 + 8LL * v27);
    if ( v31 == v26[7] )
      v31 = *(_QWORD *)(v26[9] - 8LL) + 512LL;
    v33 = *(_DWORD *)(v31 - 8);
    if ( v33 )
    {
      a5 = v33 - 1;
      v28 = *(_QWORD *)(v31 - 24);
      v19 = (unsigned int)a5 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v29 = (_QWORD *)(v28 + 16 * v19);
      v30 = *v29;
      if ( *v29 == v24 )
        goto LABEL_21;
      v108 = 1;
      v112 = 0;
      while ( v30 != -8 )
      {
        if ( !v112 )
        {
          if ( v30 != -16 )
            v29 = 0;
          v112 = v29;
        }
        v19 = (unsigned int)a5 & (v108 + (_DWORD)v19);
        v29 = (_QWORD *)(v28 + 16LL * (unsigned int)v19);
        v30 = *v29;
        if ( *v29 == v24 )
          goto LABEL_21;
        ++v108;
      }
      if ( v112 )
        v29 = v112;
      v80 = *(_DWORD *)(v31 - 16);
      ++*(_QWORD *)(v31 - 32);
      v19 = (unsigned int)(v80 + 1);
      if ( 4 * (int)v19 < 3 * v33 )
      {
        if ( v33 - *(_DWORD *)(v31 - 12) - (unsigned int)v19 <= v33 >> 3 )
        {
          v109 = a4;
          v114 = v32;
          sub_19B8820(v31 - 32, v33);
          v86 = *(_DWORD *)(v31 - 8);
          if ( !v86 )
          {
LABEL_186:
            ++*(_DWORD *)(v31 - 16);
            BUG();
          }
          v87 = v86 - 1;
          a5 = *(_QWORD *)(v31 - 24);
          v88 = 1;
          v89 = v87 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
          v32 = v114;
          a4 = v109;
          v19 = (unsigned int)(*(_DWORD *)(v31 - 16) + 1);
          v90 = 0;
          v29 = (_QWORD *)(a5 + 16LL * v89);
          v91 = *v29;
          if ( *v29 != v24 )
          {
            while ( v91 != -8 )
            {
              if ( !v90 && v91 == -16 )
                v90 = v29;
              v89 = v87 & (v88 + v89);
              v29 = (_QWORD *)(a5 + 16LL * v89);
              v91 = *v29;
              if ( *v29 == v24 )
                goto LABEL_104;
              ++v88;
            }
            if ( v90 )
              v29 = v90;
          }
        }
        goto LABEL_104;
      }
    }
    else
    {
      ++*(_QWORD *)(v31 - 32);
    }
    v106 = a4;
    v110 = v32;
    sub_19B8820(v31 - 32, 2 * v33);
    v34 = *(_DWORD *)(v31 - 8);
    if ( !v34 )
      goto LABEL_186;
    a5 = (unsigned int)(v34 - 1);
    v35 = *(_QWORD *)(v31 - 24);
    v32 = v110;
    a4 = v106;
    v36 = a5 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
    v19 = (unsigned int)(*(_DWORD *)(v31 - 16) + 1);
    v29 = (_QWORD *)(v35 + 16LL * v36);
    v37 = *v29;
    if ( *v29 != v24 )
    {
      v38 = 1;
      v39 = 0;
      while ( v37 != -8 )
      {
        if ( v37 == -16 && !v39 )
          v39 = v29;
        v36 = a5 & (v38 + v36);
        v29 = (_QWORD *)(v35 + 16LL * v36);
        v37 = *v29;
        if ( *v29 == v24 )
          goto LABEL_104;
        ++v38;
      }
      if ( v39 )
        v29 = v39;
    }
LABEL_104:
    *(_DWORD *)(v31 - 16) = v19;
    if ( *v29 != -8 )
      --*(_DWORD *)(v31 - 12);
    *v29 = v24;
    v29[1] = 0;
LABEL_21:
    v29[1] = v32;
    v24 += 40;
    ++v27;
  }
  while ( v24 != v122 );
  a2 = v115;
  v14 = v26;
LABEL_34:
  v126 = 0;
  v127 = v131;
  v128 = v131;
  v40 = *(_QWORD *)(a2 + 80);
  v129 = 32;
  v130 = 0;
  if ( !v40 )
    BUG();
  v41 = *(_QWORD *)(v40 + 24);
  v42 = v40 - 24;
  v43 = (__int64)v14;
  while ( 2 )
  {
    v125 = 0;
    v44 = sub_1AC6BD0(v43, v41, &v125, a7, a8, a9, a10, a11, a12, a13, a14, v19, a5);
    if ( !(_BYTE)v44 )
    {
      v45 = (unsigned __int64)v128;
      v46 = v127;
      v22 = v44;
      goto LABEL_131;
    }
    v45 = (unsigned __int64)v128;
    v46 = v127;
    if ( !v125 )
    {
      v22 = v44;
      v92 = sub_157EBA0(v42);
      v93 = *(_DWORD *)(v92 + 20) & 0xFFFFFFF;
      if ( !v93 )
      {
LABEL_136:
        --*(_DWORD *)(v43 + 88);
        goto LABEL_131;
      }
      v94 = *(_QWORD *)(v92 - 24LL * v93);
      if ( *(_BYTE *)(v94 + 16) <= 0x10u )
      {
LABEL_135:
        *a3 = v94;
        goto LABEL_136;
      }
      v96 = *(_QWORD *)(v43 + 48);
      if ( v96 == *(_QWORD *)(v43 + 56) )
        v96 = *(_QWORD *)(*(_QWORD *)(v43 + 72) - 8LL) + 512LL;
      v97 = *(_DWORD *)(v96 - 8);
      if ( v97 )
      {
        v98 = *(_QWORD *)(v96 - 24);
        v99 = v97 - 1;
        v100 = v99 & (((unsigned int)v94 >> 9) ^ ((unsigned int)v94 >> 4));
        v101 = (__int64 *)(v98 + 16LL * v100);
        v102 = *v101;
        if ( v94 == *v101 )
        {
LABEL_158:
          v94 = v101[1];
          goto LABEL_135;
        }
        v104 = 1;
        while ( v102 != -8 )
        {
          v105 = v104 + 1;
          v100 = v99 & (v104 + v100);
          v101 = (__int64 *)(v98 + 16LL * v100);
          v102 = *v101;
          if ( v94 == *v101 )
            goto LABEL_158;
          v104 = v105;
        }
      }
      v94 = 0;
      goto LABEL_135;
    }
    if ( v128 == v127 )
    {
      v19 = (__int64)&v127[8 * HIDWORD(v129)];
      if ( v127 == (_BYTE *)v19 )
        goto LABEL_96;
      v64 = 0;
      do
      {
        if ( v125 == *v46 )
          return 0;
        if ( *v46 == -2 )
          v64 = v46;
        ++v46;
      }
      while ( (_QWORD *)v19 != v46 );
      if ( v64 )
      {
        *v64 = v125;
        --v130;
        ++v126;
        goto LABEL_40;
      }
LABEL_96:
      if ( HIDWORD(v129) < (unsigned int)v129 )
      {
        ++HIDWORD(v129);
        *(_QWORD *)v19 = v125;
        ++v126;
LABEL_40:
        v41 = *(_QWORD *)(v125 + 48);
LABEL_41:
        if ( !v41 )
          BUG();
        if ( *(_BYTE *)(v41 - 8) != 77 )
        {
          v42 = v125;
          continue;
        }
        v48 = v41 - 24;
        v49 = 0x17FFFFFFE8LL;
        v50 = *(_BYTE *)(v41 - 1) & 0x40;
        v51 = *(_DWORD *)(v41 - 4) & 0xFFFFFFF;
        if ( v51 )
        {
          v52 = 24LL * *(unsigned int *)(v41 + 32) + 8;
          v53 = 0;
          do
          {
            v54 = v48 - 24LL * v51;
            if ( v50 )
              v54 = *(_QWORD *)(v41 - 32);
            if ( v42 == *(_QWORD *)(v54 + v52) )
            {
              v49 = 24 * v53;
              goto LABEL_50;
            }
            ++v53;
            v52 += 8;
          }
          while ( v51 != (_DWORD)v53 );
          v49 = 0x17FFFFFFE8LL;
        }
LABEL_50:
        if ( v50 )
          v55 = *(_QWORD *)(v41 - 32);
        else
          v55 = v48 - 24LL * v51;
        v56 = *(_QWORD *)(v55 + v49);
        v57 = *(_QWORD *)(v43 + 48);
        v58 = *(_QWORD *)(v43 + 56);
        v59 = *(_QWORD *)(v43 + 72);
        if ( *(_BYTE *)(v56 + 16) <= 0x10u )
        {
LABEL_53:
          if ( v57 != v58 )
            goto LABEL_54;
LABEL_73:
          v71 = *(_QWORD *)(v59 - 8);
          v60 = *(_DWORD *)(v71 + 504);
          v19 = *(_QWORD *)(v71 + 488);
          v57 = v71 + 512;
          a5 = v57 - 32;
          if ( !v60 )
          {
LABEL_74:
            ++*(_QWORD *)(v57 - 32);
            v60 = 0;
            goto LABEL_75;
          }
LABEL_56:
          v61 = (v60 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
          v62 = (__int64 *)(v19 + 16 * v61);
          v63 = *v62;
          if ( v48 == *v62 )
          {
LABEL_57:
            v62[1] = v56;
            v41 = *(_QWORD *)(v41 + 8);
            goto LABEL_41;
          }
          v111 = 1;
          v117 = 0;
          v107 = v19;
          while ( v63 != -8 )
          {
            if ( !v117 )
            {
              if ( v63 != -16 )
                v62 = 0;
              v117 = v62;
            }
            LODWORD(v61) = (v60 - 1) & (v111 + v61);
            v19 = (unsigned int)(v111 + 1);
            v62 = (__int64 *)(v107 + 16LL * (unsigned int)v61);
            v63 = *v62;
            if ( v48 == *v62 )
              goto LABEL_57;
            ++v111;
          }
          if ( v117 )
            v62 = v117;
          v79 = *(_DWORD *)(v57 - 16);
          ++*(_QWORD *)(v57 - 32);
          v19 = (unsigned int)(v79 + 1);
          if ( 4 * (int)v19 >= 3 * v60 )
          {
LABEL_75:
            v116 = v56;
            sub_19B8820(a5, 2 * v60);
            v72 = *(_DWORD *)(v57 - 8);
            if ( !v72 )
              goto LABEL_184;
            v73 = v72 - 1;
            v74 = *(_QWORD *)(v57 - 24);
            v56 = v116;
            v75 = v73 & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
            v19 = (unsigned int)(*(_DWORD *)(v57 - 16) + 1);
            v62 = (__int64 *)(v74 + 16LL * v75);
            a5 = *v62;
            if ( v48 != *v62 )
            {
              v76 = 1;
              v77 = 0;
              while ( a5 != -8 )
              {
                if ( a5 == -16 && !v77 )
                  v77 = v62;
                v75 = v73 & (v76 + v75);
                v62 = (__int64 *)(v74 + 16LL * v75);
                a5 = *v62;
                if ( *v62 == v48 )
                  goto LABEL_93;
                ++v76;
              }
LABEL_79:
              if ( v77 )
                v62 = v77;
            }
          }
          else if ( v60 - *(_DWORD *)(v57 - 12) - (unsigned int)v19 <= v60 >> 3 )
          {
            v113 = ((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4);
            v118 = v56;
            sub_19B8820(a5, v60);
            v81 = *(_DWORD *)(v57 - 8);
            if ( !v81 )
            {
LABEL_184:
              ++*(_DWORD *)(v57 - 16);
              BUG();
            }
            v82 = v81 - 1;
            v77 = 0;
            v83 = *(_QWORD *)(v57 - 24);
            v56 = v118;
            v19 = (unsigned int)(*(_DWORD *)(v57 - 16) + 1);
            v84 = 1;
            v85 = v82 & v113;
            v62 = (__int64 *)(v83 + 16LL * (v82 & v113));
            a5 = *v62;
            if ( v48 != *v62 )
            {
              while ( a5 != -8 )
              {
                if ( a5 == -16 && !v77 )
                  v77 = v62;
                v85 = v82 & (v84 + v85);
                v62 = (__int64 *)(v83 + 16LL * v85);
                a5 = *v62;
                if ( v48 == *v62 )
                  goto LABEL_93;
                ++v84;
              }
              goto LABEL_79;
            }
          }
LABEL_93:
          *(_DWORD *)(v57 - 16) = v19;
          if ( *v62 != -8 )
            --*(_DWORD *)(v57 - 12);
          *v62 = v48;
          v62[1] = 0;
          goto LABEL_57;
        }
        if ( v57 == v58 )
        {
          v78 = *(_QWORD *)(v59 - 8);
          v65 = *(_DWORD *)(v78 + 504);
          v66 = *(_QWORD *)(v78 + 488);
          if ( !v65 )
          {
            v57 = v78 + 512;
            v56 = 0;
            a5 = v78 + 480;
            goto LABEL_74;
          }
LABEL_71:
          v67 = v65 - 1;
          v68 = (v65 - 1) & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
          v69 = (__int64 *)(v66 + 16LL * v68);
          v70 = *v69;
          if ( *v69 != v56 )
          {
            v95 = 1;
            while ( v70 != -8 )
            {
              v103 = v95 + 1;
              v68 = v67 & (v95 + v68);
              v69 = (__int64 *)(v66 + 16LL * v68);
              v70 = *v69;
              if ( v56 == *v69 )
                goto LABEL_72;
              v95 = v103;
            }
            v56 = 0;
            goto LABEL_53;
          }
LABEL_72:
          v56 = v69[1];
          if ( v57 == v58 )
            goto LABEL_73;
LABEL_54:
          v60 = *(_DWORD *)(v57 - 8);
        }
        else
        {
          v65 = *(_DWORD *)(v57 - 8);
          v60 = v65;
          if ( v65 )
          {
            v66 = *(_QWORD *)(v57 - 24);
            goto LABEL_71;
          }
          v56 = 0;
        }
        v19 = *(_QWORD *)(v57 - 24);
        a5 = v57 - 32;
        if ( !v60 )
          goto LABEL_74;
        goto LABEL_56;
      }
    }
    break;
  }
  sub_16CCBA0((__int64)&v126, v125);
  v45 = (unsigned __int64)v128;
  v46 = v127;
  if ( (_BYTE)v47 )
    goto LABEL_40;
  v22 = v47;
LABEL_131:
  if ( (_QWORD *)v45 != v46 )
    _libc_free(v45);
  return v22;
}
