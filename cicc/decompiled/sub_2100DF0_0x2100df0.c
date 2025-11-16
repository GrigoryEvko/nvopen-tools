// Function: sub_2100DF0
// Address: 0x2100df0
//
void __fastcall sub_2100DF0(_QWORD *a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v8; // rax
  __int16 v9; // di
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rcx
  unsigned int v13; // esi
  __int64 *v14; // rax
  __int64 v15; // r11
  __int64 v16; // r13
  int v17; // eax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  _BYTE *v21; // r9
  __int64 v22; // rax
  __int64 v23; // r13
  unsigned int v24; // esi
  __int64 v25; // rdi
  int v26; // r8d
  __int64 v27; // r15
  unsigned __int64 v28; // rcx
  __int64 v29; // r13
  __int64 v30; // rdi
  void (*v31)(); // rax
  _BYTE *v32; // rdi
  __int64 v33; // r15
  __int64 v34; // r12
  __int64 v35; // rax
  int v36; // r13d
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r15
  unsigned int v42; // r12d
  char v43; // al
  int v44; // r13d
  unsigned int v45; // r13d
  __int64 v46; // r15
  __int64 v47; // rax
  _QWORD *v48; // rax
  __int64 v49; // rax
  __int64 v50; // r10
  unsigned int v51; // edx
  unsigned __int64 v52; // rcx
  __int64 v53; // r11
  __int64 v54; // rax
  __int64 v55; // r13
  __int64 *v56; // rax
  char v57; // al
  __int64 *v58; // rax
  __int64 v59; // rdi
  void (*v60)(); // rax
  __int64 v61; // rax
  unsigned int v62; // r13d
  __int64 v63; // rsi
  __int64 *v64; // rsi
  unsigned int v65; // edi
  __int64 v66; // rax
  __int64 v67; // rdx
  _QWORD *v68; // rdi
  _QWORD *v69; // rdx
  __int64 v70; // rcx
  char v71; // al
  _QWORD *v72; // rcx
  __int64 v73; // rax
  char *v74; // rdi
  int v75; // ecx
  char *v76; // r8
  __int64 v77; // rax
  char *v78; // rax
  char *v79; // rsi
  int v80; // r10d
  char v81; // al
  __int64 v82; // rax
  __int64 v83; // rax
  __int64 *v84; // r13
  __int64 (*v85)(); // rax
  __int64 v86; // r13
  __int64 v87; // rax
  __int64 v88; // r8
  __int64 v89; // r9
  __int64 v90; // rcx
  __int64 v91; // r15
  __int64 v92; // rax
  __int64 v93; // rdx
  unsigned __int64 v94; // rsi
  __int64 v95; // rdi
  __int64 *v96; // rax
  unsigned __int64 v97; // rdx
  unsigned __int64 v98; // rcx
  __int64 v99; // r8
  __int64 v100; // rdi
  __int64 (*v101)(); // rax
  __int64 v102; // rcx
  __int64 v103; // rsi
  unsigned int v104; // eax
  __int64 v105; // rdx
  __int64 v106; // r13
  __int64 v107; // rax
  __int64 *v108; // r8
  char *v109; // rax
  __int64 v110; // rdi
  _QWORD *v111; // rsi
  _QWORD *v112; // rcx
  __int128 v113; // [rsp-20h] [rbp-120h]
  int v114; // [rsp+1Ch] [rbp-E4h]
  unsigned __int64 v115; // [rsp+28h] [rbp-D8h]
  __int64 v116; // [rsp+38h] [rbp-C8h]
  __int64 v117; // [rsp+38h] [rbp-C8h]
  _QWORD *v118; // [rsp+40h] [rbp-C0h]
  bool v119; // [rsp+4Ah] [rbp-B6h]
  char v120; // [rsp+4Bh] [rbp-B5h]
  char v121; // [rsp+4Ch] [rbp-B4h]
  unsigned __int64 v122; // [rsp+50h] [rbp-B0h]
  __int64 v123; // [rsp+58h] [rbp-A8h]
  int v124; // [rsp+58h] [rbp-A8h]
  __int64 v125; // [rsp+60h] [rbp-A0h]
  __int64 v126; // [rsp+60h] [rbp-A0h]
  __int64 v128; // [rsp+68h] [rbp-98h]
  unsigned int v129; // [rsp+68h] [rbp-98h]
  char v130; // [rsp+7Fh] [rbp-81h] BYREF
  unsigned __int64 v131; // [rsp+80h] [rbp-80h]
  __int64 v132; // [rsp+88h] [rbp-78h]
  __int64 v133; // [rsp+90h] [rbp-70h]
  _BYTE *v134; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v135; // [rsp+A8h] [rbp-58h]
  _BYTE v136[80]; // [rsp+B0h] [rbp-50h] BYREF

  v5 = a2;
  v6 = a2;
  v8 = a1[4];
  v9 = *(_WORD *)(a2 + 46);
  v10 = *(_QWORD *)(v8 + 272);
  if ( (v9 & 4) != 0 )
  {
    do
      v5 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v5 + 46) & 4) != 0 );
  }
  v11 = *(_QWORD *)(v10 + 368);
  v12 = *(unsigned int *)(v10 + 384);
  if ( (_DWORD)v12 )
  {
    v13 = (v12 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v14 = (__int64 *)(v11 + 16LL * v13);
    v15 = *v14;
    if ( *v14 == v5 )
    {
LABEL_5:
      v16 = v14[1];
      if ( (v9 & 0xC) != 0 )
        return;
      goto LABEL_10;
    }
    v17 = 1;
    while ( v15 != -8 )
    {
      v80 = v17 + 1;
      v13 = (v12 - 1) & (v17 + v13);
      v14 = (__int64 *)(v11 + 16LL * v13);
      v15 = *v14;
      if ( *v14 == v5 )
        goto LABEL_5;
      v17 = v80;
    }
  }
  v16 = *(_QWORD *)(v11 + 16 * v12 + 8);
  if ( (v9 & 0xC) != 0 )
    return;
LABEL_10:
  if ( **(_WORD **)(v6 + 16) == 1 )
    return;
  v130 = 0;
  v120 = sub_1E17B50(v6, 0, &v130);
  if ( !v120 )
    return;
  v134 = v136;
  v135 = 0x800000000LL;
  v22 = a1[5];
  v115 = v16 & 0xFFFFFFFFFFFFFFF8LL;
  v122 = v16 & 0xFFFFFFFFFFFFFFF8LL | 4;
  v23 = *(_QWORD *)(v6 + 32);
  if ( !v22 )
    goto LABEL_23;
  if ( *(_BYTE *)v23 || (*(_BYTE *)(v23 + 3) & 0x10) == 0 || (v18 = *(_QWORD *)(v6 + 16), *(_BYTE *)(v18 + 4) != 1) )
  {
    v126 = v23 + 40LL * *(unsigned int *)(v6 + 40);
    if ( v23 == v126 )
      goto LABEL_24;
    goto LABEL_39;
  }
  v114 = *(_DWORD *)(v23 + 8);
  v24 = v114 & 0x7FFFFFFF;
  v25 = v114 & 0x7FFFFFFF;
  v26 = *(_DWORD *)(*(_QWORD *)(v22 + 312) + 4 * v25);
  if ( v26 )
  {
    v24 = v26 & 0x7FFFFFFF;
    v25 = v26 & 0x7FFFFFFF;
  }
  else
  {
    v26 = *(_DWORD *)(v23 + 8);
  }
  v27 = a1[4];
  v125 = 8 * v25;
  v28 = *(unsigned int *)(v27 + 408);
  if ( (unsigned int)v28 <= v24 || (v29 = *(_QWORD *)(*(_QWORD *)(v27 + 400) + 8 * v25)) == 0 )
  {
    v104 = v24 + 1;
    if ( (unsigned int)v28 < v24 + 1 )
    {
      v106 = v104;
      if ( v104 < v28 )
      {
        *(_DWORD *)(v27 + 408) = v104;
        v105 = *(_QWORD *)(v27 + 400);
        goto LABEL_167;
      }
      if ( v104 > v28 )
      {
        if ( v104 > (unsigned __int64)*(unsigned int *)(v27 + 412) )
        {
          v124 = v26;
          sub_16CD150(v27 + 400, (const void *)(v27 + 416), v104, 8, v26, (int)v21);
          v104 = v24 + 1;
          v26 = v124;
        }
        v105 = *(_QWORD *)(v27 + 400);
        v110 = *(_QWORD *)(v27 + 416);
        v111 = (_QWORD *)(v105 + 8 * v106);
        v112 = (_QWORD *)(v105 + 8LL * *(unsigned int *)(v27 + 408));
        if ( v111 != v112 )
        {
          do
            *v112++ = v110;
          while ( v111 != v112 );
          v105 = *(_QWORD *)(v27 + 400);
        }
        *(_DWORD *)(v27 + 408) = v104;
        goto LABEL_167;
      }
    }
    v105 = *(_QWORD *)(v27 + 400);
LABEL_167:
    *(_QWORD *)(v105 + v125) = sub_1DBA290(v26);
    v29 = *(_QWORD *)(*(_QWORD *)(v27 + 400) + v125);
    sub_1DBB110((_QWORD *)v27, v29);
  }
  v18 = sub_1DB3C70((__int64 *)v29, v122);
  if ( v18 == *(_QWORD *)v29 + 24LL * *(unsigned int *)(v29 + 8) )
  {
    v19 = *(_QWORD *)(v6 + 32);
    goto LABEL_22;
  }
  v23 = *(_QWORD *)(v6 + 32);
  v19 = v23;
  if ( (*(_DWORD *)((*(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)v18 >> 1) & 3) > (*(_DWORD *)(v115 + 24) | 2u) )
  {
LABEL_22:
    v23 = v19;
LABEL_23:
    v126 = v23 + 40LL * *(unsigned int *)(v6 + 40);
    if ( v126 == v23 )
      goto LABEL_24;
    goto LABEL_39;
  }
  v82 = *(_QWORD *)(v18 + 16);
  if ( v82 )
  {
    v119 = v115 == (*(_QWORD *)(v82 + 8) & 0xFFFFFFFFFFFFFFF8LL);
    v126 = v23 + 40LL * *(unsigned int *)(v6 + 40);
    if ( v126 == v23 )
      goto LABEL_145;
    goto LABEL_40;
  }
  v126 = v23 + 40LL * *(unsigned int *)(v6 + 40);
  if ( v23 == v126 )
    goto LABEL_24;
LABEL_39:
  v119 = 0;
LABEL_40:
  v121 = 0;
  v41 = v23;
  v123 = v6;
  do
  {
    if ( *(_BYTE *)v41 )
      goto LABEL_51;
    v42 = *(_DWORD *)(v41 + 8);
    if ( (v42 & 0x80000000) != 0 )
    {
      v50 = a1[4];
      v51 = v42 & 0x7FFFFFFF;
      v52 = *(unsigned int *)(v50 + 408);
      v53 = v42 & 0x7FFFFFFF;
      v54 = 8 * v53;
      if ( (v42 & 0x7FFFFFFF) < (unsigned int)v52 )
      {
        v55 = *(_QWORD *)(*(_QWORD *)(v50 + 400) + 8LL * v51);
        if ( v55 )
        {
LABEL_69:
          if ( (unsigned __int8)sub_1E166B0(v123, v42, 0) )
          {
            if ( **(_WORD **)(v123 + 16) == 15 || (*(_BYTE *)(v41 + 3) & 0x10) != 0 )
            {
LABEL_72:
              v56 = *(__int64 **)(a3 + 8);
              if ( *(__int64 **)(a3 + 16) == v56 )
              {
                v64 = &v56[*(unsigned int *)(a3 + 28)];
                v65 = *(_DWORD *)(a3 + 28);
                if ( v56 != v64 )
                {
                  v19 = 0;
                  while ( 1 )
                  {
                    v18 = *v56;
                    if ( v55 == *v56 )
                      goto LABEL_74;
                    if ( v18 == -2 )
                      v19 = (__int64)v56;
                    if ( v64 == ++v56 )
                    {
                      if ( !v19 )
                        break;
                      *(_QWORD *)v19 = v55;
                      --*(_DWORD *)(a3 + 32);
                      ++*(_QWORD *)a3;
                      goto LABEL_105;
                    }
                  }
                }
                if ( v65 < *(_DWORD *)(a3 + 24) )
                {
                  *(_DWORD *)(a3 + 28) = v65 + 1;
                  *v64 = v55;
                  ++*(_QWORD *)a3;
                  goto LABEL_105;
                }
              }
              sub_16CCBA0(a3, v55);
              if ( (_BYTE)v18 )
              {
LABEL_105:
                v66 = *(unsigned int *)(a3 + 112);
                if ( (unsigned int)v66 >= *(_DWORD *)(a3 + 116) )
                {
                  sub_16CD150(a3 + 104, (const void *)(a3 + 120), 0, 8, v20, (int)v21);
                  v66 = *(unsigned int *)(a3 + 112);
                }
                v18 = *(_QWORD *)(a3 + 104);
                *(_QWORD *)(v18 + 8 * v66) = v55;
                ++*(_DWORD *)(a3 + 112);
              }
LABEL_74:
              v57 = *(_BYTE *)(v41 + 3) & 0x10;
              goto LABEL_75;
            }
            v71 = *(_BYTE *)(v41 + 4);
            if ( (v71 & 1) != 0 || (v71 & 2) != 0 )
              goto LABEL_51;
          }
          else
          {
            v18 = *(unsigned __int8 *)(v41 + 4);
            v57 = *(_BYTE *)(v41 + 3) & 0x10;
            if ( (v18 & 1) != 0 || (v18 &= 2u, (_DWORD)v18) )
            {
LABEL_75:
              if ( !v57 )
                goto LABEL_51;
              goto LABEL_76;
            }
            if ( v57 && (*(_DWORD *)v41 & 0xFFF00) == 0 )
            {
LABEL_76:
              if ( a1[7] )
              {
                v58 = (__int64 *)sub_1DB3C70((__int64 *)v55, v122);
                if ( v58 != (__int64 *)(*(_QWORD *)v55 + 24LL * *(unsigned int *)(v55 + 8))
                  && (*(_DWORD *)((*v58 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v58 >> 1) & 3) <= (*(_DWORD *)(v115 + 24) | 2u) )
                {
                  if ( v58[2] )
                  {
                    v59 = a1[7];
                    v60 = *(void (**)())(*(_QWORD *)v59 + 40LL);
                    if ( v60 != nullsub_747 )
                      ((void (__fastcall *)(__int64, _QWORD))v60)(v59, *(unsigned int *)(v55 + 112));
                  }
                }
              }
              sub_1DBEA10(a1[4], v55, v122);
              v18 = *(unsigned int *)(v55 + 8);
              if ( !(_DWORD)v18 )
              {
                v61 = (unsigned int)v135;
                if ( (unsigned int)v135 >= HIDWORD(v135) )
                {
                  sub_16CD150((__int64)&v134, v136, 0, 4, v20, (int)v21);
                  v61 = (unsigned int)v135;
                }
                v18 = (__int64)v134;
                *(_DWORD *)&v134[4 * v61] = v42;
                LODWORD(v135) = v135 + 1;
              }
              goto LABEL_51;
            }
          }
          if ( !(unsigned __int8)sub_1E69E00(a1[3], v42) && !(unsigned __int8)sub_2100690((__int64)a1, v55, (int *)v41) )
            goto LABEL_74;
          goto LABEL_72;
        }
      }
      v62 = v51 + 1;
      if ( (unsigned int)v52 < v51 + 1 )
      {
        v67 = v62;
        if ( v62 < v52 )
        {
          *(_DWORD *)(v50 + 408) = v62;
        }
        else if ( v62 > v52 )
        {
          if ( v62 > (unsigned __int64)*(unsigned int *)(v50 + 412) )
          {
            v117 = a1[4];
            sub_16CD150(v50 + 400, (const void *)(v50 + 416), v62, 8, v20, (int)v21);
            v50 = v117;
            v53 = v42 & 0x7FFFFFFF;
            v54 = 8 * v53;
            v67 = v62;
            v52 = *(unsigned int *)(v117 + 408);
          }
          v63 = *(_QWORD *)(v50 + 400);
          v68 = (_QWORD *)(v63 + 8 * v67);
          v69 = (_QWORD *)(v63 + 8 * v52);
          v70 = *(_QWORD *)(v50 + 416);
          if ( v68 != v69 )
          {
            do
              *v69++ = v70;
            while ( v68 != v69 );
            v63 = *(_QWORD *)(v50 + 400);
          }
          *(_DWORD *)(v50 + 408) = v62;
          goto LABEL_88;
        }
      }
      v63 = *(_QWORD *)(v50 + 400);
LABEL_88:
      v116 = v53;
      v118 = (_QWORD *)v50;
      *(_QWORD *)(v63 + v54) = sub_1DBA290(v42);
      v55 = *(_QWORD *)(v118[50] + 8 * v116);
      sub_1DBB110(v118, v55);
      goto LABEL_69;
    }
    v43 = *(_BYTE *)(v41 + 3) & 0x10;
    if ( !v42 || (v18 = *(unsigned __int8 *)(v41 + 4), (v18 & 1) != 0) || (v18 &= 2u, (_DWORD)v18) )
    {
      if ( !v43 )
        goto LABEL_51;
      goto LABEL_96;
    }
    if ( v43 )
    {
      if ( (*(_DWORD *)v41 & 0xFFF00) != 0 )
      {
        v19 = v42;
        v18 = v42 >> 6;
        if ( (*(_QWORD *)(*(_QWORD *)(a1[3] + 304LL) + 8 * v18) & (1LL << v42)) == 0 )
        {
          v121 = v120;
          goto LABEL_51;
        }
      }
LABEL_96:
      sub_1DBE8F0(a1[4], v42, v122);
      goto LABEL_51;
    }
    v19 = v42;
    v18 = v42 >> 6;
    v81 = v121;
    if ( (*(_QWORD *)(*(_QWORD *)(a1[3] + 304LL) + 8 * v18) & (1LL << v42)) == 0 )
      v81 = v120;
    v121 = v81;
LABEL_51:
    v41 += 40;
  }
  while ( v126 != v41 );
  v6 = v123;
  if ( v121 )
  {
    v44 = *(_DWORD *)(v123 + 40);
    *(_QWORD *)(v123 + 16) = *(_QWORD *)(a1[6] + 8LL) + 384LL;
    if ( v44 )
    {
      v45 = v44 - 1;
      v46 = 40LL * v45;
      while ( 1 )
      {
        v47 = v46 + *(_QWORD *)(v123 + 32);
        if ( *(_BYTE *)v47 || *(int *)(v47 + 8) <= 0 )
          sub_1E16C90(v123, v45, v18, v19, v20, v21);
        v46 -= 40;
        if ( !v45 )
          break;
        --v45;
      }
    }
    goto LABEL_28;
  }
LABEL_145:
  if ( !v119
    || !a1[9]
    || (v83 = *(_QWORD *)(v6 + 16), *(_WORD *)v83 != 9)
    && ((*(_BYTE *)(v83 + 11) & 2) == 0
     || ((v84 = (__int64 *)a1[6], v85 = *(__int64 (**)())(*v84 + 16), v85 == sub_1E1C800)
      || !((unsigned __int8 (__fastcall *)(_QWORD, __int64, __int64))v85)(a1[6], v6, a4))
     && !(unsigned __int8)sub_1F3B9C0(v84, v6, a4)) )
  {
LABEL_24:
    v30 = a1[7];
    if ( v30 )
    {
      v31 = *(void (**)())(*(_QWORD *)v30 + 24LL);
      if ( v31 != nullsub_738 )
        ((void (__fastcall *)(__int64, __int64))v31)(v30, v6);
    }
    sub_1F10740(*(_QWORD *)(a1[4] + 272LL), v6);
    sub_1E16240(v6);
    goto LABEL_28;
  }
  v86 = sub_2100AA0(a1, v114, 0, v19, v20, (int)v21);
  v129 = *(_DWORD *)(v86 + 72);
  v87 = sub_145CBF0((__int64 *)(a1[4] + 296LL), 16, 16);
  v90 = v129;
  v91 = v87;
  *(_DWORD *)v87 = v129;
  *(_QWORD *)(v87 + 8) = v122;
  v92 = *(unsigned int *)(v86 + 72);
  if ( (unsigned int)v92 >= *(_DWORD *)(v86 + 76) )
  {
    sub_16CD150(v86 + 64, (const void *)(v86 + 80), 0, 8, v88, v89);
    v92 = *(unsigned int *)(v86 + 72);
  }
  v93 = *(_QWORD *)(v86 + 64);
  v133 = v91;
  v94 = v122;
  *(_QWORD *)(v93 + 8 * v92) = v91;
  ++*(_DWORD *)(v86 + 72);
  v132 = v115 | 6;
  *((_QWORD *)&v113 + 1) = v115 | 6;
  *(_QWORD *)&v113 = v122;
  v131 = v122;
  sub_1DB8610(v86, v122, v93, v90, v88, v89, v113, v133);
  --*(_DWORD *)(a1[2] + 8LL);
  v95 = a1[9];
  v96 = *(__int64 **)(v95 + 8);
  if ( *(__int64 **)(v95 + 16) == v96 )
  {
    v97 = *(unsigned int *)(v95 + 28);
    v108 = &v96[v97];
    v98 = v97;
    if ( v96 == v108 )
    {
LABEL_204:
      if ( (unsigned int)v98 >= *(_DWORD *)(v95 + 24) )
        goto LABEL_154;
      v98 = (unsigned int)(v98 + 1);
      *(_DWORD *)(v95 + 28) = v98;
      *v108 = v6;
      ++*(_QWORD *)v95;
    }
    else
    {
      v97 = 0;
      while ( 1 )
      {
        v94 = *v96;
        if ( v6 == *v96 )
          break;
        if ( v94 == -2 )
          v97 = (unsigned __int64)v96;
        if ( v108 == ++v96 )
        {
          if ( !v97 )
            goto LABEL_204;
          *(_QWORD *)v97 = v6;
          --*(_DWORD *)(v95 + 32);
          ++*(_QWORD *)v95;
          break;
        }
      }
    }
  }
  else
  {
LABEL_154:
    v94 = v6;
    sub_16CCBA0(v95, v6);
  }
  v99 = 0;
  v100 = *(_QWORD *)(*(_QWORD *)a1[3] + 16LL);
  v101 = *(__int64 (**)())(*(_QWORD *)v100 + 112LL);
  if ( v101 != sub_1D00B10 )
    v99 = ((__int64 (__fastcall *)(__int64, unsigned __int64, unsigned __int64, unsigned __int64, _QWORD))v101)(
            v100,
            v94,
            v97,
            v98,
            0);
  sub_1E17170(v6, v114, *(_DWORD *)(v86 + 112), 0, v99);
  *(_BYTE *)(*(_QWORD *)(v6 + 32) + 3LL) |= 0x40u;
LABEL_28:
  v32 = v134;
  if ( (_DWORD)v135 )
  {
    v33 = 4LL * (unsigned int)v135;
    v34 = 0;
    while ( 1 )
    {
      v36 = *(_DWORD *)&v32[v34];
      v37 = a1[4];
      v38 = v36 & 0x7FFFFFFF;
      if ( (unsigned int)v38 >= *(_DWORD *)(v37 + 408) )
        goto LABEL_33;
      v39 = *(_QWORD *)(*(_QWORD *)(v37 + 400) + 8 * v38);
      if ( !v39 )
        goto LABEL_33;
      v40 = a1[3];
      if ( v36 >= 0 )
        v35 = *(_QWORD *)(*(_QWORD *)(v40 + 272) + 8LL * (unsigned int)v36);
      else
        v35 = *(_QWORD *)(*(_QWORD *)(v40 + 24) + 16 * v38 + 8);
      if ( !v35 )
        goto LABEL_63;
      if ( (*(_BYTE *)(v35 + 4) & 8) != 0 )
        break;
LABEL_33:
      v34 += 4;
      if ( v34 == v33 )
        goto LABEL_94;
    }
    while ( 1 )
    {
      v35 = *(_QWORD *)(v35 + 32);
      if ( !v35 )
        break;
      if ( (*(_BYTE *)(v35 + 4) & 8) == 0 )
        goto LABEL_33;
    }
LABEL_63:
    v48 = *(_QWORD **)(a3 + 8);
    if ( *(_QWORD **)(a3 + 16) == v48 )
    {
      v72 = &v48[*(unsigned int *)(a3 + 28)];
      if ( v48 == v72 )
      {
LABEL_161:
        v48 = v72;
      }
      else
      {
        while ( v39 != *v48 )
        {
          if ( v72 == ++v48 )
            goto LABEL_161;
        }
      }
    }
    else
    {
      v128 = v39;
      v48 = sub_16CC9F0(a3, v39);
      v39 = v128;
      if ( v128 == *v48 )
      {
        v102 = *(_QWORD *)(a3 + 16);
        if ( v102 == *(_QWORD *)(a3 + 8) )
          v103 = *(unsigned int *)(a3 + 28);
        else
          v103 = *(unsigned int *)(a3 + 24);
        v72 = (_QWORD *)(v102 + 8 * v103);
      }
      else
      {
        v49 = *(_QWORD *)(a3 + 16);
        if ( v49 != *(_QWORD *)(a3 + 8) )
        {
LABEL_66:
          sub_2100590((__int64)a1, v36);
          v32 = v134;
          goto LABEL_33;
        }
        v48 = (_QWORD *)(v49 + 8LL * *(unsigned int *)(a3 + 28));
        v72 = v48;
      }
    }
    if ( v72 == v48 )
      goto LABEL_66;
    *v48 = -2;
    v73 = *(unsigned int *)(a3 + 112);
    v74 = *(char **)(a3 + 104);
    ++*(_DWORD *)(a3 + 32);
    v75 = v73;
    v73 *= 8;
    v76 = &v74[v73];
    v77 = v73 >> 5;
    if ( v77 )
    {
      v78 = &v74[32 * v77];
      while ( v39 != *(_QWORD *)v74 )
      {
        if ( v39 == *((_QWORD *)v74 + 1) )
        {
          v74 += 8;
          v79 = v74 + 8;
          goto LABEL_135;
        }
        if ( v39 == *((_QWORD *)v74 + 2) )
        {
          v74 += 16;
          v79 = v74 + 8;
          goto LABEL_135;
        }
        if ( v39 == *((_QWORD *)v74 + 3) )
        {
          v74 += 24;
          break;
        }
        v74 += 32;
        if ( v78 == v74 )
          goto LABEL_175;
      }
LABEL_134:
      v79 = v74 + 8;
LABEL_135:
      if ( v79 != v76 )
      {
        memmove(v74, v79, v76 - v79);
        v75 = *(_DWORD *)(a3 + 112);
      }
      *(_DWORD *)(a3 + 112) = v75 - 1;
      goto LABEL_66;
    }
LABEL_175:
    v107 = v76 - v74;
    if ( v76 - v74 == 16 )
    {
      v109 = v74;
    }
    else
    {
      if ( v107 != 24 )
      {
        if ( v107 != 8 )
        {
LABEL_178:
          v74 = v76;
          v79 = v76 + 8;
          goto LABEL_135;
        }
LABEL_187:
        if ( v39 == *(_QWORD *)v74 )
          goto LABEL_134;
        goto LABEL_178;
      }
      v79 = v74 + 8;
      v109 = v74 + 8;
      if ( v39 == *(_QWORD *)v74 )
        goto LABEL_135;
    }
    v74 = v109 + 8;
    if ( v39 == *(_QWORD *)v109 )
    {
      v74 = v109;
      v79 = v109 + 8;
      goto LABEL_135;
    }
    goto LABEL_187;
  }
LABEL_94:
  if ( v32 != v136 )
    _libc_free((unsigned __int64)v32);
}
