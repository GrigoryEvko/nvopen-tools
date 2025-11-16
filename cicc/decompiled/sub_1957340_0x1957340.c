// Function: sub_1957340
// Address: 0x1957340
//
__int64 __fastcall sub_1957340(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v6; // rax
  __int64 v7; // rcx
  __int64 *v8; // rdx
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 i; // r13
  int v13; // esi
  const char *v14; // r14
  const char *v15; // rcx
  unsigned int v16; // edx
  const char **v17; // rax
  const char *v18; // rdi
  __int64 v19; // rdx
  char v20; // r8
  unsigned int v21; // edi
  __int64 v22; // rcx
  __int64 v23; // rdx
  const char *v24; // rsi
  const char *v25; // r14
  __int64 v26; // r14
  __int64 v27; // rbx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rax
  unsigned int v31; // esi
  unsigned int v32; // ecx
  _QWORD *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // r10
  __int64 v37; // rax
  _QWORD *v38; // rax
  __int64 v39; // rdi
  unsigned int v40; // r8d
  __int64 *v41; // rsi
  __int64 v42; // r14
  __int64 v43; // rcx
  __int64 v44; // rdi
  unsigned __int64 v45; // rsi
  __int64 v46; // rsi
  int v47; // esi
  int v48; // r15d
  _QWORD *v49; // rax
  _QWORD *v50; // rbx
  __int64 *v51; // r13
  __int64 v52; // rsi
  unsigned int v53; // r14d
  unsigned __int64 v54; // rbx
  int v55; // r13d
  unsigned int v56; // esi
  __int64 v57; // rdi
  __int64 v58; // rbx
  __int64 v59; // r14
  __int64 v60; // r15
  _QWORD *v61; // rcx
  _QWORD *v62; // rax
  int v63; // r8d
  int v64; // r9d
  __int64 v66; // r15
  const char *v67; // rax
  __int64 v68; // rdx
  unsigned int v69; // r9d
  _QWORD *v70; // rax
  __int64 v71; // rdi
  __int64 v72; // rdx
  unsigned int j; // eax
  const char *v74; // rsi
  int v75; // r11d
  _QWORD *v76; // r10
  int v77; // edx
  __int64 v78; // rsi
  unsigned __int8 *v79; // rsi
  __int64 v80; // rcx
  __int64 v81; // rsi
  int v82; // r11d
  _QWORD *v83; // r10
  _QWORD *v84; // r9
  __int64 v85; // r15
  int v86; // r10d
  __int64 v87; // rcx
  int v88; // r13d
  __int64 v89; // rax
  int v90; // r8d
  _QWORD *v91; // rcx
  int v92; // eax
  __int64 v93; // rdx
  __int64 v94; // r9
  int v95; // edi
  _QWORD *v96; // rsi
  int v97; // edi
  __int64 v98; // rdx
  __int64 v99; // r9
  int v100; // r11d
  const char **v101; // r10
  int v102; // edi
  __int64 *v103; // r14
  __int64 v106; // [rsp+28h] [rbp-148h]
  __int64 v107; // [rsp+30h] [rbp-140h]
  __int64 v108; // [rsp+38h] [rbp-138h]
  __int64 v109; // [rsp+50h] [rbp-120h] BYREF
  __int64 v110; // [rsp+58h] [rbp-118h]
  __int64 v111; // [rsp+60h] [rbp-110h]
  unsigned int v112; // [rsp+68h] [rbp-108h]
  const char *v113; // [rsp+70h] [rbp-100h] BYREF
  __int64 v114; // [rsp+78h] [rbp-F8h]
  const char **v115; // [rsp+B0h] [rbp-C0h] BYREF
  unsigned __int64 v116; // [rsp+B8h] [rbp-B8h]
  _QWORD v117[22]; // [rsp+C0h] [rbp-B0h] BYREF

  v6 = sub_157EBA0(a2);
  if ( (unsigned int)sub_1951E40(a2, v6, *(_DWORD *)(a1 + 256)) > *(_DWORD *)(a1 + 256) )
    return 0;
  v7 = *(unsigned int *)(a3 + 8);
  v8 = *(__int64 **)a3;
  if ( v7 == 1 )
    v106 = *v8;
  else
    v106 = sub_1956A20(a1, a2, v8, v7, ".thr_comm");
  if ( sub_15CD740(*(_QWORD *)(a1 + 24)) )
    sub_13EBC00(*(__int64 **)(a1 + 8));
  else
    sub_13EBC50(*(__int64 **)(a1 + 8));
  sub_13EB620(*(__int64 **)(a1 + 8), v106, a2, a4);
  v9 = *(_QWORD *)(a2 + 56);
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v112 = 0;
  v113 = sub_1649960(a2);
  v114 = v10;
  v115 = &v113;
  LOWORD(v117[0]) = 773;
  v116 = (unsigned __int64)".thread";
  v11 = sub_157E9C0(a2);
  v108 = sub_22077B0(64);
  if ( v108 )
    sub_157FB60((_QWORD *)v108, v11, (__int64)&v115, v9, a2);
  sub_1580AC0((_QWORD *)v108, v106);
  if ( *(_BYTE *)(a1 + 48) )
  {
    v88 = sub_13774B0(*(_QWORD *)(a1 + 40), v106, a2);
    v115 = (const char **)sub_1368AA0(*(__int64 **)(a1 + 32), v106);
    v89 = sub_16AF500((__int64 *)&v115, v88);
    sub_136C010(*(__int64 **)(a1 + 32), v108, v89);
  }
  for ( i = *(_QWORD *)(a2 + 48); ; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 8) != 77 )
      break;
    v13 = v112;
    v14 = (const char *)(i - 24);
    v113 = (const char *)(i - 24);
    v15 = (const char *)(i - 24);
    if ( !v112 )
    {
      ++v109;
LABEL_163:
      v13 = 2 * v112;
LABEL_164:
      sub_19566A0((__int64)&v109, v13);
      sub_1954890((__int64)&v109, (__int64 *)&v113, &v115);
      v17 = v115;
      v15 = v113;
      v102 = v111 + 1;
      goto LABEL_154;
    }
    v16 = (v112 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v17 = (const char **)(v110 + 16LL * v16);
    v18 = *v17;
    if ( v14 == *v17 )
      goto LABEL_15;
    v100 = 1;
    v101 = 0;
    while ( v18 != (const char *)-8LL )
    {
      if ( !v101 && v18 == (const char *)-16LL )
        v101 = v17;
      v16 = (v112 - 1) & (v100 + v16);
      v17 = (const char **)(v110 + 16LL * v16);
      v18 = *v17;
      if ( v14 == *v17 )
        goto LABEL_15;
      ++v100;
    }
    if ( v101 )
      v17 = v101;
    ++v109;
    v102 = v111 + 1;
    if ( 4 * ((int)v111 + 1) >= 3 * v112 )
      goto LABEL_163;
    if ( v112 - HIDWORD(v111) - v102 <= v112 >> 3 )
      goto LABEL_164;
LABEL_154:
    LODWORD(v111) = v102;
    if ( *v17 != (const char *)-8LL )
      --HIDWORD(v111);
    *v17 = v15;
    v17[1] = 0;
LABEL_15:
    v19 = 0x17FFFFFFE8LL;
    v20 = *(_BYTE *)(i - 1) & 0x40;
    v21 = *(_DWORD *)(i - 4) & 0xFFFFFFF;
    if ( v21 )
    {
      v22 = 24LL * *(unsigned int *)(i + 32) + 8;
      v23 = 0;
      do
      {
        v24 = &v14[-24 * v21];
        if ( v20 )
          v24 = *(const char **)(i - 32);
        if ( v106 == *(_QWORD *)&v24[v22] )
        {
          v19 = 24 * v23;
          goto LABEL_22;
        }
        ++v23;
        v22 += 8;
      }
      while ( v21 != (_DWORD)v23 );
      v19 = 0x17FFFFFFE8LL;
    }
LABEL_22:
    if ( v20 )
      v25 = *(const char **)(i - 32);
    else
      v25 = &v14[-24 * v21];
    v17[1] = *(const char **)&v25[v19];
  }
  v107 = v108 + 40;
  while ( 1 )
  {
    v26 = i - 24;
    if ( (unsigned int)*(unsigned __int8 *)(i - 8) - 25 <= 9 )
      break;
    v27 = sub_15F4880(i - 24);
    v113 = sub_1649960(i - 24);
    LOWORD(v117[0]) = 261;
    v114 = v28;
    v115 = &v113;
    sub_164B780(v27, (__int64 *)&v115);
    sub_157E9D0(v107, v27);
    v29 = *(_QWORD *)(v108 + 40);
    v30 = *(_QWORD *)(v27 + 24);
    *(_QWORD *)(v27 + 32) = v107;
    v29 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v27 + 24) = v29 | v30 & 7;
    *(_QWORD *)(v29 + 8) = v27 + 24;
    v31 = v112;
    *(_QWORD *)(v108 + 40) = *(_QWORD *)(v108 + 40) & 7LL | (v27 + 24);
    if ( v31 )
    {
      v32 = (v31 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v33 = (_QWORD *)(v110 + 16LL * v32);
      v34 = *v33;
      if ( v26 == *v33 )
        goto LABEL_30;
      v75 = 1;
      v76 = 0;
      while ( v34 != -8 )
      {
        if ( v34 == -16 && !v76 )
          v76 = v33;
        v32 = (v31 - 1) & (v75 + v32);
        v33 = (_QWORD *)(v110 + 16LL * v32);
        v34 = *v33;
        if ( v26 == *v33 )
          goto LABEL_30;
        ++v75;
      }
      if ( v76 )
        v33 = v76;
      ++v109;
      v77 = v111 + 1;
      if ( 4 * ((int)v111 + 1) < 3 * v31 )
      {
        if ( v31 - HIDWORD(v111) - v77 <= v31 >> 3 )
        {
          sub_19566A0((__int64)&v109, v31);
          if ( !v112 )
          {
LABEL_194:
            LODWORD(v111) = v111 + 1;
            BUG();
          }
          v84 = 0;
          LODWORD(v85) = (v112 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
          v86 = 1;
          v77 = v111 + 1;
          v33 = (_QWORD *)(v110 + 16LL * (unsigned int)v85);
          v87 = *v33;
          if ( v26 != *v33 )
          {
            while ( v87 != -8 )
            {
              if ( v87 == -16 && !v84 )
                v84 = v33;
              v85 = (v112 - 1) & ((_DWORD)v85 + v86);
              v33 = (_QWORD *)(v110 + 16 * v85);
              v87 = *v33;
              if ( v26 == *v33 )
                goto LABEL_97;
              ++v86;
            }
            if ( v84 )
              v33 = v84;
          }
        }
        goto LABEL_97;
      }
    }
    else
    {
      ++v109;
    }
    sub_19566A0((__int64)&v109, 2 * v31);
    if ( !v112 )
      goto LABEL_194;
    v77 = v111 + 1;
    LODWORD(v80) = (v112 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
    v33 = (_QWORD *)(v110 + 16LL * (unsigned int)v80);
    v81 = *v33;
    if ( v26 != *v33 )
    {
      v82 = 1;
      v83 = 0;
      while ( v81 != -8 )
      {
        if ( v81 == -16 && !v83 )
          v83 = v33;
        v80 = (v112 - 1) & ((_DWORD)v80 + v82);
        v33 = (_QWORD *)(v110 + 16 * v80);
        v81 = *v33;
        if ( v26 == *v33 )
          goto LABEL_97;
        ++v82;
      }
      if ( v83 )
        v33 = v83;
    }
LABEL_97:
    LODWORD(v111) = v77;
    if ( *v33 != -8 )
      --HIDWORD(v111);
    *v33 = v26;
    v33[1] = 0;
LABEL_30:
    v33[1] = v27;
    if ( (*(_DWORD *)(v27 + 20) & 0xFFFFFFF) != 0 )
    {
      v35 = 0;
      v36 = 24LL * (*(_DWORD *)(v27 + 20) & 0xFFFFFFF);
      do
      {
        if ( (*(_BYTE *)(v27 + 23) & 0x40) != 0 )
          v37 = *(_QWORD *)(v27 - 8);
        else
          v37 = v27 - 24LL * (*(_DWORD *)(v27 + 20) & 0xFFFFFFF);
        v38 = (_QWORD *)(v35 + v37);
        v39 = *v38;
        if ( *(_BYTE *)(*v38 + 16LL) > 0x17u && v112 )
        {
          v40 = (v112 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
          v41 = (__int64 *)(v110 + 16LL * v40);
          v42 = *v41;
          if ( v39 == *v41 )
          {
LABEL_36:
            if ( v41 != (__int64 *)(v110 + 16LL * v112) )
            {
              v43 = v41[1];
              v44 = v38[1];
              v45 = v38[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v45 = v44;
              if ( v44 )
                *(_QWORD *)(v44 + 16) = *(_QWORD *)(v44 + 16) & 3LL | v45;
              *v38 = v43;
              if ( v43 )
              {
                v46 = *(_QWORD *)(v43 + 8);
                v38[1] = v46;
                if ( v46 )
                  *(_QWORD *)(v46 + 16) = (unsigned __int64)(v38 + 1) | *(_QWORD *)(v46 + 16) & 3LL;
                v38[2] = (v43 + 8) | v38[2] & 3LL;
                *(_QWORD *)(v43 + 8) = v38;
              }
            }
          }
          else
          {
            v47 = 1;
            while ( v42 != -8 )
            {
              v48 = v47 + 1;
              v40 = (v112 - 1) & (v47 + v40);
              v41 = (__int64 *)(v110 + 16LL * v40);
              v42 = *v41;
              if ( v39 == *v41 )
                goto LABEL_36;
              v47 = v48;
            }
          }
        }
        v35 += 24;
      }
      while ( v35 != v36 );
    }
    i = *(_QWORD *)(i + 8);
    if ( !i )
      BUG();
  }
  v49 = sub_1648A60(56, 1u);
  v50 = v49;
  if ( v49 )
    sub_15F8590((__int64)v49, a4, v108);
  v51 = v50 + 6;
  v52 = *(_QWORD *)(sub_157EBA0(a2) + 48);
  v115 = (const char **)v52;
  if ( !v52 )
  {
    if ( v51 == (__int64 *)&v115 )
      goto LABEL_58;
    v78 = v50[6];
    if ( !v78 )
      goto LABEL_58;
LABEL_104:
    sub_161E7C0((__int64)(v50 + 6), v78);
    goto LABEL_105;
  }
  sub_1623A60((__int64)&v115, v52, 2);
  if ( v51 == (__int64 *)&v115 )
  {
    if ( v115 )
      sub_161E7C0((__int64)&v115, (__int64)v115);
    goto LABEL_58;
  }
  v78 = v50[6];
  if ( v78 )
    goto LABEL_104;
LABEL_105:
  v79 = (unsigned __int8 *)v115;
  v50[6] = v115;
  if ( v79 )
    sub_1623210((__int64)&v115, v79, (__int64)(v50 + 6));
LABEL_58:
  v53 = 0;
  sub_19523F0(a4, a2, v108, (__int64)&v109);
  v54 = sub_157EBA0(v106);
  v55 = sub_15F4D60(v54);
  if ( v55 )
  {
    do
    {
      while ( a2 != sub_15F4DF0(v54, v53) )
      {
        if ( v55 == ++v53 )
          goto LABEL_63;
      }
      sub_157F2D0(a2, v106, 1);
      v56 = v53++;
      sub_15F4ED0(v54, v56, v108);
    }
    while ( v55 != v53 );
  }
LABEL_63:
  v57 = *(_QWORD *)(a1 + 24);
  v115 = (const char **)v108;
  v117[0] = v106;
  v117[2] = v106;
  v116 = a4 & 0xFFFFFFFFFFFFFFFBLL;
  v117[1] = v108 & 0xFFFFFFFFFFFFFFFBLL;
  v117[3] = a2 | 4;
  sub_15CD9D0(v57, (__int64 *)&v115, 3);
  sub_1B3B830(&v113, 0);
  v58 = *(_QWORD *)(a2 + 48);
  v115 = (const char **)v117;
  v116 = 0x1000000000LL;
  if ( v58 != a2 + 40 )
  {
    while ( 2 )
    {
      if ( !v58 )
        BUG();
      v59 = *(_QWORD *)(v58 - 16);
      v60 = (unsigned int)v116;
      if ( !v59 )
      {
LABEL_78:
        if ( !(_DWORD)v60 )
          goto LABEL_79;
        v66 = v58 - 24;
        v67 = sub_1649960(v58 - 24);
        sub_1B3B8C0(&v113, *(_QWORD *)(v58 - 24), v67, v68);
        sub_1B3BE00(&v113, a2, v58 - 24);
        if ( v112 )
        {
          v69 = (v112 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
          v70 = (_QWORD *)(v110 + 16LL * v69);
          v71 = *v70;
          if ( v66 == *v70 )
          {
            v72 = v70[1];
LABEL_86:
            sub_1B3BE00(&v113, v108, v72);
            for ( j = v116; (_DWORD)v116; j = v116 )
            {
              v74 = v115[j - 1];
              LODWORD(v116) = j - 1;
              sub_1B420C0(&v113, v74);
            }
LABEL_79:
            v58 = *(_QWORD *)(v58 + 8);
            if ( a2 + 40 == v58 )
              goto LABEL_80;
            continue;
          }
          v90 = 1;
          v91 = 0;
          while ( v71 != -8 )
          {
            if ( v71 != -16 || v91 )
              v70 = v91;
            v69 = (v112 - 1) & (v90 + v69);
            v103 = (__int64 *)(v110 + 16LL * v69);
            v71 = *v103;
            if ( v66 == *v103 )
            {
              v72 = v103[1];
              goto LABEL_86;
            }
            ++v90;
            v91 = v70;
            v70 = (_QWORD *)(v110 + 16LL * v69);
          }
          if ( !v91 )
            v91 = v70;
          ++v109;
          v92 = v111 + 1;
          if ( 4 * ((int)v111 + 1) < 3 * v112 )
          {
            if ( v112 - HIDWORD(v111) - v92 <= v112 >> 3 )
            {
              sub_19566A0((__int64)&v109, v112);
              if ( !v112 )
              {
LABEL_193:
                LODWORD(v111) = v111 + 1;
                BUG();
              }
              v97 = 1;
              v96 = 0;
              LODWORD(v98) = (v112 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
              v92 = v111 + 1;
              v91 = (_QWORD *)(v110 + 16LL * (unsigned int)v98);
              v99 = *v91;
              if ( v66 != *v91 )
              {
                while ( v99 != -8 )
                {
                  if ( !v96 && v99 == -16 )
                    v96 = v91;
                  v98 = (v112 - 1) & ((_DWORD)v98 + v97);
                  v91 = (_QWORD *)(v110 + 16 * v98);
                  v99 = *v91;
                  if ( v66 == *v91 )
                    goto LABEL_129;
                  ++v97;
                }
                goto LABEL_145;
              }
            }
            goto LABEL_129;
          }
        }
        else
        {
          ++v109;
        }
        sub_19566A0((__int64)&v109, 2 * v112);
        if ( !v112 )
          goto LABEL_193;
        LODWORD(v93) = (v112 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
        v92 = v111 + 1;
        v91 = (_QWORD *)(v110 + 16LL * (unsigned int)v93);
        v94 = *v91;
        if ( v66 != *v91 )
        {
          v95 = 1;
          v96 = 0;
          while ( v94 != -8 )
          {
            if ( v94 == -16 && !v96 )
              v96 = v91;
            v93 = (v112 - 1) & ((_DWORD)v93 + v95);
            v91 = (_QWORD *)(v110 + 16 * v93);
            v94 = *v91;
            if ( v66 == *v91 )
              goto LABEL_129;
            ++v95;
          }
LABEL_145:
          if ( v96 )
            v91 = v96;
        }
LABEL_129:
        LODWORD(v111) = v92;
        if ( *v91 != -8 )
          --HIDWORD(v111);
        *v91 = v66;
        v72 = 0;
        v91[1] = 0;
        goto LABEL_86;
      }
      break;
    }
    while ( 2 )
    {
      v62 = sub_1648700(v59);
      if ( *((_BYTE *)v62 + 16) == 77 )
      {
        if ( (*((_BYTE *)v62 + 23) & 0x40) != 0 )
          v61 = (_QWORD *)*(v62 - 1);
        else
          v61 = &v62[-3 * (*((_DWORD *)v62 + 5) & 0xFFFFFFF)];
        if ( a2 != v61[3 * *((unsigned int *)v62 + 14) + 1 + -1431655765 * (unsigned int)((v59 - (__int64)v61) >> 3)] )
        {
          if ( (unsigned int)v60 >= HIDWORD(v116) )
            goto LABEL_76;
          goto LABEL_71;
        }
      }
      else if ( a2 != v62[5] )
      {
        if ( (unsigned int)v60 >= HIDWORD(v116) )
        {
LABEL_76:
          sub_16CD150((__int64)&v115, v117, 0, 8, v63, v64);
          v60 = (unsigned int)v116;
        }
LABEL_71:
        v115[v60] = (const char *)v59;
        v60 = (unsigned int)(v116 + 1);
        LODWORD(v116) = v116 + 1;
      }
      v59 = *(_QWORD *)(v59 + 8);
      if ( !v59 )
        goto LABEL_78;
      continue;
    }
  }
LABEL_80:
  sub_1AF47C0(v108, *(_QWORD *)a1);
  sub_1953CB0(a1, v106, a2, v108, a4);
  if ( v115 != v117 )
    _libc_free((unsigned __int64)v115);
  sub_1B3B860(&v113);
  j___libc_free_0(v110);
  return 1;
}
