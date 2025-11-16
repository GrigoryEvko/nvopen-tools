// Function: sub_3105700
// Address: 0x3105700
//
__int64 __fastcall sub_3105700(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // rbx
  int v8; // ecx
  __int64 v9; // rsi
  int v10; // ecx
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // rdi
  char v14; // bl
  unsigned __int64 v15; // rax
  __int64 v16; // r14
  char v17; // al
  __int64 v18; // rbx
  __int64 v19; // r15
  unsigned int v20; // r13d
  char v21; // r12
  __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 v28; // r12
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 *v31; // rdi
  __int64 v32; // rcx
  unsigned int v33; // eax
  __int64 v34; // r14
  int v35; // eax
  __int64 *v36; // rdx
  __int64 v37; // rbx
  __int64 *v38; // rax
  unsigned int v39; // esi
  int v40; // r13d
  __int64 v41; // rax
  unsigned int v42; // ecx
  __int64 v43; // rdx
  __int64 v44; // rdi
  _BYTE *v45; // r13
  int v46; // ecx
  __int64 v47; // rsi
  int v48; // ecx
  unsigned int v49; // edx
  __int64 *v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rax
  int v54; // eax
  char v55; // dl
  unsigned int v56; // esi
  __int64 v57; // r9
  int v58; // r13d
  __int64 *v59; // rax
  __int64 *v60; // rdx
  __int64 v61; // rdi
  _BYTE *v62; // r13
  unsigned __int64 v63; // rax
  __int64 v64; // r13
  int v65; // eax
  __int64 v66; // rbx
  __int64 v67; // rdx
  int v68; // ebx
  unsigned int v69; // r14d
  __int64 *v70; // r12
  int v71; // eax
  int v72; // esi
  int v73; // esi
  __int64 v74; // r10
  __int64 v75; // rcx
  int v76; // edx
  __int64 v77; // r9
  int v78; // r8d
  __int64 *v79; // rdi
  int v80; // edi
  char v81; // al
  int v82; // ecx
  int v83; // edx
  char v84; // al
  __int64 v85; // rbx
  __int64 v86; // r14
  __int64 v87; // rax
  int v88; // esi
  int v89; // esi
  __int64 v90; // r10
  int v91; // r8d
  __int64 v92; // rcx
  __int64 v93; // r9
  int v94; // r8d
  int v95; // r8d
  int v96; // r8d
  __int64 v97; // r10
  __int64 v98; // r9
  __int64 v99; // rdi
  int v100; // esi
  __int64 v101; // rcx
  int v102; // edi
  int v103; // edi
  __int64 v104; // r10
  __int64 v105; // r9
  int v106; // esi
  __int64 v107; // r8
  __int64 v108; // [rsp+0h] [rbp-170h]
  __int64 v109; // [rsp+10h] [rbp-160h]
  __int64 v110; // [rsp+18h] [rbp-158h]
  __int64 v111; // [rsp+18h] [rbp-158h]
  __int64 v112; // [rsp+18h] [rbp-158h]
  __int64 v113; // [rsp+20h] [rbp-150h]
  __int64 v114; // [rsp+20h] [rbp-150h]
  __int64 v115; // [rsp+28h] [rbp-148h]
  int v116; // [rsp+28h] [rbp-148h]
  unsigned int v117; // [rsp+28h] [rbp-148h]
  unsigned int v118; // [rsp+28h] [rbp-148h]
  __int64 v119; // [rsp+38h] [rbp-138h]
  __int64 v120; // [rsp+38h] [rbp-138h]
  __int64 v121; // [rsp+40h] [rbp-130h]
  int v122; // [rsp+4Ch] [rbp-124h]
  unsigned int v123; // [rsp+4Ch] [rbp-124h]
  __int64 *v124; // [rsp+50h] [rbp-120h] BYREF
  __int64 v125; // [rsp+58h] [rbp-118h]
  _BYTE v126[64]; // [rsp+60h] [rbp-110h] BYREF
  __int64 v127; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 *v128; // [rsp+A8h] [rbp-C8h]
  __int64 v129; // [rsp+B0h] [rbp-C0h]
  int v130; // [rsp+B8h] [rbp-B8h]
  unsigned __int8 v131; // [rsp+BCh] [rbp-B4h]
  char v132; // [rsp+C0h] [rbp-B0h] BYREF

  v4 = *(_QWORD *)(a2 + 72);
  if ( !*(_QWORD *)(a1 + 24)
    || (v5 = a1,
        a1 += 8,
        v6 = (*(__int64 (__fastcall **)(__int64, __int64))(v5 + 32))(a1, v4),
        v4 = *(_QWORD *)(a2 + 72),
        v121 = v6,
        v7 = v6,
        !*(_QWORD *)(v5 + 88)) )
  {
    sub_4263D6(a1, v4, a3);
  }
  v115 = (*(__int64 (__fastcall **)(__int64, __int64))(v5 + 96))(v5 + 72, v4);
  v119 = *(_QWORD *)(a2 + 72);
  if ( !v7 )
  {
    if ( (unsigned __int8)sub_B2D610(*(_QWORD *)(a2 + 72), 76) )
    {
LABEL_66:
      v110 = a2;
      v113 = 0;
LABEL_9:
      v14 = sub_B2D610(v119, 41);
      goto LABEL_10;
    }
LABEL_61:
    v110 = a2;
    v14 = 0;
    v113 = 0;
    goto LABEL_10;
  }
  v8 = *(_DWORD *)(v121 + 24);
  v9 = *(_QWORD *)(v121 + 8);
  if ( !v8 )
  {
LABEL_65:
    if ( (unsigned __int8)sub_B2D610(v119, 76) )
      goto LABEL_66;
    goto LABEL_61;
  }
  v10 = v8 - 1;
  v11 = v10 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = (__int64 *)(v9 + 16LL * v11);
  v13 = *v12;
  if ( a2 != *v12 )
  {
    v54 = 1;
    while ( v13 != -4096 )
    {
      v94 = v54 + 1;
      v11 = v10 & (v54 + v11);
      v12 = (__int64 *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( a2 == *v12 )
        goto LABEL_6;
      v54 = v94;
    }
    goto LABEL_65;
  }
LABEL_6:
  v113 = v12[1];
  if ( !v113 )
    goto LABEL_65;
  v110 = **(_QWORD **)(v113 + 32);
  if ( (unsigned __int8)sub_B2D610(v119, 76) )
    goto LABEL_9;
  v14 = sub_B2D610(*(_QWORD *)(**(_QWORD **)(v113 + 32) + 72LL), 76);
  if ( v14 )
    goto LABEL_9;
LABEL_10:
  v124 = (__int64 *)v126;
  v125 = 0x800000000LL;
  v15 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v15 == a2 + 48 )
    goto LABEL_50;
  if ( !v15 )
LABEL_59:
    BUG();
  v16 = v15 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v15 - 24) - 30 > 0xA )
    goto LABEL_50;
  v122 = sub_B46E30(v16);
  if ( !v122 )
    goto LABEL_50;
  v17 = v14;
  v18 = 0;
  v109 = v5;
  v19 = v110;
  v111 = a2;
  v20 = 0;
  v21 = v17 ^ 1;
  do
  {
    while ( 1 )
    {
      v22 = sub_B46EC0(v16, v20);
      if ( v22 != v19 || v21 )
        break;
      if ( v122 == ++v20 )
        goto LABEL_21;
    }
    if ( v18 + 1 > (unsigned __int64)HIDWORD(v125) )
    {
      v108 = v22;
      sub_C8D5F0((__int64)&v124, v126, v18 + 1, 8u, v23, v24);
      v18 = (unsigned int)v125;
      v22 = v108;
    }
    ++v20;
    v124[v18] = v22;
    v18 = (unsigned int)(v125 + 1);
    LODWORD(v125) = v125 + 1;
  }
  while ( v122 != v20 );
LABEL_21:
  if ( !(_DWORD)v18 )
    goto LABEL_50;
  if ( (_DWORD)v18 == 1 )
  {
    v31 = v124;
    v28 = *v124;
    goto LABEL_51;
  }
  if ( !v115
    || (v25 = (unsigned int)(*(_DWORD *)(v111 + 44) + 1), (unsigned int)v25 >= *(_DWORD *)(v115 + 56))
    || (v26 = *(_QWORD *)(*(_QWORD *)(v115 + 48) + 8 * v25)) == 0
    || (v27 = *(__int64 **)(v26 + 8)) == 0 )
  {
    if ( (_DWORD)v18 != 2 )
    {
      if ( !v113 )
        goto LABEL_50;
      goto LABEL_56;
    }
LABEL_135:
    v28 = *v124;
    v85 = v124[1];
    v86 = sub_AA5780(*v124);
    v87 = sub_AA5780(v85);
    if ( v111 != v86 )
    {
      if ( v28 == v87 || v111 == v87 )
      {
LABEL_143:
        if ( v28 )
          goto LABEL_28;
        if ( !v113 )
        {
LABEL_57:
          if ( !v28 )
            goto LABEL_131;
          goto LABEL_28;
        }
LABEL_56:
        v28 = sub_D47600(v113);
        goto LABEL_57;
      }
      if ( v85 != v86 )
      {
        v28 = v86;
        if ( v86 != v87 )
        {
LABEL_140:
          if ( !v113 )
            goto LABEL_50;
          goto LABEL_56;
        }
        goto LABEL_143;
      }
    }
    v28 = v85;
    goto LABEL_143;
  }
  v28 = *v27;
  if ( !*v27 )
  {
    if ( (_DWORD)v18 != 2 )
      goto LABEL_140;
    goto LABEL_135;
  }
LABEL_28:
  if ( (unsigned __int8)sub_B2D610(v119, 76) && (unsigned __int8)sub_B2D610(v119, 41) )
    goto LABEL_131;
  v127 = 0;
  v128 = (__int64 *)&v132;
  v31 = v124;
  v32 = 1;
  v129 = 16;
  v130 = 0;
  v33 = ((unsigned int)v119 >> 9) ^ ((unsigned int)v119 >> 4);
  v131 = 1;
  v34 = v119;
  v120 = v109 + 104;
  v123 = v33;
  v35 = v125;
  while ( 2 )
  {
    v36 = &v31[v35];
    do
    {
      if ( !v35 )
      {
        if ( (_BYTE)v32 )
          goto LABEL_51;
        _libc_free((unsigned __int64)v128);
LABEL_131:
        v31 = v124;
        goto LABEL_51;
      }
      v37 = *(v36 - 1);
      --v35;
      --v36;
      LODWORD(v125) = v35;
    }
    while ( v28 == v37 );
    if ( !(_BYTE)v32 )
      break;
    v38 = v128;
    v32 = HIDWORD(v129);
    v36 = &v128[HIDWORD(v129)];
    if ( v128 != v36 )
    {
      while ( v37 != *v38 )
      {
        if ( v36 == ++v38 )
          goto LABEL_89;
      }
      goto LABEL_38;
    }
LABEL_89:
    if ( HIDWORD(v129) < (unsigned int)v129 )
    {
      ++HIDWORD(v129);
      *v36 = v37;
      v56 = *(_DWORD *)(v109 + 128);
      ++v127;
      if ( !v56 )
      {
LABEL_91:
        ++*(_QWORD *)(v109 + 104);
        goto LABEL_92;
      }
LABEL_69:
      v57 = *(_QWORD *)(v109 + 112);
      v58 = 1;
      v59 = 0;
      v29 = (v56 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
      v60 = (__int64 *)(v57 + 16 * v29);
      v61 = *v60;
      if ( v37 == *v60 )
      {
LABEL_70:
        v62 = v60 + 1;
        if ( *((_BYTE *)v60 + 9) )
          goto LABEL_71;
        goto LABEL_114;
      }
      while ( v61 != -4096 )
      {
        if ( !v59 && v61 == -8192 )
          v59 = v60;
        v29 = (v56 - 1) & (v58 + (_DWORD)v29);
        v60 = (__int64 *)(v57 + 16LL * (unsigned int)v29);
        v61 = *v60;
        if ( v37 == *v60 )
          goto LABEL_70;
        ++v58;
      }
      v80 = *(_DWORD *)(v109 + 120);
      if ( !v59 )
        v59 = v60;
      ++*(_QWORD *)(v109 + 104);
      v76 = v80 + 1;
      if ( 4 * (v80 + 1) < 3 * v56 )
      {
        if ( v56 - *(_DWORD *)(v109 + 124) - v76 > v56 >> 3 )
          goto LABEL_111;
        v118 = ((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4);
        sub_3105520(v120, v56);
        v88 = *(_DWORD *)(v109 + 128);
        if ( v88 )
        {
          v89 = v88 - 1;
          v90 = *(_QWORD *)(v109 + 112);
          v91 = 1;
          LODWORD(v92) = v89 & v118;
          v76 = *(_DWORD *)(v109 + 120) + 1;
          v79 = 0;
          v59 = (__int64 *)(v90 + 16LL * (v89 & v118));
          v93 = *v59;
          if ( v37 == *v59 )
            goto LABEL_111;
          while ( v93 != -4096 )
          {
            if ( v93 == -8192 && !v79 )
              v79 = v59;
            v92 = v89 & (unsigned int)(v92 + v91);
            v59 = (__int64 *)(v90 + 16 * v92);
            v93 = *v59;
            if ( v37 == *v59 )
              goto LABEL_111;
            ++v91;
          }
LABEL_149:
          if ( v79 )
            v59 = v79;
          goto LABEL_111;
        }
LABEL_183:
        ++*(_DWORD *)(v109 + 120);
        BUG();
      }
LABEL_92:
      sub_3105520(v120, 2 * v56);
      v72 = *(_DWORD *)(v109 + 128);
      if ( !v72 )
        goto LABEL_183;
      v73 = v72 - 1;
      v74 = *(_QWORD *)(v109 + 112);
      LODWORD(v75) = v73 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
      v76 = *(_DWORD *)(v109 + 120) + 1;
      v59 = (__int64 *)(v74 + 16LL * (unsigned int)v75);
      v77 = *v59;
      if ( v37 != *v59 )
      {
        v78 = 1;
        v79 = 0;
        while ( v77 != -4096 )
        {
          if ( !v79 && v77 == -8192 )
            v79 = v59;
          v75 = v73 & (unsigned int)(v75 + v78);
          v59 = (__int64 *)(v74 + 16 * v75);
          v77 = *v59;
          if ( v37 == *v59 )
            goto LABEL_111;
          ++v78;
        }
        goto LABEL_149;
      }
LABEL_111:
      *(_DWORD *)(v109 + 120) = v76;
      if ( *v59 != -4096 )
        --*(_DWORD *)(v109 + 124);
      *v59 = v37;
      v62 = v59 + 1;
      *((_BYTE *)v59 + 9) = 0;
LABEL_114:
      v81 = sub_98CE00(v37);
      v62[1] = 1;
      *v62 = v81;
LABEL_71:
      if ( !*v62 )
        goto LABEL_48;
      v63 = *(_QWORD *)(v37 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v63 == v37 + 48 )
        goto LABEL_129;
      if ( !v63 )
        goto LABEL_59;
      v64 = v63 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v63 - 24) - 30 <= 0xA )
      {
        v65 = sub_B46E30(v64);
        v66 = v65;
        v30 = v65;
        goto LABEL_76;
      }
LABEL_129:
      v66 = 0;
      v30 = 0;
      v64 = 0;
LABEL_76:
      v67 = (unsigned int)v125;
      if ( v66 + (unsigned __int64)(unsigned int)v125 > HIDWORD(v125) )
      {
        v117 = v30;
        sub_C8D5F0((__int64)&v124, v126, v66 + (unsigned int)v125, 8u, v29, v30);
        v67 = (unsigned int)v125;
        v30 = v117;
      }
      v31 = v124;
      if ( (_DWORD)v30 )
      {
        v116 = v66;
        v68 = v30;
        v114 = v34;
        v69 = 0;
        v112 = v28;
        v70 = &v124[v67];
        do
        {
          if ( v70 )
            *v70 = sub_B46EC0(v64, v69);
          ++v69;
          ++v70;
        }
        while ( v68 != v69 );
        LODWORD(v66) = v116;
        v34 = v114;
        v28 = v112;
        LODWORD(v67) = v125;
        v31 = v124;
      }
      LODWORD(v125) = v66 + v67;
      v35 = v66 + v67;
LABEL_85:
      v32 = v131;
      continue;
    }
    break;
  }
  sub_C8CC70((__int64)&v127, v37, (__int64)v36, v32, v29, v30);
  if ( v55 )
  {
    v56 = *(_DWORD *)(v109 + 128);
    if ( !v56 )
      goto LABEL_91;
    goto LABEL_69;
  }
LABEL_38:
  if ( (unsigned __int8)sub_B2D610(v34, 76) )
    goto LABEL_88;
  if ( !v121 )
    goto LABEL_48;
  v39 = *(_DWORD *)(v109 + 160);
  if ( !v39 )
  {
    ++*(_QWORD *)(v109 + 136);
    goto LABEL_157;
  }
  v30 = v39 - 1;
  v40 = 1;
  v41 = 0;
  v29 = *(_QWORD *)(v109 + 144);
  v42 = v30 & v123;
  v43 = v29 + 16LL * ((unsigned int)v30 & v123);
  v44 = *(_QWORD *)v43;
  if ( v34 == *(_QWORD *)v43 )
  {
LABEL_42:
    v45 = (_BYTE *)(v43 + 8);
    if ( *(_BYTE *)(v43 + 9) )
      goto LABEL_43;
    goto LABEL_128;
  }
  while ( v44 != -4096 )
  {
    if ( !v41 && v44 == -8192 )
      v41 = v43;
    v42 = v30 & (v40 + v42);
    v43 = v29 + 16LL * v42;
    v44 = *(_QWORD *)v43;
    if ( v34 == *(_QWORD *)v43 )
      goto LABEL_42;
    ++v40;
  }
  v82 = *(_DWORD *)(v109 + 152);
  if ( !v41 )
    v41 = v43;
  ++*(_QWORD *)(v109 + 136);
  v83 = v82 + 1;
  if ( 4 * (v82 + 1) >= 3 * v39 )
  {
LABEL_157:
    sub_3105340(v109 + 136, 2 * v39);
    v95 = *(_DWORD *)(v109 + 160);
    if ( v95 )
    {
      v96 = v95 - 1;
      v97 = *(_QWORD *)(v109 + 144);
      LODWORD(v98) = v96 & v123;
      v83 = *(_DWORD *)(v109 + 152) + 1;
      v41 = v97 + 16LL * (v96 & v123);
      v99 = *(_QWORD *)v41;
      if ( v34 == *(_QWORD *)v41 )
        goto LABEL_125;
      v100 = 1;
      v101 = 0;
      while ( v99 != -4096 )
      {
        if ( v99 == -8192 && !v101 )
          v101 = v41;
        v98 = v96 & (unsigned int)(v98 + v100);
        v41 = v97 + 16 * v98;
        v99 = *(_QWORD *)v41;
        if ( v34 == *(_QWORD *)v41 )
          goto LABEL_125;
        ++v100;
      }
LABEL_161:
      if ( v101 )
        v41 = v101;
      goto LABEL_125;
    }
LABEL_184:
    ++*(_DWORD *)(v109 + 152);
    BUG();
  }
  if ( v39 - *(_DWORD *)(v109 + 156) - v83 <= v39 >> 3 )
  {
    sub_3105340(v109 + 136, v39);
    v102 = *(_DWORD *)(v109 + 160);
    if ( v102 )
    {
      v103 = v102 - 1;
      v104 = *(_QWORD *)(v109 + 144);
      v101 = 0;
      LODWORD(v105) = v103 & v123;
      v83 = *(_DWORD *)(v109 + 152) + 1;
      v106 = 1;
      v41 = v104 + 16LL * (v103 & v123);
      v107 = *(_QWORD *)v41;
      if ( v34 == *(_QWORD *)v41 )
        goto LABEL_125;
      while ( v107 != -4096 )
      {
        if ( !v101 && v107 == -8192 )
          v101 = v41;
        v105 = v103 & (unsigned int)(v105 + v106);
        v41 = v104 + 16 * v105;
        v107 = *(_QWORD *)v41;
        if ( v34 == *(_QWORD *)v41 )
          goto LABEL_125;
        ++v106;
      }
      goto LABEL_161;
    }
    goto LABEL_184;
  }
LABEL_125:
  *(_DWORD *)(v109 + 152) = v83;
  if ( *(_QWORD *)v41 != -4096 )
    --*(_DWORD *)(v109 + 156);
  *(_QWORD *)v41 = v34;
  v45 = (_BYTE *)(v41 + 8);
  *(_BYTE *)(v41 + 9) = 0;
LABEL_128:
  v84 = sub_31052D0(v34, v121);
  v45[1] = 1;
  *v45 = v84;
LABEL_43:
  if ( !*v45 )
  {
    v46 = *(_DWORD *)(v121 + 24);
    v47 = *(_QWORD *)(v121 + 8);
    if ( v46 )
    {
      v48 = v46 - 1;
      v49 = v48 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
      v50 = (__int64 *)(v47 + 16LL * v49);
      v51 = *v50;
      if ( v37 == *v50 )
      {
LABEL_46:
        v52 = v50[1];
        if ( v52 && !(unsigned __int8)sub_B2D610(*(_QWORD *)(**(_QWORD **)(v52 + 32) + 72LL), 76) )
          goto LABEL_48;
      }
      else
      {
        v71 = 1;
        while ( v51 != -4096 )
        {
          v29 = (unsigned int)(v71 + 1);
          v49 = v48 & (v71 + v49);
          v50 = (__int64 *)(v47 + 16LL * v49);
          v51 = *v50;
          if ( v37 == *v50 )
            goto LABEL_46;
          v71 = v29;
        }
      }
    }
LABEL_88:
    v31 = v124;
    v35 = v125;
    goto LABEL_85;
  }
LABEL_48:
  if ( !v131 )
    _libc_free((unsigned __int64)v128);
LABEL_50:
  v31 = v124;
  v28 = 0;
LABEL_51:
  if ( v31 != (__int64 *)v126 )
    _libc_free((unsigned __int64)v31);
  return v28;
}
