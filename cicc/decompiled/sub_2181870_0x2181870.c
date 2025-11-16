// Function: sub_2181870
// Address: 0x2181870
//
__int64 __fastcall sub_2181870(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // r14
  __int64 v7; // r12
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rbx
  bool v10; // cc
  int v11; // eax
  __int64 v12; // rdx
  _DWORD *v13; // rax
  _DWORD *v14; // rdx
  _QWORD *v15; // r8
  __int64 v16; // r13
  __int64 v17; // r14
  __int64 v18; // r9
  unsigned int v19; // edi
  int *v20; // rdx
  int v21; // ecx
  int v22; // ebx
  int v23; // esi
  unsigned int v24; // r9d
  int v25; // r11d
  unsigned int v26; // edx
  unsigned int v27; // r15d
  int *v28; // rax
  int v29; // edi
  int v30; // r10d
  int v31; // edx
  unsigned int v32; // esi
  int v33; // eax
  int v34; // esi
  __int64 v35; // r9
  __int64 v36; // rdx
  int v37; // ecx
  int *v38; // rax
  int v39; // edi
  int *v40; // r10
  int v41; // r11d
  int v42; // ecx
  __int64 v43; // r11
  __int64 v44; // rbx
  unsigned int v45; // r13d
  __int64 v46; // r15
  unsigned int v47; // edx
  int *v48; // rcx
  unsigned int v49; // esi
  int v50; // eax
  __int64 v51; // r9
  unsigned int v52; // ecx
  int *v53; // rdi
  int v54; // r8d
  int v55; // esi
  int *v56; // rdi
  int v57; // ecx
  __int64 v58; // rdx
  unsigned __int64 v59; // rdx
  _DWORD *v60; // rax
  __int64 i; // rdx
  unsigned int v62; // ecx
  _DWORD *v63; // rdi
  unsigned int v64; // eax
  int v65; // r13d
  unsigned int v66; // eax
  int v67; // r11d
  int v68; // r11d
  int v69; // ecx
  int v70; // eax
  int v71; // edx
  __int64 v72; // rdi
  int *v73; // r9
  __int64 v74; // r15
  int v75; // r10d
  int v76; // esi
  int v77; // r11d
  int *v78; // rdx
  int v79; // ecx
  int v80; // ecx
  int v81; // esi
  int v82; // edi
  int v83; // esi
  __int64 v84; // r9
  unsigned int v85; // r8d
  int v86; // r11d
  int *v87; // r10
  int v88; // esi
  int v89; // esi
  __int64 v90; // r9
  int v91; // r11d
  unsigned int v92; // r8d
  int *v94; // r11
  int v95; // r11d
  int v96; // ecx
  int *v97; // r15
  void *v98; // rax
  __int64 v99; // rax
  __int64 v100; // rax
  __int64 v101; // rax
  __int64 v102; // rax
  __int64 v103; // rbx
  __int64 v104; // r13
  int v105; // eax
  void *v106; // rax
  __int64 v107; // rax
  __int64 v108; // rax
  int v109; // r11d
  int *v110; // r10
  _DWORD *v111; // rax
  int *v112; // r15
  int *v113; // rax
  int *v114; // r13
  int *v115; // rbx
  int *v116; // rdx
  _QWORD *v118; // [rsp+8h] [rbp-A8h]
  _QWORD *v119; // [rsp+8h] [rbp-A8h]
  _QWORD *v120; // [rsp+8h] [rbp-A8h]
  _QWORD *v121; // [rsp+8h] [rbp-A8h]
  int v122; // [rsp+8h] [rbp-A8h]
  _QWORD *v123; // [rsp+8h] [rbp-A8h]
  unsigned int v125; // [rsp+1Ch] [rbp-94h]
  int v126; // [rsp+28h] [rbp-88h] BYREF
  int v127; // [rsp+2Ch] [rbp-84h] BYREF
  __int64 v128; // [rsp+30h] [rbp-80h] BYREF
  __int64 v129; // [rsp+38h] [rbp-78h]
  __int64 v130; // [rsp+40h] [rbp-70h]
  unsigned int v131; // [rsp+48h] [rbp-68h]
  _QWORD v132[12]; // [rsp+50h] [rbp-60h] BYREF

  v6 = a2;
  v7 = a5;
  v8 = *(unsigned int *)(a2 + 8);
  v125 = *(_DWORD *)(a2 + 8);
  if ( byte_4FD2E80 )
  {
    v98 = sub_16E8CB0();
    v99 = sub_1263B40((__int64)v98, "After pre-check, ");
    v100 = sub_16E7A90(v99, *(unsigned int *)(a3 + 8));
    v101 = sub_1263B40(v100, " good candidates, ");
    v102 = sub_16E7A90(v101, *(unsigned int *)(a2 + 8));
    sub_1263B40(v102, " given second-chance\n");
    v8 = *(unsigned int *)(a2 + 8);
  }
  v9 = v125;
  v128 = 0;
  v129 = 0;
  v130 = 0;
  v131 = 0;
  v10 = v125 <= v8;
  if ( v125 < v8 )
    goto LABEL_4;
  while ( 2 )
  {
    if ( v10 )
    {
LABEL_5:
      v11 = v130;
      ++v128;
      if ( (_DWORD)v130 )
        goto LABEL_69;
    }
    else
    {
      if ( v9 > *(unsigned int *)(v6 + 12) )
      {
        sub_16CD150(v6, (const void *)(v6 + 16), v9, 4, a5, a6);
        v8 = *(unsigned int *)(v6 + 8);
      }
      v60 = (_DWORD *)(*(_QWORD *)v6 + 4 * v8);
      for ( i = *(_QWORD *)v6 + 4 * v9; (_DWORD *)i != v60; ++v60 )
      {
        if ( v60 )
          *v60 = 0;
      }
      ++v128;
      *(_DWORD *)(v6 + 8) = v125;
      v11 = v130;
      if ( (_DWORD)v130 )
      {
LABEL_69:
        v62 = 4 * v11;
        v12 = v131;
        if ( (unsigned int)(4 * v11) < 0x40 )
          v62 = 64;
        if ( v131 > v62 )
        {
          v63 = (_DWORD *)v129;
          v64 = v11 - 1;
          if ( v64 )
          {
            _BitScanReverse(&v64, v64);
            v65 = 1 << (33 - (v64 ^ 0x1F));
            if ( v65 < 64 )
              v65 = 64;
            if ( v131 == v65 )
            {
              v130 = 0;
              v111 = (_DWORD *)(v129 + 8LL * v131);
              do
              {
                if ( v63 )
                  *v63 = -1;
                v63 += 2;
              }
              while ( v111 != v63 );
              goto LABEL_11;
            }
          }
          else
          {
            v65 = 64;
          }
          j___libc_free_0(v129);
          v66 = sub_217D900(v65);
          v131 = v66;
          if ( v66 )
          {
            v129 = sub_22077B0(8LL * v66);
            sub_217ECA0((__int64)&v128);
            goto LABEL_11;
          }
          goto LABEL_77;
        }
        goto LABEL_8;
      }
    }
    if ( HIDWORD(v130) )
    {
      v12 = v131;
      if ( v131 > 0x40 )
      {
        j___libc_free_0(v129);
        v131 = 0;
LABEL_77:
        v129 = 0;
LABEL_10:
        v130 = 0;
        goto LABEL_11;
      }
LABEL_8:
      v13 = (_DWORD *)v129;
      v14 = (_DWORD *)(v129 + 8 * v12);
      if ( (_DWORD *)v129 != v14 )
      {
        do
        {
          *v13 = -1;
          v13 += 2;
        }
        while ( v14 != v13 );
      }
      goto LABEL_10;
    }
LABEL_11:
    if ( !v125 )
    {
      v59 = *(unsigned int *)(v6 + 8);
      break;
    }
    v15 = (_QWORD *)v6;
    v16 = 0;
    v17 = 4 * v9;
    while ( 1 )
    {
LABEL_21:
      v32 = *(_DWORD *)(v7 + 24);
      v22 = *(_DWORD *)(*v15 + v16);
      if ( !v32 )
      {
        ++*(_QWORD *)v7;
        goto LABEL_23;
      }
      v18 = *(_QWORD *)(v7 + 8);
      v19 = (v32 - 1) & (37 * v22);
      v20 = (int *)(v18 + 40LL * v19);
      v21 = *v20;
      if ( v22 != *v20 )
      {
        v68 = 1;
        v38 = 0;
        while ( v21 != -1 )
        {
          if ( !v38 && v21 == -2 )
            v38 = v20;
          v19 = (v32 - 1) & (v68 + v19);
          v20 = (int *)(v18 + 40LL * v19);
          v21 = *v20;
          if ( v22 == *v20 )
            goto LABEL_14;
          ++v68;
        }
        v69 = *(_DWORD *)(v7 + 16);
        if ( !v38 )
          v38 = v20;
        ++*(_QWORD *)v7;
        v37 = v69 + 1;
        if ( 4 * v37 < 3 * v32 )
        {
          if ( v32 - *(_DWORD *)(v7 + 20) - v37 <= v32 >> 3 )
          {
            v120 = v15;
            sub_1DF5170(v7, v32);
            v70 = *(_DWORD *)(v7 + 24);
            if ( !v70 )
            {
LABEL_213:
              ++*(_DWORD *)(v7 + 16);
              BUG();
            }
            v71 = v70 - 1;
            v72 = *(_QWORD *)(v7 + 8);
            v73 = 0;
            LODWORD(v74) = (v70 - 1) & (37 * v22);
            v15 = v120;
            v75 = 1;
            v37 = *(_DWORD *)(v7 + 16) + 1;
            v38 = (int *)(v72 + 40LL * (unsigned int)v74);
            v76 = *v38;
            if ( v22 != *v38 )
            {
              while ( v76 != -1 )
              {
                if ( !v73 && v76 == -2 )
                  v73 = v38;
                v74 = v71 & (unsigned int)(v74 + v75);
                v38 = (int *)(v72 + 40 * v74);
                v76 = *v38;
                if ( v22 == *v38 )
                  goto LABEL_25;
                ++v75;
              }
              if ( v73 )
                v38 = v73;
            }
          }
          goto LABEL_25;
        }
LABEL_23:
        v118 = v15;
        sub_1DF5170(v7, 2 * v32);
        v33 = *(_DWORD *)(v7 + 24);
        if ( !v33 )
          goto LABEL_213;
        v34 = v33 - 1;
        v35 = *(_QWORD *)(v7 + 8);
        v15 = v118;
        LODWORD(v36) = (v33 - 1) & (37 * v22);
        v37 = *(_DWORD *)(v7 + 16) + 1;
        v38 = (int *)(v35 + 40LL * (unsigned int)v36);
        v39 = *v38;
        if ( v22 != *v38 )
        {
          v109 = 1;
          v110 = 0;
          while ( v39 != -1 )
          {
            if ( !v110 && v39 == -2 )
              v110 = v38;
            v36 = v34 & (unsigned int)(v36 + v109);
            v38 = (int *)(v35 + 40 * v36);
            v39 = *v38;
            if ( v22 == *v38 )
              goto LABEL_25;
            ++v109;
          }
          if ( v110 )
            v38 = v110;
        }
LABEL_25:
        *(_DWORD *)(v7 + 16) = v37;
        if ( *v38 != -1 )
          --*(_DWORD *)(v7 + 20);
        *v38 = v22;
        *((_QWORD *)v38 + 1) = v38 + 6;
        *((_QWORD *)v38 + 2) = 0x400000000LL;
        goto LABEL_16;
      }
LABEL_14:
      if ( v20[4] )
        v22 = *(_DWORD *)(*((_QWORD *)v20 + 1) + 4LL * (unsigned int)v20[4] - 4);
LABEL_16:
      v23 = v131;
      v127 = v22;
      if ( !v131 )
        break;
      v24 = v131 - 1;
      v25 = 1;
      v26 = (v131 - 1) & (37 * v22);
      v27 = v26;
      v28 = (int *)(v129 + 8LL * v26);
      v29 = *v28;
      v30 = *v28;
      if ( v22 == *v28 )
      {
        if ( v28 != (int *)(v129 + 8LL * v131) )
        {
LABEL_19:
          v31 = v28[1] + 1;
          goto LABEL_20;
        }
        goto LABEL_41;
      }
      while ( 1 )
      {
        if ( v30 == -1 )
        {
          v26 = v24 & (37 * v22);
          v40 = (int *)(v129 + 8LL * v26);
          v29 = *v40;
          if ( v22 == *v40 )
          {
            v28 = (int *)(v129 + 8LL * v26);
            goto LABEL_41;
          }
LABEL_31:
          v41 = 1;
          v28 = 0;
          while ( v29 != -1 )
          {
            if ( v28 || v29 != -2 )
              v40 = v28;
            v26 = v24 & (v41 + v26);
            v28 = (int *)(v129 + 8LL * v26);
            v29 = *v28;
            if ( v22 == *v28 )
              goto LABEL_41;
            ++v41;
            v112 = v40;
            v40 = (int *)(v129 + 8LL * v26);
            v28 = v112;
          }
          if ( !v28 )
            v28 = v40;
          ++v128;
          v42 = v130 + 1;
          if ( 4 * ((int)v130 + 1) < 3 * v131 )
          {
            if ( v131 - HIDWORD(v130) - v42 <= v131 >> 3 )
            {
              v121 = v15;
              sub_1392B70((__int64)&v128, v131);
              sub_1932870((__int64)&v128, &v127, v132);
              v28 = (int *)v132[0];
              v22 = v127;
              v15 = v121;
              v42 = v130 + 1;
            }
            goto LABEL_38;
          }
          goto LABEL_79;
        }
        v27 = v24 & (v25 + v27);
        v122 = v25 + 1;
        v94 = (int *)(v129 + 8LL * v27);
        v30 = *v94;
        if ( v22 == *v94 )
          break;
        v25 = v122;
      }
      v40 = (int *)(v129 + 8LL * v26);
      if ( v94 == (int *)(v129 + 8LL * v131) )
        goto LABEL_31;
      v95 = 1;
      v28 = 0;
      while ( v29 != -1 )
      {
        if ( v29 != -2 || v28 )
          v40 = v28;
        v26 = v24 & (v95 + v26);
        v28 = (int *)(v129 + 8LL * v26);
        v29 = *v28;
        if ( v22 == *v28 )
          goto LABEL_19;
        ++v95;
        v97 = v40;
        v40 = (int *)(v129 + 8LL * v26);
        v28 = v97;
      }
      if ( !v28 )
        v28 = v40;
      ++v128;
      v96 = v130 + 1;
      if ( 4 * ((int)v130 + 1) >= 3 * v131 )
      {
        v123 = v15;
        v23 = 2 * v131;
      }
      else
      {
        if ( v131 - HIDWORD(v130) - v96 > v131 >> 3 )
          goto LABEL_140;
        v123 = v15;
      }
      sub_1392B70((__int64)&v128, v23);
      sub_1932870((__int64)&v128, &v127, v132);
      v28 = (int *)v132[0];
      v22 = v127;
      v15 = v123;
      v96 = v130 + 1;
LABEL_140:
      LODWORD(v130) = v96;
      if ( *v28 != -1 )
        --HIDWORD(v130);
      *v28 = v22;
      v31 = 1;
      v28[1] = 0;
LABEL_20:
      v16 += 4;
      v28[1] = v31;
      if ( v17 == v16 )
        goto LABEL_42;
    }
    ++v128;
LABEL_79:
    v119 = v15;
    sub_1392B70((__int64)&v128, 2 * v131);
    sub_1932870((__int64)&v128, &v127, v132);
    v28 = (int *)v132[0];
    v22 = v127;
    v15 = v119;
    v42 = v130 + 1;
LABEL_38:
    LODWORD(v130) = v42;
    if ( *v28 != -1 )
      --HIDWORD(v130);
    *v28 = v22;
    v28[1] = 0;
LABEL_41:
    v16 += 4;
    v28[1] = 1;
    if ( v17 != v16 )
      goto LABEL_21;
LABEL_42:
    v43 = v17;
    v44 = 0;
    v6 = (__int64)v15;
    v45 = 0;
    v46 = v43;
    while ( 1 )
    {
LABEL_47:
      v49 = *(_DWORD *)(v7 + 24);
      v50 = *(_DWORD *)(*(_QWORD *)v6 + v44);
      v126 = v50;
      if ( v49 )
      {
        v51 = *(_QWORD *)(v7 + 8);
        v52 = (v49 - 1) & (37 * v50);
        v53 = (int *)(v51 + 40LL * v52);
        v54 = *v53;
        if ( v50 == *v53 )
        {
LABEL_49:
          if ( v53[4] )
            v50 = *(_DWORD *)(*((_QWORD *)v53 + 1) + 4LL * (unsigned int)v53[4] - 4);
          goto LABEL_51;
        }
        v77 = 1;
        v78 = 0;
        while ( v54 != -1 )
        {
          if ( v54 == -2 && !v78 )
            v78 = v53;
          v52 = (v49 - 1) & (v77 + v52);
          v53 = (int *)(v51 + 40LL * v52);
          v54 = *v53;
          if ( v50 == *v53 )
            goto LABEL_49;
          ++v77;
        }
        v79 = *(_DWORD *)(v7 + 16);
        if ( !v78 )
          v78 = v53;
        ++*(_QWORD *)v7;
        v80 = v79 + 1;
        if ( 4 * v80 < 3 * v49 )
        {
          if ( v49 - *(_DWORD *)(v7 + 20) - v80 > v49 >> 3 )
            goto LABEL_105;
          sub_1DF5170(v7, v49);
          v88 = *(_DWORD *)(v7 + 24);
          if ( !v88 )
          {
LABEL_212:
            ++*(_DWORD *)(v7 + 16);
            BUG();
          }
          v89 = v88 - 1;
          v90 = *(_QWORD *)(v7 + 8);
          v87 = 0;
          v82 = v126;
          v91 = 1;
          v80 = *(_DWORD *)(v7 + 16) + 1;
          v92 = v89 & (37 * v126);
          v78 = (int *)(v90 + 40LL * v92);
          v50 = *v78;
          if ( v126 == *v78 )
            goto LABEL_105;
          while ( v50 != -1 )
          {
            if ( !v87 && v50 == -2 )
              v87 = v78;
            v92 = v89 & (v91 + v92);
            v78 = (int *)(v90 + 40LL * v92);
            v50 = *v78;
            if ( v126 == *v78 )
              goto LABEL_105;
            ++v91;
          }
          goto LABEL_113;
        }
      }
      else
      {
        ++*(_QWORD *)v7;
      }
      sub_1DF5170(v7, 2 * v49);
      v81 = *(_DWORD *)(v7 + 24);
      if ( !v81 )
        goto LABEL_212;
      v82 = v126;
      v83 = v81 - 1;
      v84 = *(_QWORD *)(v7 + 8);
      v80 = *(_DWORD *)(v7 + 16) + 1;
      v85 = v83 & (37 * v126);
      v78 = (int *)(v84 + 40LL * v85);
      v50 = *v78;
      if ( v126 == *v78 )
        goto LABEL_105;
      v86 = 1;
      v87 = 0;
      while ( v50 != -1 )
      {
        if ( v50 == -2 && !v87 )
          v87 = v78;
        v85 = v83 & (v86 + v85);
        v78 = (int *)(v84 + 40LL * v85);
        v50 = *v78;
        if ( v126 == *v78 )
          goto LABEL_105;
        ++v86;
      }
LABEL_113:
      v50 = v82;
      if ( v87 )
        v78 = v87;
LABEL_105:
      *(_DWORD *)(v7 + 16) = v80;
      if ( *v78 != -1 )
        --*(_DWORD *)(v7 + 20);
      *v78 = v50;
      *((_QWORD *)v78 + 1) = v78 + 6;
      *((_QWORD *)v78 + 2) = 0x400000000LL;
      v50 = v126;
LABEL_51:
      v55 = v131;
      v127 = v50;
      if ( !v131 )
      {
        ++v128;
        goto LABEL_53;
      }
      a6 = v129;
      v47 = (v131 - 1) & (37 * v50);
      v48 = (int *)(v129 + 8LL * v47);
      LODWORD(a5) = *v48;
      if ( v50 != *v48 )
        break;
LABEL_44:
      if ( v48[1] != 1 || *(_DWORD *)(v6 + 8) == 1 )
        goto LABEL_58;
      v44 += 4;
      sub_217F7B0((__int64)v132, a4, &v126);
      if ( v46 == v44 )
        goto LABEL_59;
    }
    v67 = 1;
    v56 = 0;
    while ( (_DWORD)a5 != -1 )
    {
      if ( !v56 && (_DWORD)a5 == -2 )
        v56 = v48;
      v47 = (v131 - 1) & (v67 + v47);
      v48 = (int *)(v129 + 8LL * v47);
      LODWORD(a5) = *v48;
      if ( v50 == *v48 )
        goto LABEL_44;
      ++v67;
    }
    if ( !v56 )
      v56 = v48;
    ++v128;
    v57 = v130 + 1;
    if ( 4 * ((int)v130 + 1) >= 3 * v131 )
    {
LABEL_53:
      v55 = 2 * v131;
LABEL_54:
      sub_1392B70((__int64)&v128, v55);
      sub_1932870((__int64)&v128, &v127, v132);
      v56 = (int *)v132[0];
      v50 = v127;
      v57 = v130 + 1;
      goto LABEL_55;
    }
    LODWORD(a5) = v131 >> 3;
    if ( v131 - HIDWORD(v130) - v57 <= v131 >> 3 )
      goto LABEL_54;
LABEL_55:
    LODWORD(v130) = v57;
    if ( *v56 != -1 )
      --HIDWORD(v130);
    *v56 = v50;
    v56[1] = 0;
LABEL_58:
    v58 = v45;
    v44 += 4;
    ++v45;
    *(_DWORD *)(*(_QWORD *)v6 + 4 * v58) = v126;
    if ( v46 != v44 )
      goto LABEL_47;
LABEL_59:
    v8 = *(unsigned int *)(v6 + 8);
    v59 = v8;
    if ( v125 != v45 )
    {
      v125 = v45;
      v9 = v45;
      v10 = v45 <= v8;
      if ( v45 >= v8 )
        continue;
LABEL_4:
      *(_DWORD *)(v6 + 8) = v9;
      goto LABEL_5;
    }
    break;
  }
  if ( (unsigned int)v130 <= (unsigned int)v59 )
  {
    if ( (_DWORD)v130 )
    {
      v113 = (int *)v129;
      v114 = (int *)(v129 + 8LL * v131);
      if ( (int *)v129 != v114 )
      {
        while ( 1 )
        {
          v115 = v113;
          if ( (unsigned int)*v113 <= 0xFFFFFFFD )
            break;
          v113 += 2;
          if ( v114 == v113 )
            goto LABEL_162;
        }
        if ( v114 != v113 )
        {
          do
          {
            v116 = v115;
            v115 += 2;
            sub_217F7B0((__int64)v132, a4, v116);
            if ( v115 == v114 )
              break;
            while ( (unsigned int)*v115 > 0xFFFFFFFD )
            {
              v115 += 2;
              if ( v114 == v115 )
                goto LABEL_190;
            }
          }
          while ( v115 != v114 );
LABEL_190:
          v59 = *(unsigned int *)(v6 + 8);
        }
      }
    }
LABEL_162:
    if ( (_DWORD)v59 )
    {
      v103 = 4 * v59;
      v104 = 0;
      do
      {
        v105 = *(_DWORD *)(*(_QWORD *)v6 + v104);
        v104 += 4;
        LODWORD(v132[0]) = v105;
        sub_1525B90(a3, v132);
      }
      while ( v103 != v104 );
    }
    if ( byte_4FD2E80 )
    {
      v106 = sub_16E8CB0();
      v107 = sub_1263B40((__int64)v106, "ADD ");
      v108 = sub_16E7A90(v107, *(unsigned int *)(v6 + 8) - (unsigned __int64)(unsigned int)v130);
      sub_1263B40(v108, " candidates from second-chance\n");
    }
  }
  return j___libc_free_0(v129);
}
