// Function: sub_D1EE30
// Address: 0xd1ee30
//
__int64 __fastcall sub_D1EE30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 *v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 *v14; // rax
  __int64 *v15; // rax
  _QWORD *v16; // rax
  _QWORD *v17; // r15
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 result; // rax
  __int64 *v21; // rcx
  __int64 *v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  unsigned int v25; // eax
  unsigned int v26; // eax
  __int64 *v27; // rax
  __int64 v28; // r13
  _QWORD *v29; // rax
  _QWORD *v30; // r12
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // rax
  bool v34; // zf
  __int64 *v35; // rax
  __int64 *v36; // rdx
  __int64 *v37; // r13
  __int64 v38; // r12
  __int64 *v39; // rbx
  __int64 *v40; // rax
  unsigned int v41; // esi
  int v42; // r11d
  __int64 v43; // r9
  __int64 *v44; // rdx
  unsigned int v45; // r8d
  _QWORD *v46; // rdi
  __int64 v47; // rcx
  unsigned __int64 *v48; // rdi
  __int64 *v49; // rax
  __int64 *v50; // rax
  __int64 *v51; // r11
  __int64 v52; // rbx
  __int64 *v53; // r13
  __int64 *v54; // r12
  __int64 *v55; // rax
  unsigned int v56; // esi
  __int64 v57; // r8
  int v58; // r11d
  __int64 *v59; // r10
  unsigned int v60; // edx
  __int64 *v61; // rdi
  __int64 v62; // rcx
  __int64 *v63; // rax
  int v64; // ecx
  int v65; // ecx
  char v66; // dl
  _QWORD *v67; // rax
  __int64 v68; // rsi
  _QWORD *v69; // r8
  __int64 v70; // rax
  int v71; // r10d
  int v72; // r10d
  __int64 v73; // r8
  unsigned int v74; // eax
  __int64 v75; // rsi
  int v76; // r11d
  __int64 *v77; // rdi
  int v78; // r10d
  int v79; // r10d
  __int64 v80; // r8
  int v81; // r11d
  unsigned int v82; // eax
  __int64 v83; // rsi
  char v84; // dl
  _QWORD *v85; // rax
  __int64 v86; // rsi
  _QWORD *v87; // r8
  __int64 v88; // rax
  int v89; // r11d
  int v90; // r11d
  __int64 v91; // r9
  unsigned int v92; // eax
  int v93; // edx
  __int64 v94; // r8
  int v95; // esi
  __int64 *v96; // rcx
  int v97; // ecx
  int v98; // r11d
  int v99; // r11d
  __int64 v100; // r9
  int v101; // esi
  unsigned int v102; // eax
  __int64 v103; // r8
  __int64 v104; // [rsp+0h] [rbp-2D0h]
  __int64 v105; // [rsp+8h] [rbp-2C8h]
  __int64 v106; // [rsp+18h] [rbp-2B8h]
  _QWORD *v107; // [rsp+18h] [rbp-2B8h]
  unsigned int v108; // [rsp+18h] [rbp-2B8h]
  __int64 v109; // [rsp+18h] [rbp-2B8h]
  _QWORD *v110; // [rsp+18h] [rbp-2B8h]
  unsigned int v111; // [rsp+18h] [rbp-2B8h]
  __int64 v112; // [rsp+28h] [rbp-2A8h]
  __int64 v113; // [rsp+30h] [rbp-2A0h]
  __int64 v114; // [rsp+38h] [rbp-298h]
  __int64 v115; // [rsp+38h] [rbp-298h]
  __int64 v116; // [rsp+40h] [rbp-290h] BYREF
  void *s; // [rsp+48h] [rbp-288h]
  _BYTE v118[12]; // [rsp+50h] [rbp-280h]
  unsigned __int8 v119; // [rsp+5Ch] [rbp-274h]
  char v120; // [rsp+60h] [rbp-270h] BYREF
  __int64 v121; // [rsp+E0h] [rbp-1F0h] BYREF
  void *v122; // [rsp+E8h] [rbp-1E8h]
  _BYTE v123[12]; // [rsp+F0h] [rbp-1E0h]
  char v124; // [rsp+FCh] [rbp-1D4h]
  char v125; // [rsp+100h] [rbp-1D0h] BYREF
  __int64 v126; // [rsp+180h] [rbp-150h] BYREF
  __int64 *v127; // [rsp+188h] [rbp-148h]
  __int64 v128; // [rsp+190h] [rbp-140h]
  int v129; // [rsp+198h] [rbp-138h]
  char v130; // [rsp+19Ch] [rbp-134h]
  char v131; // [rsp+1A0h] [rbp-130h] BYREF

  v7 = a2;
  v8 = a2 + 24;
  v9 = *(_QWORD *)(a2 + 32);
  v127 = (__int64 *)&v131;
  v126 = 0;
  v128 = 32;
  v129 = 0;
  v130 = 1;
  if ( v9 == a2 + 24 )
    goto LABEL_23;
  v112 = a2;
  do
  {
    while ( 1 )
    {
      if ( !v9 )
        BUG();
      if ( (*(_BYTE *)(v9 - 24) & 0xFu) - 7 > 1 )
        goto LABEL_4;
      v10 = v9 - 56;
      a2 = v9 - 56;
      if ( !(unsigned __int8)sub_D1B9A0(a1, v9 - 56, 0, 0, 0, a6) )
        break;
      *(_BYTE *)(a1 + 136) = 1;
LABEL_4:
      v9 = *(_QWORD *)(v9 + 8);
      if ( v8 == v9 )
        goto LABEL_22;
    }
    if ( !*(_BYTE *)(a1 + 68) )
      goto LABEL_51;
    v14 = *(__int64 **)(a1 + 48);
    v12 = *(unsigned int *)(a1 + 60);
    v11 = &v14[v12];
    if ( v14 == v11 )
    {
LABEL_50:
      if ( (unsigned int)v12 >= *(_DWORD *)(a1 + 56) )
      {
LABEL_51:
        sub_C8CC70(a1 + 40, v9 - 56, (__int64)v11, v12, v13, a6);
        goto LABEL_13;
      }
      v12 = (unsigned int)(v12 + 1);
      *(_DWORD *)(a1 + 60) = v12;
      *v11 = v10;
      ++*(_QWORD *)(a1 + 40);
    }
    else
    {
      while ( v10 != *v14 )
      {
        if ( v11 == ++v14 )
          goto LABEL_50;
      }
    }
LABEL_13:
    if ( !v130 )
      goto LABEL_49;
    v15 = v127;
    v12 = HIDWORD(v128);
    v11 = &v127[HIDWORD(v128)];
    if ( v127 == v11 )
    {
LABEL_48:
      if ( HIDWORD(v128) < (unsigned int)v128 )
      {
        ++HIDWORD(v128);
        *v11 = v10;
        ++v126;
        goto LABEL_18;
      }
LABEL_49:
      sub_C8CC70((__int64)&v126, v9 - 56, (__int64)v11, v12, v13, a6);
      goto LABEL_18;
    }
    while ( v10 != *v15 )
    {
      if ( v11 == ++v15 )
        goto LABEL_48;
    }
LABEL_18:
    v114 = *(_QWORD *)(a1 + 336);
    v16 = (_QWORD *)sub_22077B0(64);
    a2 = v114;
    v16[3] = 2;
    v17 = v16;
    v16[4] = 0;
    v16[5] = v10;
    if ( v9 != -8136 && v9 != -4040 )
    {
      sub_BD73F0((__int64)(v16 + 3));
      a2 = v114;
    }
    v17[6] = a1;
    v17[7] = 0;
    v17[2] = &unk_49DDE50;
    sub_2208C80(v17, a2);
    v18 = *(_QWORD *)(a1 + 336);
    ++*(_QWORD *)(a1 + 352);
    *(_QWORD *)(v18 + 56) = v18;
    v9 = *(_QWORD *)(v9 + 8);
  }
  while ( v8 != v9 );
LABEL_22:
  v7 = v112;
LABEL_23:
  v19 = *(_QWORD *)(v7 + 16);
  v116 = 0;
  s = &v120;
  v122 = &v125;
  result = v7 + 8;
  *(_QWORD *)v118 = 16;
  *(_DWORD *)&v118[8] = 0;
  v119 = 1;
  v121 = 0;
  *(_QWORD *)v123 = 16;
  *(_DWORD *)&v123[8] = 0;
  v124 = 1;
  v115 = v7 + 8;
  if ( v7 + 8 == v19 )
    goto LABEL_46;
  while ( 2 )
  {
    if ( !v19 )
      BUG();
    if ( (*(_BYTE *)(v19 - 24) & 0xFu) - 7 <= 1 )
    {
      a2 = v19 - 56;
      v21 = &v121;
      v113 = v19 - 56;
      if ( (*(_BYTE *)(v19 + 24) & 1) != 0 )
        v21 = 0;
      if ( (unsigned __int8)sub_D1B9A0(a1, a2, (__int64)&v116, (__int64)v21, 0, a6) )
        goto LABEL_29;
      if ( !*(_BYTE *)(a1 + 68) )
        goto LABEL_71;
      v27 = *(__int64 **)(a1 + 48);
      v23 = *(unsigned int *)(a1 + 60);
      v22 = &v27[v23];
      if ( v27 != v22 )
      {
        while ( v113 != *v27 )
        {
          if ( v22 == ++v27 )
            goto LABEL_70;
        }
        goto LABEL_57;
      }
LABEL_70:
      if ( (unsigned int)v23 < *(_DWORD *)(a1 + 56) )
      {
        *(_DWORD *)(a1 + 60) = v23 + 1;
        *v22 = v113;
        ++*(_QWORD *)(a1 + 40);
      }
      else
      {
LABEL_71:
        sub_C8CC70(a1 + 40, v113, (__int64)v22, v23, v24, a6);
      }
LABEL_57:
      v28 = *(_QWORD *)(a1 + 336);
      v29 = (_QWORD *)sub_22077B0(64);
      v29[3] = 2;
      v30 = v29;
      v29[4] = 0;
      v29[5] = v113;
      if ( v19 != -4040 && v19 != -8136 )
        sub_BD73F0((__int64)(v29 + 3));
      v30[6] = a1;
      a2 = v28;
      v30[7] = 0;
      v30[2] = &unk_49DDE50;
      sub_2208C80(v30, v28);
      v33 = *(_QWORD *)(a1 + 336);
      ++*(_QWORD *)(a1 + 352);
      v34 = v119 == 0;
      *(_QWORD *)(v33 + 56) = v33;
      v35 = (__int64 *)s;
      if ( v34 )
      {
        v36 = (__int64 *)*(unsigned int *)v118;
        v37 = (__int64 *)((char *)s + 8 * *(unsigned int *)v118);
      }
      else
      {
        v36 = (__int64 *)*(unsigned int *)&v118[4];
        v37 = (__int64 *)((char *)s + 8 * *(unsigned int *)&v118[4]);
      }
      if ( s != v37 )
      {
        while ( 1 )
        {
          v38 = *v35;
          v39 = v35;
          if ( (unsigned __int64)*v35 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v37 == ++v35 )
            goto LABEL_65;
        }
        if ( v37 != v35 )
        {
          v104 = a1 + 272;
          if ( v130 )
          {
LABEL_74:
            v40 = v127;
            v31 = HIDWORD(v128);
            v36 = &v127[HIDWORD(v128)];
            if ( v127 != v36 )
            {
              do
              {
                if ( *v40 == v38 )
                  goto LABEL_78;
                ++v40;
              }
              while ( v36 != v40 );
            }
            if ( HIDWORD(v128) >= (unsigned int)v128 )
              goto LABEL_118;
            ++HIDWORD(v128);
            *v36 = v38;
            ++v126;
LABEL_119:
            v106 = *(_QWORD *)(a1 + 336);
            v67 = (_QWORD *)sub_22077B0(64);
            v68 = v106;
            v67[5] = v38;
            v69 = v67;
            v67[3] = 2;
            v67[4] = 0;
            if ( v38 != -4096 && v38 != 0 && v38 != -8192 )
            {
              v107 = v67;
              sub_BD73F0((__int64)(v67 + 3));
              v69 = v107;
            }
            v69[6] = a1;
            v69[7] = 0;
            v69[2] = &unk_49DDE50;
            sub_2208C80(v69, v68);
            v70 = *(_QWORD *)(a1 + 336);
            ++*(_QWORD *)(a1 + 352);
            *(_QWORD *)(v70 + 56) = v70;
            v41 = *(_DWORD *)(a1 + 296);
            if ( v41 )
              goto LABEL_79;
LABEL_123:
            ++*(_QWORD *)(a1 + 272);
            goto LABEL_124;
          }
          while ( 1 )
          {
LABEL_118:
            sub_C8CC70((__int64)&v126, v38, (__int64)v36, v31, v32, a6);
            if ( v66 )
              goto LABEL_119;
LABEL_78:
            v41 = *(_DWORD *)(a1 + 296);
            if ( !v41 )
              goto LABEL_123;
LABEL_79:
            v42 = 1;
            v43 = *(_QWORD *)(a1 + 280);
            v44 = 0;
            v45 = (v41 - 1) & (((unsigned int)v38 >> 4) ^ ((unsigned int)v38 >> 9));
            v46 = (_QWORD *)(v43 + 16LL * v45);
            v47 = *v46;
            if ( *v46 != v38 )
              break;
LABEL_80:
            v48 = v46 + 1;
LABEL_81:
            a2 = v19 - 56;
            sub_D1E0D0(v48, v113, 1);
            v49 = v39 + 1;
            if ( v39 + 1 == v37 )
              goto LABEL_84;
            while ( 1 )
            {
              v38 = *v49;
              v39 = v49;
              if ( (unsigned __int64)*v49 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v37 == ++v49 )
                goto LABEL_84;
            }
            if ( v37 == v49 )
            {
LABEL_84:
              if ( (*(_BYTE *)(v19 + 24) & 1) == 0 )
                goto LABEL_85;
              goto LABEL_66;
            }
            if ( v130 )
              goto LABEL_74;
          }
          while ( v47 != -4096 )
          {
            if ( !v44 && v47 == -8192 )
              v44 = v46;
            v45 = (v41 - 1) & (v42 + v45);
            v46 = (_QWORD *)(v43 + 16LL * v45);
            v47 = *v46;
            if ( *v46 == v38 )
              goto LABEL_80;
            ++v42;
          }
          v64 = *(_DWORD *)(a1 + 288);
          if ( !v44 )
            v44 = v46;
          ++*(_QWORD *)(a1 + 272);
          v65 = v64 + 1;
          if ( 4 * v65 >= 3 * v41 )
          {
LABEL_124:
            sub_D1E430(v104, 2 * v41);
            v71 = *(_DWORD *)(a1 + 296);
            if ( v71 )
            {
              v72 = v71 - 1;
              v73 = *(_QWORD *)(a1 + 280);
              v74 = v72 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
              v65 = *(_DWORD *)(a1 + 288) + 1;
              v44 = (__int64 *)(v73 + 16LL * v74);
              v75 = *v44;
              if ( *v44 == v38 )
                goto LABEL_113;
              v76 = 1;
              v77 = 0;
              while ( v75 != -4096 )
              {
                if ( !v77 && v75 == -8192 )
                  v77 = v44;
                v74 = v72 & (v76 + v74);
                v44 = (__int64 *)(v73 + 16LL * v74);
                v75 = *v44;
                if ( *v44 == v38 )
                  goto LABEL_113;
                ++v76;
              }
LABEL_142:
              if ( v77 )
                v44 = v77;
              goto LABEL_113;
            }
          }
          else
          {
            if ( v41 - *(_DWORD *)(a1 + 292) - v65 > v41 >> 3 )
            {
LABEL_113:
              *(_DWORD *)(a1 + 288) = v65;
              if ( *v44 != -4096 )
                --*(_DWORD *)(a1 + 292);
              *v44 = v38;
              v48 = (unsigned __int64 *)(v44 + 1);
              v44[1] = 0;
              goto LABEL_81;
            }
            v108 = ((unsigned int)v38 >> 4) ^ ((unsigned int)v38 >> 9);
            sub_D1E430(v104, v41);
            v78 = *(_DWORD *)(a1 + 296);
            if ( v78 )
            {
              v79 = v78 - 1;
              v77 = 0;
              v80 = *(_QWORD *)(a1 + 280);
              v81 = 1;
              v82 = v79 & v108;
              v65 = *(_DWORD *)(a1 + 288) + 1;
              v44 = (__int64 *)(v80 + 16LL * (v79 & v108));
              v83 = *v44;
              if ( *v44 == v38 )
                goto LABEL_113;
              while ( v83 != -4096 )
              {
                if ( v83 == -8192 && !v77 )
                  v77 = v44;
                v82 = v79 & (v81 + v82);
                v44 = (__int64 *)(v80 + 16LL * v82);
                v83 = *v44;
                if ( *v44 == v38 )
                  goto LABEL_113;
                ++v81;
              }
              goto LABEL_142;
            }
          }
          ++*(_DWORD *)(a1 + 288);
          BUG();
        }
      }
LABEL_65:
      if ( (*(_BYTE *)(v19 + 24) & 1) != 0 )
        goto LABEL_66;
LABEL_85:
      v50 = (__int64 *)v122;
      if ( v124 )
      {
        v36 = (__int64 *)*(unsigned int *)&v123[4];
        v51 = (__int64 *)((char *)v122 + 8 * *(unsigned int *)&v123[4]);
      }
      else
      {
        v36 = (__int64 *)*(unsigned int *)v123;
        v51 = (__int64 *)((char *)v122 + 8 * *(unsigned int *)v123);
      }
      if ( v122 != v51 )
      {
        while ( 1 )
        {
          v52 = *v50;
          v53 = v50;
          if ( (unsigned __int64)*v50 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v51 == ++v50 )
            goto LABEL_66;
        }
        if ( v51 != v50 )
        {
          v54 = v51;
          v105 = a1 + 272;
          if ( v130 )
          {
LABEL_93:
            v55 = v127;
            v31 = HIDWORD(v128);
            v36 = &v127[HIDWORD(v128)];
            if ( v127 != v36 )
            {
              do
              {
                if ( *v55 == v52 )
                  goto LABEL_97;
                ++v55;
              }
              while ( v36 != v55 );
            }
            if ( HIDWORD(v128) >= (unsigned int)v128 )
              goto LABEL_148;
            ++HIDWORD(v128);
            *v36 = v52;
            ++v126;
LABEL_149:
            v109 = *(_QWORD *)(a1 + 336);
            v85 = (_QWORD *)sub_22077B0(64);
            v86 = v109;
            v85[5] = v52;
            v87 = v85;
            v85[3] = 2;
            v85[4] = 0;
            if ( v52 != 0 && v52 != -4096 && v52 != -8192 )
            {
              v110 = v85;
              sub_BD73F0((__int64)(v85 + 3));
              v87 = v110;
            }
            v87[6] = a1;
            v87[7] = 0;
            v87[2] = &unk_49DDE50;
            sub_2208C80(v87, v86);
            v88 = *(_QWORD *)(a1 + 336);
            ++*(_QWORD *)(a1 + 352);
            *(_QWORD *)(v88 + 56) = v88;
            v56 = *(_DWORD *)(a1 + 296);
            if ( v56 )
              goto LABEL_98;
LABEL_153:
            ++*(_QWORD *)(a1 + 272);
            goto LABEL_154;
          }
          while ( 1 )
          {
LABEL_148:
            sub_C8CC70((__int64)&v126, v52, (__int64)v36, v31, v32, a6);
            if ( v84 )
              goto LABEL_149;
LABEL_97:
            v56 = *(_DWORD *)(a1 + 296);
            if ( !v56 )
              goto LABEL_153;
LABEL_98:
            v57 = *(_QWORD *)(a1 + 280);
            v58 = 1;
            v59 = 0;
            v60 = (v56 - 1) & (((unsigned int)v52 >> 4) ^ ((unsigned int)v52 >> 9));
            v61 = (__int64 *)(v57 + 16LL * v60);
            v62 = *v61;
            if ( *v61 != v52 )
              break;
LABEL_99:
            a2 = v19 - 56;
            sub_D1E0D0((unsigned __int64 *)v61 + 1, v113, 2);
            v63 = v53 + 1;
            if ( v53 + 1 == v54 )
              goto LABEL_66;
            while ( 1 )
            {
              v52 = *v63;
              v53 = v63;
              if ( (unsigned __int64)*v63 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v54 == ++v63 )
                goto LABEL_66;
            }
            if ( v54 == v63 )
              goto LABEL_66;
            if ( v130 )
              goto LABEL_93;
          }
          while ( v62 != -4096 )
          {
            if ( !v59 && v62 == -8192 )
              v59 = v61;
            v60 = (v56 - 1) & (v58 + v60);
            v61 = (__int64 *)(v57 + 16LL * v60);
            v62 = *v61;
            if ( *v61 == v52 )
              goto LABEL_99;
            ++v58;
          }
          v97 = *(_DWORD *)(a1 + 288);
          if ( v59 )
            v61 = v59;
          ++*(_QWORD *)(a1 + 272);
          v93 = v97 + 1;
          if ( 4 * (v97 + 1) >= 3 * v56 )
          {
LABEL_154:
            sub_D1E430(v105, 2 * v56);
            v89 = *(_DWORD *)(a1 + 296);
            if ( v89 )
            {
              v90 = v89 - 1;
              v91 = *(_QWORD *)(a1 + 280);
              v92 = v90 & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
              v93 = *(_DWORD *)(a1 + 288) + 1;
              v61 = (__int64 *)(v91 + 16LL * v92);
              v94 = *v61;
              if ( *v61 == v52 )
              {
LABEL_174:
                *(_DWORD *)(a1 + 288) = v93;
                if ( *v61 != -4096 )
                  --*(_DWORD *)(a1 + 292);
                *v61 = v52;
                v61[1] = 0;
                goto LABEL_99;
              }
              v95 = 1;
              v96 = 0;
              while ( v94 != -4096 )
              {
                if ( !v96 && v94 == -8192 )
                  v96 = v61;
                v92 = v90 & (v95 + v92);
                v61 = (__int64 *)(v91 + 16LL * v92);
                v94 = *v61;
                if ( *v61 == v52 )
                  goto LABEL_174;
                ++v95;
              }
LABEL_158:
              if ( v96 )
                v61 = v96;
              goto LABEL_174;
            }
          }
          else
          {
            if ( v56 - *(_DWORD *)(a1 + 292) - v93 > v56 >> 3 )
              goto LABEL_174;
            v111 = ((unsigned int)v52 >> 4) ^ ((unsigned int)v52 >> 9);
            sub_D1E430(v105, v56);
            v98 = *(_DWORD *)(a1 + 296);
            if ( v98 )
            {
              v99 = v98 - 1;
              v100 = *(_QWORD *)(a1 + 280);
              v101 = 1;
              v102 = v99 & v111;
              v93 = *(_DWORD *)(a1 + 288) + 1;
              v96 = 0;
              v61 = (__int64 *)(v100 + 16LL * (v99 & v111));
              v103 = *v61;
              if ( *v61 == v52 )
                goto LABEL_174;
              while ( v103 != -4096 )
              {
                if ( !v96 && v103 == -8192 )
                  v96 = v61;
                v102 = v99 & (v101 + v102);
                v61 = (__int64 *)(v100 + 16LL * v102);
                v103 = *v61;
                if ( *v61 == v52 )
                  goto LABEL_174;
                ++v101;
              }
              goto LABEL_158;
            }
          }
          ++*(_DWORD *)(a1 + 288);
          BUG();
        }
      }
LABEL_66:
      if ( *(_BYTE *)(*(_QWORD *)(v19 - 32) + 8LL) == 14 )
      {
        a2 = v19 - 56;
        sub_D1E880(a1, v113, v36, v31, v32, a6);
      }
LABEL_29:
      ++v116;
      if ( v119 )
      {
LABEL_34:
        *(_QWORD *)&v118[4] = 0;
      }
      else
      {
        v25 = 4 * (*(_DWORD *)&v118[4] - *(_DWORD *)&v118[8]);
        if ( v25 < 0x20 )
          v25 = 32;
        if ( *(_DWORD *)v118 <= v25 )
        {
          a2 = 0xFFFFFFFFLL;
          memset(s, -1, 8LL * *(unsigned int *)v118);
          goto LABEL_34;
        }
        sub_C8C990((__int64)&v116, a2);
      }
      ++v121;
      if ( v124 )
      {
LABEL_40:
        *(_QWORD *)&v123[4] = 0;
      }
      else
      {
        v26 = 4 * (*(_DWORD *)&v123[4] - *(_DWORD *)&v123[8]);
        if ( v26 < 0x20 )
          v26 = 32;
        if ( v26 >= *(_DWORD *)v123 )
        {
          a2 = 0xFFFFFFFFLL;
          memset(v122, -1, 8LL * *(unsigned int *)v123);
          goto LABEL_40;
        }
        sub_C8C990((__int64)&v121, a2);
      }
    }
    v19 = *(_QWORD *)(v19 + 8);
    if ( v115 != v19 )
      continue;
    break;
  }
  if ( !v124 )
    _libc_free(v122, a2);
  result = v119;
  if ( !v119 )
    result = _libc_free(s, a2);
LABEL_46:
  if ( !v130 )
    return _libc_free(v127, a2);
  return result;
}
