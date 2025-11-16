// Function: sub_37D1F90
// Address: 0x37d1f90
//
void __fastcall sub_37D1F90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // r12
  int v9; // eax
  __int64 v10; // r15
  char v11; // al
  unsigned int v12; // r14d
  char *v13; // rax
  int v14; // esi
  unsigned int *v15; // rbx
  unsigned int *v16; // rax
  unsigned __int64 v17; // r8
  char v18; // r14
  char v19; // r13
  __int64 v20; // r14
  unsigned int *i; // rax
  __int64 v22; // rdi
  int v23; // r15d
  int v24; // r8d
  __int64 v25; // r10
  unsigned int v26; // esi
  __int64 v27; // rdi
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // r13
  int v30; // eax
  int v31; // eax
  char v32; // al
  __int64 v33; // rdx
  _DWORD *v34; // rax
  _DWORD *v35; // rdx
  __int64 v36; // rax
  __int64 v37; // r14
  unsigned __int64 v38; // rdx
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  unsigned int v41; // eax
  __int64 *v42; // r14
  __int64 *v43; // r12
  __int64 v44; // rsi
  unsigned __int64 v45; // r15
  char v46; // r12
  __int64 *v47; // r14
  unsigned int *v48; // rax
  __int64 v49; // rdi
  __int64 v50; // r8
  unsigned int v51; // esi
  __int64 v52; // rdi
  __int64 v53; // rax
  __int64 v54; // rcx
  unsigned int v55; // r9d
  int *v56; // rdx
  int v57; // r8d
  __int64 v58; // rax
  unsigned int v59; // r14d
  unsigned __int64 v60; // r8
  unsigned __int64 v61; // r10
  __int64 v62; // r13
  __int64 v63; // rbx
  unsigned __int64 v64; // r15
  unsigned int v65; // r14d
  int v66; // eax
  char v67; // al
  __int64 v68; // rdx
  __int64 v69; // rax
  __int64 v70; // rdi
  __int64 v71; // rsi
  __int64 v72; // rcx
  __int64 v73; // rdx
  unsigned __int64 v74; // r12
  unsigned int v75; // eax
  unsigned int v76; // r14d
  __int64 v77; // r15
  __int64 *v78; // r13
  __int64 v79; // rbx
  int v80; // edx
  int v81; // edx
  int v82; // r11d
  __int64 v83; // r12
  unsigned int *v84; // r14
  unsigned int v85; // esi
  unsigned int v86; // eax
  __int64 v87; // rsi
  unsigned int v88; // eax
  int v89; // edi
  unsigned int v90; // ecx
  __int64 v91; // rax
  __int64 v92; // rdx
  __int64 v93; // rsi
  int v94; // r9d
  unsigned __int64 v95; // rdx
  __int64 v96; // rdx
  unsigned int v97; // eax
  unsigned int v98; // r14d
  int v99; // r12d
  __int64 v100; // rax
  _DWORD *v101; // rax
  _DWORD *v102; // rdx
  __int64 v103; // rax
  __int64 v104; // rsi
  unsigned int *v105; // r13
  bool v106; // r8
  __int64 v107; // rax
  __int64 v108; // rax
  __int64 v109; // rdx
  _QWORD *v110; // r15
  __int64 v111; // [rsp+8h] [rbp-1E8h]
  __int64 v112; // [rsp+10h] [rbp-1E0h]
  __int64 v113; // [rsp+20h] [rbp-1D0h]
  unsigned __int64 v114; // [rsp+28h] [rbp-1C8h]
  unsigned __int64 v115; // [rsp+38h] [rbp-1B8h]
  unsigned __int64 v116; // [rsp+38h] [rbp-1B8h]
  char v117; // [rsp+5Fh] [rbp-191h]
  unsigned int *v118; // [rsp+60h] [rbp-190h]
  __int64 v119; // [rsp+60h] [rbp-190h]
  char v120; // [rsp+60h] [rbp-190h]
  int v121; // [rsp+68h] [rbp-188h]
  unsigned int v122; // [rsp+68h] [rbp-188h]
  __int64 v123; // [rsp+70h] [rbp-180h]
  __int64 *v124; // [rsp+78h] [rbp-178h]
  unsigned __int64 v125; // [rsp+78h] [rbp-178h]
  unsigned int *v126; // [rsp+78h] [rbp-178h]
  unsigned __int64 v127; // [rsp+78h] [rbp-178h]
  int v128; // [rsp+8Ch] [rbp-164h] BYREF
  char *v129; // [rsp+90h] [rbp-160h] BYREF
  char *v130; // [rsp+98h] [rbp-158h]
  _BYTE *v131; // [rsp+A0h] [rbp-150h] BYREF
  __int64 v132; // [rsp+A8h] [rbp-148h]
  _BYTE v133[32]; // [rsp+B0h] [rbp-140h] BYREF
  __int64 *v134; // [rsp+D0h] [rbp-120h] BYREF
  __int64 v135; // [rsp+D8h] [rbp-118h]
  _BYTE v136[32]; // [rsp+E0h] [rbp-110h] BYREF
  unsigned int *v137; // [rsp+100h] [rbp-F0h] BYREF
  __int64 v138; // [rsp+108h] [rbp-E8h]
  _BYTE v139[128]; // [rsp+110h] [rbp-E0h] BYREF
  __int64 v140; // [rsp+190h] [rbp-60h] BYREF
  __int64 v141; // [rsp+198h] [rbp-58h] BYREF
  unsigned __int64 v142; // [rsp+1A0h] [rbp-50h]
  __int64 *v143; // [rsp+1A8h] [rbp-48h]
  __int64 *v144; // [rsp+1B0h] [rbp-40h]
  __int64 v145; // [rsp+1B8h] [rbp-38h]

  v6 = a2;
  v7 = a1;
  if ( *(_WORD *)(a2 + 68) == 10 )
  {
    v83 = *(_QWORD *)(a1 + 408);
    v84 = (unsigned int *)(*(_QWORD *)(v83 + 64) + 4LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL));
    v85 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL);
    v86 = *v84;
    if ( *v84 == -1 )
    {
      v86 = sub_37BA230(*(_QWORD *)(a1 + 408), v85);
      *v84 = v86;
    }
    if ( *(_DWORD *)(*(_QWORD *)(v83 + 32) + 8LL * v86 + 4) >> 8 )
      return;
  }
  else if ( (*(_BYTE *)(*(_QWORD *)(a2 + 16) + 24LL) & 0x10) != 0 )
  {
    return;
  }
  v117 = *(_BYTE *)(a1 + 2360);
  if ( v117 )
  {
    v30 = *(_DWORD *)(v6 + 44);
    if ( (v30 & 4) != 0 || (v30 & 8) == 0 )
      v117 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v6 + 16) + 24LL) >> 7;
    else
      v117 = sub_2E88A90(v6, 128, 1);
    v8 = *(_QWORD *)(v6 + 32);
    if ( v117 )
    {
      v117 = 0;
      if ( *(_BYTE *)v8 == 9 )
        v117 = strcmp(*(const char **)(v8 + 24), *(const char **)(a1 + 2368)) == 0;
    }
  }
  else
  {
    v8 = *(_QWORD *)(v6 + 32);
  }
  LODWORD(v141) = 0;
  v137 = (unsigned int *)v139;
  v138 = 0x2000000000LL;
  v143 = &v141;
  v144 = &v141;
  v131 = v133;
  v132 = 0x400000000LL;
  v135 = 0x400000000LL;
  v9 = *(_DWORD *)(v6 + 40);
  v142 = 0;
  v145 = 0;
  v10 = v8 + 40LL * (v9 & 0xFFFFFF);
  v134 = (__int64 *)v136;
  if ( v10 == v8 )
  {
    v17 = (unsigned __int64)v139;
    v124 = (__int64 *)v139;
    goto LABEL_118;
  }
  do
  {
    while ( 1 )
    {
      v11 = *(_BYTE *)v8;
      if ( !*(_BYTE *)v8 )
      {
        if ( (*(_BYTE *)(v8 + 3) & 0x10) == 0 )
          goto LABEL_20;
        v12 = *(_DWORD *)(v8 + 8);
        if ( v12 - 1 > 0x3FFFFFFE )
          goto LABEL_20;
        if ( v117 )
          goto LABEL_10;
        v31 = *(_DWORD *)(v6 + 44);
        if ( (v31 & 4) == 0 && (v31 & 8) != 0 )
          v32 = sub_2E88A90(v6, 128, 1);
        else
          v32 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v6 + 16) + 24LL) >> 7;
        if ( !v32 )
        {
LABEL_140:
          v12 = *(_DWORD *)(v8 + 8);
LABEL_10:
          sub_37B9A30(&v129, v12, *(_QWORD **)(v7 + 16), 1);
          v13 = v129;
          if ( v130 == v129 )
            goto LABEL_20;
          v113 = v10;
          v112 = v7;
          v111 = v6;
          while ( 2 )
          {
            v14 = *(unsigned __int16 *)v13;
            v128 = v14;
            if ( !v145 )
            {
              a6 = (unsigned __int64)v137;
              v15 = &v137[(unsigned int)v138];
              if ( v137 == v15 )
              {
                if ( (unsigned int)v138 <= 0x1FuLL )
                  goto LABEL_161;
              }
              else
              {
                v16 = v137;
                while ( v14 != *v16 )
                {
                  if ( v15 == ++v16 )
                    goto LABEL_152;
                }
                if ( v15 != v16 )
                  goto LABEL_18;
LABEL_152:
                v105 = v137;
                if ( (unsigned int)v138 <= 0x1FuLL )
                {
LABEL_161:
                  sub_9C8C60((__int64)&v137, v14);
LABEL_18:
                  v13 = v129 + 2;
                  v129 = v13;
                  if ( v130 == v13 )
                  {
                    v10 = v113;
                    v7 = v112;
                    v6 = v111;
                    goto LABEL_20;
                  }
                  continue;
                }
                do
                {
                  v108 = sub_B9AB10(&v140, (__int64)&v141, v105);
                  v110 = (_QWORD *)v109;
                  if ( v109 )
                  {
                    v106 = v108 || (__int64 *)v109 == &v141 || *v105 < *(_DWORD *)(v109 + 32);
                    v120 = v106;
                    v107 = sub_22077B0(0x28u);
                    *(_DWORD *)(v107 + 32) = *v105;
                    sub_220F040(v120, v107, v110, &v141);
                    ++v145;
                  }
                  ++v105;
                }
                while ( v15 != v105 );
              }
              LODWORD(v138) = 0;
            }
            break;
          }
          sub_B99770((__int64)&v140, (unsigned int *)&v128);
          goto LABEL_18;
        }
        v33 = *(_QWORD *)(v7 + 408);
        if ( *(_QWORD *)(v33 + 200) )
        {
          v103 = *(_QWORD *)(v33 + 176);
          if ( !v103 )
            goto LABEL_140;
          v104 = v33 + 168;
          do
          {
            if ( v12 > *(_DWORD *)(v103 + 32) )
            {
              v103 = *(_QWORD *)(v103 + 24);
            }
            else
            {
              v104 = v103;
              v103 = *(_QWORD *)(v103 + 16);
            }
          }
          while ( v103 );
          if ( v33 + 168 == v104 || v12 < *(_DWORD *)(v104 + 32) )
            goto LABEL_140;
        }
        else
        {
          v34 = *(_DWORD **)(v33 + 112);
          v35 = &v34[*(unsigned int *)(v33 + 120)];
          if ( v34 == v35 )
            goto LABEL_140;
          while ( v12 != *v34 )
          {
            if ( v35 == ++v34 )
              goto LABEL_140;
          }
          if ( v35 == v34 )
            goto LABEL_140;
        }
        v11 = *(_BYTE *)v8;
      }
      if ( v11 == 12 )
        break;
LABEL_20:
      v8 += 40;
      if ( v10 == v8 )
        goto LABEL_21;
    }
    v36 = (unsigned int)v132;
    v37 = *(_QWORD *)(v8 + 24);
    v38 = (unsigned int)v132 + 1LL;
    if ( v38 > HIDWORD(v132) )
    {
      sub_C8D5F0((__int64)&v131, v133, v38, 8u, a5, a6);
      v36 = (unsigned int)v132;
    }
    *(_QWORD *)&v131[8 * v36] = v37;
    v39 = (unsigned int)v135;
    LODWORD(v132) = v132 + 1;
    v40 = (unsigned int)v135 + 1LL;
    if ( v40 > HIDWORD(v135) )
    {
      sub_C8D5F0((__int64)&v134, v136, v40, 8u, a5, a6);
      v39 = (unsigned int)v135;
    }
    v134[v39] = v8;
    v8 += 40;
    LODWORD(v135) = v135 + 1;
  }
  while ( v10 != v8 );
LABEL_21:
  if ( v145 )
  {
    v17 = (unsigned __int64)v143;
    v18 = 0;
    v124 = &v141;
    goto LABEL_23;
  }
  v17 = (unsigned __int64)v137;
  v124 = (__int64 *)&v137[(unsigned int)v138];
LABEL_118:
  v18 = 1;
LABEL_23:
  v115 = v6;
  v19 = v18;
  v20 = v17;
  if ( v19 )
    goto LABEL_30;
  while ( (__int64 *)v20 != v124 )
  {
    for ( i = (unsigned int *)(v20 + 32); ; i = (unsigned int *)v20 )
    {
      v22 = *(_QWORD *)(v7 + 408);
      v23 = *(_DWORD *)(v7 + 420);
      v24 = *(_DWORD *)(v7 + 416);
      v25 = *i;
      v26 = *(_DWORD *)(*(_QWORD *)(v22 + 64) + 4 * v25);
      if ( v26 == -1 )
      {
        v118 = (unsigned int *)(*(_QWORD *)(v22 + 64) + 4 * v25);
        v121 = *(_DWORD *)(v7 + 416);
        v41 = sub_37BA230(v22, v25);
        v24 = v121;
        v26 = v41;
        *v118 = v41;
      }
      a6 = v26;
      v27 = *(_QWORD *)(v22 + 32) + 8LL * v26;
      v28 = v24 & 0xFFFFF | ((unsigned __int64)(v23 & 0xFFFFF) << 20);
      v17 = *(_QWORD *)v27 & 0xFFFFFF0000000000LL;
      *(_QWORD *)v27 = v17 | v28;
      *(_DWORD *)(v27 + 4) = (unsigned __int8)((v17 | v28) >> 32) | (v26 << 8);
      if ( !v19 )
        break;
      v20 += 4;
LABEL_30:
      if ( (__int64 *)v20 == v124 )
        goto LABEL_65;
    }
    v20 = sub_220EF30(v20);
  }
LABEL_65:
  v42 = v134;
  v29 = v115;
  v43 = &v134[(unsigned int)v135];
  if ( v43 != v134 )
  {
    do
    {
      v44 = *v42++;
      sub_37BA470(*(_QWORD *)(v7 + 408), v44, *(_DWORD *)(v7 + 416), *(_DWORD *)(v7 + 420), v17, a6);
    }
    while ( v43 != v42 );
  }
  if ( (unsigned __int8)sub_37B9BA0(v7, v115) )
  {
    v129 = (char *)sub_37C70E0(v7, v115);
    if ( BYTE4(v129) )
    {
      v87 = *(_QWORD *)(v7 + 408);
      v88 = *(_DWORD *)(v87 + 288);
      if ( v88 )
      {
        v89 = (_DWORD)v129 - 1;
        v90 = 0;
        do
        {
          v91 = *(_DWORD *)(v87 + 284) + v90 + v89 * v88;
          ++v90;
          v92 = *(unsigned int *)(*(_QWORD *)(v87 + 64) + 4 * v91);
          v93 = *(_QWORD *)(v87 + 32) + 8 * v92;
          v94 = (_DWORD)v92 << 8;
          v95 = *(_QWORD *)v93 & 0xFFFFFF0000000000LL
              | *(_DWORD *)(v7 + 416) & 0xFFFFF
              | ((unsigned __int64)(*(_DWORD *)(v7 + 420) & 0xFFFFF) << 20);
          *(_QWORD *)v93 = v95;
          *(_DWORD *)(v93 + 4) = v94 | BYTE4(v95);
          v87 = *(_QWORD *)(v7 + 408);
          v88 = *(_DWORD *)(v87 + 288);
        }
        while ( v88 > v90 );
      }
    }
  }
  if ( !*(_QWORD *)(v7 + 432) )
    goto LABEL_34;
  if ( v145 )
  {
    v45 = (unsigned __int64)v143;
    v47 = &v141;
    v46 = 0;
  }
  else
  {
    v45 = (unsigned __int64)v137;
    v46 = 1;
    v47 = (__int64 *)&v137[(unsigned int)v138];
  }
  while ( v47 != (__int64 *)v45 )
  {
    while ( 1 )
    {
      v48 = (unsigned int *)(v45 + 32);
      v49 = *(_QWORD *)(v7 + 408);
      if ( v46 )
        v48 = (unsigned int *)v45;
      v50 = *v48;
      v51 = *(_DWORD *)(*(_QWORD *)(v49 + 64) + 4 * v50);
      if ( v51 == -1 )
      {
        v126 = (unsigned int *)(*(_QWORD *)(v49 + 64) + 4 * v50);
        v51 = sub_37BA230(v49, v50);
        *v126 = v51;
      }
      v52 = *(_QWORD *)(v7 + 432);
      v53 = *(unsigned int *)(v52 + 3432);
      v54 = *(_QWORD *)(v52 + 3416);
      if ( (_DWORD)v53 )
      {
        v55 = v51 & (v53 - 1);
        v56 = (int *)(v54 + 88LL * v55);
        v57 = *v56;
        if ( *v56 == v51 )
        {
LABEL_78:
          if ( v56 != (int *)(v54 + 88 * v53) )
            sub_37CFFC0(v52, v51, *(_QWORD *)(*(_QWORD *)(v52 + 3136) + 8LL * v51), v115, 0);
        }
        else
        {
          v81 = 1;
          while ( v57 != -1 )
          {
            v82 = v81 + 1;
            v55 = (v53 - 1) & (v81 + v55);
            v56 = (int *)(v54 + 88LL * v55);
            v57 = *v56;
            if ( *v56 == v51 )
              goto LABEL_78;
            v81 = v82;
          }
        }
      }
      if ( !v46 )
        break;
      v45 += 4LL;
      if ( v47 == (__int64 *)v45 )
        goto LABEL_82;
    }
    v45 = sub_220EF30(v45);
  }
LABEL_82:
  if ( !(_DWORD)v135 )
    goto LABEL_33;
  v58 = *(_QWORD *)(v7 + 408);
  v59 = *(_DWORD *)(v58 + 40);
  if ( !v59 )
    goto LABEL_33;
  v60 = v114;
  v61 = v115;
  v123 = v59;
  v62 = v7;
  v63 = 0;
  while ( 2 )
  {
    v64 = v60 & 0xFFFFFFFF00000000LL | (unsigned int)v63;
    v65 = *(_DWORD *)(*(_QWORD *)(v58 + 88) + 4 * v63);
    v60 = v64;
    if ( v65 < *(_DWORD *)(v58 + 284) )
    {
      if ( v117 )
        goto LABEL_99;
      v66 = *(_DWORD *)(v61 + 44);
      if ( (v66 & 4) == 0 && (v66 & 8) != 0 )
      {
        v127 = v61;
        v67 = sub_2E88A90(v61, 128, 1);
        v61 = v127;
        v60 = v64;
      }
      else
      {
        v67 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v61 + 16) + 24LL) >> 7;
      }
      if ( !v67 )
        goto LABEL_99;
      v68 = *(_QWORD *)(v62 + 408);
      if ( *(_QWORD *)(v68 + 200) )
      {
        v69 = *(_QWORD *)(v68 + 176);
        v70 = v68 + 168;
        if ( !v69 )
          goto LABEL_99;
        v71 = v68 + 168;
        do
        {
          while ( 1 )
          {
            v72 = *(_QWORD *)(v69 + 16);
            v73 = *(_QWORD *)(v69 + 24);
            if ( v65 <= *(_DWORD *)(v69 + 32) )
              break;
            v69 = *(_QWORD *)(v69 + 24);
            if ( !v73 )
              goto LABEL_97;
          }
          v71 = v69;
          v69 = *(_QWORD *)(v69 + 16);
        }
        while ( v72 );
LABEL_97:
        if ( v70 == v71 || v65 < *(_DWORD *)(v71 + 32) )
          goto LABEL_99;
      }
      else
      {
        v101 = *(_DWORD **)(v68 + 112);
        v102 = &v101[*(unsigned int *)(v68 + 120)];
        if ( v101 == v102 )
          goto LABEL_99;
        while ( v65 != *v101 )
        {
          if ( v102 == ++v101 )
            goto LABEL_99;
        }
        if ( v101 == v102 )
        {
LABEL_99:
          v74 = (unsigned __int64)v134;
          if ( &v134[(unsigned int)v135] != v134 )
          {
            v75 = v65;
            v119 = v63;
            v76 = v65 & 0x1F;
            v122 = v64;
            v77 = v62;
            v78 = &v134[(unsigned int)v135];
            v116 = v60;
            v79 = 4LL * (v75 >> 5);
            do
            {
              while ( 1 )
              {
                v80 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v74 + 24LL) + v79);
                if ( !_bittest(&v80, v76) )
                  break;
                v74 += 8LL;
                if ( v78 == (__int64 *)v74 )
                  goto LABEL_104;
              }
              v74 += 8LL;
              v125 = v61;
              sub_37D0D70(*(_QWORD *)(v77 + 432), v122, v61, 0);
              v61 = v125;
            }
            while ( v78 != (__int64 *)v74 );
LABEL_104:
            v63 = v119;
            v60 = v116;
            v62 = v77;
          }
        }
      }
    }
    if ( ++v63 != v123 )
    {
      v58 = *(_QWORD *)(v62 + 408);
      continue;
    }
    break;
  }
  v7 = v62;
  v29 = v61;
LABEL_33:
  if ( (unsigned __int8)sub_37B9BA0(v7, v29) )
  {
    v129 = (char *)sub_37C70E0(v7, v29);
    if ( BYTE4(v129) )
    {
      v96 = *(_QWORD *)(v7 + 408);
      v97 = *(_DWORD *)(v96 + 288);
      if ( v97 )
      {
        v98 = 0;
        v99 = (_DWORD)v129 - 1;
        do
        {
          v100 = *(_DWORD *)(v96 + 284) + v98 + v99 * v97;
          ++v98;
          sub_37D0D70(*(_QWORD *)(v7 + 432), *(_DWORD *)(*(_QWORD *)(v96 + 64) + 4 * v100), v29, 1u);
          v96 = *(_QWORD *)(v7 + 408);
          v97 = *(_DWORD *)(v96 + 288);
        }
        while ( v97 > v98 );
      }
    }
  }
LABEL_34:
  if ( v134 != (__int64 *)v136 )
    _libc_free((unsigned __int64)v134);
  if ( v131 != v133 )
    _libc_free((unsigned __int64)v131);
  sub_37B80B0(v142);
  if ( v137 != (unsigned int *)v139 )
    _libc_free((unsigned __int64)v137);
}
