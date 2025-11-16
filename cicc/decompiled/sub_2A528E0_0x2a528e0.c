// Function: sub_2A528E0
// Address: 0x2a528e0
//
__int64 __fastcall sub_2A528E0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        unsigned __int8 a9)
{
  unsigned __int64 v9; // r13
  __int64 v11; // r14
  unsigned int v12; // r12d
  _BYTE *v13; // r15
  unsigned __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  unsigned __int64 *v18; // rax
  __int64 v19; // r14
  unsigned int *v20; // r13
  unsigned __int64 v21; // rax
  __int64 v22; // r11
  unsigned int v23; // ecx
  unsigned int v24; // edx
  unsigned int *v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rdx
  unsigned int *v28; // rsi
  __int64 v29; // r14
  __int64 v30; // rax
  __int64 v31; // r15
  unsigned __int8 v32; // al
  int v33; // eax
  _DWORD *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  int v39; // eax
  _DWORD *v40; // rax
  _BYTE *v41; // rdi
  __int64 v42; // r13
  int v43; // ecx
  __int64 v44; // rsi
  int v45; // ecx
  unsigned int v46; // edx
  __int64 *v47; // rax
  __int64 v48; // r8
  __int64 *v49; // rax
  __int64 v50; // rax
  __int64 v51; // rsi
  _QWORD *v52; // r15
  __int64 *v53; // rax
  __int64 *v54; // r12
  __int64 *i; // r14
  __int64 v56; // rdi
  __int64 v57; // rax
  __int64 *v58; // rax
  __int64 *v59; // r12
  __int64 *j; // r14
  __int64 v61; // rdi
  int v62; // ecx
  __int64 v63; // rdi
  int v64; // ecx
  unsigned int v65; // edx
  _QWORD *v66; // rax
  _QWORD *v67; // r8
  __int64 *v68; // r15
  __int64 *k; // rbx
  __int64 v70; // r14
  __int64 v71; // rax
  __int64 v72; // rbx
  __int64 m; // r12
  _QWORD *v74; // r14
  __int64 v75; // rax
  __int64 v77; // rax
  __int64 v78; // rdx
  __int64 v79; // r13
  int v80; // r13d
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // rcx
  __int64 v84; // rdx
  __int64 v85; // r13
  __int64 v86; // r12
  __int64 v87; // rcx
  bool v88; // cl
  int v89; // eax
  _DWORD *v90; // rax
  __int64 v91; // rdi
  _BYTE *v92; // rax
  __int64 v93; // rax
  __int64 v94; // rdx
  __int64 v95; // rdx
  int v96; // eax
  _DWORD *v97; // rax
  __int64 v98; // rdx
  __int64 v99; // rcx
  __int64 v100; // r8
  __int64 v101; // r9
  __int64 v102; // rdi
  _BYTE *v103; // rax
  __int64 v104; // rdx
  __int64 *v105; // r9
  __int64 v106; // rax
  __int64 v107; // rax
  int v108; // eax
  _DWORD *v109; // rax
  __int64 v110; // rdx
  __int64 v111; // r8
  __int64 v112; // r9
  __int64 v113; // rdi
  _BYTE *v114; // rax
  __int64 v115; // rcx
  __int64 v116; // rax
  __int64 v117; // rsi
  __int64 v118; // rsi
  __int64 v119; // rsi
  _BYTE *v120; // rax
  __int64 v121; // rdx
  __int64 v122; // rsi
  int v123; // ecx
  int v124; // ecx
  unsigned int v125; // edx
  _QWORD *v126; // rax
  _QWORD *v127; // rdi
  int v128; // eax
  int v129; // eax
  int v130; // edi
  __int64 v131; // rcx
  int v132; // eax
  int v133; // r8d
  __int64 v134; // rax
  __int64 v135; // r9
  __int64 v136; // rax
  bool v137; // [rsp+Eh] [rbp-632h]
  unsigned __int8 v138; // [rsp+Fh] [rbp-631h]
  char v141; // [rsp+28h] [rbp-618h]
  __int64 v143; // [rsp+38h] [rbp-608h]
  _QWORD *v144; // [rsp+38h] [rbp-608h]
  unsigned __int64 v145; // [rsp+38h] [rbp-608h]
  __int64 v148; // [rsp+50h] [rbp-5F0h] BYREF
  __int64 v149; // [rsp+58h] [rbp-5E8h]
  _BYTE *v150; // [rsp+200h] [rbp-440h] BYREF
  __int64 v151; // [rsp+208h] [rbp-438h]
  _BYTE v152[1072]; // [rsp+210h] [rbp-430h] BYREF

  v11 = a1[2];
  v150 = v152;
  v12 = a9;
  v151 = 0x4000000000LL;
  if ( v11 )
  {
    do
    {
      while ( 1 )
      {
        v13 = *(_BYTE **)(v11 + 24);
        if ( *v13 == 62 )
          break;
        v11 = *(_QWORD *)(v11 + 8);
        if ( !v11 )
          goto LABEL_8;
      }
      v14 = (unsigned int)sub_2A4E220(a3, *(_QWORD *)(v11 + 24)) | v9 & 0xFFFFFFFF00000000LL;
      v16 = (unsigned int)v151;
      v9 = v14;
      v17 = (unsigned int)v151 + 1LL;
      if ( v17 > HIDWORD(v151) )
      {
        v145 = v14;
        sub_C8D5F0((__int64)&v150, v152, v17, 0x10u, v14, v15);
        v16 = (unsigned int)v151;
        v14 = v145;
      }
      v18 = (unsigned __int64 *)&v150[16 * v16];
      *v18 = v14;
      v18[1] = (unsigned __int64)v13;
      LODWORD(v151) = v151 + 1;
      v11 = *(_QWORD *)(v11 + 8);
    }
    while ( v11 );
LABEL_8:
    v19 = 16LL * (unsigned int)v151;
    v20 = (unsigned int *)&v150[v19];
    if ( v150 != &v150[v19] )
    {
      v143 = (__int64)v150;
      _BitScanReverse64(&v21, v19 >> 4);
      sub_2A4FAA0((__int64)v150, (unsigned __int64)&v150[v19], 2LL * (int)(63 - (v21 ^ 0x3F)));
      if ( (unsigned __int64)v19 <= 0x100 )
      {
        sub_2A4BC20(v143, v20);
      }
      else
      {
        sub_2A4BC20(v143, (unsigned int *)(v143 + 256));
        for ( ; v20 != (unsigned int *)v22; *((_QWORD *)v28 + 1) = v26 )
        {
          while ( 1 )
          {
            v23 = *(_DWORD *)v22;
            v24 = *(_DWORD *)(v22 - 16);
            v25 = (unsigned int *)(v22 - 16);
            v26 = *(_QWORD *)(v22 + 8);
            if ( *(_DWORD *)v22 < v24 )
              break;
            v119 = v22;
            v22 += 16;
            *(_DWORD *)v119 = v23;
            *(_QWORD *)(v119 + 8) = v26;
            if ( v20 == (unsigned int *)v22 )
              goto LABEL_14;
          }
          do
          {
            v25[4] = v24;
            v27 = *((_QWORD *)v25 + 1);
            v28 = v25;
            v25 -= 4;
            *((_QWORD *)v25 + 5) = v27;
            v24 = *v25;
          }
          while ( v23 < *v25 );
          v22 += 16;
          *v28 = v23;
        }
      }
    }
LABEL_14:
    v29 = a1[2];
    while ( v29 )
    {
      while ( 1 )
      {
        v30 = v29;
        v29 = *(_QWORD *)(v29 + 8);
        v31 = *(_QWORD *)(v30 + 24);
        v32 = *(_BYTE *)v31;
        if ( !(_BYTE)v12 )
        {
          if ( v32 != 61 )
            goto LABEL_17;
          goto LABEL_26;
        }
        if ( v32 <= 0x1Cu )
          goto LABEL_17;
        if ( v32 == 85 )
        {
          if ( *(char *)(v31 + 7) < 0 )
          {
            v77 = sub_BD2BC0(v31);
            v79 = v77 + v78;
            if ( *(char *)(v31 + 7) >= 0 )
            {
              if ( (unsigned int)(v79 >> 4) )
                goto LABEL_164;
            }
            else if ( (unsigned int)((v79 - sub_BD2BC0(v31)) >> 4) )
            {
              if ( *(char *)(v31 + 7) < 0 )
              {
                v80 = *(_DWORD *)(sub_BD2BC0(v31) + 8);
                if ( *(char *)(v31 + 7) >= 0 )
                  BUG();
                v81 = sub_BD2BC0(v31);
                v83 = 32LL * (unsigned int)(*(_DWORD *)(v81 + v82 - 4) - v80);
LABEL_78:
                v84 = (32LL * (*(_DWORD *)(v31 + 4) & 0x7FFFFFF) - 32 - v83) >> 5;
                if ( !(_DWORD)v84 )
                  goto LABEL_17;
                v138 = v12;
                v85 = 0;
                v86 = (unsigned int)v84;
                v87 = *(_DWORD *)(v31 + 4) & 0x7FFFFFF;
                while ( 2 )
                {
                  v88 = a1 == *(_QWORD **)(v31 + 32 * (v85 - v87)) && *(_QWORD *)(v31 + 32 * (v85 - v87)) != 0;
                  if ( !v88 )
                  {
LABEL_81:
                    if ( v86 == ++v85 )
                    {
                      v12 = v138;
                      goto LABEL_17;
                    }
                    v87 = *(_DWORD *)(v31 + 4) & 0x7FFFFFF;
                    continue;
                  }
                  break;
                }
                v137 = v88;
                v108 = sub_2A4E220(a3, v31);
                v149 = 0;
                LODWORD(v148) = v108;
                v109 = sub_2A4BE80((__int64)&v150, &v148);
                v41 = v150;
                if ( v109 == (_DWORD *)v150 )
                {
                  v12 = v137;
                  goto LABEL_67;
                }
                v113 = *((_QWORD *)v109 - 1);
                v114 = *(_BYTE **)(v113 - 64);
                if ( *v114 == 61 )
                {
                  v115 = *((_QWORD *)v114 - 4);
                  v116 = v31 + 32 * (v85 - (*(_DWORD *)(v31 + 4) & 0x7FFFFFF));
                  v122 = *(_QWORD *)v116;
                  if ( !v115 )
                  {
                    if ( v122 )
                    {
                      v131 = *(_QWORD *)(v116 + 8);
                      **(_QWORD **)(v116 + 16) = v131;
                      if ( v131 )
                        *(_QWORD *)(v131 + 16) = *(_QWORD *)(v116 + 16);
                      *(_QWORD *)v116 = 0;
                    }
                    goto LABEL_81;
                  }
                  if ( v122 )
                  {
                    v117 = *(_QWORD *)(v116 + 8);
                    **(_QWORD **)(v116 + 16) = v117;
                    if ( v117 )
                    {
LABEL_108:
                      *(_QWORD *)(v117 + 16) = *(_QWORD *)(v116 + 16);
                      goto LABEL_109;
                    }
                  }
                  *(_QWORD *)v116 = v115;
                }
                else
                {
                  v115 = sub_2A4E920(v113, (__int64)&v148, v110, v137, v111, v112);
                  v116 = v31 + 32 * (v85 - (*(_DWORD *)(v31 + 4) & 0x7FFFFFF));
                  if ( *(_QWORD *)v116 )
                  {
                    v117 = *(_QWORD *)(v116 + 8);
                    **(_QWORD **)(v116 + 16) = v117;
                    if ( v117 )
                      goto LABEL_108;
                  }
LABEL_109:
                  *(_QWORD *)v116 = v115;
                  if ( !v115 )
                    goto LABEL_81;
                }
                v118 = *(_QWORD *)(v115 + 16);
                *(_QWORD *)(v116 + 8) = v118;
                if ( v118 )
                  *(_QWORD *)(v118 + 16) = v116 + 8;
                *(_QWORD *)(v116 + 16) = v115 + 16;
                *(_QWORD *)(v115 + 16) = v116;
                goto LABEL_81;
              }
LABEL_164:
              BUG();
            }
          }
          v83 = 0;
          goto LABEL_78;
        }
        if ( v32 != 78 )
          break;
        v89 = sub_2A4E220(a3, v31);
        v149 = 0;
        LODWORD(v148) = v89;
        v90 = sub_2A4BE80((__int64)&v150, &v148);
        v41 = v150;
        if ( v90 == (_DWORD *)v150 )
          goto LABEL_67;
        v91 = *((_QWORD *)v90 - 1);
        v92 = *(_BYTE **)(v91 - 64);
        if ( *v92 != 61 )
        {
LABEL_85:
          v93 = sub_2A4E920(v91, (__int64)&v148, v35, v36, v37, v38);
          if ( *(_QWORD *)(v31 - 32) )
          {
            v94 = *(_QWORD *)(v31 - 24);
            **(_QWORD **)(v31 - 16) = v94;
            if ( v94 )
              goto LABEL_87;
          }
          goto LABEL_88;
        }
        v93 = *((_QWORD *)v92 - 4);
        if ( v93 )
        {
          if ( *(_QWORD *)(v31 - 32) )
            goto LABEL_138;
          goto LABEL_119;
        }
        if ( *(_QWORD *)(v31 - 32) )
          goto LABEL_155;
LABEL_17:
        if ( !v29 )
          goto LABEL_33;
      }
      if ( v32 == 63 )
      {
        v96 = sub_2A4E220(a3, v31);
        v149 = 0;
        LODWORD(v148) = v96;
        v97 = sub_2A4BE80((__int64)&v150, &v148);
        v41 = v150;
        if ( v97 == (_DWORD *)v150 )
          goto LABEL_67;
        v102 = *((_QWORD *)v97 - 1);
        v103 = *(_BYTE **)(v102 - 64);
        if ( *v103 == 61 )
        {
          v104 = *((_QWORD *)v103 - 4);
          if ( !v104 )
          {
            v135 = v31 - 32LL * (*(_DWORD *)(v31 + 4) & 0x7FFFFFF);
            if ( *(_QWORD *)v135 )
            {
              v136 = *(_QWORD *)(v135 + 8);
              **(_QWORD **)(v135 + 16) = v136;
              if ( v136 )
                *(_QWORD *)(v136 + 16) = *(_QWORD *)(v135 + 16);
              *(_QWORD *)v135 = 0;
            }
            goto LABEL_17;
          }
          v105 = (__int64 *)(v31 - 32LL * (*(_DWORD *)(v31 + 4) & 0x7FFFFFF));
          if ( !*v105 || (v106 = v105[1], (*(_QWORD *)v105[2] = v106) == 0) )
          {
            *v105 = v104;
LABEL_101:
            v107 = *(_QWORD *)(v104 + 16);
            v105[1] = v107;
            if ( v107 )
              *(_QWORD *)(v107 + 16) = v105 + 1;
            v105[2] = v104 + 16;
            *(_QWORD *)(v104 + 16) = v105;
            goto LABEL_17;
          }
LABEL_99:
          *(_QWORD *)(v106 + 16) = v105[2];
        }
        else
        {
          v104 = sub_2A4E920(v102, (__int64)&v148, v98, v99, v100, v101);
          v105 = (__int64 *)(v31 - 32LL * (*(_DWORD *)(v31 + 4) & 0x7FFFFFF));
          if ( *v105 )
          {
            v106 = v105[1];
            *(_QWORD *)v105[2] = v106;
            if ( v106 )
              goto LABEL_99;
          }
        }
        *v105 = v104;
        if ( v104 )
          goto LABEL_101;
        goto LABEL_17;
      }
      if ( v32 != 61 )
        goto LABEL_17;
      v33 = sub_2A4E220(a3, v31);
      v149 = 0;
      LODWORD(v148) = v33;
      v34 = sub_2A4BE80((__int64)&v150, &v148);
      if ( v34 != (_DWORD *)v150 )
      {
        v91 = *((_QWORD *)v34 - 1);
        v120 = *(_BYTE **)(v91 - 64);
        if ( *v120 != 61 )
          goto LABEL_85;
        v93 = *((_QWORD *)v120 - 4);
        v121 = *(_QWORD *)(v31 - 32);
        if ( v93 )
        {
          if ( !v121 )
            goto LABEL_119;
LABEL_138:
          v94 = *(_QWORD *)(v31 - 24);
          **(_QWORD **)(v31 - 16) = v94;
          if ( !v94 )
          {
LABEL_119:
            *(_QWORD *)(v31 - 32) = v93;
LABEL_89:
            v95 = *(_QWORD *)(v93 + 16);
            *(_QWORD *)(v31 - 24) = v95;
            if ( v95 )
              *(_QWORD *)(v95 + 16) = v31 - 24;
            *(_QWORD *)(v31 - 16) = v93 + 16;
            *(_QWORD *)(v93 + 16) = v31 - 32;
            goto LABEL_17;
          }
LABEL_87:
          *(_QWORD *)(v94 + 16) = *(_QWORD *)(v31 - 16);
LABEL_88:
          *(_QWORD *)(v31 - 32) = v93;
          if ( !v93 )
            goto LABEL_17;
          goto LABEL_89;
        }
        if ( v121 )
        {
LABEL_155:
          v134 = *(_QWORD *)(v31 - 24);
          **(_QWORD **)(v31 - 16) = v134;
          if ( v134 )
            *(_QWORD *)(v134 + 16) = *(_QWORD *)(v31 - 16);
          *(_QWORD *)(v31 - 32) = 0;
          goto LABEL_17;
        }
        goto LABEL_17;
      }
      if ( *(_BYTE *)v31 != 61 )
        goto LABEL_17;
LABEL_26:
      v39 = sub_2A4E220(a3, v31);
      v149 = 0;
      LODWORD(v148) = v39;
      v40 = sub_2A4BE80((__int64)&v150, &v148);
      v41 = v150;
      if ( v40 == (_DWORD *)v150 )
      {
        if ( (_DWORD)v151 )
        {
          v12 = 0;
          goto LABEL_67;
        }
        v42 = sub_ACA8A0(*(__int64 ***)(v31 + 8));
      }
      else
      {
        v42 = *(_QWORD *)(*((_QWORD *)v40 - 1) - 64LL);
        if ( !v42 )
        {
          sub_2A4C510(v31, 0, a4, a6, a5);
          goto LABEL_30;
        }
      }
      sub_2A4C510(v31, (unsigned __int8 *)v42, a4, a6, a5);
      if ( v42 == v31 )
        v42 = sub_ACADE0(*(__int64 ***)(v42 + 8));
LABEL_30:
      sub_BD84D0(v31, v42);
      sub_B43D60((_QWORD *)v31);
      v43 = *(_DWORD *)(a3 + 24);
      v44 = *(_QWORD *)(a3 + 8);
      if ( !v43 )
        goto LABEL_17;
      v45 = v43 - 1;
      v46 = v45 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v47 = (__int64 *)(v44 + 16LL * v46);
      v48 = *v47;
      if ( v31 != *v47 )
      {
        v129 = 1;
        while ( v48 != -4096 )
        {
          v130 = v129 + 1;
          v46 = v45 & (v129 + v46);
          v47 = (__int64 *)(v44 + 16LL * v46);
          v48 = *v47;
          if ( v31 == *v47 )
            goto LABEL_32;
          v129 = v130;
        }
        goto LABEL_17;
      }
LABEL_32:
      *v47 = -8192;
      --*(_DWORD *)(a3 + 16);
      ++*(_DWORD *)(a3 + 20);
    }
  }
LABEL_33:
  v49 = (__int64 *)sub_B43CA0((__int64)a1);
  sub_AE0470((__int64)&v148, v49, 0, 0);
  v50 = a1[2];
  v51 = a2 + 616;
  if ( v50 )
  {
    v141 = v12;
    do
    {
      v52 = *(_QWORD **)(v50 + 24);
      v144 = (_QWORD *)*(v52 - 8);
      sub_2A518A0(a2 + 616, (__int64)v52, &v148, a7, a8);
      v51 = a2;
      v53 = *(__int64 **)(a2 + 568);
      v54 = &v53[*(unsigned int *)(a2 + 576)];
      for ( i = v53; v54 != i; ++i )
      {
        v56 = *i;
        v57 = *(_QWORD *)(*i - 32);
        if ( !v57 || *(_BYTE *)v57 || (v51 = *(_QWORD *)(v56 + 80), *(_QWORD *)(v57 + 24) != v51) )
          BUG();
        if ( *(_DWORD *)(v57 + 36) == 69 )
        {
          v51 = (__int64)v52;
          sub_F519F0(v56, (__int64)v52, &v148);
        }
      }
      v58 = *(__int64 **)(a2 + 592);
      v59 = &v58[*(unsigned int *)(a2 + 600)];
      for ( j = v58; v59 != j; ++j )
      {
        while ( 1 )
        {
          v61 = *j;
          if ( !*(_BYTE *)(*j + 64) )
            break;
          if ( v59 == ++j )
            goto LABEL_47;
        }
        v51 = (__int64)v52;
        sub_F51C80(v61, (__int64)v52, &v148);
      }
LABEL_47:
      sub_B43D60(v52);
      v62 = *(_DWORD *)(a3 + 24);
      v63 = *(_QWORD *)(a3 + 8);
      if ( v62 )
      {
        v64 = v62 - 1;
        v65 = v64 & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
        v66 = (_QWORD *)(v63 + 16LL * v65);
        v67 = (_QWORD *)*v66;
        if ( (_QWORD *)*v66 == v52 )
        {
LABEL_49:
          *v66 = -8192;
          --*(_DWORD *)(a3 + 16);
          ++*(_DWORD *)(a3 + 20);
        }
        else
        {
          v128 = 1;
          while ( v67 != (_QWORD *)-4096LL )
          {
            v51 = (unsigned int)(v128 + 1);
            v65 = v64 & (v128 + v65);
            v66 = (_QWORD *)(v63 + 16LL * v65);
            v67 = (_QWORD *)*v66;
            if ( v52 == (_QWORD *)*v66 )
              goto LABEL_49;
            v128 = v51;
          }
        }
      }
      if ( v141 )
      {
        if ( !v144[2] )
        {
          sub_B43D60(v144);
          v123 = *(_DWORD *)(a3 + 24);
          v51 = *(_QWORD *)(a3 + 8);
          if ( v123 )
          {
            v124 = v123 - 1;
            v125 = v124 & (((unsigned int)v144 >> 9) ^ ((unsigned int)v144 >> 4));
            v126 = (_QWORD *)(v51 + 16LL * v125);
            v127 = (_QWORD *)*v126;
            if ( v144 == (_QWORD *)*v126 )
            {
LABEL_126:
              *v126 = -8192;
              --*(_DWORD *)(a3 + 16);
              ++*(_DWORD *)(a3 + 20);
            }
            else
            {
              v132 = 1;
              while ( v127 != (_QWORD *)-4096LL )
              {
                v133 = v132 + 1;
                v125 = v124 & (v132 + v125);
                v126 = (_QWORD *)(v51 + 16LL * v125);
                v127 = (_QWORD *)*v126;
                if ( v144 == (_QWORD *)*v126 )
                  goto LABEL_126;
                v132 = v133;
              }
            }
          }
        }
      }
      v50 = a1[2];
    }
    while ( v50 );
  }
  sub_AE94E0((__int64)a1);
  sub_B43D60(a1);
  v68 = *(__int64 **)(a2 + 568);
  for ( k = &v68[*(unsigned int *)(a2 + 576)]; k != v68; ++v68 )
  {
    v70 = *v68;
    v71 = *(_QWORD *)(*v68 - 32);
    if ( !v71 || *(_BYTE *)v71 || (v51 = *(_QWORD *)(v70 + 80), *(_QWORD *)(v71 + 24) != v51) )
      BUG();
    if ( *(_DWORD *)(v71 + 36) == 69
      || sub_AF4730(*(_QWORD *)(*(_QWORD *)(v70 + 32 * (2LL - (*(_DWORD *)(v70 + 4) & 0x7FFFFFF))) + 24LL)) )
    {
      sub_B43D60((_QWORD *)v70);
    }
  }
  v72 = *(_QWORD *)(a2 + 592);
  for ( m = v72 + 8LL * *(unsigned int *)(a2 + 600); m != v72; v72 += 8 )
  {
    v74 = *(_QWORD **)v72;
    if ( *(_BYTE *)(*(_QWORD *)v72 + 64LL) )
    {
      v75 = sub_B11F60((__int64)(v74 + 10));
      if ( !sub_AF4730(v75) )
        continue;
    }
    sub_B14290(v74);
  }
  v12 = 1;
  sub_AE9130((__int64)&v148, v51);
  v41 = v150;
LABEL_67:
  if ( v41 != v152 )
    _libc_free((unsigned __int64)v41);
  return v12;
}
