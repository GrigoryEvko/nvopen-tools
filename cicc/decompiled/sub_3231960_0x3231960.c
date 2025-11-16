// Function: sub_3231960
// Address: 0x3231960
//
void __fastcall sub_3231960(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int16 v8; // ax
  __int64 v9; // rdi
  __int64 v10; // r15
  __int64 (*v11)(void); // rax
  __int64 v12; // r14
  __int64 v13; // r15
  __int64 (__fastcall *v14)(__int64, __int64); // rax
  __int64 v15; // rax
  int v16; // ebx
  int v17; // r12d
  char (__fastcall *v18)(__int64, __int64); // rax
  int v19; // eax
  int v20; // esi
  int v21; // ecx
  __int64 v22; // rdx
  __int64 v23; // rsi
  int v24; // eax
  int v25; // ecx
  int v26; // r9d
  int v27; // r8d
  int v28; // eax
  _BYTE *v29; // r12
  __int64 v30; // rax
  _QWORD *v31; // r13
  char v32; // al
  int v33; // r8d
  int v34; // eax
  __int64 v35; // rcx
  __int64 v36; // rax
  unsigned int v37; // edx
  __int64 *v38; // r13
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rbx
  unsigned __int64 v42; // rbx
  __int64 *v43; // rax
  __int64 v44; // rax
  unsigned int *v45; // r12
  __int64 v46; // r15
  __int64 v47; // r9
  int v48; // r10d
  unsigned __int64 *v49; // rdx
  __int64 v50; // r8
  unsigned __int64 v51; // r14
  unsigned int v52; // edi
  unsigned __int64 *v53; // rax
  unsigned __int64 v54; // rcx
  __int64 v55; // rax
  unsigned __int64 v56; // rbx
  __int64 v57; // rcx
  int v58; // eax
  unsigned __int64 v59; // rsi
  int v60; // eax
  __int64 *v61; // rdx
  __int64 v62; // r12
  __int64 v63; // r13
  __int64 v64; // r12
  __int64 v65; // rdi
  __int64 v66; // r8
  unsigned int v67; // ecx
  __int64 v68; // rax
  __int64 v69; // r9
  __int64 *v70; // rsi
  __int64 v71; // rax
  int v72; // eax
  __int64 v73; // rax
  int v74; // eax
  unsigned __int64 v75; // rbx
  unsigned __int64 v76; // rdi
  __int64 *v77; // rbx
  __int64 *v78; // r12
  unsigned __int64 v79; // rdi
  _BYTE *v80; // rbx
  unsigned __int64 v81; // r12
  unsigned __int64 v82; // rdi
  unsigned __int64 *v83; // rsi
  int v84; // r10d
  unsigned __int64 v85; // rcx
  unsigned __int64 v86; // rax
  unsigned __int64 v87; // rbx
  unsigned __int64 v88; // rdi
  __int64 *v89; // rbx
  unsigned __int64 v90; // rdi
  int i; // r8d
  __int64 v92; // r12
  int v93; // eax
  char v94; // al
  __int64 v95; // rax
  __int64 v96; // rcx
  __int64 v97; // r8
  __int64 v98; // r9
  __int64 v99; // rdx
  __int64 *v100; // rbx
  unsigned __int64 *v101; // rax
  unsigned __int64 v102; // rdx
  int v103; // r14d
  int v104; // eax
  int v105; // r10d
  __int64 *v106; // rax
  _QWORD *v107; // rbx
  __int64 *v108; // r12
  __int64 *v109; // r13
  __int64 *v110; // rcx
  __int64 v111; // r8
  int v112; // esi
  int v113; // r11d
  unsigned __int64 *v114; // r10
  unsigned __int64 v115; // [rsp+10h] [rbp-310h]
  __int64 v116; // [rsp+18h] [rbp-308h]
  __int64 v117; // [rsp+20h] [rbp-300h]
  __int64 v118; // [rsp+20h] [rbp-300h]
  __int64 v119; // [rsp+28h] [rbp-2F8h]
  __int64 v120; // [rsp+40h] [rbp-2E0h]
  __int64 v121; // [rsp+48h] [rbp-2D8h]
  int v122; // [rsp+50h] [rbp-2D0h]
  __int64 v123; // [rsp+58h] [rbp-2C8h]
  int v124; // [rsp+68h] [rbp-2B8h]
  unsigned int *v125; // [rsp+68h] [rbp-2B8h]
  __int64 v127; // [rsp+78h] [rbp-2A8h]
  __int64 v129; // [rsp+88h] [rbp-298h]
  unsigned __int64 v130; // [rsp+98h] [rbp-288h] BYREF
  __int64 v131; // [rsp+A0h] [rbp-280h] BYREF
  __int64 v132; // [rsp+A8h] [rbp-278h]
  __int64 v133; // [rsp+B0h] [rbp-270h]
  unsigned int v134; // [rsp+B8h] [rbp-268h]
  __int64 *v135; // [rsp+C0h] [rbp-260h] BYREF
  __int64 v136; // [rsp+C8h] [rbp-258h]
  __int64 v137; // [rsp+D0h] [rbp-250h] BYREF
  __int64 v138; // [rsp+D8h] [rbp-248h]
  _QWORD v139[4]; // [rsp+E0h] [rbp-240h] BYREF
  __int64 *v140; // [rsp+100h] [rbp-220h] BYREF
  unsigned __int64 v141; // [rsp+108h] [rbp-218h] BYREF
  __int64 v142; // [rsp+110h] [rbp-210h] BYREF
  _QWORD v143[8]; // [rsp+118h] [rbp-208h] BYREF
  int v144; // [rsp+158h] [rbp-1C8h] BYREF
  unsigned __int64 v145; // [rsp+160h] [rbp-1C0h]
  int *v146; // [rsp+168h] [rbp-1B8h]
  int *v147; // [rsp+170h] [rbp-1B0h]
  __int64 v148; // [rsp+178h] [rbp-1A8h]
  _BYTE *v149; // [rsp+180h] [rbp-1A0h] BYREF
  __int64 v150; // [rsp+188h] [rbp-198h]
  _BYTE v151[400]; // [rsp+190h] [rbp-190h] BYREF

  v122 = a4;
  if ( (*(_BYTE *)(a2 + 35) & 0x20) == 0 || (*(_BYTE *)(a2 + 36) & 8) == 0 )
    return;
  v8 = sub_37361D0(a3, 122);
  v9 = a3;
  v10 = 0;
  sub_3249FA0(v9, a4, v8);
  v11 = *(__int64 (**)(void))(**(_QWORD **)(a5 + 16) + 128LL);
  if ( v11 != sub_2DAC790 )
    v10 = v11();
  v121 = a5 + 320;
  v127 = *(_QWORD *)(a5 + 328);
  if ( v127 == a5 + 320 )
    return;
  v129 = v10;
  do
  {
    v12 = *(_QWORD *)(v127 + 56);
    v13 = v127 + 48;
    if ( v12 == v127 + 48 )
      goto LABEL_36;
    while ( 2 )
    {
      if ( *(_WORD *)(v12 + 68) == 21 )
        goto LABEL_25;
      if ( !sub_2E88ED0(v12, 0) )
        goto LABEL_25;
      v28 = *(_DWORD *)(v12 + 44);
      if ( (v28 & 1) != 0 )
        goto LABEL_25;
      if ( (v28 & 4) != 0 || (v28 & 8) == 0 )
      {
        if ( (*(_QWORD *)(*(_QWORD *)(v12 + 16) + 24LL) & 0x20000LL) == 0 )
          goto LABEL_12;
      }
      else if ( !sub_2E88A90(v12, 0x20000, 1) )
      {
        goto LABEL_12;
      }
      if ( (*(_BYTE *)(v12 + 44) & 8) == 0 )
        return;
LABEL_12:
      v14 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v129 + 1504LL);
      if ( v14 == sub_2FDC820 )
        v15 = *(_QWORD *)(v12 + 32);
      else
        v15 = v14(v129, v12);
      if ( *(_BYTE *)v15 == 10 )
      {
        v29 = *(_BYTE **)(v15 + 24);
        if ( *v29 || !sub_B92180(*(_QWORD *)(v15 + 24)) )
          goto LABEL_25;
        v16 = 0;
        v17 = sub_B92180((__int64)v29);
      }
      else
      {
        if ( *(_BYTE *)v15 )
          goto LABEL_25;
        v16 = *(_DWORD *)(v15 + 8);
        v17 = 0;
        if ( (unsigned int)(v16 - 1) > 0x3FFFFFFE )
          goto LABEL_25;
      }
      v18 = *(char (__fastcall **)(__int64, __int64))(*(_QWORD *)v129 + 1328LL);
      if ( v18 == sub_2FDE950 )
      {
        v19 = *(_DWORD *)(v12 + 44);
        LOBYTE(v20) = v19;
        v21 = v19 & 4;
        if ( (v19 & 4) != 0 || (v19 & 8) == 0 )
        {
          v30 = *(_QWORD *)(v12 + 16);
          v22 = (*(_QWORD *)(v30 + 24) >> 5) & 1LL;
          if ( (*(_QWORD *)(v30 + 24) & 0x20LL) == 0 )
            goto LABEL_21;
        }
        else
        {
          LOBYTE(v22) = sub_2E88A90(v12, 32, 1);
          v20 = *(_DWORD *)(v12 + 44);
          v21 = v20 & 4;
          if ( !(_BYTE)v22 )
            goto LABEL_21;
        }
        if ( v21 || (v20 & 8) == 0 )
        {
          LOBYTE(v22) = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v12 + 16) + 24LL) >> 7;
        }
        else
        {
          LOBYTE(v22) = sub_2E88A90(v12, 128, 1);
          v21 = *(_DWORD *)(v12 + 44) & 4;
        }
        if ( !(_BYTE)v22 )
        {
LABEL_21:
          if ( !v21 )
          {
            v23 = v12;
            goto LABEL_23;
          }
          v31 = (_QWORD *)v12;
          goto LABEL_130;
        }
        v31 = (_QWORD *)v12;
        if ( !v21 )
          goto LABEL_44;
      }
      else
      {
        v31 = (_QWORD *)v12;
        LOBYTE(v22) = v18(v129, v12);
        if ( (*(_BYTE *)(v12 + 44) & 4) == 0 )
        {
          v23 = v12;
          if ( (_BYTE)v22 )
            goto LABEL_44;
          goto LABEL_23;
        }
      }
      do
      {
LABEL_130:
        v86 = *v31 & 0xFFFFFFFFFFFFFFF8LL;
        v31 = (_QWORD *)v86;
      }
      while ( (*(_BYTE *)(v86 + 44) & 4) != 0 );
      v23 = v86;
      if ( (_BYTE)v22 )
      {
LABEL_44:
        v32 = sub_3736140(a3);
        v33 = 0;
        if ( v32 )
          v33 = sub_3211FB0(a1, (__int64)v31);
        v124 = v33;
        v34 = sub_3211F40(a1, (__int64)v31);
        v27 = v124;
        v25 = 1;
        v26 = v34;
        goto LABEL_24;
      }
LABEL_23:
      v24 = sub_3211FB0(a1, v23);
      v25 = 0;
      v26 = 0;
      v27 = v24;
LABEL_24:
      v123 = sub_37391A0(a3, v122, v17, v25, v27, v26, v16);
      if ( !*(_BYTE *)(a1 + 3771) )
        goto LABEL_25;
      v149 = v151;
      v150 = 0x400000000LL;
      v120 = sub_2E88D60(v12);
      v35 = *(_QWORD *)(v120 + 696);
      v36 = *(unsigned int *)(v120 + 712);
      if ( !(_DWORD)v36 )
        goto LABEL_107;
      v37 = (v36 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v38 = (__int64 *)(v35 + 32LL * v37);
      v39 = *v38;
      if ( v12 != *v38 )
      {
        for ( i = 1; ; ++i )
        {
          if ( v39 == -4096 )
            goto LABEL_107;
          v37 = (v36 - 1) & (i + v37);
          v38 = (__int64 *)(v35 + 32LL * v37);
          v39 = *v38;
          if ( *v38 == v12 )
            break;
        }
      }
      if ( v38 == (__int64 *)(v35 + 32 * v36) )
        goto LABEL_107;
      v40 = *(_QWORD *)(v12 + 24);
      v41 = *(_QWORD *)v12;
      v131 = 0;
      v132 = 0;
      v119 = v40;
      v42 = v41 & 0xFFFFFFFFFFFFFFF8LL;
      v135 = &v137;
      v133 = 0;
      v134 = 0;
      v136 = 0;
      v43 = (__int64 *)sub_B2BE50(*(_QWORD *)v120);
      v44 = sub_B0D000(v43, 0, 0, 0, 1);
      v45 = (unsigned int *)v38[1];
      v125 = &v45[2 * *((unsigned int *)v38 + 4)];
      if ( v45 == v125 )
        goto LABEL_70;
      v117 = v13;
      v46 = v44;
      v116 = v12;
      v115 = v42;
      do
      {
        while ( 1 )
        {
          v55 = *v45;
          v139[1] = v46;
          v138 = 0x200000001LL;
          v137 = (__int64)v139;
          v139[0] = v55;
          v56 = *v45;
          v141 = (unsigned __int64)v143;
          v140 = (__int64 *)v56;
          v143[0] = v55;
          v143[1] = v46;
          v142 = 0x200000001LL;
          if ( !v134 )
          {
            ++v131;
LABEL_57:
            sub_9E25D0((__int64)&v131, 2 * v134);
            if ( !v134 )
              goto LABEL_192;
            v47 = v134 - 1;
            LODWORD(v57) = v47 & (((0xBF58476D1CE4E5B9LL * v56) >> 31) ^ (484763065 * v56));
            v58 = v133 + 1;
            v49 = (unsigned __int64 *)(v132 + 16LL * (unsigned int)v57);
            v59 = *v49;
            if ( v56 != *v49 )
            {
              v113 = 1;
              v114 = 0;
              while ( v59 != -1 )
              {
                if ( !v114 && v59 == -2 )
                  v114 = v49;
                v50 = (unsigned int)(v113 + 1);
                v57 = (unsigned int)v47 & ((_DWORD)v57 + v113);
                v49 = (unsigned __int64 *)(v132 + 16 * v57);
                v59 = *v49;
                if ( v56 == *v49 )
                  goto LABEL_59;
                ++v113;
              }
              if ( v114 )
                v49 = v114;
            }
            goto LABEL_59;
          }
          v47 = v134 - 1;
          v48 = 1;
          v49 = 0;
          v50 = v132;
          v51 = ((0xBF58476D1CE4E5B9LL * v56) >> 31) ^ (0xBF58476D1CE4E5B9LL * v56);
          v52 = v51 & (v134 - 1);
          v53 = (unsigned __int64 *)(v132 + 16LL * v52);
          v54 = *v53;
          if ( v56 != *v53 )
            break;
LABEL_54:
          v45 += 2;
          if ( v125 == v45 )
            goto LABEL_69;
        }
        while ( v54 != -1 )
        {
          if ( v54 != -2 || v49 )
            v53 = v49;
          v52 = v47 & (v48 + v52);
          v54 = *(_QWORD *)(v132 + 16LL * v52);
          if ( v56 == v54 )
            goto LABEL_54;
          ++v48;
          v49 = v53;
          v53 = (unsigned __int64 *)(v132 + 16LL * v52);
        }
        if ( !v49 )
          v49 = v53;
        ++v131;
        v58 = v133 + 1;
        if ( 4 * ((int)v133 + 1) >= 3 * v134 )
          goto LABEL_57;
        if ( v134 - HIDWORD(v133) - v58 <= v134 >> 3 )
        {
          sub_9E25D0((__int64)&v131, v134);
          if ( !v134 )
          {
LABEL_192:
            LODWORD(v133) = v133 + 1;
            BUG();
          }
          v47 = v134 - 1;
          v50 = v132;
          v83 = 0;
          LODWORD(v51) = v47 & v51;
          v84 = 1;
          v58 = v133 + 1;
          v49 = (unsigned __int64 *)(v132 + 16LL * (unsigned int)v51);
          v85 = *v49;
          if ( v56 != *v49 )
          {
            while ( v85 != -1 )
            {
              if ( !v83 && v85 == -2 )
                v83 = v49;
              v51 = (unsigned int)v47 & ((_DWORD)v51 + v84);
              v49 = (unsigned __int64 *)(v132 + 16 * v51);
              v85 = *v49;
              if ( v56 == *v49 )
                goto LABEL_59;
              ++v84;
            }
            if ( v83 )
              v49 = v83;
          }
        }
LABEL_59:
        LODWORD(v133) = v58;
        if ( *v49 != -1 )
          --HIDWORD(v133);
        *((_DWORD *)v49 + 2) = 0;
        *v49 = v56;
        *((_DWORD *)v49 + 2) = v136;
        v60 = v136;
        if ( HIDWORD(v136) <= (unsigned int)v136 )
        {
          v95 = sub_C8D7D0((__int64)&v135, (__int64)&v137, 0, 0x38u, &v130, v47);
          v99 = (unsigned int)v136;
          v100 = (__int64 *)v95;
          v101 = (unsigned __int64 *)(v95 + 56LL * (unsigned int)v136);
          if ( v101 )
          {
            v102 = (unsigned __int64)v140;
            v101[2] = 0x200000000LL;
            *v101 = v102;
            v99 = (__int64)(v101 + 3);
            v101[1] = (unsigned __int64)(v101 + 3);
            if ( (_DWORD)v142 )
              sub_32187E0((__int64)(v101 + 1), (char **)&v141, v99, v96, v97, v98);
          }
          sub_3228460(&v135, (__int64)v100, v99, v96, v97, v98);
          v103 = v130;
          if ( v135 != &v137 )
            _libc_free((unsigned __int64)v135);
          LODWORD(v136) = v136 + 1;
          v135 = v100;
          HIDWORD(v136) = v103;
        }
        else
        {
          v61 = &v135[7 * (unsigned int)v136];
          if ( v61 )
          {
            *v61 = (__int64)v140;
            v61[1] = (__int64)(v61 + 3);
            v61[2] = 0x200000000LL;
            if ( (_DWORD)v142 )
              sub_32187E0((__int64)(v61 + 1), (char **)&v141, (__int64)v61, (unsigned int)v142, v50, v47);
            v60 = v136;
          }
          LODWORD(v136) = v60 + 1;
        }
        if ( (_QWORD *)v141 == v143 )
          goto LABEL_54;
        _libc_free(v141);
        v45 += 2;
      }
      while ( v125 != v45 );
LABEL_69:
      v13 = v117;
      v12 = v116;
      v42 = v115;
LABEL_70:
      v62 = *(_QWORD *)(v12 + 32);
      v63 = v62 + 40LL * (*(_DWORD *)(v12 + 40) & 0xFFFFFF);
      v64 = v62 + 40LL * (unsigned int)sub_2E88FE0(v12);
      if ( v63 != v64 )
      {
        while ( 1 )
        {
          if ( *(_BYTE *)v64 )
            goto LABEL_72;
          if ( (*(_BYTE *)(v64 + 4) & 1) == 0 )
            goto LABEL_72;
          v65 = *(unsigned int *)(v64 + 8);
          if ( !v134 )
            goto LABEL_72;
          v66 = v134 - 1;
          v67 = v66 & (((0xBF58476D1CE4E5B9LL * v65) >> 31) ^ (484763065 * v65));
          v68 = v132 + 16LL * v67;
          v69 = *(_QWORD *)v68;
          if ( v65 != *(_QWORD *)v68 )
          {
            v104 = 1;
            while ( v69 != -1 )
            {
              v105 = v104 + 1;
              v67 = v66 & (v104 + v67);
              v68 = v132 + 16LL * v67;
              v69 = *(_QWORD *)v68;
              if ( v65 == *(_QWORD *)v68 )
                goto LABEL_77;
              v104 = v105;
            }
            goto LABEL_72;
          }
LABEL_77:
          if ( v68 == v132 + 16LL * v134
            || (v70 = &v135[7 * *(unsigned int *)(v68 + 8)], v70 == &v135[7 * (unsigned int)v136]) )
          {
LABEL_72:
            v64 += 40;
            if ( v63 == v64 )
              break;
          }
          else
          {
            v64 += 40;
            sub_32198F0((__int64)&v131, v70, (__int64)v135, (unsigned int)v136, v66, v69);
            if ( v63 == v64 )
              break;
          }
        }
      }
      v71 = *(_QWORD *)(v120 + 328);
      v144 = 0;
      v145 = 0;
      v118 = v71;
      v140 = &v142;
      v141 = 0x1000000000LL;
      v146 = &v144;
      v147 = &v144;
      v148 = 0;
      v72 = *(_DWORD *)(v12 + 44);
      if ( (v72 & 4) == 0 && (v72 & 8) != 0 )
        LOBYTE(v73) = sub_2E88A90(v12, 0x20000, 1);
      else
        v73 = (*(_QWORD *)(*(_QWORD *)(v12 + 16) + 24LL) >> 17) & 1LL;
      if ( !(_BYTE)v73 )
        goto LABEL_84;
      v92 = *(_QWORD *)(v12 + 8);
      if ( *(_WORD *)(v92 + 68) == 21 )
        goto LABEL_84;
      v93 = *(_DWORD *)(v92 + 44);
      if ( (v93 & 4) == 0 && (v93 & 8) != 0 )
        v94 = sub_2E88A90(*(_QWORD *)(v12 + 8), 128, 1);
      else
        v94 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v92 + 16) + 24LL) >> 7;
      if ( v94 || !(_DWORD)v136 )
      {
LABEL_95:
        v75 = v145;
        while ( v75 )
        {
          sub_321A260(*(_QWORD *)(v75 + 24));
          v76 = v75;
          v75 = *(_QWORD *)(v75 + 16);
          j_j___libc_free_0(v76);
        }
        if ( v140 != &v142 )
          _libc_free((unsigned __int64)v140);
        v77 = v135;
        v78 = &v135[7 * (unsigned int)v136];
        if ( v135 != v78 )
        {
          do
          {
            v78 -= 7;
            v79 = v78[1];
            if ( (__int64 *)v79 != v78 + 3 )
              _libc_free(v79);
          }
          while ( v77 != v78 );
LABEL_103:
          v78 = v135;
        }
      }
      else
      {
        if ( (*(_DWORD *)(v92 + 40) & 0xFFFFFF) != 0 )
          sub_3230220(v92, (__int64)&v131, (__int64)&v149, (__int64)&v140);
LABEL_84:
        while ( v42 != v119 + 48 )
        {
          if ( *(_WORD *)(v42 + 68) != 21 )
          {
            v74 = *(_DWORD *)(v42 + 44);
            if ( (v74 & 4) != 0 || (v74 & 8) == 0 )
            {
              if ( (*(_QWORD *)(*(_QWORD *)(v42 + 16) + 24LL) & 0x80u) != 0LL )
                goto LABEL_95;
            }
            else if ( sub_2E88A90(v42, 128, 1) )
            {
              goto LABEL_95;
            }
            if ( !(_DWORD)v136 )
              goto LABEL_95;
            if ( (*(_DWORD *)(v42 + 40) & 0xFFFFFF) != 0 )
              sub_3230220(v42, (__int64)&v131, (__int64)&v149, (__int64)&v140);
          }
          v42 = *(_QWORD *)v42 & 0xFFFFFFFFFFFFFFF8LL;
        }
        if ( v118 == v119 )
        {
          v137 = 4099;
          v138 = 1;
          v106 = (__int64 *)sub_B2BE50(*(_QWORD *)v120);
          v107 = (_QWORD *)sub_B0D000(v106, &v137, 2, 0, 1);
          if ( v135 != &v135[7 * (unsigned int)v136] )
          {
            v108 = v135;
            v109 = &v135[7 * (unsigned int)v136];
            do
            {
              v110 = (__int64 *)v108[1];
              v111 = *((unsigned int *)v108 + 4);
              v112 = *(_DWORD *)v108;
              v108 += 7;
              sub_32280F0(1, v112, v107, v110, v111, (__int64)&v149);
            }
            while ( v109 != v108 );
          }
        }
        v87 = v145;
        while ( v87 )
        {
          sub_321A260(*(_QWORD *)(v87 + 24));
          v88 = v87;
          v87 = *(_QWORD *)(v87 + 16);
          j_j___libc_free_0(v88);
        }
        if ( v140 != &v142 )
          _libc_free((unsigned __int64)v140);
        v89 = v135;
        v78 = &v135[7 * (unsigned int)v136];
        if ( v135 != v78 )
        {
          do
          {
            v78 -= 7;
            v90 = v78[1];
            if ( (__int64 *)v90 != v78 + 3 )
              _libc_free(v90);
          }
          while ( v89 != v78 );
          goto LABEL_103;
        }
      }
      if ( v78 != &v137 )
        _libc_free((unsigned __int64)v78);
      sub_C7D6A0(v132, 16LL * v134, 8);
LABEL_107:
      sub_373A5A0(a3, v123, &v149);
      v80 = v149;
      v81 = (unsigned __int64)&v149[88 * (unsigned int)v150];
      if ( v149 != (_BYTE *)v81 )
      {
        do
        {
          v81 -= 88LL;
          v82 = *(_QWORD *)(v81 + 16);
          if ( v82 != v81 + 32 )
            _libc_free(v82);
        }
        while ( v80 != (_BYTE *)v81 );
        v81 = (unsigned __int64)v149;
      }
      if ( (_BYTE *)v81 != v151 )
        _libc_free(v81);
LABEL_25:
      v12 = *(_QWORD *)(v12 + 8);
      if ( v13 != v12 )
        continue;
      break;
    }
LABEL_36:
    v127 = *(_QWORD *)(v127 + 8);
  }
  while ( v121 != v127 );
}
