// Function: sub_27867C0
// Address: 0x27867c0
//
__int64 __fastcall sub_27867C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned int v11; // ecx
  __int64 v12; // rdx
  unsigned __int8 *v13; // r8
  __int64 v14; // rax
  __int64 v15; // r13
  int v17; // edx
  int v18; // r13d
  __int64 v19; // r13
  unsigned __int8 *v20; // r8
  unsigned __int8 *v21; // r13
  unsigned __int8 *i; // r15
  unsigned __int8 *v23; // r8
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // r8
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  __int64 v30; // rdx
  unsigned int v31; // eax
  void **v32; // r8
  __int64 v33; // rax
  __int64 v34; // rdx
  unsigned __int64 v35; // r8
  __int64 v36; // rdx
  int v37; // edx
  unsigned __int8 *v38; // r11
  unsigned __int8 *v39; // r10
  unsigned int v40; // r14d
  __int64 (__fastcall *v41)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  char v42; // al
  unsigned __int8 *v43; // r10
  unsigned __int8 *v44; // rdx
  unsigned __int8 *v45; // rsi
  __int64 v46; // rax
  int v47; // r14d
  __int64 v48; // r14
  unsigned int *v49; // r14
  unsigned int *v50; // rbx
  __int64 v51; // rdx
  unsigned int v52; // esi
  __int64 v53; // rdx
  unsigned __int8 *v54; // r14
  unsigned __int8 *v55; // r10
  __int64 (__fastcall *v56)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v57; // rax
  __int64 v58; // r14
  unsigned int *v59; // r14
  unsigned int *v60; // rbx
  __int64 v61; // rdx
  unsigned int v62; // esi
  __int64 **v63; // rdx
  _QWORD *v64; // rdi
  _QWORD *v65; // rsi
  unsigned int v66; // esi
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // rdi
  __int64 v70; // r11
  unsigned int v71; // ecx
  __int64 v72; // rax
  unsigned __int8 *v73; // rdx
  __int64 v74; // rax
  __int64 v75; // rdx
  int v76; // eax
  __int64 v77; // rcx
  int v78; // edx
  unsigned int v79; // eax
  unsigned __int8 *v80; // rsi
  unsigned int v81; // r14d
  __int64 v82; // rax
  __int64 v83; // rdx
  int v84; // edi
  int v85; // eax
  int v86; // edx
  __int64 v87; // rax
  unsigned __int8 **v88; // rax
  int v89; // r8d
  __int64 v90; // rsi
  __int64 v91; // rax
  __int64 v92; // rcx
  int v93; // r14d
  __int64 v94; // r10
  int v95; // edi
  int v96; // edi
  int v97; // r10d
  __int64 v98; // rcx
  __int64 v99; // r14
  __int64 v100; // rax
  __int64 v101; // rax
  __int64 v102; // rax
  unsigned __int8 *v103; // [rsp+0h] [rbp-180h]
  unsigned __int8 *v104; // [rsp+0h] [rbp-180h]
  unsigned __int8 *v105; // [rsp+0h] [rbp-180h]
  unsigned __int8 *v106; // [rsp+8h] [rbp-178h]
  unsigned __int8 *v107; // [rsp+8h] [rbp-178h]
  unsigned __int8 *v108; // [rsp+8h] [rbp-178h]
  unsigned __int8 *v109; // [rsp+8h] [rbp-178h]
  __int64 v110; // [rsp+8h] [rbp-178h]
  unsigned __int8 *v111; // [rsp+8h] [rbp-178h]
  unsigned __int8 *v112; // [rsp+8h] [rbp-178h]
  unsigned __int8 *v113; // [rsp+10h] [rbp-170h]
  unsigned __int8 *v114; // [rsp+10h] [rbp-170h]
  __int64 v115; // [rsp+10h] [rbp-170h]
  __int64 v116; // [rsp+10h] [rbp-170h]
  __int64 v117; // [rsp+28h] [rbp-158h] BYREF
  const char *v118; // [rsp+30h] [rbp-150h] BYREF
  __int64 v119; // [rsp+38h] [rbp-148h]
  __int16 v120; // [rsp+50h] [rbp-130h]
  const char *v121; // [rsp+60h] [rbp-120h] BYREF
  __int64 v122; // [rsp+68h] [rbp-118h]
  __int16 v123; // [rsp+80h] [rbp-100h]
  unsigned __int8 **v124; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v125; // [rsp+98h] [rbp-E8h]
  _BYTE v126[32]; // [rsp+A0h] [rbp-E0h] BYREF
  unsigned int *v127; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v128; // [rsp+C8h] [rbp-B8h]
  _BYTE v129[32]; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v130; // [rsp+F0h] [rbp-90h]
  __int64 v131; // [rsp+F8h] [rbp-88h]
  __int64 v132; // [rsp+100h] [rbp-80h]
  __int64 v133; // [rsp+108h] [rbp-78h]
  void **v134; // [rsp+110h] [rbp-70h]
  void **v135; // [rsp+118h] [rbp-68h]
  __int64 v136; // [rsp+120h] [rbp-60h]
  int v137; // [rsp+128h] [rbp-58h]
  __int16 v138; // [rsp+12Ch] [rbp-54h]
  char v139; // [rsp+12Eh] [rbp-52h]
  __int64 v140; // [rsp+130h] [rbp-50h]
  __int64 v141; // [rsp+138h] [rbp-48h]
  void *v142; // [rsp+140h] [rbp-40h] BYREF
  void *v143; // [rsp+148h] [rbp-38h] BYREF

  v8 = (unsigned __int8 *)a2;
  v9 = *(unsigned int *)(a1 + 232);
  v117 = a2;
  v10 = *(_QWORD *)(a1 + 216);
  if ( (_DWORD)v9 )
  {
    v11 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v12 = v10 + 16LL * v11;
    v13 = *(unsigned __int8 **)v12;
    if ( v8 == *(unsigned __int8 **)v12 )
    {
LABEL_3:
      if ( v12 != v10 + 16 * v9 )
      {
        v14 = *(_QWORD *)(a1 + 240) + 16LL * *(unsigned int *)(v12 + 8);
        if ( v14 != *(_QWORD *)(a1 + 240) + 16LL * *(unsigned int *)(a1 + 248) )
          return *(_QWORD *)(v14 + 8);
      }
    }
    else
    {
      v17 = 1;
      while ( v13 != (unsigned __int8 *)-4096LL )
      {
        a6 = (unsigned int)(v17 + 1);
        v11 = (v9 - 1) & (v17 + v11);
        v12 = v10 + 16LL * v11;
        v13 = *(unsigned __int8 **)v12;
        if ( v8 == *(unsigned __int8 **)v12 )
          goto LABEL_3;
        v17 = a6;
      }
    }
  }
  v18 = *((_DWORD *)v8 + 1);
  v124 = (unsigned __int8 **)v126;
  v125 = 0x400000000LL;
  v19 = 32LL * (v18 & 0x7FFFFFF);
  if ( (v8[7] & 0x40) != 0 )
  {
    v20 = (unsigned __int8 *)*((_QWORD *)v8 - 1);
    v21 = &v20[v19];
  }
  else
  {
    v20 = &v8[-v19];
    v21 = v8;
  }
  for ( i = v20; v21 != i; i += 32 )
  {
    v23 = *(unsigned __int8 **)i;
    if ( (unsigned __int8)(*v8 - 72) <= 1u )
    {
      v28 = (unsigned int)v125;
      v29 = (unsigned int)v125 + 1LL;
      if ( v29 > HIDWORD(v125) )
      {
        v114 = *(unsigned __int8 **)i;
        sub_C8D5F0((__int64)&v124, v126, v29, 8u, (__int64)v23, a6);
        v28 = (unsigned int)v125;
        v23 = v114;
      }
      v124[v28] = v23;
      LODWORD(v125) = v125 + 1;
    }
    else if ( *v23 <= 0x1Cu )
    {
      v113 = *(unsigned __int8 **)i;
      if ( *v23 != 18 )
        BUG();
      v127 = (unsigned int *)sub_BCAE30(a3);
      v128 = v30;
      v31 = sub_CA1930(&v127);
      v32 = (void **)v113;
      LODWORD(v122) = v31;
      if ( v31 > 0x40 )
      {
        sub_C43690((__int64)&v121, 0, 0);
        v32 = (void **)v113;
      }
      else
      {
        v121 = 0;
      }
      BYTE4(v122) = 0;
      sub_C41980(v32 + 3, (__int64)&v121, 1, &v127);
      v33 = sub_AD8D80(a3, (__int64)&v121);
      v34 = (unsigned int)v125;
      v35 = (unsigned int)v125 + 1LL;
      if ( v35 > HIDWORD(v125) )
      {
        v116 = v33;
        sub_C8D5F0((__int64)&v124, v126, (unsigned int)v125 + 1LL, 8u, v35, a6);
        v34 = (unsigned int)v125;
        v33 = v116;
      }
      v124[v34] = (unsigned __int8 *)v33;
      LODWORD(v125) = v125 + 1;
      if ( (unsigned int)v122 > 0x40 && v121 )
        j_j___libc_free_0_0((unsigned __int64)v121);
    }
    else
    {
      v24 = sub_27867C0(a1, *(_QWORD *)i, a3);
      v25 = (unsigned int)v125;
      v26 = (unsigned int)v125 + 1LL;
      if ( v26 > HIDWORD(v125) )
      {
        v115 = v24;
        sub_C8D5F0((__int64)&v124, v126, (unsigned int)v125 + 1LL, 8u, v26, a6);
        v25 = (unsigned int)v125;
        v24 = v115;
      }
      v124[v25] = (unsigned __int8 *)v24;
      LODWORD(v125) = v125 + 1;
    }
  }
  v27 = sub_BD5C60((__int64)v8);
  v135 = &v143;
  v133 = v27;
  v134 = &v142;
  v138 = 512;
  LOWORD(v132) = 0;
  v127 = (unsigned int *)v129;
  v142 = &unk_49DA100;
  v128 = 0x200000000LL;
  v136 = 0;
  v143 = &unk_49DA0B0;
  v137 = 0;
  v139 = 7;
  v140 = 0;
  v141 = 0;
  v130 = 0;
  v131 = 0;
  sub_D5F1F0((__int64)&v127, (__int64)v8);
  switch ( *v8 )
  {
    case ')':
      v118 = sub_BD5D20((__int64)v8);
      v120 = 261;
      v119 = v53;
      v54 = *v124;
      v55 = (unsigned __int8 *)sub_AD6530(*((_QWORD *)*v124 + 1), (__int64)v8);
      v56 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))*((_QWORD *)*v134 + 4);
      if ( v56 == sub_9201A0 )
      {
        if ( *v55 > 0x15u || *v54 > 0x15u )
          goto LABEL_58;
        v109 = v55;
        if ( (unsigned __int8)sub_AC47B0(15) )
          v57 = sub_AD5570(15, (__int64)v109, v54, 0, 0);
        else
          v57 = sub_AABE40(0xFu, v109, v54);
        v55 = v109;
        v15 = v57;
      }
      else
      {
        v112 = v55;
        v102 = v56((__int64)v134, 15u, v55, v54, 0, 0);
        v55 = v112;
        v15 = v102;
      }
      if ( v15 )
        goto LABEL_64;
LABEL_58:
      v123 = 257;
      v15 = sub_B504D0(15, (__int64)v55, (__int64)v54, (__int64)&v121, 0, 0);
      (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v135 + 2))(
        v135,
        v15,
        &v118,
        v131,
        v132);
      v58 = 4LL * (unsigned int)v128;
      if ( v127 == &v127[v58] )
        goto LABEL_64;
      v108 = v8;
      v59 = &v127[v58];
      v60 = v127;
      do
      {
        v61 = *((_QWORD *)v60 + 1);
        v62 = *v60;
        v60 += 4;
        sub_B99FD0(v15, v62, v61);
      }
      while ( v59 != v60 );
      goto LABEL_48;
    case '+':
    case '-':
    case '/':
      v118 = sub_BD5D20((__int64)v8);
      v119 = v36;
      v37 = *v8;
      v120 = 261;
      v38 = v124[1];
      v39 = *v124;
      if ( v37 == 45 )
      {
        v40 = 15;
      }
      else
      {
        v40 = 17;
        if ( v37 != 47 )
        {
          if ( v37 != 43 )
LABEL_134:
            BUG();
          v40 = 13;
        }
      }
      v41 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v134 + 2);
      if ( v41 == sub_9202E0 )
      {
        if ( *v39 > 0x15u || *v38 > 0x15u )
          goto LABEL_41;
        v103 = *v124;
        v106 = v124[1];
        v42 = sub_AC47B0(v40);
        v43 = v103;
        v44 = v106;
        v45 = v103;
        v104 = v106;
        v107 = v43;
        if ( v42 )
          v46 = sub_AD5570(v40, (__int64)v45, v44, 0, 0);
        else
          v46 = sub_AABE40(v40, v45, v44);
        v39 = v107;
        v38 = v104;
        v15 = v46;
      }
      else
      {
        v105 = v124[1];
        v111 = *v124;
        v101 = v41((__int64)v134, v40, v39, v38);
        v38 = v105;
        v39 = v111;
        v15 = v101;
      }
      if ( v15 )
        goto LABEL_64;
LABEL_41:
      v123 = 257;
      v15 = sub_B504D0(v40, (__int64)v39, (__int64)v38, (__int64)&v121, 0, 0);
      if ( (unsigned __int8)sub_920620(v15) )
      {
        v47 = v137;
        if ( v136 )
          sub_B99FD0(v15, 3u, v136);
        sub_B45150(v15, v47);
      }
      (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v135 + 2))(
        v135,
        v15,
        &v118,
        v131,
        v132);
      v48 = 4LL * (unsigned int)v128;
      if ( v127 != &v127[v48] )
      {
        v108 = v8;
        v49 = &v127[v48];
        v50 = v127;
        do
        {
          v51 = *((_QWORD *)v50 + 1);
          v52 = *v50;
          v50 += 4;
          sub_B99FD0(v15, v52, v51);
        }
        while ( v49 != v50 );
LABEL_48:
        v8 = v108;
      }
LABEL_64:
      if ( !*(_DWORD *)(a1 + 64) )
      {
LABEL_65:
        v64 = *(_QWORD **)(a1 + 80);
        v65 = &v64[*(unsigned int *)(a1 + 88)];
        if ( v65 == sub_27849A0(v64, (__int64)v65, &v117) )
          goto LABEL_66;
LABEL_77:
        sub_BD84D0((__int64)v8, v15);
        goto LABEL_66;
      }
LABEL_75:
      v76 = *(_DWORD *)(a1 + 72);
      v77 = *(_QWORD *)(a1 + 56);
      if ( !v76 )
        goto LABEL_66;
      v78 = v76 - 1;
      v79 = (v76 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v80 = *(unsigned __int8 **)(v77 + 8LL * v79);
      if ( v8 == v80 )
        goto LABEL_77;
      v84 = 1;
      while ( v80 != (unsigned __int8 *)-4096LL )
      {
        v79 = v78 & (v84 + v79);
        v80 = *(unsigned __int8 **)(v77 + 8LL * v79);
        if ( v8 == v80 )
          goto LABEL_77;
        ++v84;
      }
LABEL_66:
      v66 = *(_DWORD *)(a1 + 232);
      v67 = a1 + 208;
      if ( !v66 )
      {
        ++*(_QWORD *)(a1 + 208);
        goto LABEL_106;
      }
      v68 = 1;
      v69 = *(_QWORD *)(a1 + 216);
      v70 = 0;
      v71 = (v66 - 1) & (((unsigned int)v8 >> 4) ^ ((unsigned int)v8 >> 9));
      v72 = v69 + 16LL * v71;
      v73 = *(unsigned __int8 **)v72;
      if ( v8 != *(unsigned __int8 **)v72 )
      {
        while ( v73 != (unsigned __int8 *)-4096LL )
        {
          if ( v73 == (unsigned __int8 *)-8192LL && !v70 )
            v70 = v72;
          v71 = (v66 - 1) & (v68 + v71);
          v72 = v69 + 16LL * v71;
          v73 = *(unsigned __int8 **)v72;
          if ( v8 == *(unsigned __int8 **)v72 )
            goto LABEL_68;
          v68 = (unsigned int)(v68 + 1);
        }
        if ( !v70 )
          v70 = v72;
        v85 = *(_DWORD *)(a1 + 224);
        ++*(_QWORD *)(a1 + 208);
        v86 = v85 + 1;
        if ( 4 * (v85 + 1) < 3 * v66 )
        {
          if ( v66 - *(_DWORD *)(a1 + 228) - v86 > v66 >> 3 )
          {
LABEL_100:
            *(_DWORD *)(a1 + 224) = v86;
            if ( *(_QWORD *)v70 != -4096 )
              --*(_DWORD *)(a1 + 228);
            *(_QWORD *)v70 = v8;
            *(_DWORD *)(v70 + 8) = 0;
            v87 = *(unsigned int *)(a1 + 248);
            if ( v87 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 252) )
            {
              v110 = v70;
              sub_C8D5F0(a1 + 240, (const void *)(a1 + 256), v87 + 1, 0x10u, v67, v68);
              v87 = *(unsigned int *)(a1 + 248);
              v70 = v110;
            }
            v88 = (unsigned __int8 **)(*(_QWORD *)(a1 + 240) + 16 * v87);
            *v88 = v8;
            v88[1] = 0;
            v74 = *(unsigned int *)(a1 + 248);
            *(_DWORD *)(a1 + 248) = v74 + 1;
            *(_DWORD *)(v70 + 8) = v74;
            goto LABEL_69;
          }
          sub_9BAAD0(a1 + 208, v66);
          v95 = *(_DWORD *)(a1 + 232);
          if ( v95 )
          {
            v96 = v95 - 1;
            v97 = 1;
            v67 = 0;
            v98 = *(_QWORD *)(a1 + 216);
            LODWORD(v99) = v96 & (((unsigned int)v8 >> 4) ^ ((unsigned int)v8 >> 9));
            v86 = *(_DWORD *)(a1 + 224) + 1;
            v70 = v98 + 16LL * (unsigned int)v99;
            v100 = *(_QWORD *)v70;
            if ( v8 != *(unsigned __int8 **)v70 )
            {
              while ( v100 != -4096 )
              {
                if ( v100 == -8192 && !v67 )
                  v67 = v70;
                v99 = v96 & (unsigned int)(v99 + v97);
                v70 = v98 + 16 * v99;
                v100 = *(_QWORD *)v70;
                if ( v8 == *(unsigned __int8 **)v70 )
                  goto LABEL_100;
                ++v97;
              }
              if ( v67 )
                v70 = v67;
            }
            goto LABEL_100;
          }
LABEL_133:
          ++*(_DWORD *)(a1 + 224);
          BUG();
        }
LABEL_106:
        sub_9BAAD0(a1 + 208, 2 * v66);
        v89 = *(_DWORD *)(a1 + 232);
        if ( v89 )
        {
          v67 = (unsigned int)(v89 - 1);
          v90 = *(_QWORD *)(a1 + 216);
          LODWORD(v91) = v67 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
          v86 = *(_DWORD *)(a1 + 224) + 1;
          v70 = v90 + 16LL * (unsigned int)v91;
          v92 = *(_QWORD *)v70;
          if ( v8 != *(unsigned __int8 **)v70 )
          {
            v93 = 1;
            v94 = 0;
            while ( v92 != -4096 )
            {
              if ( !v94 && v92 == -8192 )
                v94 = v70;
              v91 = (unsigned int)v67 & ((_DWORD)v91 + v93);
              v70 = v90 + 16 * v91;
              v92 = *(_QWORD *)v70;
              if ( v8 == *(unsigned __int8 **)v70 )
                goto LABEL_100;
              ++v93;
            }
            if ( v94 )
              v70 = v94;
          }
          goto LABEL_100;
        }
        goto LABEL_133;
      }
LABEL_68:
      v74 = *(unsigned int *)(v72 + 8);
LABEL_69:
      *(_QWORD *)(*(_QWORD *)(a1 + 240) + 16 * v74 + 8) = v15;
      nullsub_61();
      v142 = &unk_49DA100;
      nullsub_63();
      if ( v127 != (unsigned int *)v129 )
        _libc_free((unsigned __int64)v127);
      if ( v124 != (unsigned __int8 **)v126 )
        _libc_free((unsigned __int64)v124);
      return v15;
    case 'F':
      v75 = *((_QWORD *)v8 + 1);
      v123 = 257;
      goto LABEL_74;
    case 'G':
      v63 = (__int64 **)*((_QWORD *)v8 + 1);
      v123 = 257;
      goto LABEL_63;
    case 'H':
      v75 = a3;
      v123 = 257;
LABEL_74:
      v15 = sub_A830B0(&v127, (__int64)*v124, v75, (__int64)&v121);
      if ( *(_DWORD *)(a1 + 64) )
        goto LABEL_75;
      goto LABEL_65;
    case 'I':
      v63 = (__int64 **)a3;
      v123 = 257;
LABEL_63:
      v15 = sub_2784C30((__int64 *)&v127, (unsigned __int64)*v124, v63, (__int64)&v121);
      goto LABEL_64;
    case 'S':
      v81 = 42;
      v82 = (*((unsigned __int16 *)v8 + 1) & 0x3Fu) - 1;
      if ( (unsigned int)v82 <= 0xD )
        v81 = dword_4393720[v82];
      v121 = sub_BD5D20((__int64)v8);
      v123 = 261;
      v122 = v83;
      v15 = sub_92B530(&v127, v81, (__int64)*v124, v124[1], (__int64)&v121);
      goto LABEL_64;
    default:
      goto LABEL_134;
  }
}
