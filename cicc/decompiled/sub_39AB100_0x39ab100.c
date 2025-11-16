// Function: sub_39AB100
// Address: 0x39ab100
//
void __fastcall sub_39AB100(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, signed __int64 a5, char *a6)
{
  _QWORD *v7; // rbx
  __int64 v8; // rsi
  unsigned __int64 v9; // rdx
  int v10; // r14d
  unsigned int v11; // edi
  __int64 v12; // rax
  unsigned int v13; // r13d
  __int64 v14; // r15
  char *v15; // r15
  __int64 v16; // r14
  char *v17; // r13
  __int64 *v18; // rsi
  unsigned __int64 v19; // rax
  char *v20; // r14
  __int64 v21; // r10
  char *i; // rsi
  __int64 v23; // rdi
  char *v24; // rcx
  char *v25; // rax
  char *v26; // rdx
  __int64 v27; // rdi
  __int64 v28; // r14
  __int64 v29; // rdi
  __int64 v30; // r14
  __int64 v31; // rax
  __int64 v32; // rdi
  __int64 v33; // r14
  void (__fastcall *v34)(__int64, __int64, _QWORD); // rbx
  __int64 v35; // rax
  __int64 v36; // rdi
  char v37; // bl
  __int64 (*v38)(); // rax
  __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // r14
  unsigned __int64 v42; // r13
  __int64 v43; // r14
  __int64 v44; // rdi
  unsigned __int64 v45; // rsi
  __int64 v46; // rdi
  __int64 *v47; // r8
  __int64 v48; // rax
  void (*v49)(); // rax
  _QWORD *v50; // rdi
  _BYTE *v51; // r8
  _BYTE *v52; // r15
  __int64 v53; // r11
  void (*v54)(); // rax
  __int64 v55; // rdi
  __int64 v56; // r11
  void (*v57)(); // rcx
  __int64 *v58; // rax
  __int64 v59; // rdx
  _QWORD *v60; // rax
  __int64 *v61; // rax
  __int64 v62; // rdx
  _QWORD *v63; // rax
  __int64 v64; // rax
  __int64 v65; // r8
  _BYTE *v66; // rax
  void (*v67)(); // rcx
  char *v68; // rdx
  const char *v69; // rax
  int v70; // r14d
  int *j; // r15
  __int64 v72; // rdi
  __int64 v73; // rsi
  __int64 v74; // rdi
  __int64 v75; // r8
  void (*v76)(); // r9
  void (*v77)(); // r9
  int v78; // edx
  __int64 v79; // r9
  void (*v80)(); // r8
  const char *v81; // rax
  void (*v82)(); // rax
  void (*v83)(); // rcx
  char **v84; // rax
  unsigned __int64 v85; // r14
  unsigned int v86; // r13d
  __int64 v87; // rdi
  unsigned int v88; // eax
  __int64 v89; // rdi
  unsigned __int64 v90; // r8
  __int64 v91; // r9
  void (*v92)(); // rax
  void (*v93)(); // rax
  __int64 v94; // r8
  void (*v95)(); // rcx
  __int64 v96; // rdi
  __int64 v97; // rax
  __int64 v98; // rdi
  __int64 v99; // r14
  char v100; // al
  __int64 v101; // [rsp+8h] [rbp-DD8h]
  unsigned int v102; // [rsp+30h] [rbp-DB0h]
  char v103; // [rsp+37h] [rbp-DA9h]
  __int64 v104; // [rsp+40h] [rbp-DA0h]
  _BYTE *v105; // [rsp+48h] [rbp-D98h]
  _BYTE *v106; // [rsp+48h] [rbp-D98h]
  int v107; // [rsp+58h] [rbp-D88h]
  unsigned __int64 v108; // [rsp+58h] [rbp-D88h]
  _BYTE *v109; // [rsp+60h] [rbp-D80h]
  int v110; // [rsp+68h] [rbp-D78h]
  int *v111; // [rsp+68h] [rbp-D78h]
  _BYTE *v112; // [rsp+68h] [rbp-D78h]
  _QWORD v113[2]; // [rsp+70h] [rbp-D70h] BYREF
  _QWORD v114[2]; // [rsp+80h] [rbp-D60h] BYREF
  _QWORD v115[2]; // [rsp+90h] [rbp-D50h] BYREF
  __int16 v116; // [rsp+A0h] [rbp-D40h]
  const char *v117; // [rsp+B0h] [rbp-D30h] BYREF
  char *v118; // [rsp+B8h] [rbp-D28h]
  __int16 v119; // [rsp+C0h] [rbp-D20h]
  const char *v120; // [rsp+D0h] [rbp-D10h] BYREF
  char *v121; // [rsp+D8h] [rbp-D08h]
  __int16 v122; // [rsp+E0h] [rbp-D00h]
  __int64 v123[2]; // [rsp+F0h] [rbp-CF0h] BYREF
  _BYTE v124[256]; // [rsp+100h] [rbp-CE0h] BYREF
  int *v125; // [rsp+200h] [rbp-BE0h] BYREF
  __int64 v126; // [rsp+208h] [rbp-BD8h]
  _BYTE v127[384]; // [rsp+210h] [rbp-BD0h] BYREF
  void *src; // [rsp+390h] [rbp-A50h] BYREF
  __int64 v129; // [rsp+398h] [rbp-A48h]
  _BYTE v130[512]; // [rsp+3A0h] [rbp-A40h] BYREF
  _BYTE *v131; // [rsp+5A0h] [rbp-840h] BYREF
  __int64 v132; // [rsp+5A8h] [rbp-838h]
  _BYTE v133[2096]; // [rsp+5B0h] [rbp-830h] BYREF

  v7 = *(_QWORD **)(a1[1] + 264);
  v8 = v7[51];
  src = v130;
  v129 = 0x4000000000LL;
  v9 = 0xEEEEEEEEEEEEEEEFLL * ((v7[52] - v8) >> 3);
  if ( (unsigned __int64)(v7[52] - v8) > 0x1E00 )
  {
    sub_16CD150((__int64)&src, v130, v9, 8, a5, (int)a6);
    v8 = v7[51];
    v12 = (unsigned int)v129;
    v10 = -286331153 * ((v7[52] - v8) >> 3);
    if ( !v10 )
      goto LABEL_9;
    v11 = HIDWORD(v129);
  }
  else
  {
    v10 = -286331153 * ((v7[52] - v8) >> 3);
    if ( !(_DWORD)v9 )
      goto LABEL_22;
    v11 = 64;
    v12 = 0;
  }
  v13 = 0;
  while ( 1 )
  {
    v14 = v8 + 120LL * v13;
    if ( (unsigned int)v12 >= v11 )
    {
      sub_16CD150((__int64)&src, v130, 0, 8, a5, (int)a6);
      v12 = (unsigned int)v129;
    }
    ++v13;
    *((_QWORD *)src + v12) = v14;
    v12 = (unsigned int)(v129 + 1);
    LODWORD(v129) = v129 + 1;
    if ( v13 == v10 )
      break;
    v8 = v7[51];
    v11 = HIDWORD(v129);
  }
LABEL_9:
  v15 = (char *)src;
  v16 = 8 * v12;
  v17 = (char *)src + 8 * v12;
  if ( v17 != src )
  {
    v18 = (__int64 *)((char *)src + 8 * v12);
    _BitScanReverse64(&v19, v16 >> 3);
    sub_39A97E0((__int64 *)src, v18, 2LL * (int)(63 - (v19 ^ 0x3F)));
    if ( (unsigned __int64)v16 <= 0x80 )
    {
      sub_39A94A0(v15, v17);
    }
    else
    {
      v20 = v15 + 128;
      sub_39A94A0(v15, v15 + 128);
      if ( v17 != v15 + 128 )
      {
        do
        {
          v21 = *(_QWORD *)v20;
          for ( i = v20; ; i -= 8 )
          {
            v23 = *((_QWORD *)i - 1);
            v24 = *(char **)(v21 + 104);
            v25 = *(char **)(v21 + 96);
            a6 = *(char **)(v23 + 104);
            v26 = *(char **)(v23 + 96);
            a5 = a6 - v26;
            if ( v24 - v25 > a6 - v26 )
              v24 = &v25[a5];
            if ( v25 == v24 )
              break;
            while ( *(_DWORD *)v25 >= *(_DWORD *)v26 )
            {
              if ( *(_DWORD *)v25 > *(_DWORD *)v26 )
                goto LABEL_100;
              v25 += 4;
              v26 += 4;
              if ( v24 == v25 )
                goto LABEL_99;
            }
LABEL_20:
            *(_QWORD *)i = v23;
          }
LABEL_99:
          if ( v26 != a6 )
            goto LABEL_20;
LABEL_100:
          v20 += 8;
          *(_QWORD *)i = v21;
        }
        while ( v17 != v20 );
      }
    }
  }
LABEL_22:
  v125 = (int *)v127;
  v126 = 0x2000000000LL;
  v123[0] = (__int64)v124;
  v123[1] = 0x4000000000LL;
  sub_39A9FD0((__int64)a1, (__int64 **)&src, (__int64)&v125, (__int64)v123, a5, (int)a6);
  v132 = 0x4000000000LL;
  v131 = v133;
  sub_39AAA90((__int64)a1, (__int64)&v131, (__int64)&src, v123);
  v27 = a1[1];
  v110 = *(_DWORD *)(*(_QWORD *)(v27 + 240) + 348LL);
  if ( v7[67] == v7[66] && v7[70] == v7[69] )
  {
    v28 = *(_QWORD *)(sub_396DD80(v27) + 64);
    v103 = 0;
    v102 = 255;
  }
  else
  {
    v28 = *(_QWORD *)(sub_396DD80(v27) + 64);
    v103 = 1;
    v102 = *(_DWORD *)(sub_396DD80(a1[1]) + 24);
  }
  if ( v28 )
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1[1] + 256) + 160LL))(
      *(_QWORD *)(a1[1] + 256),
      v28,
      0);
  sub_396F480(a1[1], 2u, 0);
  v29 = a1[1];
  v30 = *(_QWORD *)(v29 + 248);
  LODWORD(v117) = sub_396DD70(v29);
  v120 = "GCC_except_table";
  v121 = (char *)v117;
  v122 = 2307;
  v31 = sub_38BF510(v30, (__int64)&v120);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1[1] + 256) + 176LL))(
    *(_QWORD *)(a1[1] + 256),
    v31,
    0);
  v32 = a1[1];
  v33 = *(_QWORD *)(v32 + 256);
  v34 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v33 + 176LL);
  v35 = sub_396F550(v32);
  v34(v33, v35, 0);
  sub_397C150(a1[1], 0xFFu);
  sub_397C150(a1[1], v102);
  v101 = 0;
  if ( v103 )
  {
    v96 = a1[1];
    v120 = "ttbaseref";
    v122 = 259;
    v97 = sub_396F530(v96, (__int64)&v120);
    v98 = a1[1];
    v99 = v97;
    v122 = 259;
    v120 = "ttbase";
    v101 = sub_396F530(v98, (__int64)&v120);
    sub_397C140(a1[1]);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1[1] + 256) + 176LL))(
      *(_QWORD *)(a1[1] + 256),
      v99,
      0);
  }
  v36 = a1[1];
  v37 = 0;
  v38 = *(__int64 (**)())(**(_QWORD **)(v36 + 256) + 80LL);
  if ( v38 != sub_168DB50 )
  {
    v100 = ((__int64 (__fastcall *)(_QWORD))v38)(*(_QWORD *)(v36 + 256));
    v36 = a1[1];
    v37 = v100;
  }
  v120 = "cst_begin";
  v122 = 259;
  v39 = sub_396F530(v36, (__int64)&v120);
  v40 = a1[1];
  v41 = v39;
  v122 = 259;
  v120 = "cst_end";
  v104 = sub_396F530(v40, (__int64)&v120);
  sub_397C150(a1[1], 2 * (v110 == 2) + 1);
  sub_397C140(a1[1]);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1[1] + 256) + 176LL))(
    *(_QWORD *)(a1[1] + 256),
    v41,
    0);
  if ( v110 == 2 )
  {
    v85 = (unsigned __int64)v131;
    v86 = 0;
    v112 = &v131[32 * (unsigned int)v132];
    if ( v131 == v112 )
      goto LABEL_60;
    while ( 1 )
    {
      v89 = a1[1];
      v90 = v86;
      if ( v37 )
      {
        v91 = *(_QWORD *)(v89 + 256);
        v92 = *(void (**)())(*(_QWORD *)v91 + 104LL);
        LODWORD(v115[0]) = v86;
        v122 = 770;
        v117 = ">> Call Site ";
        v118 = (char *)v115[0];
        v119 = 2307;
        v120 = (const char *)&v117;
        v121 = " <<";
        if ( v92 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, const char **, __int64))v92)(v91, &v120, 1);
          v89 = a1[1];
          v90 = v86;
          v91 = *(_QWORD *)(v89 + 256);
        }
        v93 = *(void (**)())(*(_QWORD *)v91 + 104LL);
        LODWORD(v117) = v86;
        v120 = "  On exception at call site ";
        v122 = 2307;
        v121 = (char *)v117;
        if ( v93 != nullsub_580 )
        {
          v108 = v90;
          ((void (__fastcall *)(__int64, const char **, __int64))v93)(v91, &v120, 1);
          v89 = a1[1];
          v90 = v108;
        }
        sub_397C0C0(v89, v90, 0);
        v87 = a1[1];
        v88 = *(_DWORD *)(v85 + 24);
        v94 = *(_QWORD *)(v87 + 256);
        v95 = *(void (**)())(*(_QWORD *)v94 + 104LL);
        if ( v88 )
        {
          v120 = "  Action: ";
          v122 = 2307;
          LODWORD(v117) = ((v88 - 1) >> 1) + 1;
          v121 = (char *)v117;
          if ( v95 == nullsub_580 )
            goto LABEL_107;
LABEL_115:
          ((void (__fastcall *)(__int64, const char **, __int64))v95)(v94, &v120, 1);
          v87 = a1[1];
          v88 = *(_DWORD *)(v85 + 24);
          goto LABEL_107;
        }
        v120 = "  Action: cleanup";
        v122 = 259;
        if ( v95 != nullsub_580 )
          goto LABEL_115;
      }
      else
      {
        sub_397C0C0(v89, v86, 0);
        v87 = a1[1];
        v88 = *(_DWORD *)(v85 + 24);
      }
LABEL_107:
      v85 += 32LL;
      ++v86;
      sub_397C0C0(v87, v88, 0);
      if ( v112 == (_BYTE *)v85 )
        goto LABEL_60;
    }
  }
  v42 = (unsigned __int64)v131;
  v43 = 32LL * (unsigned int)v132;
  v109 = &v131[v43];
  if ( v131 != &v131[v43] )
  {
    v107 = 0;
    do
    {
      v50 = (_QWORD *)a1[1];
      v51 = *(_BYTE **)v42;
      v52 = *(_BYTE **)(v42 + 8);
      if ( !*(_QWORD *)v42 )
        v51 = (_BYTE *)v50[48];
      if ( !v52 )
        v52 = (_BYTE *)v50[49];
      if ( v37 )
      {
        v53 = v50[32];
        ++v107;
        v54 = *(void (**)())(*(_QWORD *)v53 + 104LL);
        LODWORD(v115[0]) = v107;
        v117 = ">> Call Site ";
        v120 = (const char *)&v117;
        v118 = (char *)v115[0];
        v121 = " <<";
        v119 = 2307;
        v122 = 770;
        if ( v54 != nullsub_580 )
        {
          v106 = v51;
          ((void (__fastcall *)(__int64, const char **, __int64))v54)(v53, &v120, 1);
          v50 = (_QWORD *)a1[1];
          v51 = v106;
        }
        v105 = v51;
        sub_397C140((__int64)v50);
        v55 = a1[1];
        v56 = *(_QWORD *)(v55 + 256);
        v57 = *(void (**)())(*(_QWORD *)v56 + 104LL);
        if ( (*v52 & 4) != 0 )
        {
          v58 = (__int64 *)*((_QWORD *)v52 - 1);
          v59 = *v58;
          v60 = v58 + 2;
        }
        else
        {
          v59 = 0;
          v60 = 0;
        }
        v114[0] = v60;
        v114[1] = v59;
        if ( (*v105 & 4) != 0 )
        {
          v61 = (__int64 *)*((_QWORD *)v105 - 1);
          v62 = *v61;
          v63 = v61 + 2;
        }
        else
        {
          v62 = 0;
          v63 = 0;
        }
        v113[0] = v63;
        v115[0] = "  Call between ";
        v115[1] = v113;
        v116 = 1283;
        v117 = (const char *)v115;
        v118 = " and ";
        v119 = 770;
        v113[1] = v62;
        v120 = (const char *)&v117;
        v121 = (char *)v114;
        v122 = 1282;
        if ( v57 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, const char **, __int64))v57)(v56, &v120, 1);
          v55 = a1[1];
        }
        sub_397C140(v55);
        v44 = a1[1];
        v64 = *(_QWORD *)(v42 + 16);
        v65 = *(_QWORD *)(v44 + 256);
        if ( !v64 )
        {
          v82 = *(void (**)())(*(_QWORD *)v65 + 104LL);
          v120 = "    has no landing pad";
          v122 = 259;
          if ( v82 != nullsub_580 )
          {
            ((void (__fastcall *)(__int64, const char **, __int64))v82)(v65, &v120, 1);
LABEL_91:
            v44 = a1[1];
          }
          sub_397C0C0(v44, 0, 0);
          goto LABEL_36;
        }
        v66 = *(_BYTE **)(v64 + 88);
        v67 = *(void (**)())(*(_QWORD *)v65 + 104LL);
        if ( (*v66 & 4) != 0 )
        {
          v84 = (char **)*((_QWORD *)v66 - 1);
          v68 = *v84;
          v69 = (const char *)(v84 + 2);
        }
        else
        {
          v68 = 0;
          v69 = 0;
        }
        v117 = v69;
        v120 = "    jumps to ";
        v118 = v68;
        v121 = (char *)&v117;
        v122 = 1283;
        if ( v67 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, const char **, __int64))v67)(v65, &v120, 1);
          v44 = a1[1];
        }
      }
      else
      {
        sub_397C140((__int64)v50);
        sub_397C140(a1[1]);
        if ( !*(_QWORD *)(v42 + 16) )
          goto LABEL_91;
        v44 = a1[1];
      }
      sub_397C140(v44);
LABEL_36:
      v45 = *(unsigned int *)(v42 + 24);
      v46 = a1[1];
      if ( v37 )
      {
        v47 = *(__int64 **)(v46 + 256);
        v48 = *v47;
        if ( (_DWORD)v45 )
        {
          v83 = *(void (**)())(v48 + 104);
          v122 = 2307;
          LODWORD(v117) = ((unsigned int)(v45 - 1) >> 1) + 1;
          v120 = "  On action: ";
          v121 = (char *)v117;
          if ( v83 != nullsub_580 )
          {
            ((void (__fastcall *)(__int64 *, const char **, __int64))v83)(v47, &v120, 1);
            v46 = a1[1];
            v45 = *(unsigned int *)(v42 + 24);
          }
        }
        else
        {
          v49 = *(void (**)())(v48 + 104);
          v120 = "  On action: cleanup";
          v122 = 259;
          if ( v49 != nullsub_580 )
          {
            ((void (__fastcall *)(__int64 *, const char **, __int64))v49)(v47, &v120, 1);
            v46 = a1[1];
            v45 = *(unsigned int *)(v42 + 24);
          }
        }
      }
      v42 += 32LL;
      sub_397C0C0(v46, v45, 0);
    }
    while ( v109 != (_BYTE *)v42 );
  }
LABEL_60:
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1[1] + 256) + 176LL))(
    *(_QWORD *)(a1[1] + 256),
    v104,
    0);
  v111 = &v125[3 * (unsigned int)v126];
  if ( v125 != v111 )
  {
    v70 = 0;
    for ( j = v125; v111 != j; j += 3 )
    {
      v74 = a1[1];
      if ( !v37 )
      {
        sub_397C040(v74, *j, 0);
        v72 = a1[1];
        goto LABEL_63;
      }
      v75 = *(_QWORD *)(v74 + 256);
      ++v70;
      v76 = *(void (**)())(*(_QWORD *)v75 + 104LL);
      LODWORD(v115[0]) = v70;
      v117 = ">> Action Record ";
      v119 = 2563;
      v118 = (char *)v115[0];
      v120 = (const char *)&v117;
      v121 = " <<";
      v122 = 770;
      if ( v76 != nullsub_580 )
      {
        ((void (__fastcall *)(__int64, const char **, __int64))v76)(v75, &v120, 1);
        v74 = a1[1];
        v75 = *(_QWORD *)(v74 + 256);
      }
      v77 = *(void (**)())(*(_QWORD *)v75 + 104LL);
      if ( *j > 0 )
      {
        LODWORD(v117) = *j;
        v81 = "  Catch TypeInfo ";
      }
      else
      {
        if ( !*j )
        {
          v120 = "  Cleanup";
          v122 = 259;
          if ( v77 != nullsub_580 )
            goto LABEL_75;
          goto LABEL_70;
        }
        LODWORD(v117) = *j;
        v81 = "  Filter TypeInfo ";
      }
      v120 = v81;
      v121 = (char *)v117;
      v122 = 2563;
      if ( v77 != nullsub_580 )
      {
LABEL_75:
        ((void (__fastcall *)(__int64, const char **, __int64))v77)(v75, &v120, 1);
        v74 = a1[1];
      }
LABEL_70:
      sub_397C040(v74, *j, 0);
      v78 = j[1];
      if ( v78 )
      {
        v72 = a1[1];
        v79 = *(_QWORD *)(v72 + 256);
        v80 = *(void (**)())(*(_QWORD *)v79 + 104LL);
        v120 = "  Continue to action ";
        v122 = 2307;
        LODWORD(v117) = v70 + (v78 + 1) / 2;
        v121 = (char *)v117;
        if ( v80 == nullsub_580 )
          goto LABEL_63;
LABEL_72:
        ((void (__fastcall *)(__int64, const char **, __int64))v80)(v79, &v120, 1);
        v72 = a1[1];
        goto LABEL_63;
      }
      v72 = a1[1];
      v79 = *(_QWORD *)(v72 + 256);
      v80 = *(void (**)())(*(_QWORD *)v79 + 104LL);
      v120 = "  No further actions";
      v122 = 259;
      if ( v80 != nullsub_580 )
        goto LABEL_72;
LABEL_63:
      v73 = j[1];
      sub_397C040(v72, v73, 0);
    }
  }
  if ( v103 )
  {
    sub_396F480(a1[1], 2u, 0);
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 104))(a1, v102, v101);
  }
  sub_396F480(a1[1], 2u, 0);
  if ( v131 != v133 )
    _libc_free((unsigned __int64)v131);
  if ( (_BYTE *)v123[0] != v124 )
    _libc_free(v123[0]);
  if ( v125 != (int *)v127 )
    _libc_free((unsigned __int64)v125);
  if ( src != v130 )
    _libc_free((unsigned __int64)src);
}
