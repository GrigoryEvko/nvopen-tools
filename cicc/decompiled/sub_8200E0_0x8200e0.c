// Function: sub_8200E0
// Address: 0x8200e0
//
__int64 __fastcall sub_8200E0(unsigned __int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const char *v6; // r13
  size_t v7; // r8
  unsigned int *v8; // rsi
  unsigned __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  char v14; // al
  void *v15; // rax
  unsigned __int16 v16; // ax
  int v17; // r15d
  char v18; // r15d^2
  size_t v19; // r8
  char *v20; // r9
  char *v21; // rdx
  unsigned __int16 v22; // ax
  _DWORD *v23; // rdx
  char *v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  __int16 v27; // ax
  __int16 v28; // r9
  char *v29; // rax
  char v30; // r8d^2
  char *v31; // rax
  _BYTE *v32; // r13
  __int64 v33; // r15
  __int64 v34; // rbx
  unsigned __int8 v35; // r12
  void *v36; // rdi
  unsigned __int64 v37; // rsi
  __int64 v38; // rcx
  unsigned int v39; // esi
  size_t v40; // r13
  void *v41; // r15
  int v42; // eax
  char *v43; // rax
  char v44; // r8d^2
  char *v45; // rax
  unsigned __int64 v46; // rdx
  unsigned int *v47; // rsi
  unsigned int v48; // edi
  char *v49; // rax
  char *v50; // rax
  size_t v51; // r13
  __int64 v53; // rax
  unsigned __int8 v54; // di
  size_t v55; // r13
  unsigned __int8 v56; // dl
  __int64 v57; // r12
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // rdx
  char v65; // cl
  unsigned __int16 v66; // ax
  __int64 v67; // rsi
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // rax
  __int64 *v73; // r14
  void *v74; // rax
  unsigned __int64 v75; // rdi
  __int64 v76; // rcx
  __int64 v77; // r8
  __int64 v78; // r9
  __int64 v79; // rdx
  __int64 v80; // r9
  bool v81; // cf
  bool v82; // zf
  const char *v83; // rdi
  __int64 v84; // rax
  _QWORD *v85; // r12
  __int64 v86; // rax
  __int64 v87; // rdx
  __int64 v88; // rcx
  __int64 v89; // r8
  __int64 v90; // r9
  _DWORD *v91; // rax
  unsigned __int8 v92; // di
  unsigned int v93; // esi
  __int64 v94; // rdx
  __int64 v95; // rcx
  __int64 v96; // r8
  __int64 v97; // r9
  unsigned __int64 v98; // r14
  __int64 v99; // rbx
  __int64 v100; // rdx
  _QWORD *v101; // [rsp+0h] [rbp-120h]
  int v102; // [rsp+8h] [rbp-118h]
  int v103; // [rsp+Ch] [rbp-114h]
  __int64 v104; // [rsp+10h] [rbp-110h]
  __int64 v105; // [rsp+18h] [rbp-108h]
  __int64 v106; // [rsp+20h] [rbp-100h]
  int v107; // [rsp+28h] [rbp-F8h]
  __int16 v108; // [rsp+30h] [rbp-F0h]
  __int64 v109; // [rsp+38h] [rbp-E8h]
  __int64 *v110; // [rsp+38h] [rbp-E8h]
  __int64 v111; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v112; // [rsp+48h] [rbp-D8h]
  unsigned int v113; // [rsp+5Ch] [rbp-C4h] BYREF
  unsigned __int64 v114; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v115; // [rsp+68h] [rbp-B8h] BYREF
  unsigned __int8 *v116; // [rsp+70h] [rbp-B0h] BYREF
  void *src; // [rsp+78h] [rbp-A8h] BYREF
  char *v118; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v119; // [rsp+88h] [rbp-98h] BYREF
  size_t v120; // [rsp+90h] [rbp-90h] BYREF
  size_t v121; // [rsp+98h] [rbp-88h] BYREF
  __int64 v122; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v123; // [rsp+A8h] [rbp-78h] BYREF
  _QWORD v124[2]; // [rsp+B0h] [rbp-70h] BYREF
  _QWORD v125[2]; // [rsp+C0h] [rbp-60h] BYREF
  _QWORD v126[2]; // [rsp+D0h] [rbp-50h] BYREF
  _QWORD v127[8]; // [rsp+E0h] [rbp-40h] BYREF

  v118 = 0;
  v119 = 0;
  v116 = 0;
  src = 0;
  v104 = unk_4D03BE0;
  v124[0] = unk_4D03BE0;
  v124[1] = &v116;
  v125[0] = v124;
  v125[1] = &src;
  v126[0] = v125;
  v126[1] = &v118;
  v127[0] = v126;
  v127[1] = &v119;
  unk_4D03BE0 = v127;
  if ( dword_4D03BA0 )
    v122 = *(_QWORD *)&dword_4F063F8;
  unk_4D03BA4 = 1;
  sub_7B8B50(a1, a2, a3, a4, a5, a6);
  unk_4D03BA4 = 0;
  if ( !dword_4D03BA0 )
    v122 = *(_QWORD *)&dword_4F063F8;
  if ( word_4F06418[0] != 1 )
  {
    sub_6851C0(0x28u, dword_4F07508);
    v105 = 0;
    unk_4D03CE0 = 1;
    goto LABEL_114;
  }
  v6 = qword_4F06410;
  v7 = qword_4F06400;
  v120 = qword_4F06400;
  if ( unk_4F061E4 )
  {
    v53 = sub_7B3EE0((unsigned __int8 *)qword_4F06410, &v120);
    v7 = v120;
    v6 = (const char *)v53;
    if ( !dword_4D04788 )
      goto LABEL_9;
  }
  else if ( !dword_4D04788 )
  {
    goto LABEL_9;
  }
  if ( v7 == 11 )
  {
    if ( !memcmp(v6, "__VA_ARGS__", 0xBu) )
    {
      sub_6851C0(0x3C9u, dword_4F07508);
      v7 = v120;
    }
    goto LABEL_11;
  }
LABEL_9:
  if ( unk_4D041B8 && v7 == 10 && !memcmp(v6, "__VA_OPT__", 0xAu) )
  {
    sub_6851C0(0xB7Bu, dword_4F07508);
    v7 = v120;
  }
LABEL_11:
  v8 = (unsigned int *)v7;
  v9 = (unsigned __int64)v6;
  v103 = 1;
  v105 = sub_87A510(v6, v7, &qword_4D04A00);
  if ( !v105 )
  {
    v103 = 0;
    if ( (word_4D04A10 & 0x2000) == 0 )
    {
      v8 = (unsigned int *)&qword_4D04A00;
      v9 = 1;
      qword_4D04A08 = v122;
      v105 = sub_885AD0(1, &qword_4D04A00, (unsigned int)-(dword_4F04C64 == -1), 1);
    }
  }
  v14 = *qword_4F06460;
  if ( *qword_4F06460 != 40 )
  {
    if ( dword_4F077C4 != 2 && dword_4D04964 && v14 && (unsigned int)sub_7B3970(v14) )
    {
      v123 = *(_QWORD *)&dword_4F077C8;
      sub_7B0EB0((unsigned __int64)qword_4F06460, (__int64)&v123);
      sub_6851C0(0x2FDu, &v123);
    }
    v102 = 0;
    v107 = 1;
    v112 = 0;
    v101 = 0;
    goto LABEL_17;
  }
  sub_7B8B50(v9, v8, v10, v11, v12, v13);
  sub_7B8B50(v9, v8, v58, v59, v60, v61);
  v114 = 0;
  v64 = qword_4F061C8;
  v65 = *(_BYTE *)(qword_4F061C8 + 36LL);
  *(_BYTE *)(qword_4F061C8 + 36LL) = v65 + 1;
  v66 = word_4F06418[0];
  if ( word_4F06418[0] == 28 )
  {
    v112 = 0;
    v101 = 0;
    v102 = 0;
    goto LABEL_171;
  }
  ++*(_BYTE *)(v64 + 75);
  v112 = 0;
  v110 = 0;
  while ( !dword_4D04788 || v66 != 76 )
  {
    if ( v66 != 1 )
    {
      v67 = 40;
      sub_7BE280(1u, 40, 0, 0, v62, v63);
      goto LABEL_150;
    }
    if ( sub_819210(v112, &v115) )
    {
      v67 = (__int64)dword_4F07508;
      sub_6851C0(0x31u, dword_4F07508);
      sub_7B8B50(0x31u, dword_4F07508, v94, v95, v96, v97);
      goto LABEL_150;
    }
    ++v114;
    v123 = *(_QWORD *)&dword_4F063F8;
    v72 = sub_823970(24);
    *(_QWORD *)v72 = 0;
    v73 = (__int64 *)v72;
    *(_WORD *)(v72 + 16) = 0;
    *(_QWORD *)(v72 + 8) = 0;
    v74 = (void *)sub_823970(qword_4F06400 + 1);
    *v73 = (__int64)v74;
    v75 = (unsigned __int64)v74;
    v67 = (__int64)qword_4F06410;
    memcpy(v74, qword_4F06410, qword_4F06400);
    v79 = *v73;
    *(_BYTE *)(*v73 + qword_4F06400) = 0;
    if ( v112 )
    {
      v110[1] = (__int64)v73;
      sub_7B8B50(v75, (unsigned int *)v67, v79, v76, v77, v78);
      v67 = dword_4D04780;
      if ( dword_4D04780 )
        goto LABEL_158;
    }
    else
    {
      sub_7B8B50(v75, (unsigned int *)v67, v79, v76, v77, v78);
      v112 = (unsigned __int64)v73;
      if ( dword_4D04780 )
      {
LABEL_158:
        if ( word_4F06418[0] == 76 )
        {
          v101 = v73;
          sub_7B8B50(v75, (unsigned int *)v67, v68, v69, v70, v80);
          v102 = 1;
          goto LABEL_168;
        }
      }
    }
    v71 = dword_4D04788;
    v110 = v73;
    v81 = 0;
    v82 = dword_4D04788 == 0;
    if ( dword_4D04788 )
    {
      v67 = *v73;
      v69 = 12;
      v83 = "__VA_ARGS__";
      do
      {
        if ( !v69 )
          break;
        v81 = *(_BYTE *)v67 < *v83;
        v82 = *(_BYTE *)v67++ == *v83++;
        --v69;
      }
      while ( v82 );
      if ( (!v81 && !v82) == v81 )
      {
        v67 = (__int64)&v123;
        sub_6851C0(0x3C9u, &v123);
      }
    }
LABEL_150:
    if ( !(unsigned int)sub_7BE800(0x43u, (unsigned int *)v67, v68, v69, v70, v71) )
    {
      v102 = 0;
      v101 = v110;
      goto LABEL_168;
    }
    v66 = word_4F06418[0];
  }
  ++v114;
  v123 = *(_QWORD *)&dword_4F063F8;
  v84 = sub_823970(24);
  *(_QWORD *)v84 = 0;
  v85 = (_QWORD *)v84;
  *(_QWORD *)(v84 + 8) = 0;
  *(_WORD *)(v84 + 16) = 0;
  v101 = (_QWORD *)v84;
  v86 = sub_823970(12);
  *v85 = v86;
  *(_QWORD *)v86 = 0x4752415F41565F5FLL;
  *(_DWORD *)(v86 + 8) = (_DWORD)&loc_5F5F53;
  if ( v112 )
  {
    v110[1] = (__int64)v101;
    sub_7B8B50(0xCu, (unsigned int *)0x4752415F41565F5FLL, v87, v88, v89, v90);
  }
  else
  {
    sub_7B8B50(0xCu, (unsigned int *)0x4752415F41565F5FLL, v87, v88, v89, v90);
    v112 = (unsigned __int64)v101;
  }
  v102 = 1;
LABEL_168:
  v64 = qword_4F061C8;
  --*(_BYTE *)(qword_4F061C8 + 75LL);
  if ( word_4F06418[0] != 28 )
  {
    sub_6851C0(0x12u, dword_4F07508);
    v64 = qword_4F061C8;
  }
  v65 = *(_BYTE *)(v64 + 36) - 1;
LABEL_171:
  *(_BYTE *)(v64 + 36) = v65;
  v107 = 0;
LABEL_17:
  if ( qword_4D042B8 && !dword_4D042B0[0] )
  {
    if ( HIDWORD(qword_4F077B4) )
    {
      sub_7BC390();
      if ( !dword_4F063EC )
      {
        v57 = qword_4D042B8;
        qword_4D042B8 = 0;
        sub_685190(0x670u, *(_QWORD *)(qword_4D04A00 + 8));
        qword_4D042B8 = v57;
      }
      goto LABEL_19;
    }
    if ( *qword_4F06460 != 61 )
      sub_684920(0x3E0u, qword_4D042B8);
    ++qword_4F06460;
    if ( !qword_4F06438 )
      goto LABEL_111;
  }
  else
  {
LABEL_19:
    if ( qword_4F06438 )
      goto LABEL_20;
LABEL_111:
    if ( !qword_4D03BD8 )
    {
      v15 = qword_4F195B0;
      qword_4F19590 = 0;
      qword_4F19598 = 0;
      qword_4F195A0 = qword_4F195B0;
      goto LABEL_21;
    }
  }
LABEL_20:
  v15 = qword_4F195A0;
LABEL_21:
  qword_4F19588 = v15;
  src = v15;
  qword_4F194B8 = 0;
  v116 = 0;
  sub_819F50(v112, (unsigned int *)&v114, &v115, &v113);
  v16 = word_4F06418[0];
  if ( qword_4D042B8 || word_4F06418[0] == 10 || !v107 )
    goto LABEL_22;
  if ( v113 )
  {
    v113 = 0;
    goto LABEL_23;
  }
  v54 = 5;
  if ( dword_4D04964 )
  {
    if ( dword_4F077C4 == 2 )
    {
      if ( unk_4F07778 > 201102 || dword_4F07774 )
LABEL_128:
        v54 = unk_4F07471;
    }
    else if ( unk_4F07778 > 199900 )
    {
      goto LABEL_128;
    }
  }
  sub_6849F0(v54, 0x671u, &dword_4F063F8, *(_QWORD *)(qword_4D04A00 + 8));
  v16 = word_4F06418[0];
LABEL_22:
  v113 = 0;
  if ( v16 != 10 )
  {
LABEL_23:
    v17 = 0;
    v109 = 0;
    while ( 1 )
    {
      v21 = v118;
      if ( !v118 )
        goto LABEL_46;
      if ( v16 == 27 )
      {
        ++v109;
        v22 = word_4F06418[0];
      }
      else
      {
        if ( v16 != 28 )
        {
          if ( v16 == 1 && qword_4F06400 == 10 && !memcmp(qword_4F06410, "__VA_OPT__", 0xAu) )
          {
            sub_6851C0(0xB7Cu, &dword_4F063F8);
            v49 = v118;
            *(_WORD *)(v118 + 1) = 0;
            v49[3] = 0;
            v118 = 0;
            sub_819F50(v112, (unsigned int *)&v114, &v115, &v113);
          }
LABEL_46:
          v22 = word_4F06418[0];
          goto LABEL_47;
        }
        if ( --v109 )
          goto LABEL_46;
        v42 = (_DWORD)qword_4F195A0 - (_DWORD)v118 - 4;
        *(_WORD *)(v118 + 1) = v42;
        v21[3] = BYTE2(v42);
        v118 = 0;
        sub_819F50(v112, (unsigned int *)&v114, &v115, &v113);
        v22 = word_4F06418[0];
        v116 = 0;
        if ( word_4F06418[0] == 10 )
          break;
      }
LABEL_47:
      if ( v22 == 69 )
      {
        if ( qword_4F195A0 == src )
        {
          v47 = dword_4F07508;
          v48 = 50;
LABEL_98:
          sub_6851C0(v48, v47);
          v17 = 0;
          sub_819F50(v112, (unsigned int *)&v114, &v115, &v113);
          v16 = word_4F06418[0];
          goto LABEL_41;
        }
        if ( (unsigned __int16)sub_819F50(v112, (unsigned int *)&v114, &v115, &v113) == 28 && v118 && v109 == 1 )
        {
          sub_6851C0(0xB80u, dword_4F07508);
          v50 = v118;
          *(_WORD *)(v118 + 1) = 0;
          v50[3] = 0;
          v118 = 0;
          sub_819F50(v112, (unsigned int *)&v114, &v115, &v113);
          if ( word_4F06418[0] == 10 )
          {
LABEL_104:
            v17 = 0;
            sub_6851C0(0x33u, dword_4F07508);
            v16 = word_4F06418[0];
            goto LABEL_41;
          }
        }
        else if ( word_4F06418[0] == 10 )
        {
          goto LABEL_104;
        }
        v23 = qword_4F195A0;
        v116 = 0;
        if ( (unsigned __int64)(qword_4F195A8 - (_QWORD)qword_4F195A0) <= 3 )
        {
          sub_81AC10(4u);
          v23 = qword_4F195A0;
        }
        *v23 = 2;
        v24 = (char *)(v23 + 1);
        qword_4F195A0 = v23 + 1;
        if ( !v114 )
        {
          v113 = 0;
          v16 = word_4F06418[0];
          v17 = 0;
          goto LABEL_41;
        }
        v116 = 0;
        if ( (unsigned __int64)(qword_4F195A8 - (_QWORD)v24) <= 3 )
        {
          sub_81AC10(4u);
          v24 = (char *)qword_4F195A0;
        }
        *v24 = 3;
        v25 = v114 >> 16;
        *(_WORD *)(v24 + 1) = v114;
        v24[3] = v25;
        qword_4F195A0 = v24 + 4;
        *(_BYTE *)(v115 + 17) = 1;
        goto LABEL_40;
      }
      if ( v17 && !dword_4D04954 )
        sub_81B3B0(&unk_4B7C358, 2u, &v116);
      if ( v113 )
      {
        if ( qword_4F194B0 )
          sub_81B3B0(qword_4F194B0, qword_4F06410 - (_BYTE *)qword_4F194B0, &v116);
        else
          sub_81B3B0(" ", 1u, &v116);
        v113 = 0;
      }
      if ( word_4F06418[0] == 68 && (unsigned __int8)v107 != 1 && !qword_4F194B8 )
      {
        sub_819F50(v112, (unsigned int *)&v114, &v115, &v113);
        if ( word_4F06418[0] == 1 && unk_4D041B8 && qword_4F06400 == 10 && !memcmp(qword_4F06410, "__VA_OPT__", 0xAu) )
        {
          v114 = 0xFFFFFF;
        }
        else if ( !v114 )
        {
          v17 = 0;
          sub_6851C0(0x34u, dword_4F07508);
          v16 = word_4F06418[0];
          goto LABEL_41;
        }
        v45 = (char *)qword_4F195A0;
        v116 = 0;
        if ( (unsigned __int64)(qword_4F195A8 - (_QWORD)qword_4F195A0) <= 3 )
        {
          sub_81AC10(4u);
          v45 = (char *)qword_4F195A0;
        }
        *v45 = 4;
        v46 = v114 >> 16;
        *(_WORD *)(v45 + 1) = v114;
        v45[3] = v46;
        qword_4F195A0 = v45 + 4;
        if ( v114 == 0xFFFFFF )
        {
LABEL_92:
          if ( v102 )
          {
            if ( (unsigned __int16)sub_819F50(v112, (unsigned int *)&v114, &v115, &v113) == 27 )
            {
              v91 = qword_4F195A0;
              v116 = 0;
              v118 = (char *)qword_4F195A0;
              v123 = *(_QWORD *)&dword_4F063F8;
              if ( (unsigned __int64)(qword_4F195A8 - (_QWORD)qword_4F195A0) <= 3 )
              {
                sub_81AC10(4u);
                v91 = qword_4F195A0;
              }
              *v91 = 9;
              qword_4F195A0 = v91 + 1;
              sub_819F50(v112, (unsigned int *)&v114, &v115, &v113);
              v16 = word_4F06418[0];
              if ( word_4F06418[0] == 69 )
              {
                sub_6851C0(0xB7Fu, dword_4F07508);
                sub_819F50(v112, (unsigned int *)&v114, &v115, &v113);
                v16 = word_4F06418[0];
              }
              v17 = 0;
              v109 = 1;
              *((_BYTE *)v101 + 16) = 1;
            }
            else
            {
              v17 = 0;
              sub_6851C0(0xB7Eu, &dword_4F063F8);
              v16 = word_4F06418[0];
            }
            goto LABEL_41;
          }
          v47 = &dword_4F063F8;
          v48 = 2939;
          goto LABEL_98;
        }
LABEL_40:
        v17 = 1;
        sub_819F50(v112, (unsigned int *)&v114, &v115, &v113);
        v16 = word_4F06418[0];
LABEL_41:
        if ( v16 == 10 )
          break;
      }
      else
      {
        v18 = BYTE2(v114);
        if ( !v114 )
        {
          v19 = qword_4F06400;
          v20 = (char *)qword_4F06410;
          if ( unk_4D041B8 )
          {
            if ( word_4F06418[0] != 1 )
            {
              v121 = qword_4F06400;
              goto LABEL_37;
            }
            if ( qword_4F06400 == 10 && !memcmp(qword_4F06410, "__VA_OPT__", 0xAu) )
              goto LABEL_92;
            v121 = qword_4F06400;
            if ( !unk_4F061E4 )
              goto LABEL_37;
          }
          else
          {
            v121 = qword_4F06400;
            if ( word_4F06418[0] != 1 || !unk_4F061E4 )
            {
LABEL_37:
              sub_81B3B0(v20, v19, &v116);
              if ( !word_4F06418[0] && !qword_4F194B8 )
                sub_684B00(unk_4F06208, dword_4F07508);
              goto LABEL_40;
            }
          }
          v26 = sub_7B3EE0((unsigned __int8 *)qword_4F06410, &v121);
          v19 = v121;
          v20 = (char *)v26;
          goto LABEL_37;
        }
        v108 = v114;
        v106 = v115;
        v27 = sub_819F50(v112, (unsigned int *)&v114, &v115, &v113);
        v28 = v108;
        if ( v27 == 69 || dword_4D04954 )
        {
          v43 = (char *)qword_4F195A0;
          v116 = 0;
          if ( (unsigned __int64)(qword_4F195A8 - (_QWORD)qword_4F195A0) <= 3 )
          {
            sub_81AC10(4u);
            v43 = (char *)qword_4F195A0;
            v28 = v108;
          }
          v44 = v18;
          *v43 = 3;
          v17 = 0;
          *(_WORD *)(v43 + 1) = v28;
          v43[3] = v44;
          qword_4F195A0 = v43 + 4;
          *(_BYTE *)(v106 + 17) = 1;
          v16 = word_4F06418[0];
          goto LABEL_41;
        }
        v29 = (char *)qword_4F195A0;
        v116 = 0;
        if ( (unsigned __int64)(qword_4F195A8 - (_QWORD)qword_4F195A0) <= 3 )
        {
          sub_81AC10(4u);
          v29 = (char *)qword_4F195A0;
          v28 = v108;
        }
        v30 = v18;
        *v29 = 6;
        v17 = 1;
        *(_WORD *)(v29 + 1) = v28;
        v29[3] = v30;
        qword_4F195A0 = v29 + 4;
        *(_BYTE *)(v106 + 16) = 1;
        v16 = word_4F06418[0];
        if ( word_4F06418[0] == 10 )
          break;
      }
    }
  }
  if ( v118 )
  {
    sub_6851C0(0xB7Du, &v123);
    v31 = v118;
    *(_WORD *)(v118 + 1) = 0;
    v31[3] = 0;
    v118 = 0;
  }
  v32 = qword_4F195A0;
  qword_4F194B8 = 0;
  *(_BYTE *)qword_4F195A0 = 0;
  if ( !v103 )
  {
    v51 = v32 - (_BYTE *)src;
    v41 = (void *)sub_823970(v51 + 1);
    memcpy(v41, src, v51);
    *((_BYTE *)v41 + v51) = 0;
    if ( !v105 )
      goto LABEL_114;
LABEL_134:
    v34 = sub_823970(24);
    sub_81B550((unsigned __int8 *)v34);
    goto LABEL_135;
  }
  v33 = qword_4D042B8;
  v34 = *(_QWORD *)(v105 + 88);
  if ( !qword_4D042B8 )
  {
    v35 = *(_BYTE *)v34;
    v36 = src;
    if ( v107 != (*(_BYTE *)v34 & 1) )
      goto LABEL_74;
    v37 = v32 - (_BYTE *)src;
    if ( v102 != ((v35 & 8) != 0) )
      goto LABEL_74;
    goto LABEL_73;
  }
  if ( dword_4D042B0[0] )
  {
    v35 = *(_BYTE *)v34;
    v36 = src;
    v37 = v32 - (_BYTE *)src;
    if ( v107 != (*(_BYTE *)v34 & 1) || v102 != ((v35 & 8) != 0) )
      goto LABEL_222;
    goto LABEL_73;
  }
  v35 = *(_BYTE *)v34;
  if ( (*(_BYTE *)v34 & 2) == 0 )
  {
    v55 = v32 - (_BYTE *)src;
    v41 = (void *)sub_823970(v55 + 1);
    memcpy(v41, src, v55);
    *((_BYTE *)v41 + v55) = 0;
    goto LABEL_134;
  }
  v36 = src;
  if ( v107 != (v35 & 1) )
  {
    if ( !HIDWORD(qword_4F077B4) )
      goto LABEL_178;
    goto LABEL_187;
  }
  v37 = v32 - (_BYTE *)src;
  if ( v102 != ((v35 & 8) != 0) )
  {
LABEL_176:
    if ( !HIDWORD(qword_4F077B4) )
    {
      if ( v33 )
      {
LABEL_178:
        v38 = v105;
        v39 = 46;
        goto LABEL_78;
      }
LABEL_204:
      qword_4D042B8 = 0;
      sub_6853B0(7u, 0x2Eu, (FILE *)&v122, v105);
      qword_4D042B8 = v33;
      goto LABEL_114;
    }
LABEL_187:
    v92 = 5;
    v93 = 1561;
    goto LABEL_188;
  }
LABEL_73:
  if ( !sub_81A4B0((__int64)v36, v37, *(_BYTE **)(v34 + 16)) )
    goto LABEL_74;
  v100 = *(_QWORD *)(v34 + 8);
  if ( v112 )
  {
    v98 = v112;
    if ( v100 )
    {
      v111 = v34;
      v99 = *(_QWORD *)(v34 + 8);
      while ( !strcmp(*(const char **)v98, *(const char **)v99) )
      {
        v98 = *(_QWORD *)(v98 + 8);
        v99 = *(_QWORD *)(v99 + 8);
        if ( !v98 || !v99 )
        {
          v100 = v99;
          v34 = v111;
          goto LABEL_201;
        }
      }
      v34 = v111;
      goto LABEL_74;
    }
  }
  v98 = v112;
LABEL_201:
  if ( v100 != v98 )
  {
LABEL_74:
    if ( !dword_4D042B0[0] )
    {
      if ( (v35 & 2) == 0 )
      {
        if ( v33 )
        {
          v38 = v105;
          v39 = 47;
LABEL_78:
          sub_6853B0(8u, v39, (FILE *)&v122, v38);
          v32 = qword_4F195A0;
          qword_4D042B8 = v33;
LABEL_79:
          v40 = v32 - (_BYTE *)src;
          v41 = (void *)sub_823970(v40 + 1);
          memcpy(v41, src, v40);
          *((_BYTE *)v41 + v40) = 0;
          sub_81B550((unsigned __int8 *)v34);
LABEL_135:
          v56 = *(_BYTE *)v34;
          *(_QWORD *)(v34 + 16) = v41;
          *(_QWORD *)(v34 + 8) = v112;
          *(_BYTE *)v34 = v56 & 0xF6 | (v107 | (8 * v102)) & 9;
          *(_BYTE *)v34 = (16 * (dword_4D042B0[0] & 1)) | v56 & 0xE6 | (v107 | (8 * v102)) & 9;
          *(_BYTE *)(v105 + 81) &= ~2u;
          *(_QWORD *)(v105 + 88) = v34;
          sub_8756F0(3, v105, &v122, 0);
          goto LABEL_114;
        }
        if ( dword_4D04964 )
        {
          if ( byte_4F07472[0] == 3 )
            goto LABEL_79;
          v92 = byte_4F07472[0];
          v93 = 47;
          if ( byte_4F07472[0] > 7u )
          {
            sub_6853B0(byte_4F07472[0], 0x2Fu, (FILE *)&v122, v105);
            v32 = qword_4F195A0;
            qword_4D042B8 = 0;
            goto LABEL_79;
          }
        }
        else
        {
          v92 = 5;
          v93 = 47;
        }
LABEL_188:
        qword_4D042B8 = 0;
        sub_6853B0(v92, v93, (FILE *)&v122, v105);
        v32 = qword_4F195A0;
        qword_4D042B8 = v33;
        goto LABEL_79;
      }
      goto LABEL_176;
    }
LABEL_222:
    sub_685220(0x53Au, *(_QWORD *)(*(_QWORD *)v105 + 8LL));
  }
  if ( !v33 )
  {
    if ( (v35 & 2) == 0 )
      goto LABEL_114;
    goto LABEL_204;
  }
  if ( !dword_4D042B0[0] && (v35 & 2) == 0 )
    goto LABEL_79;
LABEL_114:
  qword_4F19588 = 0;
  unk_4D03BE0 = v104;
  return v105;
}
