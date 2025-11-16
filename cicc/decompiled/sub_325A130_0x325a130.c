// Function: sub_325A130
// Address: 0x325a130
//
void __fastcall sub_325A130(__int64 a1, __int64 a2)
{
  __int64 v3; // r15
  const char *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  bool v7; // zf
  __int64 v8; // rdi
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 (*v12)(); // rdx
  unsigned __int8 *v13; // rax
  unsigned __int8 *v14; // rax
  unsigned __int8 *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  void (*v18)(); // rcx
  void (*v19)(); // rax
  __int64 v20; // rax
  int *v21; // rbx
  __int64 v22; // r13
  __int64 v23; // rax
  void (*v24)(); // r9
  void (*v25)(); // rax
  unsigned __int8 *v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // rdi
  __int64 v29; // r12
  __int64 v30; // rbx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rax
  void (*v34)(); // rcx
  __int64 v35; // rax
  void (*v36)(); // rcx
  __int64 v37; // rax
  void (*v38)(); // rcx
  __int64 v39; // rax
  void (*v40)(); // rcx
  void (*v41)(); // rax
  unsigned __int8 *v42; // rsi
  __int64 v43; // r13
  __int64 v44; // r12
  int *v45; // rbx
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rdi
  _BYTE *v50; // rdi
  unsigned __int8 **v51; // rdi
  unsigned __int8 **v52; // rbx
  unsigned __int8 **v53; // r14
  void (*v54)(); // rcx
  __int64 v55; // rcx
  void (*v56)(); // r9
  __int64 v57; // rsi
  __int64 v58; // rsi
  __int64 v59; // rbx
  int *v60; // r13
  __int64 v61; // rax
  void (*v62)(); // rcx
  void (*v63)(); // rax
  __int64 v64; // rsi
  __int64 v65; // rdi
  __int64 v66; // rax
  unsigned __int8 *v67; // rsi
  void (*v68)(); // rax
  void (*v69)(); // rax
  unsigned __int8 *v70; // rax
  unsigned int v71; // esi
  int v72; // eax
  __int64 v73; // rax
  unsigned __int64 v74; // rdi
  __int64 v75; // r12
  __int64 v76; // rax
  void (*v77)(); // rcx
  __int64 v78; // rax
  void (*v79)(); // rcx
  void (*v80)(); // rax
  unsigned __int8 *v81; // rax
  __int64 v82; // rax
  void (*v83)(); // rcx
  void (*v84)(); // rax
  unsigned __int8 *v85; // rax
  __int64 v86; // rax
  void (*v87)(); // rcx
  void (*v88)(); // rax
  __int64 (*v89)(); // rax
  __int64 v90; // rax
  unsigned __int8 *v91; // rax
  __int64 v92; // rax
  void (*v93)(); // rcx
  int v94; // eax
  unsigned int v95; // esi
  int v96; // eax
  __int64 v97; // rax
  void (*v98)(); // rcx
  __int64 v99; // rdi
  __int64 v100; // rdi
  __int64 v101; // rdi
  __int64 v102; // [rsp+18h] [rbp-188h]
  __int64 v104; // [rsp+20h] [rbp-180h]
  __int64 v105; // [rsp+28h] [rbp-178h]
  unsigned int v106; // [rsp+28h] [rbp-178h]
  const char *v107; // [rsp+38h] [rbp-168h]
  __int64 v108; // [rsp+38h] [rbp-168h]
  const char *v109; // [rsp+40h] [rbp-160h]
  int *v110; // [rsp+40h] [rbp-160h]
  __int64 v111; // [rsp+48h] [rbp-158h]
  int v112; // [rsp+50h] [rbp-150h]
  int *v113; // [rsp+50h] [rbp-150h]
  __int64 v114; // [rsp+50h] [rbp-150h]
  unsigned __int8 *v115; // [rsp+50h] [rbp-150h]
  char v116; // [rsp+5Fh] [rbp-141h]
  __int64 v117; // [rsp+68h] [rbp-138h] BYREF
  _BYTE *v118; // [rsp+70h] [rbp-130h] BYREF
  __int64 v119; // [rsp+78h] [rbp-128h]
  _BYTE v120[16]; // [rsp+80h] [rbp-120h] BYREF
  _QWORD v121[4]; // [rsp+90h] [rbp-110h] BYREF
  __int16 v122; // [rsp+B0h] [rbp-F0h]
  _QWORD v123[4]; // [rsp+C0h] [rbp-E0h] BYREF
  __int16 v124; // [rsp+E0h] [rbp-C0h]
  const char *v125[2]; // [rsp+F0h] [rbp-B0h] BYREF
  const char *v126; // [rsp+100h] [rbp-A0h]
  const char *v127; // [rsp+108h] [rbp-98h]
  __int16 v128; // [rsp+110h] [rbp-90h]
  unsigned __int8 **v129; // [rsp+120h] [rbp-80h] BYREF
  __int64 v130; // [rsp+128h] [rbp-78h]
  _BYTE v131[112]; // [rsp+130h] [rbp-70h] BYREF

  v111 = *(_QWORD *)(a2 + 88);
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL);
  v4 = sub_BD5D20(*(_QWORD *)a2);
  v107 = v4;
  v109 = (const char *)v5;
  if ( v5 && *v4 == 1 )
  {
    v6 = v5 - 1;
    v107 = v4 + 1;
    v109 = (const char *)(v5 - 1);
  }
  v7 = *(_BYTE *)(a1 + 24) == 0;
  v129 = (unsigned __int8 **)v131;
  v130 = 0x400000000LL;
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 216LL);
  if ( v7 )
  {
    v128 = 261;
    v125[0] = v107;
    v125[1] = v109;
    v9 = sub_E6C840(v8, (__int64)v125, v5, v6);
  }
  else
  {
    v128 = 1283;
    v125[0] = "$cppxdata$";
    v126 = v107;
    v127 = v109;
    v9 = sub_E6C460(v8, v125);
    sub_3259C30(a1, a2, v111, (__int64)&v129);
  }
  v112 = 0;
  v10 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 208LL);
  if ( *(_DWORD *)(v10 + 336) == 4 )
  {
    v94 = *(_DWORD *)(v10 + 344);
    if ( v94 == 6 || !v94 )
    {
      v112 = 0;
    }
    else
    {
      v95 = *(_DWORD *)(v111 + 736);
      if ( v95 != 0x7FFFFFFF )
        v112 = sub_32590E0(a1, v95, v111);
    }
  }
  v11 = 0;
  if ( *(_DWORD *)(v111 + 168) )
  {
    v101 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 216LL);
    v128 = 1283;
    v125[0] = "$stateUnwindMap$";
    v126 = v107;
    v127 = v109;
    v11 = sub_E6C460(v101, v125);
  }
  v105 = 0;
  if ( *(_DWORD *)(v111 + 248) )
  {
    v100 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 216LL);
    v128 = 1283;
    v125[0] = "$tryMap$";
    v126 = v107;
    v127 = v109;
    v105 = sub_E6C460(v100, v125);
  }
  v102 = 0;
  if ( (_DWORD)v130 )
  {
    v99 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 216LL);
    v128 = 1283;
    v125[0] = "$ip2state$";
    v126 = v107;
    v127 = v109;
    v102 = sub_E6C460(v99, v125);
  }
  v12 = *(__int64 (**)())(*(_QWORD *)v3 + 96LL);
  if ( v12 == sub_C13EE0 )
  {
    (*(void (__fastcall **)(__int64, __int64, _QWORD, __int64, _QWORD))(*(_QWORD *)v3 + 608LL))(v3, 2, 0, 1, 0);
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v3 + 208LL))(v3, v9, 0);
LABEL_15:
    (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v3 + 536LL))(v3, 429065506, 4);
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v3 + 536LL))(v3, *(unsigned int *)(v111 + 168), 4);
    v13 = (unsigned __int8 *)sub_3258F50(a1, v11);
    sub_E9A5B0(v3, v13);
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v3 + 536LL))(v3, *(unsigned int *)(v111 + 248), 4);
    v14 = (unsigned __int8 *)sub_3258F50(a1, v105);
    sub_E9A5B0(v3, v14);
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v3 + 536LL))(v3, (unsigned int)v130, 4);
    v116 = 0;
    goto LABEL_16;
  }
  v116 = ((__int64 (__fastcall *)(__int64))v12)(v3);
  (*(void (__fastcall **)(__int64, __int64, _QWORD, __int64, _QWORD))(*(_QWORD *)v3 + 608LL))(v3, 2, 0, 1, 0);
  (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v3 + 208LL))(v3, v9, 0);
  v125[0] = "MagicNumber";
  v128 = 259;
  if ( !v116 )
    goto LABEL_15;
  v76 = *(_QWORD *)v3;
  v77 = *(void (**)())(*(_QWORD *)v3 + 120LL);
  if ( v77 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, const char **, __int64))v77)(v3, v125, 1);
    v76 = *(_QWORD *)v3;
  }
  (*(void (__fastcall **)(__int64, __int64, __int64))(v76 + 536))(v3, 429065506, 4);
  v125[0] = "MaxState";
  v128 = 259;
  v78 = *(_QWORD *)v3;
  v79 = *(void (**)())(*(_QWORD *)v3 + 120LL);
  if ( v79 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, const char **, __int64))v79)(v3, v125, 1);
    v78 = *(_QWORD *)v3;
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(v78 + 536))(v3, *(unsigned int *)(v111 + 168), 4);
  v125[0] = "UnwindMap";
  v128 = 259;
  v80 = *(void (**)())(*(_QWORD *)v3 + 120LL);
  if ( v80 != nullsub_98 )
    ((void (__fastcall *)(__int64, const char **, __int64))v80)(v3, v125, 1);
  v81 = (unsigned __int8 *)sub_3258F50(a1, v11);
  sub_E9A5B0(v3, v81);
  v125[0] = "NumTryBlocks";
  v128 = 259;
  v82 = *(_QWORD *)v3;
  v83 = *(void (**)())(*(_QWORD *)v3 + 120LL);
  if ( v83 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, const char **, __int64))v83)(v3, v125, 1);
    v82 = *(_QWORD *)v3;
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(v82 + 536))(v3, *(unsigned int *)(v111 + 248), 4);
  v125[0] = "TryBlockMap";
  v128 = 259;
  v84 = *(void (**)())(*(_QWORD *)v3 + 120LL);
  if ( v84 != nullsub_98 )
    ((void (__fastcall *)(__int64, const char **, __int64))v84)(v3, v125, 1);
  v85 = (unsigned __int8 *)sub_3258F50(a1, v105);
  sub_E9A5B0(v3, v85);
  v125[0] = "IPMapEntries";
  v128 = 259;
  v86 = *(_QWORD *)v3;
  v87 = *(void (**)())(*(_QWORD *)v3 + 120LL);
  if ( v87 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64, const char **, __int64))v87)(v3, v125, 1);
    v86 = *(_QWORD *)v3;
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(v86 + 536))(v3, (unsigned int)v130, 4);
  v125[0] = "IPToStateXData";
  v128 = 259;
  v88 = *(void (**)())(*(_QWORD *)v3 + 120LL);
  if ( v88 != nullsub_98 )
    ((void (__fastcall *)(__int64, const char **, __int64))v88)(v3, v125, 1);
LABEL_16:
  v15 = (unsigned __int8 *)sub_3258F50(a1, v102);
  sub_E9A5B0(v3, v15);
  v16 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 208LL);
  if ( *(_DWORD *)(v16 + 336) == 4 )
  {
    v96 = *(_DWORD *)(v16 + 344);
    if ( v96 )
    {
      if ( v96 != 6 && *(_DWORD *)(v111 + 736) != 0x7FFFFFFF )
      {
        v125[0] = "UnwindHelp";
        v128 = 259;
        v97 = *(_QWORD *)v3;
        if ( v116 )
        {
          v98 = *(void (**)())(v97 + 120);
          if ( v98 != nullsub_98 )
          {
            ((void (__fastcall *)(__int64, const char **, __int64))v98)(v3, v125, 1);
            v97 = *(_QWORD *)v3;
          }
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(v97 + 536))(v3, v112, 4);
      }
    }
  }
  v125[0] = "ESTypeList";
  v128 = 259;
  v17 = *(_QWORD *)v3;
  if ( v116 )
  {
    v18 = *(void (**)())(v17 + 120);
    if ( v18 != nullsub_98 )
    {
      ((void (__fastcall *)(__int64, const char **, __int64))v18)(v3, v125, 1);
      v17 = *(_QWORD *)v3;
    }
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(v17 + 536))(v3, 0, 4);
    v125[0] = "EHFlags";
    v128 = 259;
    v19 = *(void (**)())(*(_QWORD *)v3 + 120LL);
    if ( v19 != nullsub_98 )
      ((void (__fastcall *)(__int64, const char **, __int64))v19)(v3, v125, 1);
  }
  else
  {
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(v17 + 536))(v3, 0, 4);
  }
  v7 = sub_BA91D0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 2488LL), "eh-asynch", 9u) == 0;
  v20 = *(_QWORD *)v3;
  if ( v7 )
    (*(void (__fastcall **)(__int64, __int64, __int64))(v20 + 536))(v3, 1, 4);
  else
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(v20 + 536))(v3, 0, 4);
  if ( v11 )
  {
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v3 + 208LL))(v3, v11, 0);
    v21 = *(int **)(v111 + 160);
    v22 = 4LL * *(unsigned int *)(v111 + 168);
    v113 = &v21[v22];
    while ( v113 != v21 )
    {
      v27 = *((_QWORD *)v21 + 1);
      v28 = 0;
      if ( v27 )
      {
        v28 = v27 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v27 & 4) == 0 )
          v28 = 0;
      }
      v29 = sub_3258B50(v28);
      v128 = 259;
      v125[0] = "ToState";
      if ( v116 )
      {
        v23 = *(_QWORD *)v3;
        v24 = *(void (**)())(*(_QWORD *)v3 + 120LL);
        if ( v24 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, const char **, __int64))v24)(v3, v125, 1);
          v23 = *(_QWORD *)v3;
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(v23 + 536))(v3, *v21, 4);
        v125[0] = "Action";
        v128 = 259;
        v25 = *(void (**)())(*(_QWORD *)v3 + 120LL);
        if ( v25 != nullsub_98 )
          ((void (__fastcall *)(__int64, const char **, __int64))v25)(v3, v125, 1);
      }
      else
      {
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v3 + 536LL))(v3, *v21, 4);
      }
      v21 += 4;
      v26 = (unsigned __int8 *)sub_3258F50(a1, v29);
      sub_E9A5B0(v3, v26);
    }
  }
  if ( v105 )
  {
    v30 = 0;
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v3 + 208LL))(v3, v105, 0);
    v117 = 0;
    v118 = v120;
    v119 = 0x100000000LL;
    v114 = *(unsigned int *)(v111 + 248);
    if ( *(_DWORD *)(v111 + 248) )
    {
      do
      {
        v43 = 0;
        v44 = 0;
        v45 = (int *)(*(_QWORD *)(v111 + 240) + (v30 << 6));
        if ( v45[6] )
        {
          v48 = *(_QWORD *)(a1 + 8);
          v124 = 770;
          v49 = *(_QWORD *)(v48 + 216);
          v128 = 1282;
          v121[0] = "$handlerMap$";
          v121[2] = &v117;
          v122 = 2819;
          v123[0] = v121;
          v123[2] = "$";
          v125[0] = (const char *)v123;
          v126 = v107;
          v127 = v109;
          v44 = sub_E6C460(v49, v125);
          v43 = v44;
        }
        v46 = (unsigned int)v119;
        v47 = (unsigned int)v119 + 1LL;
        if ( v47 > HIDWORD(v119) )
        {
          sub_C8D5F0((__int64)&v118, v120, v47, 8u, v31, v32);
          v46 = (unsigned int)v119;
        }
        *(_QWORD *)&v118[8 * v46] = v43;
        LODWORD(v119) = v119 + 1;
        v125[0] = "TryLow";
        v128 = 259;
        if ( v116 )
        {
          v33 = *(_QWORD *)v3;
          v34 = *(void (**)())(*(_QWORD *)v3 + 120LL);
          if ( v34 != nullsub_98 )
          {
            ((void (__fastcall *)(__int64, const char **, __int64))v34)(v3, v125, 1);
            v33 = *(_QWORD *)v3;
          }
          (*(void (__fastcall **)(__int64, _QWORD, __int64))(v33 + 536))(v3, *v45, 4);
          v125[0] = "TryHigh";
          v128 = 259;
          v35 = *(_QWORD *)v3;
          v36 = *(void (**)())(*(_QWORD *)v3 + 120LL);
          if ( v36 != nullsub_98 )
          {
            ((void (__fastcall *)(__int64, const char **, __int64))v36)(v3, v125, 1);
            v35 = *(_QWORD *)v3;
          }
          (*(void (__fastcall **)(__int64, _QWORD, __int64))(v35 + 536))(v3, v45[1], 4);
          v125[0] = "CatchHigh";
          v128 = 259;
          v37 = *(_QWORD *)v3;
          v38 = *(void (**)())(*(_QWORD *)v3 + 120LL);
          if ( v38 != nullsub_98 )
          {
            ((void (__fastcall *)(__int64, const char **, __int64))v38)(v3, v125, 1);
            v37 = *(_QWORD *)v3;
          }
          (*(void (__fastcall **)(__int64, _QWORD, __int64))(v37 + 536))(v3, v45[2], 4);
          v125[0] = "NumCatches";
          v128 = 259;
          v39 = *(_QWORD *)v3;
          v40 = *(void (**)())(*(_QWORD *)v3 + 120LL);
          if ( v40 != nullsub_98 )
          {
            ((void (__fastcall *)(__int64, const char **, __int64))v40)(v3, v125, 1);
            v39 = *(_QWORD *)v3;
          }
          (*(void (__fastcall **)(__int64, _QWORD, __int64))(v39 + 536))(v3, (unsigned int)v45[6], 4);
          v125[0] = "HandlerArray";
          v128 = 259;
          v41 = *(void (**)())(*(_QWORD *)v3 + 120LL);
          if ( v41 != nullsub_98 )
            ((void (__fastcall *)(__int64, const char **, __int64))v41)(v3, v125, 1);
        }
        else
        {
          (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v3 + 536LL))(v3, *v45, 4);
          (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v3 + 536LL))(v3, v45[1], 4);
          (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v3 + 536LL))(v3, v45[2], 4);
          (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v3 + 536LL))(v3, (unsigned int)v45[6], 4);
        }
        v42 = (unsigned __int8 *)sub_3258F50(a1, v44);
        sub_E9A5B0(v3, v42);
        v30 = v117 + 1;
        v117 = v30;
      }
      while ( v30 != v114 );
    }
    if ( *(_BYTE *)(a1 + 24) )
    {
      v89 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 136LL);
      if ( v89 == sub_2DD19D0 )
        BUG();
      v90 = v89();
      v106 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v90 + 288LL))(v90, a2);
    }
    else
    {
      v106 = 0;
    }
    v50 = v118;
    v104 = *(unsigned int *)(v111 + 248);
    if ( *(_DWORD *)(v111 + 248) )
    {
      v108 = 0;
      while ( 1 )
      {
        v58 = *(_QWORD *)&v50[8 * v108];
        if ( !v58 )
          goto LABEL_58;
        v59 = *(_QWORD *)(v111 + 240) + (v108 << 6);
        (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v3 + 208LL))(v3, v58, 0);
        v60 = *(int **)(v59 + 16);
        v110 = &v60[8 * *(unsigned int *)(v59 + 24)];
        if ( v110 != v60 )
          break;
LABEL_57:
        v50 = v118;
LABEL_58:
        if ( ++v108 == v104 )
          goto LABEL_59;
      }
      while ( 1 )
      {
        v71 = v60[2];
        if ( v71 == 0x7FFFFFFF )
        {
          v115 = (unsigned __int8 *)sub_E81A90(0, *(_QWORD **)(*(_QWORD *)(a1 + 8) + 216LL), 0, 0);
        }
        else
        {
          v72 = sub_32590E0(a1, v71, v111);
          v115 = (unsigned __int8 *)sub_E81A90(v72, *(_QWORD **)(*(_QWORD *)(a1 + 8) + 216LL), 0, 0);
        }
        v73 = *((_QWORD *)v60 + 3);
        v74 = 0;
        if ( v73 && (v73 & 4) != 0 )
          v74 = v73 & 0xFFFFFFFFFFFFFFF8LL;
        v75 = sub_3258B50(v74);
        v128 = 259;
        v125[0] = "Adjectives";
        if ( v116 )
        {
          v61 = *(_QWORD *)v3;
          v62 = *(void (**)())(*(_QWORD *)v3 + 120LL);
          if ( v62 != nullsub_98 )
          {
            ((void (__fastcall *)(__int64, const char **, __int64))v62)(v3, v125, 1);
            v61 = *(_QWORD *)v3;
          }
          (*(void (__fastcall **)(__int64, _QWORD, __int64))(v61 + 536))(v3, *v60, 4);
          v125[0] = "Type";
          v128 = 259;
          v63 = *(void (**)())(*(_QWORD *)v3 + 120LL);
          if ( v63 != nullsub_98 )
            ((void (__fastcall *)(__int64, const char **, __int64))v63)(v3, v125, 1);
          v64 = *((_QWORD *)v60 + 2);
          v65 = *(_QWORD *)(a1 + 8);
          if ( v64 )
          {
LABEL_87:
            v66 = sub_31DB510(v65, v64);
            v67 = (unsigned __int8 *)sub_3258F50(a1, v66);
            goto LABEL_88;
          }
        }
        else
        {
          (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v3 + 536LL))(v3, *v60, 4);
          v64 = *((_QWORD *)v60 + 2);
          v65 = *(_QWORD *)(a1 + 8);
          if ( v64 )
            goto LABEL_87;
        }
        v67 = (unsigned __int8 *)sub_E81A90(0, *(_QWORD **)(v65 + 216), 0, 0);
LABEL_88:
        sub_E9A5B0(v3, v67);
        v125[0] = "CatchObjOffset";
        v128 = 259;
        if ( v116 )
        {
          v68 = *(void (**)())(*(_QWORD *)v3 + 120LL);
          if ( v68 != nullsub_98 )
            ((void (__fastcall *)(__int64, const char **, __int64))v68)(v3, v125, 1);
          sub_E9A5B0(v3, v115);
          v125[0] = "Handler";
          v128 = 259;
          v69 = *(void (**)())(*(_QWORD *)v3 + 120LL);
          if ( v69 != nullsub_98 )
            ((void (__fastcall *)(__int64, const char **, __int64))v69)(v3, v125, 1);
          v70 = (unsigned __int8 *)sub_3258F50(a1, v75);
          sub_E9A5B0(v3, v70);
          if ( !*(_BYTE *)(a1 + 24) )
            goto LABEL_94;
          v125[0] = "ParentFrameOffset";
          v128 = 259;
          v92 = *(_QWORD *)v3;
          v93 = *(void (**)())(*(_QWORD *)v3 + 120LL);
          if ( v93 != nullsub_98 )
          {
            ((void (__fastcall *)(__int64, const char **, __int64))v93)(v3, v125, 1);
LABEL_123:
            v92 = *(_QWORD *)v3;
          }
          (*(void (__fastcall **)(__int64, _QWORD, __int64))(v92 + 536))(v3, v106, 4);
          goto LABEL_94;
        }
        sub_E9A5B0(v3, v115);
        v91 = (unsigned __int8 *)sub_3258F50(a1, v75);
        sub_E9A5B0(v3, v91);
        if ( *(_BYTE *)(a1 + 24) )
          goto LABEL_123;
LABEL_94:
        v60 += 8;
        if ( v110 == v60 )
          goto LABEL_57;
      }
    }
LABEL_59:
    if ( v50 != v120 )
      _libc_free((unsigned __int64)v50);
  }
  if ( v102 )
  {
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v3 + 208LL))(v3, v102, 0);
    v51 = v129;
    v52 = &v129[2 * (unsigned int)v130];
    if ( v52 == v129 )
      goto LABEL_72;
    v53 = v129;
    do
    {
      v125[0] = "IP";
      v128 = 259;
      if ( v116 )
      {
        v54 = *(void (**)())(*(_QWORD *)v3 + 120LL);
        if ( v54 != nullsub_98 )
          ((void (__fastcall *)(__int64, const char **, __int64))v54)(v3, v125, 1);
        sub_E9A5B0(v3, *v53);
        v125[0] = "ToState";
        v128 = 259;
        v55 = *(_QWORD *)v3;
        v56 = *(void (**)())(*(_QWORD *)v3 + 120LL);
        if ( v56 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, const char **, __int64))v56)(v3, v125, 1);
          v55 = *(_QWORD *)v3;
        }
      }
      else
      {
        sub_E9A5B0(v3, *v53);
        v55 = *(_QWORD *)v3;
      }
      v57 = *((int *)v53 + 2);
      v53 += 2;
      (*(void (__fastcall **)(__int64, __int64, __int64))(v55 + 536))(v3, v57, 4);
    }
    while ( v52 != v53 );
  }
  v51 = v129;
LABEL_72:
  if ( v51 != (unsigned __int8 **)v131 )
    _libc_free((unsigned __int64)v51);
}
