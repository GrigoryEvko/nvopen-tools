// Function: sub_39ADBC0
// Address: 0x39adbc0
//
void __fastcall sub_39ADBC0(__int64 a1, __int64 a2)
{
  __int64 v3; // r15
  const char *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rsi
  bool v7; // zf
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 (*v13)(); // rdx
  unsigned int *v14; // rax
  unsigned int *v15; // rax
  unsigned int *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  void (*v19)(); // rcx
  __int64 v20; // rax
  void (*v21)(); // rcx
  int *v22; // rbx
  __int64 v23; // r13
  __int64 v24; // rax
  void (*v25)(); // r9
  void (*v26)(); // rax
  unsigned int *v27; // rax
  unsigned __int64 v28; // rdi
  __int64 v29; // r12
  __int64 v30; // rbx
  int v31; // r8d
  int v32; // r9d
  __int64 v33; // rax
  void (*v34)(); // rcx
  __int64 v35; // rax
  void (*v36)(); // rcx
  __int64 v37; // rax
  void (*v38)(); // rcx
  __int64 v39; // rax
  void (*v40)(); // rcx
  void (*v41)(); // rax
  unsigned int *v42; // rsi
  __int64 v43; // r13
  __int64 v44; // r12
  int *v45; // rbx
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rdi
  _BYTE *v49; // rdi
  unsigned int **v50; // rdi
  unsigned int **v51; // rbx
  unsigned int **v52; // r14
  void (*v53)(); // rax
  __int64 v54; // rax
  void (*v55)(); // rcx
  __int64 v56; // rsi
  __int64 v57; // rsi
  __int64 v58; // r12
  int *v59; // rbx
  void (*v60)(); // rax
  void (*v61)(); // rax
  unsigned int *v62; // rax
  unsigned int v63; // esi
  int v64; // eax
  unsigned __int64 v65; // rdi
  __int64 v66; // r12
  __int64 v67; // rax
  void (*v68)(); // rcx
  void (*v69)(); // rax
  __int64 v70; // rsi
  __int64 v71; // rdi
  __int64 v72; // rax
  unsigned int *v73; // rsi
  unsigned int *v74; // rax
  __int64 v75; // rax
  __int64 v76; // rax
  void (*v77)(); // rcx
  __int64 v78; // rax
  void (*v79)(); // rcx
  void (*v80)(); // rax
  unsigned int *v81; // rax
  __int64 v82; // rax
  void (*v83)(); // rcx
  void (*v84)(); // rax
  unsigned int *v85; // rax
  __int64 v86; // rax
  void (*v87)(); // rcx
  void (*v88)(); // rax
  __int64 (*v89)(); // rax
  __int64 v90; // rax
  void (*v91)(); // rcx
  int v92; // eax
  __int64 v93; // rax
  void (*v94)(); // rcx
  int v95; // eax
  __int64 v96; // rdi
  __int64 v97; // rdi
  __int64 v98; // rdi
  __int64 v99; // [rsp+10h] [rbp-160h]
  __int64 v100; // [rsp+18h] [rbp-158h]
  __int64 v102; // [rsp+28h] [rbp-148h]
  __int64 v103; // [rsp+30h] [rbp-140h]
  int *v104; // [rsp+30h] [rbp-140h]
  __int64 v105; // [rsp+38h] [rbp-138h]
  int v106; // [rsp+40h] [rbp-130h]
  int *v107; // [rsp+40h] [rbp-130h]
  __int64 v108; // [rsp+40h] [rbp-130h]
  unsigned int *v109; // [rsp+40h] [rbp-130h]
  unsigned int v110; // [rsp+48h] [rbp-128h]
  char v111; // [rsp+4Fh] [rbp-121h]
  __int64 v112; // [rsp+58h] [rbp-118h] BYREF
  _QWORD v113[2]; // [rsp+60h] [rbp-110h] BYREF
  _BYTE *v114; // [rsp+70h] [rbp-100h] BYREF
  __int64 v115; // [rsp+78h] [rbp-F8h]
  _BYTE v116[16]; // [rsp+80h] [rbp-F0h] BYREF
  _QWORD v117[2]; // [rsp+90h] [rbp-E0h] BYREF
  __int16 v118; // [rsp+A0h] [rbp-D0h]
  _QWORD v119[2]; // [rsp+B0h] [rbp-C0h] BYREF
  __int16 v120; // [rsp+C0h] [rbp-B0h]
  char *v121; // [rsp+D0h] [rbp-A0h] BYREF
  _QWORD *v122; // [rsp+D8h] [rbp-98h]
  __int16 v123; // [rsp+E0h] [rbp-90h]
  unsigned int **v124; // [rsp+F0h] [rbp-80h] BYREF
  __int64 v125; // [rsp+F8h] [rbp-78h]
  _BYTE v126[112]; // [rsp+100h] [rbp-70h] BYREF

  v105 = *(_QWORD *)(a2 + 88);
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL);
  v4 = sub_1649960(*(_QWORD *)a2);
  v6 = (__int64)v4;
  if ( v5 && *v4 == 1 )
  {
    --v5;
    v6 = (__int64)(v4 + 1);
  }
  v7 = *(_BYTE *)(a1 + 24) == 0;
  v113[0] = v6;
  v124 = (unsigned int **)v126;
  v125 = 0x400000000LL;
  v8 = *(_QWORD *)(a1 + 8);
  v113[1] = v5;
  v9 = *(_QWORD *)(v8 + 248);
  if ( v7 )
  {
    v10 = sub_38BF870(v9, v6, v5);
  }
  else
  {
    v123 = 1283;
    v121 = "$cppxdata$";
    v122 = v113;
    v10 = sub_38BF510(v9, (__int64)&v121);
    sub_39AD750(a1, a2, v105, (__int64)&v124);
  }
  v106 = 0;
  v11 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 240LL);
  if ( *(_DWORD *)(v11 + 348) == 4 )
  {
    v95 = *(_DWORD *)(v11 + 352);
    if ( !v95 || v95 == 6 )
      v106 = 0;
    else
      v106 = sub_39ACCA0(a1, *(_DWORD *)(v105 + 704), v105);
  }
  v12 = 0;
  if ( *(_DWORD *)(v105 + 136) )
  {
    v96 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 248LL);
    v123 = 1283;
    v121 = "$stateUnwindMap$";
    v122 = v113;
    v12 = sub_38BF510(v96, (__int64)&v121);
  }
  v103 = 0;
  if ( *(_DWORD *)(v105 + 216) )
  {
    v98 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 248LL);
    v123 = 1283;
    v121 = "$tryMap$";
    v122 = v113;
    v103 = sub_38BF510(v98, (__int64)&v121);
  }
  v99 = 0;
  if ( (_DWORD)v125 )
  {
    v97 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 248LL);
    v123 = 1283;
    v121 = "$ip2state$";
    v122 = v113;
    v99 = sub_38BF510(v97, (__int64)&v121);
  }
  v13 = *(__int64 (**)())(*(_QWORD *)v3 + 80LL);
  if ( v13 == sub_168DB50 )
  {
    (*(void (__fastcall **)(__int64, __int64, _QWORD, __int64, _QWORD))(*(_QWORD *)v3 + 512LL))(v3, 4, 0, 1, 0);
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v3 + 176LL))(v3, v10, 0);
LABEL_15:
    (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v3 + 424LL))(v3, 429065506, 4);
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v3 + 424LL))(v3, *(unsigned int *)(v105 + 136), 4);
    v14 = (unsigned int *)sub_39ACBF0(a1, v12);
    sub_38DDD30(v3, v14);
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v3 + 424LL))(v3, *(unsigned int *)(v105 + 216), 4);
    v15 = (unsigned int *)sub_39ACBF0(a1, v103);
    sub_38DDD30(v3, v15);
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v3 + 424LL))(v3, (unsigned int)v125, 4);
    v111 = 0;
    goto LABEL_16;
  }
  v111 = ((__int64 (__fastcall *)(__int64))v13)(v3);
  (*(void (__fastcall **)(__int64, __int64, _QWORD, __int64, _QWORD))(*(_QWORD *)v3 + 512LL))(v3, 4, 0, 1, 0);
  (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v3 + 176LL))(v3, v10, 0);
  v121 = "MagicNumber";
  v123 = 259;
  if ( !v111 )
    goto LABEL_15;
  v76 = *(_QWORD *)v3;
  v77 = *(void (**)())(*(_QWORD *)v3 + 104LL);
  if ( v77 != nullsub_580 )
  {
    ((void (__fastcall *)(__int64, char **, __int64))v77)(v3, &v121, 1);
    v76 = *(_QWORD *)v3;
  }
  (*(void (__fastcall **)(__int64, __int64, __int64))(v76 + 424))(v3, 429065506, 4);
  v121 = "MaxState";
  v123 = 259;
  v78 = *(_QWORD *)v3;
  v79 = *(void (**)())(*(_QWORD *)v3 + 104LL);
  if ( v79 != nullsub_580 )
  {
    ((void (__fastcall *)(__int64, char **, __int64))v79)(v3, &v121, 1);
    v78 = *(_QWORD *)v3;
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(v78 + 424))(v3, *(unsigned int *)(v105 + 136), 4);
  v121 = "UnwindMap";
  v123 = 259;
  v80 = *(void (**)())(*(_QWORD *)v3 + 104LL);
  if ( v80 != nullsub_580 )
    ((void (__fastcall *)(__int64, char **, __int64))v80)(v3, &v121, 1);
  v81 = (unsigned int *)sub_39ACBF0(a1, v12);
  sub_38DDD30(v3, v81);
  v121 = "NumTryBlocks";
  v123 = 259;
  v82 = *(_QWORD *)v3;
  v83 = *(void (**)())(*(_QWORD *)v3 + 104LL);
  if ( v83 != nullsub_580 )
  {
    ((void (__fastcall *)(__int64, char **, __int64))v83)(v3, &v121, 1);
    v82 = *(_QWORD *)v3;
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(v82 + 424))(v3, *(unsigned int *)(v105 + 216), 4);
  v121 = "TryBlockMap";
  v123 = 259;
  v84 = *(void (**)())(*(_QWORD *)v3 + 104LL);
  if ( v84 != nullsub_580 )
    ((void (__fastcall *)(__int64, char **, __int64))v84)(v3, &v121, 1);
  v85 = (unsigned int *)sub_39ACBF0(a1, v103);
  sub_38DDD30(v3, v85);
  v121 = "IPMapEntries";
  v123 = 259;
  v86 = *(_QWORD *)v3;
  v87 = *(void (**)())(*(_QWORD *)v3 + 104LL);
  if ( v87 != nullsub_580 )
  {
    ((void (__fastcall *)(__int64, char **, __int64))v87)(v3, &v121, 1);
    v86 = *(_QWORD *)v3;
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(v86 + 424))(v3, (unsigned int)v125, 4);
  v121 = "IPToStateXData";
  v123 = 259;
  v88 = *(void (**)())(*(_QWORD *)v3 + 104LL);
  if ( v88 != nullsub_580 )
    ((void (__fastcall *)(__int64, char **, __int64))v88)(v3, &v121, 1);
LABEL_16:
  v16 = (unsigned int *)sub_39ACBF0(a1, v99);
  sub_38DDD30(v3, v16);
  v17 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 240LL);
  if ( *(_DWORD *)(v17 + 348) == 4 )
  {
    v92 = *(_DWORD *)(v17 + 352);
    if ( v92 )
    {
      if ( v92 != 6 )
      {
        v121 = "UnwindHelp";
        v123 = 259;
        v93 = *(_QWORD *)v3;
        if ( v111 )
        {
          v94 = *(void (**)())(v93 + 104);
          if ( v94 != nullsub_580 )
          {
            ((void (__fastcall *)(__int64, char **, __int64))v94)(v3, &v121, 1);
            v93 = *(_QWORD *)v3;
          }
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(v93 + 424))(v3, v106, 4);
      }
    }
  }
  v121 = "ESTypeList";
  v123 = 259;
  v18 = *(_QWORD *)v3;
  if ( v111 )
  {
    v19 = *(void (**)())(v18 + 104);
    if ( v19 != nullsub_580 )
    {
      ((void (__fastcall *)(__int64, char **, __int64))v19)(v3, &v121, 1);
      v18 = *(_QWORD *)v3;
    }
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(v18 + 424))(v3, 0, 4);
    v121 = "EHFlags";
    v123 = 259;
    v20 = *(_QWORD *)v3;
    v21 = *(void (**)())(*(_QWORD *)v3 + 104LL);
    if ( v21 != nullsub_580 )
    {
      ((void (__fastcall *)(__int64, char **, __int64))v21)(v3, &v121, 1);
      v20 = *(_QWORD *)v3;
    }
  }
  else
  {
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(v18 + 424))(v3, 0, 4);
    v20 = *(_QWORD *)v3;
  }
  (*(void (__fastcall **)(__int64, __int64, __int64))(v20 + 424))(v3, 1, 4);
  if ( v12 )
  {
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v3 + 176LL))(v3, v12, 0);
    v22 = *(int **)(v105 + 128);
    v23 = 4LL * *(unsigned int *)(v105 + 136);
    v107 = &v22[v23];
    while ( v107 != v22 )
    {
      v28 = *((_QWORD *)v22 + 1) & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*((_QWORD *)v22 + 1) & 4) == 0 )
        v28 = 0;
      v29 = sub_39AC850(v28);
      v123 = 259;
      v121 = "ToState";
      if ( v111 )
      {
        v24 = *(_QWORD *)v3;
        v25 = *(void (**)())(*(_QWORD *)v3 + 104LL);
        if ( v25 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, char **, __int64))v25)(v3, &v121, 1);
          v24 = *(_QWORD *)v3;
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(v24 + 424))(v3, *v22, 4);
        v121 = "Action";
        v123 = 259;
        v26 = *(void (**)())(*(_QWORD *)v3 + 104LL);
        if ( v26 != nullsub_580 )
          ((void (__fastcall *)(__int64, char **, __int64))v26)(v3, &v121, 1);
      }
      else
      {
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v3 + 424LL))(v3, *v22, 4);
      }
      v22 += 4;
      v27 = (unsigned int *)sub_39ACBF0(a1, v29);
      sub_38DDD30(v3, v27);
    }
  }
  if ( !v103 )
    goto LABEL_58;
  v30 = 0;
  (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v3 + 176LL))(v3, v103, 0);
  v112 = 0;
  v114 = v116;
  v115 = 0x100000000LL;
  v108 = *(unsigned int *)(v105 + 216);
  if ( *(_DWORD *)(v105 + 216) )
  {
    do
    {
      v43 = 0;
      v44 = 0;
      v45 = (int *)(*(_QWORD *)(v105 + 208) + (v30 << 6));
      if ( v45[6] )
      {
        v47 = *(_QWORD *)(a1 + 8);
        v120 = 770;
        v48 = *(_QWORD *)(v47 + 248);
        v123 = 1282;
        v117[0] = "$handlerMap$";
        v117[1] = &v112;
        v118 = 2819;
        v119[0] = v117;
        v119[1] = "$";
        v121 = (char *)v119;
        v122 = v113;
        v44 = sub_38BF510(v48, (__int64)&v121);
        v43 = v44;
        v46 = (unsigned int)v115;
        if ( (unsigned int)v115 >= HIDWORD(v115) )
        {
LABEL_53:
          sub_16CD150((__int64)&v114, v116, 0, 8, v31, v32);
          v46 = (unsigned int)v115;
        }
      }
      else
      {
        v46 = (unsigned int)v115;
        if ( (unsigned int)v115 >= HIDWORD(v115) )
          goto LABEL_53;
      }
      *(_QWORD *)&v114[8 * v46] = v43;
      LODWORD(v115) = v115 + 1;
      v121 = "TryLow";
      v123 = 259;
      if ( v111 )
      {
        v33 = *(_QWORD *)v3;
        v34 = *(void (**)())(*(_QWORD *)v3 + 104LL);
        if ( v34 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, char **, __int64))v34)(v3, &v121, 1);
          v33 = *(_QWORD *)v3;
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(v33 + 424))(v3, *v45, 4);
        v121 = "TryHigh";
        v123 = 259;
        v35 = *(_QWORD *)v3;
        v36 = *(void (**)())(*(_QWORD *)v3 + 104LL);
        if ( v36 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, char **, __int64))v36)(v3, &v121, 1);
          v35 = *(_QWORD *)v3;
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(v35 + 424))(v3, v45[1], 4);
        v121 = "CatchHigh";
        v123 = 259;
        v37 = *(_QWORD *)v3;
        v38 = *(void (**)())(*(_QWORD *)v3 + 104LL);
        if ( v38 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, char **, __int64))v38)(v3, &v121, 1);
          v37 = *(_QWORD *)v3;
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(v37 + 424))(v3, v45[2], 4);
        v121 = "NumCatches";
        v123 = 259;
        v39 = *(_QWORD *)v3;
        v40 = *(void (**)())(*(_QWORD *)v3 + 104LL);
        if ( v40 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, char **, __int64))v40)(v3, &v121, 1);
          v39 = *(_QWORD *)v3;
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(v39 + 424))(v3, (unsigned int)v45[6], 4);
        v121 = "HandlerArray";
        v123 = 259;
        v41 = *(void (**)())(*(_QWORD *)v3 + 104LL);
        if ( v41 != nullsub_580 )
          ((void (__fastcall *)(__int64, char **, __int64))v41)(v3, &v121, 1);
      }
      else
      {
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v3 + 424LL))(v3, *v45, 4);
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v3 + 424LL))(v3, v45[1], 4);
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v3 + 424LL))(v3, v45[2], 4);
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v3 + 424LL))(v3, (unsigned int)v45[6], 4);
      }
      v42 = (unsigned int *)sub_39ACBF0(a1, v44);
      sub_38DDD30(v3, v42);
      v30 = v112 + 1;
      v112 = v30;
    }
    while ( v30 != v108 );
  }
  if ( *(_BYTE *)(a1 + 24) )
  {
    v89 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 48LL);
    if ( v89 == sub_1D90020 )
      BUG();
    v90 = v89();
    v110 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v90 + 208LL))(v90, a2);
  }
  else
  {
    v110 = 0;
  }
  v49 = v114;
  v100 = *(unsigned int *)(v105 + 216);
  if ( *(_DWORD *)(v105 + 216) )
  {
    v102 = 0;
    while ( 1 )
    {
      v57 = *(_QWORD *)&v49[8 * v102];
      if ( !v57 )
        goto LABEL_55;
      v58 = *(_QWORD *)(v105 + 208) + (v102 << 6);
      (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v3 + 176LL))(v3, v57, 0);
      v59 = *(int **)(v58 + 16);
      v104 = &v59[8 * *(unsigned int *)(v58 + 24)];
      if ( v104 != v59 )
        break;
LABEL_54:
      v49 = v114;
LABEL_55:
      if ( ++v102 == v100 )
        goto LABEL_56;
    }
    while ( 1 )
    {
      v63 = v59[2];
      if ( v63 == 0x7FFFFFFF )
      {
        v109 = (unsigned int *)sub_38CB470(0, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 248LL));
      }
      else
      {
        v64 = sub_39ACCA0(a1, v63, v105);
        v109 = (unsigned int *)sub_38CB470(v64, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 248LL));
      }
      v65 = *((_QWORD *)v59 + 3) & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*((_QWORD *)v59 + 3) & 4) == 0 )
        v65 = 0;
      v66 = sub_39AC850(v65);
      v123 = 259;
      v121 = "Adjectives";
      if ( v111 )
      {
        v67 = *(_QWORD *)v3;
        v68 = *(void (**)())(*(_QWORD *)v3 + 104LL);
        if ( v68 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, char **, __int64))v68)(v3, &v121, 1);
          v67 = *(_QWORD *)v3;
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(v67 + 424))(v3, *v59, 4);
        v121 = "Type";
        v123 = 259;
        v69 = *(void (**)())(*(_QWORD *)v3 + 104LL);
        if ( v69 != nullsub_580 )
          ((void (__fastcall *)(__int64, char **, __int64))v69)(v3, &v121, 1);
        v70 = *((_QWORD *)v59 + 2);
        v71 = *(_QWORD *)(a1 + 8);
        if ( v70 )
        {
LABEL_95:
          v72 = sub_396EAF0(v71, v70);
          v73 = (unsigned int *)sub_39ACBF0(a1, v72);
          goto LABEL_96;
        }
      }
      else
      {
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v3 + 424LL))(v3, *v59, 4);
        v70 = *((_QWORD *)v59 + 2);
        v71 = *(_QWORD *)(a1 + 8);
        if ( v70 )
          goto LABEL_95;
      }
      v73 = (unsigned int *)sub_38CB470(0, *(_QWORD *)(v71 + 248));
LABEL_96:
      sub_38DDD30(v3, v73);
      v121 = "CatchObjOffset";
      v123 = 259;
      if ( v111 )
      {
        v60 = *(void (**)())(*(_QWORD *)v3 + 104LL);
        if ( v60 != nullsub_580 )
          ((void (__fastcall *)(__int64, char **, __int64))v60)(v3, &v121, 1);
        sub_38DDD30(v3, v109);
        v121 = "Handler";
        v123 = 259;
        v61 = *(void (**)())(*(_QWORD *)v3 + 104LL);
        if ( v61 != nullsub_580 )
          ((void (__fastcall *)(__int64, char **, __int64))v61)(v3, &v121, 1);
        v62 = (unsigned int *)sub_39ACBF0(a1, v66);
        sub_38DDD30(v3, v62);
        if ( !*(_BYTE *)(a1 + 24) )
          goto LABEL_84;
        v121 = "ParentFrameOffset";
        v123 = 259;
        v75 = *(_QWORD *)v3;
        v91 = *(void (**)())(*(_QWORD *)v3 + 104LL);
        if ( v91 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, char **, __int64))v91)(v3, &v121, 1);
LABEL_98:
          v75 = *(_QWORD *)v3;
        }
        (*(void (__fastcall **)(__int64, _QWORD, __int64))(v75 + 424))(v3, v110, 4);
        goto LABEL_84;
      }
      sub_38DDD30(v3, v109);
      v74 = (unsigned int *)sub_39ACBF0(a1, v66);
      sub_38DDD30(v3, v74);
      if ( *(_BYTE *)(a1 + 24) )
        goto LABEL_98;
LABEL_84:
      v59 += 8;
      if ( v104 == v59 )
        goto LABEL_54;
    }
  }
LABEL_56:
  if ( v49 != v116 )
    _libc_free((unsigned __int64)v49);
LABEL_58:
  if ( v99 )
  {
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v3 + 176LL))(v3, v99, 0);
    v50 = v124;
    v51 = &v124[2 * (unsigned int)v125];
    if ( v51 == v124 )
      goto LABEL_69;
    v52 = v124;
    do
    {
      v121 = "IP";
      v123 = 259;
      if ( v111 )
      {
        v53 = *(void (**)())(*(_QWORD *)v3 + 104LL);
        if ( v53 != nullsub_580 )
          ((void (__fastcall *)(__int64, char **, __int64))v53)(v3, &v121, 1);
        sub_38DDD30(v3, *v52);
        v121 = "ToState";
        v123 = 259;
        v54 = *(_QWORD *)v3;
        v55 = *(void (**)())(*(_QWORD *)v3 + 104LL);
        if ( v55 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, char **, __int64))v55)(v3, &v121, 1);
          v54 = *(_QWORD *)v3;
        }
      }
      else
      {
        sub_38DDD30(v3, *v52);
        v54 = *(_QWORD *)v3;
      }
      v56 = *((int *)v52 + 2);
      v52 += 2;
      (*(void (__fastcall **)(__int64, __int64, __int64))(v54 + 424))(v3, v56, 4);
    }
    while ( v51 != v52 );
  }
  v50 = v124;
LABEL_69:
  if ( v50 != (unsigned int **)v126 )
    _libc_free((unsigned __int64)v50);
}
