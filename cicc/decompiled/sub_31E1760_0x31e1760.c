// Function: sub_31E1760
// Address: 0x31e1760
//
char __fastcall sub_31E1760(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 **v4; // r13
  __int64 v5; // rax
  __int64 **v6; // r14
  void (*v7)(); // rax
  __int64 *v8; // rdi
  __int64 v9; // rax
  void (*v10)(); // rdx
  __int64 **v11; // r13
  __int64 v12; // rax
  __int64 **v13; // r14
  void (*v14)(); // rax
  __int64 *v15; // rdi
  __int64 v16; // rax
  void (*v17)(void); // rdx
  __int64 *v18; // rax
  __int64 *v19; // r13
  __int64 *i; // r14
  __int64 v21; // rdi
  unsigned int v22; // esi
  char v23; // al
  __int64 v24; // rsi
  __int64 *v25; // rax
  __int64 v26; // rdx
  __int64 *v27; // r13
  __int64 *j; // r14
  __int64 v29; // rsi
  unsigned __int8 *v30; // r13
  __int64 v31; // rax
  int v32; // ecx
  __int64 v33; // rsi
  int v34; // ecx
  unsigned int v35; // edx
  __int64 *v36; // rax
  __int64 v37; // rdi
  __int64 v38; // r13
  __int64 *v39; // r15
  __int64 v40; // rdx
  _QWORD *v41; // rax
  void (*v42)(); // r14
  int v43; // edx
  int v44; // eax
  _QWORD *v45; // rax
  int v46; // edx
  _BYTE *v47; // rax
  unsigned int v48; // eax
  _QWORD *v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rdx
  const char *v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r13
  __int64 (__fastcall *v55)(__int64, __int64, _QWORD); // r14
  __int64 v56; // rax
  __int64 v57; // rdi
  int v58; // edx
  __int64 (__fastcall *v59)(__int64, _QWORD *, _QWORD); // rax
  __int64 v60; // rdi
  void (*v61)(); // rax
  void (*v62)(); // rax
  __int64 *v63; // r13
  __int64 *v64; // r14
  __int64 v65; // rdi
  __int64 *v66; // r13
  __int64 *k; // r12
  __int64 v68; // rdi
  __int64 v69; // r13
  void (__fastcall *v70)(__int64, __int64, _QWORD); // r14
  __int64 v71; // rax
  __int64 v72; // rsi
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // rdi
  void (*v77)(); // rax
  __int64 v78; // r14
  __int64 v79; // rsi
  __int64 v80; // rdi
  _BYTE *v81; // rax
  __int64 v82; // r13
  __int64 (__fastcall *v83)(__int64, __int64, _QWORD); // r14
  __int64 v84; // rax
  __int64 v85; // r14
  unsigned int v86; // eax
  _WORD *v87; // rdx
  _QWORD *v88; // rax
  int v89; // edx
  int v90; // esi
  unsigned int v91; // esi
  __int64 v92; // rdx
  int v93; // eax
  int v94; // r8d
  __int64 v95; // rdx
  __int64 v97; // [rsp+8h] [rbp-218h]
  __int64 v98; // [rsp+18h] [rbp-208h]
  __int64 v99; // [rsp+28h] [rbp-1F8h]
  __int128 v101; // [rsp+40h] [rbp-1E0h] BYREF
  __int128 v102; // [rsp+50h] [rbp-1D0h]
  __int64 v103; // [rsp+60h] [rbp-1C0h]
  char *v104; // [rsp+70h] [rbp-1B0h]
  __int64 v105; // [rsp+78h] [rbp-1A8h]
  char v106; // [rsp+90h] [rbp-190h]
  char v107; // [rsp+91h] [rbp-18Fh]
  __int128 v108; // [rsp+A0h] [rbp-180h] BYREF
  char *v109; // [rsp+B0h] [rbp-170h]
  __int64 v110; // [rsp+B8h] [rbp-168h]
  __int64 v111; // [rsp+C0h] [rbp-160h]
  __int128 v112; // [rsp+D0h] [rbp-150h]
  __int16 v113; // [rsp+F0h] [rbp-130h]
  _QWORD v114[2]; // [rsp+100h] [rbp-120h] BYREF
  __int128 v115; // [rsp+110h] [rbp-110h]
  __int64 v116; // [rsp+120h] [rbp-100h]
  char *v117; // [rsp+130h] [rbp-F0h]
  __int64 v118; // [rsp+138h] [rbp-E8h]
  __int64 v119; // [rsp+150h] [rbp-D0h]
  _QWORD v120[4]; // [rsp+160h] [rbp-C0h] BYREF
  __int64 v121; // [rsp+180h] [rbp-A0h]
  __int128 v122; // [rsp+190h] [rbp-90h] BYREF
  __int128 v123; // [rsp+1A0h] [rbp-80h]
  __int64 v124; // [rsp+1B0h] [rbp-70h]
  _QWORD v125[2]; // [rsp+1C0h] [rbp-60h] BYREF
  __int128 v126; // [rsp+1D0h] [rbp-50h]
  __int64 v127; // [rsp+1E0h] [rbp-40h]

  v2 = a1;
  if ( !*(_BYTE *)(a2 + 235) )
    goto LABEL_17;
  v4 = *(__int64 ***)(a1 + 576);
  v5 = *(unsigned int *)(a1 + 584);
  if ( v4 == &v4[v5] )
    goto LABEL_10;
  v6 = &v4[v5];
  do
  {
    while ( 1 )
    {
      v8 = *v4;
      v9 = **v4;
      v10 = *(void (**)())(v9 + 112);
      if ( v10 != nullsub_1843 )
        break;
      v7 = *(void (**)())(v9 + 104);
      if ( v7 != nullsub_1842 )
        goto LABEL_8;
LABEL_5:
      if ( v6 == ++v4 )
        goto LABEL_9;
    }
    v10();
    v8 = *v4;
    v7 = *(void (**)())(**v4 + 104);
    if ( v7 == nullsub_1842 )
      goto LABEL_5;
LABEL_8:
    ++v4;
    ((void (__fastcall *)(__int64 *, __int64, _QWORD))v7)(v8, a2, 0);
  }
  while ( v6 != v4 );
LABEL_9:
  v2 = a1;
LABEL_10:
  v11 = *(__int64 ***)(v2 + 552);
  v12 = *(unsigned int *)(v2 + 560);
  if ( v11 != &v11[v12] )
  {
    v13 = &v11[v12];
    while ( 1 )
    {
      v15 = *v11;
      v16 = **v11;
      v17 = *(void (**)(void))(v16 + 112);
      if ( v17 == nullsub_1843 )
      {
        v14 = *(void (**)())(v16 + 104);
        if ( v14 == nullsub_1842 )
          goto LABEL_13;
LABEL_16:
        ++v11;
        ((void (__fastcall *)(__int64 *, __int64, _QWORD))v14)(v15, a2, 0);
        if ( v13 == v11 )
          break;
      }
      else
      {
        v17();
        v15 = *v11;
        v14 = *(void (**)())(**v11 + 104);
        if ( v14 != nullsub_1842 )
          goto LABEL_16;
LABEL_13:
        if ( v13 == ++v11 )
          break;
      }
    }
  }
LABEL_17:
  if ( *(_BYTE *)(a2 + 260) && !sub_2E31AB0(a2) )
  {
    v69 = *(_QWORD *)(v2 + 224);
    v70 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v69 + 176LL);
    v71 = sub_31DA6B0(v2);
    v72 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, _QWORD))(*(_QWORD *)v71 + 72LL))(
            v71,
            **(_QWORD **)(v2 + 232),
            a2,
            *(_QWORD *)(v2 + 200));
    v70(v69, v72, 0);
    *(_QWORD *)(v2 + 440) = sub_2E309C0(a2, v72, v73, v74, v75);
  }
  v18 = *(__int64 **)(v2 + 576);
  v19 = &v18[*(unsigned int *)(v2 + 584)];
  for ( i = v18; v19 != i; ++i )
  {
    v21 = *i;
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v21 + 96LL))(v21, a2);
  }
  v22 = *(unsigned __int8 *)(a2 + 208);
  if ( (_BYTE)v22 )
    sub_31DCA70(v2, v22, 0, *(_DWORD *)(a2 + 212));
  v23 = *(_BYTE *)(v2 + 488);
  v24 = *(_QWORD *)(a2 + 224);
  if ( v24 )
  {
    if ( v23 )
    {
      v60 = *(_QWORD *)(v2 + 224);
      v61 = *(void (**)())(*(_QWORD *)v60 + 120LL);
      v125[0] = "Block address taken";
      LOWORD(v127) = 259;
      if ( v61 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v61)(v60, v125, 1);
        v24 = *(_QWORD *)(a2 + 224);
      }
    }
    v25 = sub_31E0DA0(v2, v24);
    v27 = &v25[v26];
    for ( j = v25; v27 != j; ++j )
    {
      v29 = *j;
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(v2 + 224) + 208LL))(*(_QWORD *)(v2 + 224), v29, 0);
    }
LABEL_26:
    if ( *(_BYTE *)(v2 + 488) )
      goto LABEL_27;
  }
  else if ( v23 )
  {
    if ( *(_BYTE *)(a2 + 217) )
    {
      v76 = *(_QWORD *)(v2 + 224);
      v77 = *(void (**)())(*(_QWORD *)v76 + 120LL);
      v125[0] = "Block address taken";
      LOWORD(v127) = 259;
      if ( v77 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64, _QWORD *, __int64))v77)(v76, v125, 1);
        goto LABEL_26;
      }
    }
LABEL_27:
    v30 = *(unsigned __int8 **)(a2 + 16);
    if ( v30 && (v30[7] & 0x10) != 0 )
    {
      v78 = sub_AA4B30(*(_QWORD *)(a2 + 16));
      v79 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v2 + 224) + 128LL))(*(_QWORD *)(v2 + 224));
      sub_A5BF40(v30, v79, 0, v78);
      v80 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v2 + 224) + 128LL))(*(_QWORD *)(v2 + 224));
      v81 = *(_BYTE **)(v80 + 32);
      if ( (unsigned __int64)v81 >= *(_QWORD *)(v80 + 24) )
      {
        sub_CB5D20(v80, 10);
      }
      else
      {
        *(_QWORD *)(v80 + 32) = v81 + 1;
        *v81 = 10;
      }
    }
    v31 = *(_QWORD *)(v2 + 256);
    v32 = *(_DWORD *)(v31 + 24);
    v33 = *(_QWORD *)(v31 + 8);
    if ( v32 )
    {
      v34 = v32 - 1;
      v35 = v34 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v36 = (__int64 *)(v33 + 16LL * v35);
      v37 = *v36;
      if ( a2 == *v36 )
      {
LABEL_31:
        v38 = v36[1];
        if ( v38 )
        {
          v39 = *(__int64 **)(v2 + 224);
          v40 = *v39;
          if ( a2 == **(_QWORD **)(v38 + 32) )
          {
            v85 = (*(__int64 (__fastcall **)(_QWORD))(v40 + 128))(*(_QWORD *)(v2 + 224));
            v86 = sub_31DA6A0(v2);
            sub_31D87E0(v85, *(_QWORD *)v38, v86);
            v87 = *(_WORD **)(v85 + 32);
            if ( *(_QWORD *)(v85 + 24) - (_QWORD)v87 <= 1u )
            {
              sub_CB6200(v85, (unsigned __int8 *)"=>", 2u);
            }
            else
            {
              *v87 = 15933;
              *(_QWORD *)(v85 + 32) += 2LL;
            }
            v88 = *(_QWORD **)v38;
            if ( *(_QWORD *)v38 )
            {
              v89 = 1;
              do
              {
                v88 = (_QWORD *)*v88;
                v90 = v89++;
              }
              while ( v88 );
              v91 = 2 * v90;
            }
            else
            {
              v91 = 0;
            }
            sub_CB69B0(v85, v91);
            v92 = *(_QWORD *)(v85 + 32);
            if ( (unsigned __int64)(*(_QWORD *)(v85 + 24) - v92) <= 4 )
            {
              sub_CB6200(v85, "This ", 5u);
            }
            else
            {
              *(_DWORD *)v92 = 1936287828;
              *(_BYTE *)(v92 + 4) = 32;
              *(_QWORD *)(v85 + 32) += 5LL;
            }
            if ( *(_QWORD *)(v38 + 16) == *(_QWORD *)(v38 + 8) )
            {
              v95 = *(_QWORD *)(v85 + 32);
              if ( (unsigned __int64)(*(_QWORD *)(v85 + 24) - v95) <= 5 )
              {
                sub_CB6200(v85, "Inner ", 6u);
              }
              else
              {
                *(_DWORD *)v95 = 1701736009;
                *(_WORD *)(v95 + 4) = 8306;
                *(_QWORD *)(v85 + 32) += 6LL;
              }
            }
            v45 = *(_QWORD **)v38;
            v46 = 1;
            if ( *(_QWORD *)v38 )
            {
              do
              {
                v45 = (_QWORD *)*v45;
                ++v46;
              }
              while ( v45 );
            }
            LODWORD(v126) = v46;
            LOWORD(v127) = 2307;
            v125[0] = "Loop Header: Depth=";
            sub_CA0E80((__int64)v125, v85);
            v47 = *(_BYTE **)(v85 + 32);
            if ( (unsigned __int64)v47 >= *(_QWORD *)(v85 + 24) )
            {
              sub_CB5D20(v85, 10);
            }
            else
            {
              *(_QWORD *)(v85 + 32) = v47 + 1;
              *v47 = 10;
            }
            v48 = sub_31DA6A0(v2);
            sub_31D71D0(v85, v38, v48);
          }
          else
          {
            v41 = *(_QWORD **)v38;
            v42 = *(void (**)())(v40 + 120);
            v43 = 1;
            if ( *(_QWORD *)v38 )
            {
              do
              {
                v41 = (_QWORD *)*v41;
                ++v43;
              }
              while ( v41 );
            }
            LODWORD(v122) = v43;
            LOWORD(v124) = 265;
            LOWORD(v119) = 259;
            v117 = " Depth=";
            v44 = *(_DWORD *)(**(_QWORD **)(v38 + 32) + 24LL);
            v113 = 266;
            v107 = 1;
            LODWORD(v112) = v44;
            v104 = "_";
            v106 = 3;
            LODWORD(v102) = sub_31DA6A0(v2);
            *(_QWORD *)&v101 = "  in Loop: Header=BB";
            LOWORD(v103) = 2307;
            v109 = "_";
            *(_QWORD *)&v108 = &v101;
            v110 = v105;
            LOWORD(v111) = 770;
            v114[0] = &v108;
            v114[1] = v99;
            v115 = v112;
            LOWORD(v116) = 2562;
            v120[0] = v114;
            v120[2] = " Depth=";
            v120[1] = v98;
            LOWORD(v121) = 770;
            v120[3] = v118;
            v125[0] = v120;
            v126 = v122;
            v125[1] = v97;
            LOWORD(v127) = 2306;
            if ( v42 != nullsub_98 )
              ((void (__fastcall *)(__int64 *, _QWORD *, __int64))v42)(v39, v125, 1);
          }
        }
      }
      else
      {
        v93 = 1;
        while ( v37 != -4096 )
        {
          v94 = v93 + 1;
          v35 = v34 & (v93 + v35);
          v36 = (__int64 *)(v33 + 16LL * v35);
          v37 = *v36;
          if ( a2 == *v36 )
            goto LABEL_31;
          v93 = v94;
        }
      }
    }
  }
  v49 = (_QWORD *)a2;
  LOBYTE(v50) = sub_31DF300((_QWORD *)v2, a2);
  if ( (_BYTE)v50 )
  {
    v54 = *(_QWORD *)(v2 + 224);
    if ( *(_BYTE *)(v2 + 488) )
    {
      if ( *(_BYTE *)(a2 + 232) )
      {
        v52 = "Label of block must be emitted";
        v62 = *(void (**)())(*(_QWORD *)v54 + 120LL);
        v125[0] = "Label of block must be emitted";
        LOWORD(v127) = 259;
        if ( v62 != nullsub_98 )
        {
          v49 = v125;
          ((void (__fastcall *)(__int64, _QWORD *, __int64))v62)(v54, v125, 1);
          v54 = *(_QWORD *)(v2 + 224);
        }
      }
    }
    v55 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v54 + 208LL);
    v56 = sub_2E309C0(a2, (__int64)v49, v51, (__int64)v52, v53);
    LOBYTE(v50) = v55(v54, v56, 0);
  }
  else if ( *(_BYTE *)(v2 + 488) )
  {
    v57 = *(_QWORD *)(v2 + 224);
    v58 = *(_DWORD *)(a2 + 24);
    v59 = *(__int64 (__fastcall **)(__int64, _QWORD *, _QWORD))(*(_QWORD *)v57 + 136LL);
    *(_QWORD *)&v122 = " %bb.";
    LODWORD(v123) = v58;
    LOWORD(v124) = 2563;
    *(_QWORD *)&v126 = ":";
    v125[0] = &v122;
    LOWORD(v127) = 770;
    LOBYTE(v50) = v59(v57, v125, 0);
  }
  if ( *(_BYTE *)(a2 + 234) )
  {
    v50 = *(_QWORD *)(v2 + 208);
    if ( *(_DWORD *)(v50 + 336) == 4 )
    {
      v82 = *(_QWORD *)(v2 + 224);
      v83 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v82 + 208LL);
      v84 = sub_2E30D70(a2);
      LOBYTE(v50) = v83(v82, v84, 0);
    }
  }
  if ( *(_BYTE *)(a2 + 260) )
  {
    LOBYTE(v50) = sub_2E31AB0(a2);
    if ( !(_BYTE)v50 )
    {
      v63 = *(__int64 **)(v2 + 576);
      v64 = &v63[*(unsigned int *)(v2 + 584)];
      while ( v64 != v63 )
      {
        v65 = *v63++;
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v65 + 56LL))(v65, a2);
      }
      v50 = *(_QWORD *)(v2 + 552);
      v66 = (__int64 *)(v50 + 8LL * *(unsigned int *)(v2 + 560));
      for ( k = (__int64 *)v50;
            v66 != k;
            LOBYTE(v50) = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v68 + 56LL))(v68, a2) )
      {
        v68 = *k++;
      }
    }
  }
  return v50;
}
