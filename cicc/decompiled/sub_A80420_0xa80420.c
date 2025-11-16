// Function: sub_A80420
// Address: 0xa80420
//
__int64 __fastcall sub_A80420(__int64 a1, unsigned __int64 a2, unsigned __int8 *a3, __int64 a4, __int64 a5)
{
  int v7; // edx
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rcx
  int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // rdx
  int v19; // ebx
  __int64 v20; // rax
  __int64 v21; // rdx
  int v22; // eax
  int v23; // edx
  unsigned __int8 *v24; // r8
  __int64 v25; // rax
  __int64 v26; // rcx
  unsigned __int8 *v27; // r14
  _QWORD *v28; // rax
  __int64 v29; // r9
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v35; // r8
  int v36; // eax
  __int64 v37; // rax
  __int64 v38; // rax
  _BYTE *v39; // rsi
  __int64 v40; // rdx
  __int64 v41; // rax
  __int128 *v42; // rdi
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // r14
  int v47; // eax
  __int64 v48; // rdx
  _BYTE *v49; // r13
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // r15
  _BYTE *v53; // rax
  __int64 v54; // r15
  __int64 v55; // rax
  __int64 v56; // r9
  __int64 v57; // rdi
  _BYTE *v58; // r10
  __int64 (__fastcall *v59)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v60; // rax
  __int64 v61; // rax
  unsigned int *v62; // rbx
  __int64 v63; // r12
  __int64 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // rax
  int v67; // eax
  __int64 v68; // r14
  __int64 v69; // rax
  __int64 v70; // r9
  __int64 v71; // rdi
  _BYTE *v72; // r10
  __int64 (__fastcall *v73)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v74; // rax
  __int64 v75; // r15
  int v76; // edx
  __int64 v77; // rcx
  bool v78; // zf
  __int64 v79; // rdx
  __int64 v80; // r9
  _BYTE *v81; // r13
  __int64 v82; // rdi
  int v83; // r11d
  int v84; // r10d
  __int64 (__fastcall *v85)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // r13
  unsigned int *v89; // r12
  unsigned int *v90; // r13
  __int64 v91; // rdx
  __int64 v92; // rax
  unsigned int *v93; // r14
  __int64 v94; // rdx
  __int64 v95; // rsi
  __int64 v96; // rax
  __int64 v97; // rax
  int v98; // [rsp-1E8h] [rbp-1E8h]
  int v99; // [rsp-1E8h] [rbp-1E8h]
  int v100; // [rsp-1E8h] [rbp-1E8h]
  int v101; // [rsp-1D8h] [rbp-1D8h]
  int v102; // [rsp-1D8h] [rbp-1D8h]
  int v103; // [rsp-1D8h] [rbp-1D8h]
  int v104; // [rsp-1D8h] [rbp-1D8h]
  unsigned int v105; // [rsp-1D0h] [rbp-1D0h]
  __int128 *v106; // [rsp-1D0h] [rbp-1D0h]
  int v107; // [rsp-1D0h] [rbp-1D0h]
  int v108; // [rsp-1D0h] [rbp-1D0h]
  int v109; // [rsp-1D0h] [rbp-1D0h]
  int v110; // [rsp-1D0h] [rbp-1D0h]
  __int64 v111; // [rsp-1D0h] [rbp-1D0h]
  int v112; // [rsp-1D0h] [rbp-1D0h]
  int v113; // [rsp-1D0h] [rbp-1D0h]
  __int64 v114; // [rsp-1C0h] [rbp-1C0h] BYREF
  _QWORD v115[4]; // [rsp-1B8h] [rbp-1B8h] BYREF
  __int16 v116; // [rsp-198h] [rbp-198h]
  __int64 v117[4]; // [rsp-188h] [rbp-188h] BYREF
  __int16 v118; // [rsp-168h] [rbp-168h]
  _DWORD *v119; // [rsp-158h] [rbp-158h] BYREF
  __int64 v120; // [rsp-150h] [rbp-150h]
  _DWORD v121[2]; // [rsp-148h] [rbp-148h] BYREF
  __int64 v122; // [rsp-140h] [rbp-140h]
  __int16 v123; // [rsp-138h] [rbp-138h]
  __int128 *v124; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v125; // [rsp-C0h] [rbp-C0h]
  __int128 v126; // [rsp-B8h] [rbp-B8h] BYREF
  __int128 v127; // [rsp-A8h] [rbp-A8h]
  __int64 v128; // [rsp-8h] [rbp-8h] BYREF

  if ( a2 <= 9 )
  {
    if ( a2 > 7 && *(_QWORD *)a1 == 0x747663662E657673LL )
      BUG();
    BUG();
  }
  if ( *(_QWORD *)a1 != 0x6366622E6E6F656ELL || *(_WORD *)(a1 + 8) != 29814 )
  {
    if ( *(_QWORD *)a1 != 0x747663662E657673LL )
      goto LABEL_4;
    if ( a2 == 16 )
    {
      if ( *(_QWORD *)(a1 + 8) == 0x323366363166622ELL )
      {
        v105 = 1291;
        goto LABEL_10;
      }
    }
    else if ( a2 == 18 && *(_QWORD *)(a1 + 8) == 0x66363166622E746ELL && *(_WORD *)(a1 + 16) == 12851 )
    {
      v105 = 1304;
LABEL_10:
      v7 = *a3;
      if ( v7 == 40 )
      {
        v8 = -32 - 32LL * (unsigned int)sub_B491D0(a3);
      }
      else
      {
        v8 = -32;
        if ( v7 != 85 )
        {
          v8 = -96;
          if ( v7 != 34 )
            BUG();
        }
      }
      if ( (a3[7] & 0x80u) != 0 )
      {
        v15 = sub_BD2BC0(a3);
        v17 = v15 + v16;
        v18 = 0;
        if ( (a3[7] & 0x80u) != 0 )
          v18 = sub_BD2BC0(a3);
        if ( (unsigned int)((v17 - v18) >> 4) )
        {
          if ( (a3[7] & 0x80u) == 0 )
            BUG();
          v19 = *(_DWORD *)(sub_BD2BC0(a3) + 8);
          if ( (a3[7] & 0x80u) == 0 )
            BUG();
          v20 = sub_BD2BC0(a3);
          v8 -= 32LL * (unsigned int)(*(_DWORD *)(v20 + v21 - 4) - v19);
        }
      }
      v22 = *((_DWORD *)a3 + 1);
      v23 = 0;
      v119 = v121;
      v24 = &a3[v8];
      v120 = 0x300000000LL;
      v25 = 32LL * (v22 & 0x7FFFFFF);
      v26 = v8 + v25;
      v27 = &a3[-v25];
      v28 = v121;
      v29 = v26 >> 5;
      if ( (unsigned __int64)v26 > 0x60 )
      {
        v101 = v26 >> 5;
        sub_C8D5F0(&v119, v121, v26 >> 5, 8);
        v23 = v120;
        v24 = &a3[v8];
        LODWORD(v29) = v101;
        v28 = &v119[2 * (unsigned int)v120];
      }
      if ( v27 != v24 )
      {
        do
        {
          if ( v28 )
            *v28 = *(_QWORD *)v27;
          v27 += 32;
          ++v28;
        }
        while ( v24 != v27 );
        v23 = v120;
      }
      v30 = *(_QWORD *)(a5 + 72);
      LODWORD(v120) = v29 + v23;
      v31 = sub_BCB2A0(v30);
      v32 = sub_BCDE10(v31, 8);
      v33 = *(_QWORD *)(a5 + 72);
      v114 = v32;
      v34 = sub_BCB2A0(v33);
      v115[0] = sub_BCDE10(v34, 4);
      if ( v114 == *(_QWORD *)(*((_QWORD *)v119 + 1) + 8LL) )
      {
        HIDWORD(v117[0]) = 0;
        LOWORD(v127) = 257;
        v35 = sub_B33D10(a5, 1249, (unsigned int)&v114, 1, (int)v119 + 8, 1, LODWORD(v117[0]), (__int64)&v124);
        v36 = (int)v119;
        *((_QWORD *)v119 + 1) = v35;
        HIDWORD(v117[0]) = 0;
        LOWORD(v127) = 257;
        v37 = sub_B33D10(a5, 1248, (unsigned int)v115, 1, v36 + 8, 1, LODWORD(v117[0]), (__int64)&v124);
        *((_QWORD *)v119 + 1) = v37;
        v38 = sub_BD5D20(a3);
        BYTE4(v117[0]) = 0;
        LOWORD(v127) = 261;
        v39 = (_BYTE *)v105;
        v125 = v40;
        v124 = (__int128 *)v38;
        v41 = sub_B33D10(a5, v105, 0, 0, (_DWORD)v119, v120, v117[0], (__int64)&v124);
        v42 = (__int128 *)v119;
        v13 = v41;
        if ( v119 == v121 )
          return v13;
LABEL_45:
        _libc_free(v42, v39);
        return v13;
      }
LABEL_4:
      BUG();
    }
    BUG();
  }
  if ( a2 <= 0xB )
  {
    if ( a2 == 10 )
    {
LABEL_23:
      LOWORD(v127) = 257;
      v9 = sub_B2BE50(a4);
      HIDWORD(v117[0]) = 0;
      v10 = sub_BCB150(v9);
      v11 = *((_DWORD *)a3 + 1);
      v119 = (_DWORD *)LODWORD(v117[0]);
      v12 = *(_QWORD *)&a3[-32 * (v11 & 0x7FFFFFF)];
      if ( *(_BYTE *)(a5 + 108) )
        return sub_B358C0(a5, 113, v12, v10, v117[0], (__int64)&v124, 0, 0);
      else
        return sub_A7EAA0((unsigned int **)a5, 0x2Du, v12, v10, (__int64)&v124, 0, v117[0], 0);
    }
LABEL_22:
    if ( *(_QWORD *)a1 == 0x6366622E6E6F656ELL && *(_WORD *)(a1 + 8) == 29814 && *(_BYTE *)(a1 + 10) == 110 )
    {
      v124 = &v126;
      v125 = 0x2000000008LL;
      v43 = 0;
      v126 = 0;
      v127 = 0;
      do
      {
        *((_DWORD *)&v126 + v43) = v43;
        ++v43;
      }
      while ( v43 != 8 );
      v44 = sub_B2BE50(a4);
      v45 = sub_BCB150(v44);
      HIDWORD(v115[0]) = 0;
      v46 = sub_BCDA70(v45, 4);
      v47 = *((_DWORD *)a3 + 1);
      v123 = 257;
      v117[0] = LODWORD(v115[0]);
      v48 = *(_QWORD *)&a3[-32 * (v47 & 0x7FFFFFF)];
      if ( *(_BYTE *)(a5 + 108) )
        v49 = (_BYTE *)sub_B358C0(a5, 113, v48, v46, v115[0], (__int64)&v119, 0, 0);
      else
        v49 = (_BYTE *)sub_A7EAA0((unsigned int **)a5, 0x2Du, v48, v46, (__int64)&v119, 0, v115[0], 0);
      v50 = sub_C5F790();
      v51 = *(_QWORD *)(v50 + 32);
      v52 = v50;
      if ( (unsigned __int64)(*(_QWORD *)(v50 + 24) - v51) <= 6 )
      {
        v52 = sub_CB6200(v50, "Trunc: ", 7);
      }
      else
      {
        *(_DWORD *)v51 = 1853190740;
        *(_WORD *)(v51 + 4) = 14947;
        *(_BYTE *)(v51 + 6) = 32;
        *(_QWORD *)(v50 + 32) += 7LL;
      }
      sub_A69870((__int64)v49, (_BYTE *)v52, 0);
      v53 = *(_BYTE **)(v52 + 32);
      if ( *(_BYTE **)(v52 + 24) == v53 )
      {
        sub_CB6200(v52, "\n", 1);
      }
      else
      {
        *v53 = 10;
        ++*(_QWORD *)(v52 + 32);
      }
      v118 = 257;
      v54 = (unsigned int)v125;
      v106 = v124;
      v55 = sub_AC9350(v46);
      v57 = *(_QWORD *)(a5 + 80);
      v58 = (_BYTE *)v55;
      v59 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v57 + 112LL);
      if ( v59 == sub_9B6630 )
      {
        if ( *v49 > 0x15u || *v58 > 0x15u )
        {
LABEL_69:
          v103 = (int)v58;
          v123 = 257;
          v61 = sub_BD2C40(112, unk_3F1FE60);
          v13 = v61;
          if ( v61 )
            sub_B4E9E0(v61, (_DWORD)v49, v103, (_DWORD)v106, v54, (unsigned int)&v119, 0, 0);
          v39 = (_BYTE *)v13;
          (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
            *(_QWORD *)(a5 + 88),
            v13,
            v117,
            *(_QWORD *)(a5 + 56),
            *(_QWORD *)(a5 + 64));
          v62 = *(unsigned int **)a5;
          v63 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
          while ( (unsigned int *)v63 != v62 )
          {
            v64 = *((_QWORD *)v62 + 1);
            v39 = (_BYTE *)*v62;
            v62 += 4;
            sub_B99FD0(v13, v39, v64);
          }
LABEL_64:
          v42 = v124;
          if ( v124 == &v126 )
            return v13;
          goto LABEL_45;
        }
        v39 = v58;
        v102 = (int)v58;
        v60 = sub_AD5CE0(v49, v58, v106, v54, 0, v56);
        LODWORD(v58) = v102;
        v13 = v60;
      }
      else
      {
        v104 = (int)v58;
        v39 = v49;
        v65 = ((__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *, __int128 *, __int64))v59)(v57, v49, v58, v106, v54);
        LODWORD(v58) = v104;
        v13 = v65;
      }
      if ( v13 )
        goto LABEL_64;
      goto LABEL_69;
    }
    goto LABEL_23;
  }
  if ( *(_DWORD *)(a1 + 8) != 846099574 )
    goto LABEL_22;
  v121[0] = 0;
  v120 = 0x2000000004LL;
  v122 = 0x300000002LL;
  v119 = v121;
  v121[1] = 1;
  v124 = &v126;
  v125 = 0x2000000008LL;
  v66 = 0;
  v126 = 0;
  v127 = 0;
  do
  {
    *((_DWORD *)&v126 + v66) = v66;
    ++v66;
  }
  while ( v66 != 8 );
  v67 = *((_DWORD *)a3 + 1);
  v116 = 257;
  v68 = *(_QWORD *)&a3[-32 * (v67 & 0x7FFFFFF)];
  v69 = sub_ACADE0(*(_QWORD *)(v68 + 8));
  v71 = *(_QWORD *)(a5 + 80);
  v72 = (_BYTE *)v69;
  v73 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v71 + 112LL);
  if ( v73 != sub_9B6630 )
  {
    v113 = (int)v72;
    v97 = ((__int64 (__fastcall *)(__int64, __int64, _BYTE *, _DWORD *, __int64))v73)(v71, v68, v72, v121, 4);
    LODWORD(v72) = v113;
    v75 = v97;
LABEL_81:
    if ( v75 )
      goto LABEL_82;
    goto LABEL_99;
  }
  if ( *(_BYTE *)v68 <= 0x15u && *v72 <= 0x15u )
  {
    v107 = (int)v72;
    v74 = sub_AD5CE0(v68, v72, v121, 4, 0, v70);
    LODWORD(v72) = v107;
    v75 = v74;
    goto LABEL_81;
  }
LABEL_99:
  v110 = (int)v72;
  v118 = 257;
  v92 = sub_BD2C40(112, unk_3F1FE60);
  v75 = v92;
  if ( v92 )
    sub_B4E9E0(v92, v68, v110, (unsigned int)&v128 - 320, 4, (unsigned int)&v128 - 384, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
    *(_QWORD *)(a5 + 88),
    v75,
    v115,
    *(_QWORD *)(a5 + 56),
    *(_QWORD *)(a5 + 64));
  v93 = *(unsigned int **)a5;
  v111 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
  if ( *(_QWORD *)a5 != v111 )
  {
    do
    {
      v94 = *((_QWORD *)v93 + 1);
      v95 = *v93;
      v93 += 4;
      sub_B99FD0(v75, v95, v94);
    }
    while ( (unsigned int *)v111 != v93 );
  }
LABEL_82:
  v76 = *((_DWORD *)a3 + 1);
  v118 = 257;
  v77 = *(_QWORD *)(v75 + 8);
  HIDWORD(v114) = 0;
  v78 = *(_BYTE *)(a5 + 108) == 0;
  v115[0] = (unsigned int)v114;
  v79 = *(_QWORD *)&a3[32 * (1LL - (v76 & 0x7FFFFFF))];
  if ( v78 )
    v81 = (_BYTE *)sub_A7EAA0((unsigned int **)a5, 0x2Du, v79, v77, (__int64)v117, 0, v114, 0);
  else
    v81 = (_BYTE *)sub_B358C0(a5, 113, v79, v77, v114, (__int64)v117, 0, 0);
  v82 = *(_QWORD *)(a5 + 80);
  v83 = v125;
  v116 = 257;
  v84 = (int)v124;
  v85 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v82 + 112LL);
  if ( v85 != sub_9B6630 )
  {
    v100 = (int)v124;
    v112 = v125;
    v39 = (_BYTE *)v75;
    v96 = ((__int64 (__fastcall *)(__int64, __int64, _BYTE *, __int128 *, _QWORD))v85)(
            v82,
            v75,
            v81,
            v124,
            (unsigned int)v125);
    v84 = v100;
    v83 = v112;
    v13 = v96;
LABEL_88:
    if ( v13 )
      goto LABEL_89;
    goto LABEL_94;
  }
  if ( *(_BYTE *)v75 <= 0x15u && *v81 <= 0x15u )
  {
    v39 = v81;
    v98 = (int)v124;
    v108 = v125;
    v86 = sub_AD5CE0(v75, v81, v124, (unsigned int)v125, 0, v80);
    v83 = v108;
    v84 = v98;
    v13 = v86;
    goto LABEL_88;
  }
LABEL_94:
  v99 = v83;
  v109 = v84;
  v118 = 257;
  v87 = sub_BD2C40(112, unk_3F1FE60);
  v13 = v87;
  if ( v87 )
    sub_B4E9E0(v87, v75, (_DWORD)v81, v109, v99, (unsigned int)v117, 0, 0);
  v39 = (_BYTE *)v13;
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
    *(_QWORD *)(a5 + 88),
    v13,
    v115,
    *(_QWORD *)(a5 + 56),
    *(_QWORD *)(a5 + 64));
  v88 = 4LL * *(unsigned int *)(a5 + 8);
  v89 = *(unsigned int **)a5;
  v90 = &v89[v88];
  while ( v90 != v89 )
  {
    v91 = *((_QWORD *)v89 + 1);
    v39 = (_BYTE *)*v89;
    v89 += 4;
    sub_B99FD0(v13, v39, v91);
  }
LABEL_89:
  if ( v124 != &v126 )
    _libc_free(v124, v39);
  v42 = (__int128 *)v119;
  if ( v119 != v121 )
    goto LABEL_45;
  return v13;
}
