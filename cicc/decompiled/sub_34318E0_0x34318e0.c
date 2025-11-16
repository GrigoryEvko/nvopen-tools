// Function: sub_34318E0
// Address: 0x34318e0
//
__int64 __fastcall sub_34318E0(__int64 a1, __int64 *a2)
{
  __int64 v3; // r12
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 (*v8)(); // rax
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r14
  unsigned __int64 v14; // rdi
  __int64 v15; // r8
  __int64 v16; // r9
  int v17; // eax
  int v18; // eax
  __int64 v19; // rcx
  int *v20; // r15
  int v21; // r13d
  int v22; // r12d
  __int64 v23; // r8
  _DWORD *v24; // rax
  int v25; // edi
  int v26; // ecx
  _DWORD *v27; // rdx
  int v28; // edx
  __int64 v29; // r12
  unsigned int v30; // r14d
  __int64 v31; // r13
  __int64 v32; // rcx
  __int64 v33; // rax
  unsigned int v34; // edx
  unsigned int *v35; // r10
  __int64 v36; // r15
  __int64 v37; // rax
  unsigned int v38; // r8d
  unsigned __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // rcx
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // r15
  __int64 v48; // r14
  __int64 v49; // r13
  __int64 v50; // r12
  __int64 j; // r15
  __int64 v52; // rax
  __int64 v54; // rax
  unsigned __int64 v55; // rcx
  unsigned int *v56; // r10
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rcx
  __int64 v61; // r11
  __int16 v62; // dx
  unsigned __int64 v63; // rax
  __int64 v64; // rdx
  signed int v65; // esi
  __int64 v66; // rdx
  __int64 v67; // r13
  unsigned __int64 v68; // r14
  __int64 v69; // rax
  unsigned __int64 v70; // rdx
  __int64 v71; // rdx
  __int64 v72; // rdx
  int *v73; // rsi
  int *v74; // r15
  int v75; // ecx
  int *v76; // rax
  int v77; // r9d
  int *v78; // r12
  int v79; // r14d
  int *v80; // r10
  int v81; // r8d
  unsigned int v82; // eax
  int *v83; // r13
  int v84; // edx
  __int64 v85; // rax
  __int64 v86; // rax
  int v87; // edi
  _BYTE *v88; // rax
  int v89; // r10d
  int v90; // r9d
  __int64 v91; // rax
  char v92; // al
  __int64 v93; // rdx
  unsigned __int64 *v94; // rcx
  __int64 v95; // rdx
  unsigned __int64 v96; // rsi
  unsigned int v97; // ecx
  int v98; // r9d
  int v99; // edi
  unsigned int v100; // r14d
  _DWORD *v101; // rcx
  int v102; // r8d
  unsigned int *v103; // [rsp+0h] [rbp-D0h]
  unsigned int *v105; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v106; // [rsp+8h] [rbp-C8h]
  unsigned int *v107; // [rsp+8h] [rbp-C8h]
  __int64 v108; // [rsp+8h] [rbp-C8h]
  __int64 v109; // [rsp+10h] [rbp-C0h]
  __int64 v110; // [rsp+10h] [rbp-C0h]
  _QWORD *v111; // [rsp+18h] [rbp-B8h]
  char v112; // [rsp+27h] [rbp-A9h]
  _QWORD *v113; // [rsp+28h] [rbp-A8h]
  __int64 *v114; // [rsp+28h] [rbp-A8h]
  unsigned int v115; // [rsp+30h] [rbp-A0h]
  __int64 v116; // [rsp+30h] [rbp-A0h]
  int v117; // [rsp+30h] [rbp-A0h]
  int v118; // [rsp+30h] [rbp-A0h]
  unsigned __int64 *v119; // [rsp+30h] [rbp-A0h]
  int *v120; // [rsp+38h] [rbp-98h]
  __int64 *v121; // [rsp+38h] [rbp-98h]
  __int64 *v122; // [rsp+38h] [rbp-98h]
  __int64 i; // [rsp+38h] [rbp-98h]
  unsigned __int64 v124; // [rsp+38h] [rbp-98h]
  unsigned int *v125; // [rsp+38h] [rbp-98h]
  char v126; // [rsp+38h] [rbp-98h]
  __int64 v127; // [rsp+38h] [rbp-98h]
  unsigned __int8 *v128; // [rsp+48h] [rbp-88h] BYREF
  unsigned __int64 v129; // [rsp+50h] [rbp-80h]
  __int64 v130; // [rsp+58h] [rbp-78h]
  unsigned __int64 v131; // [rsp+60h] [rbp-70h]
  __int64 v132; // [rsp+68h] [rbp-68h]
  __int64 *v133; // [rsp+70h] [rbp-60h] BYREF
  __int64 v134; // [rsp+78h] [rbp-58h]
  __int64 v135; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v136; // [rsp+88h] [rbp-48h]

  sub_35D45B0(*(_QWORD *)(a1 + 32));
  v3 = *a2;
  v112 = sub_2E79980(a2);
  sub_374DD20(*(_QWORD *)(a1 + 24), **(_QWORD **)(a1 + 40), *(_QWORD *)(a1 + 40), *(_QWORD *)(a1 + 64));
  v4 = *(_QWORD *)(a1 + 72);
  v5 = a1 + 80;
  if ( !*(_BYTE *)(a1 + 760) )
    v5 = 0;
  sub_3372D50(v4, *(_QWORD *)(a1 + 776), v5, *(_QWORD *)(a1 + 768), *(_QWORD *)(a1 + 16));
  *(_BYTE *)(*(_QWORD *)(a1 + 40) + 342LL) = 0;
  *(_BYTE *)(*(_QWORD *)(a1 + 24) + 49LL) = 0;
  if ( *(_DWORD *)(a1 + 792) )
  {
    v6 = *(_QWORD *)(a1 + 808);
    v7 = *(_QWORD *)(a1 + 40);
    v8 = *(__int64 (**)())(*(_QWORD *)v6 + 2224LL);
    if ( v8 == sub_302E1C0 )
      goto LABEL_5;
    if ( ((unsigned __int8 (__fastcall *)(__int64, __int64))v8)(v6, v7) )
    {
      *(_BYTE *)(*(_QWORD *)(a1 + 24) + 49LL) = 1;
      v67 = *(_QWORD *)(v3 + 80);
      if ( v67 != v3 + 72 )
      {
        while ( 1 )
        {
          if ( !v67 )
            BUG();
          v68 = *(_QWORD *)(v67 + 24) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v68 == v67 + 24 )
            goto LABEL_203;
          if ( !v68 )
            BUG();
          if ( (unsigned int)*(unsigned __int8 *)(v68 - 24) - 30 > 0xA )
LABEL_203:
            BUG();
          if ( !(unsigned int)sub_B46E30(v68 - 24) )
          {
            v92 = *(_BYTE *)(v68 - 24);
            if ( v92 != 36 && v92 != 30 )
              break;
          }
          v67 = *(_QWORD *)(v67 + 8);
          if ( v67 == v3 + 72 )
            goto LABEL_109;
        }
        *(_BYTE *)(*(_QWORD *)(a1 + 24) + 49LL) = 0;
        v7 = *(_QWORD *)(a1 + 40);
        goto LABEL_5;
      }
    }
  }
LABEL_109:
  v7 = *(_QWORD *)(a1 + 40);
LABEL_5:
  v113 = *(_QWORD **)(v7 + 328);
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 24) + 49LL) )
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 808) + 2248LL))(*(_QWORD *)(a1 + 808));
  sub_342F5C0(a1, v3);
  if ( *(_BYTE *)(a1 + 816) && (_BYTE)qword_5039D28 )
  {
    v135 = v3;
    v134 = 0x10000000BLL;
    v133 = (__int64 *)&unk_49D9EE8;
    v91 = sub_B2BE50(v3);
    sub_B6EB20(v91, (__int64)&v133);
  }
  v9 = *(_QWORD *)(a1 + 40);
  v111 = *(_QWORD **)(v9 + 32);
  v10 = *(_QWORD *)(a1 + 24);
  if ( *(_DWORD *)(v10 + 480) )
  {
    v72 = *(unsigned int *)(v10 + 488);
    v73 = *(int **)(v10 + 472);
    v74 = &v73[2 * v72];
    v75 = *(_DWORD *)(v10 + 488);
    if ( v73 != v74 )
    {
      v76 = *(int **)(v10 + 472);
      while ( 1 )
      {
        v77 = *v76;
        v78 = v76;
        if ( (unsigned int)*v76 <= 0xFFFFFFFD )
          break;
        v76 += 2;
        if ( v74 == v76 )
          goto LABEL_11;
      }
      if ( v74 != v76 )
      {
LABEL_129:
        v79 = v78[1];
        v80 = &v73[2 * v72];
        v81 = v75 - 1;
        while ( 1 )
        {
          if ( v75 )
          {
            v82 = v81 & (37 * v79);
            v83 = &v73[2 * v82];
            v84 = *v83;
            if ( v79 == *v83 )
            {
LABEL_131:
              if ( v74 == v83 )
                goto LABEL_135;
              goto LABEL_132;
            }
            v87 = 1;
            while ( v84 != -1 )
            {
              v82 = v81 & (v87 + v82);
              v83 = &v73[2 * v82];
              v84 = *v83;
              if ( v79 == *v83 )
                goto LABEL_131;
              ++v87;
            }
          }
          v83 = v80;
          if ( v74 == v80 )
          {
LABEL_135:
            if ( v77 < 0 )
            {
              if ( v79 >= 0 )
                goto LABEL_154;
              v118 = v77;
              sub_2EBE590(
                (__int64)v111,
                v79,
                *(_QWORD *)(v111[7] + 16LL * (v77 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                0);
              v77 = v118;
LABEL_137:
              v85 = *(_QWORD *)(v111[7] + 16LL * (v79 & 0x7FFFFFFF) + 8);
            }
            else
            {
              if ( v79 < 0 )
                goto LABEL_137;
LABEL_154:
              v85 = *(_QWORD *)(v111[38] + 8LL * (unsigned int)v79);
            }
            if ( v85 )
            {
              if ( (*(_BYTE *)(v85 + 3) & 0x10) != 0 )
              {
                while ( 1 )
                {
                  v85 = *(_QWORD *)(v85 + 32);
                  if ( !v85 )
                    break;
                  if ( (*(_BYTE *)(v85 + 3) & 0x10) == 0 )
                    goto LABEL_140;
                }
              }
              else
              {
LABEL_140:
                v117 = v77;
                sub_2EBF120((__int64)v111, v77);
                v77 = v117;
              }
            }
            sub_2EBECB0(v111, v77, v79);
            while ( 1 )
            {
              v78 += 2;
              if ( v78 == v83 )
                break;
              if ( (unsigned int)*v78 <= 0xFFFFFFFD )
              {
                if ( v78 == v83 )
                  break;
                v77 = *v78;
                v86 = *(_QWORD *)(a1 + 24);
                v72 = *(unsigned int *)(v86 + 488);
                v73 = *(int **)(v86 + 472);
                v75 = *(_DWORD *)(v86 + 488);
                goto LABEL_129;
              }
            }
            v9 = *(_QWORD *)(a1 + 40);
            break;
          }
LABEL_132:
          v79 = v83[1];
        }
      }
    }
  }
LABEL_11:
  v11 = (__int64)v113;
  v109 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v9 + 16) + 200LL))(*(_QWORD *)(v9 + 16));
  sub_2EBFDA0(*(_QWORD **)(a1 + 56), v113, v109, *(_QWORD *)(a1 + 800));
  v12 = *(_QWORD *)(a1 + 24);
  if ( *(_BYTE *)(v12 + 49) )
  {
    v134 = 0x400000000LL;
    v133 = &v135;
    v13 = a2[41];
    if ( (__int64 *)v13 != a2 + 40 )
    {
      do
      {
        if ( !*(_DWORD *)(v13 + 120) )
        {
          v14 = sub_2E313E0(v13);
          if ( v14 != v13 + 48 )
          {
            v17 = *(_DWORD *)(v14 + 44);
            if ( (v17 & 4) == 0 && (v17 & 8) != 0 )
            {
              if ( sub_2E88A90(v14, 32, 1) )
              {
LABEL_115:
                v69 = (unsigned int)v134;
                v70 = (unsigned int)v134 + 1LL;
                if ( v70 > HIDWORD(v134) )
                {
                  sub_C8D5F0((__int64)&v133, &v135, v70, 8u, v15, v16);
                  v69 = (unsigned int)v134;
                }
                v133[v69] = v13;
                LODWORD(v134) = v134 + 1;
              }
            }
            else if ( (*(_QWORD *)(*(_QWORD *)(v14 + 16) + 24LL) & 0x20LL) != 0 )
            {
              goto LABEL_115;
            }
          }
        }
        v13 = *(_QWORD *)(v13 + 8);
      }
      while ( a2 + 40 != (__int64 *)v13 );
    }
    v11 = (__int64)v113;
    (*(void (__fastcall **)(_QWORD, _QWORD *, __int64 **))(**(_QWORD **)(a1 + 808) + 2256LL))(
      *(_QWORD *)(a1 + 808),
      v113,
      &v133);
    if ( v133 != &v135 )
      _libc_free((unsigned __int64)v133);
    v12 = *(_QWORD *)(a1 + 24);
  }
  v133 = 0;
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v18 = *(_DWORD *)(v12 + 320);
  if ( !v18 )
    goto LABEL_54;
  v19 = *(_QWORD *)(a1 + 56);
  v20 = *(int **)(v19 + 488);
  v120 = *(int **)(v19 + 496);
  if ( v20 == v120 )
    goto LABEL_39;
  do
  {
    while ( 1 )
    {
      v21 = v20[1];
      if ( v21 )
      {
        v11 = v136;
        v22 = *v20;
        if ( !v136 )
        {
          v133 = (__int64 *)((char *)v133 + 1);
LABEL_172:
          v11 = 2 * v136;
          sub_342D120((__int64)&v133, v11);
          if ( !v136 )
            goto LABEL_202;
          v97 = (v136 - 1) & (37 * v22);
          v28 = v135 + 1;
          v24 = (_DWORD *)(v134 + 8LL * v97);
          v98 = *v24;
          if ( v22 != *v24 )
          {
            v99 = 1;
            v11 = 0;
            while ( v98 != -1 )
            {
              if ( !v11 && v98 == -2 )
                v11 = (__int64)v24;
              v97 = (v136 - 1) & (v99 + v97);
              v24 = (_DWORD *)(v134 + 8LL * v97);
              v98 = *v24;
              if ( *v24 == v22 )
                goto LABEL_35;
              ++v99;
            }
            if ( v11 )
              v24 = (_DWORD *)v11;
          }
          goto LABEL_35;
        }
        LODWORD(v23) = (v136 - 1) & (37 * v22);
        v24 = (_DWORD *)(v134 + 8LL * (unsigned int)v23);
        v25 = *v24;
        if ( *v24 != v22 )
          break;
      }
LABEL_25:
      v20 += 2;
      if ( v120 == v20 )
        goto LABEL_38;
    }
    v26 = 1;
    v27 = 0;
    while ( v25 != -1 )
    {
      if ( v25 == -2 && !v27 )
        v27 = v24;
      v23 = (v136 - 1) & ((_DWORD)v23 + v26);
      v24 = (_DWORD *)(v134 + 8 * v23);
      v25 = *v24;
      if ( *v24 == v22 )
        goto LABEL_25;
      ++v26;
    }
    if ( v27 )
      v24 = v27;
    v133 = (__int64 *)((char *)v133 + 1);
    v28 = v135 + 1;
    if ( 4 * ((int)v135 + 1) >= 3 * v136 )
      goto LABEL_172;
    if ( v136 - HIDWORD(v135) - v28 <= v136 >> 3 )
    {
      sub_342D120((__int64)&v133, v136);
      if ( !v136 )
      {
LABEL_202:
        LODWORD(v135) = v135 + 1;
        BUG();
      }
      v11 = 1;
      v100 = (v136 - 1) & (37 * v22);
      v28 = v135 + 1;
      v101 = 0;
      v24 = (_DWORD *)(v134 + 8LL * v100);
      v102 = *v24;
      if ( v22 != *v24 )
      {
        while ( v102 != -1 )
        {
          if ( v102 == -2 && !v101 )
            v101 = v24;
          v100 = (v136 - 1) & (v11 + v100);
          v24 = (_DWORD *)(v134 + 8LL * v100);
          v102 = *v24;
          if ( *v24 == v22 )
            goto LABEL_35;
          v11 = (unsigned int)(v11 + 1);
        }
        if ( v101 )
          v24 = v101;
      }
    }
LABEL_35:
    LODWORD(v135) = v28;
    if ( *v24 != -1 )
      --HIDWORD(v135);
    *v24 = v22;
    v20 += 2;
    v24[1] = v21;
  }
  while ( v120 != v20 );
LABEL_38:
  v12 = *(_QWORD *)(a1 + 24);
  v18 = *(_DWORD *)(v12 + 320);
  if ( v18 )
  {
LABEL_39:
    v29 = (__int64)v113;
    v30 = v18 - 1;
    v31 = v109;
    v114 = v113 + 5;
    while ( 1 )
    {
      v36 = *(_QWORD *)(*(_QWORD *)(v12 + 312) + 8LL * v30);
      v37 = *(_QWORD *)(v36 + 32);
      if ( *(_WORD *)(v36 + 68) == 14 )
      {
        if ( *(_BYTE *)v37 != 5 )
          goto LABEL_49;
      }
      else if ( *(_BYTE *)(v37 + 80) != 5 )
      {
        v37 += 80;
LABEL_49:
        v38 = *(_DWORD *)(v37 + 8);
        goto LABEL_50;
      }
      v38 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v31 + 680LL))(v31, *(_QWORD *)(a1 + 40));
LABEL_50:
      if ( v38 - 1 <= 0x3FFFFFFE )
      {
        v11 = v36;
        v115 = v38;
        v121 = *(__int64 **)(v29 + 56);
        sub_2E31040(v114, v36);
        v32 = *v121;
        v33 = *(_QWORD *)v36 & 7LL;
        *(_QWORD *)(v36 + 8) = v121;
        v32 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v36 = v32 | v33;
        *(_QWORD *)(v32 + 8) = v36;
        *v121 = v36 | *v121 & 7;
        if ( !v112 && v136 )
        {
          v34 = (v136 - 1) & (37 * v115);
          v35 = (unsigned int *)(v134 + 8LL * v34);
          v11 = *v35;
          if ( v115 == (_DWORD)v11 )
          {
LABEL_43:
            if ( v35 != (unsigned int *)(v134 + 8LL * v136) )
            {
              v105 = v35;
              v124 = sub_2EBEE10(*(_QWORD *)(a1 + 56), v35[1]);
              v116 = sub_2E89170(v36);
              v54 = sub_2E891C0(v36);
              v55 = v124;
              v110 = v54;
              v56 = v105;
              v128 = *(unsigned __int8 **)(v36 + 56);
              if ( v128 )
              {
                v106 = v124;
                v125 = v56;
                sub_B96E90((__int64)&v128, (__int64)v128, 1);
                v55 = v106;
                v56 = v125;
              }
              v126 = 0;
              if ( *(_WORD *)(v36 + 68) == 14 )
              {
                v88 = *(_BYTE **)(v36 + 32);
                if ( v88[40] == 1 )
                  v126 = *v88 == 0;
              }
              if ( !v55 )
                BUG();
              if ( (*(_BYTE *)v55 & 4) == 0 )
              {
                while ( (*(_BYTE *)(v55 + 44) & 8) != 0 )
                  v55 = *(_QWORD *)(v55 + 8);
              }
              v107 = v56;
              sub_2E90120(
                v29,
                *(unsigned __int64 **)(v55 + 8),
                &v128,
                *(_QWORD *)(*(_QWORD *)(a1 + 800) + 8LL) - 560LL,
                v126,
                v56[1],
                v116,
                v110);
              v57 = *(_QWORD *)(a1 + 56);
              v58 = v107[1];
              if ( (int)v58 < 0 )
                v59 = *(_QWORD *)(*(_QWORD *)(v57 + 56) + 16 * (v58 & 0x7FFFFFFF) + 8);
              else
                v59 = *(_QWORD *)(*(_QWORD *)(v57 + 304) + 8 * v58);
              if ( v59 )
              {
                if ( (*(_BYTE *)(v59 + 3) & 0x10) != 0 )
                {
                  while ( 1 )
                  {
                    v59 = *(_QWORD *)(v59 + 32);
                    if ( !v59 )
                      break;
                    if ( (*(_BYTE *)(v59 + 3) & 0x10) == 0 )
                      goto LABEL_90;
                  }
                }
                else
                {
LABEL_90:
                  v60 = *(_QWORD *)(v59 + 16);
                  v61 = 0;
                  v62 = *(_WORD *)(v60 + 68);
                  if ( (unsigned __int16)(v62 - 14) <= 1u )
                    goto LABEL_93;
LABEL_120:
                  if ( !v61 && v62 == 20 && v29 == *(_QWORD *)(v60 + 24) )
                  {
                    v61 = v60;
LABEL_93:
                    while ( 1 )
                    {
                      v59 = *(_QWORD *)(v59 + 32);
                      if ( !v59 )
                        break;
                      if ( (*(_BYTE *)(v59 + 3) & 0x10) == 0 )
                      {
                        v71 = *(_QWORD *)(v59 + 16);
                        if ( v71 != v60 )
                        {
                          v60 = *(_QWORD *)(v59 + 16);
                          v62 = *(_WORD *)(v71 + 68);
                          if ( (unsigned __int16)(v62 - 14) > 1u )
                            goto LABEL_120;
                        }
                      }
                    }
                    v103 = v107;
                    if ( v61 )
                    {
                      v108 = v61;
                      v63 = sub_2FF6F50(v31, *(_DWORD *)(*(_QWORD *)(v61 + 32) + 8LL), (__int64)v111);
                      v132 = v64;
                      v65 = v103[1];
                      v131 = v63;
                      v129 = sub_2FF6F50(v31, v65, (__int64)v111);
                      v130 = v66;
                      if ( v129 == v131 && (_BYTE)v130 == (_BYTE)v132 )
                      {
                        sub_2E8FEC0(
                          *(_QWORD **)(a1 + 40),
                          &v128,
                          *(_QWORD *)(*(_QWORD *)(a1 + 800) + 8LL) - 560LL,
                          v126,
                          *(_DWORD *)(*(_QWORD *)(v108 + 32) + 8LL),
                          v116,
                          v110);
                        if ( v29 + 48 == (*(_QWORD *)(v29 + 48) & 0xFFFFFFFFFFFFFFF8LL) )
                          v94 = *(unsigned __int64 **)(v29 + 56);
                        else
                          v94 = *(unsigned __int64 **)(v108 + 8);
                        v119 = v94;
                        v127 = v93;
                        sub_2E31040(v114, v93);
                        v95 = *(_QWORD *)v127;
                        v96 = *v119;
                        *(_QWORD *)(v127 + 8) = v119;
                        v96 &= 0xFFFFFFFFFFFFFFF8LL;
                        *(_QWORD *)v127 = v96 | v95 & 7;
                        *(_QWORD *)(v96 + 8) = v127;
                        *v119 = *v119 & 7 | v127;
                      }
                    }
                  }
                }
              }
              v11 = (__int64)v128;
              if ( v128 )
                sub_B91220((__int64)&v128, (__int64)v128);
            }
          }
          else
          {
            v89 = 1;
            while ( (_DWORD)v11 != -1 )
            {
              v90 = v89 + 1;
              v34 = (v136 - 1) & (v34 + v89);
              v35 = (unsigned int *)(v134 + 8LL * v34);
              v11 = *v35;
              if ( v115 == (_DWORD)v11 )
                goto LABEL_43;
              v89 = v90;
            }
          }
        }
LABEL_44:
        if ( !v30 )
          break;
        goto LABEL_45;
      }
      v11 = v38;
      v39 = sub_2EBEE10(*(_QWORD *)(a1 + 56), v38);
      if ( !v39 )
        goto LABEL_44;
      v40 = *(_QWORD *)(v39 + 24);
      if ( (*(_BYTE *)v39 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v39 + 44) & 8) != 0 )
          v39 = *(_QWORD *)(v39 + 8);
      }
      v11 = v36;
      v122 = *(__int64 **)(v39 + 8);
      sub_2E31040((__int64 *)(v40 + 40), v36);
      v41 = *v122;
      v42 = *(_QWORD *)v36 & 7LL;
      *(_QWORD *)(v36 + 8) = v122;
      v41 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v36 = v41 | v42;
      *(_QWORD *)(v41 + 8) = v36;
      *v122 = *v122 & 7 | v36;
      if ( !v30 )
        break;
LABEL_45:
      v12 = *(_QWORD *)(a1 + 24);
      --v30;
    }
  }
LABEL_54:
  if ( (unsigned __int8)sub_2E799E0(*(_QWORD *)(a1 + 40)) )
    sub_2E7F590(*(_QWORD *)(a1 + 40), v11, v43, v44, v45, v46);
  v47 = *(_QWORD *)(a1 + 40);
  v48 = *(_QWORD *)(v47 + 328);
  v49 = *(_QWORD *)(v47 + 48);
  for ( i = v47 + 320; i != v48; v48 = *(_QWORD *)(v48 + 8) )
  {
    if ( *(_BYTE *)(v49 + 66) && *(_BYTE *)(*(_QWORD *)(a1 + 40) + 342LL) )
      break;
    v50 = *(_QWORD *)(v48 + 56);
    for ( j = v48 + 48; j != v50; v50 = *(_QWORD *)(v50 + 8) )
    {
      while ( 1 )
      {
        v52 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 800) + 8LL) - 40LL * *(unsigned __int16 *)(v50 + 68) + 24);
        if ( (v52 & 0x80u) != 0LL && (v52 & 0x20) == 0 || (unsigned __int8)sub_2E89070(v50) )
          *(_BYTE *)(v49 + 66) = 1;
        if ( (unsigned int)*(unsigned __int16 *)(v50 + 68) - 1 <= 1 )
          *(_BYTE *)(*(_QWORD *)(a1 + 40) + 342LL) = 1;
        if ( (*(_BYTE *)v50 & 4) == 0 )
          break;
        v50 = *(_QWORD *)(v50 + 8);
        if ( j == v50 )
          goto LABEL_69;
      }
      while ( (*(_BYTE *)(v50 + 44) & 8) != 0 )
        v50 = *(_QWORD *)(v50 + 8);
    }
LABEL_69:
    ;
  }
  sub_374B790(*(_QWORD *)(a1 + 24));
  sub_C7D6A0(v134, 8LL * v136, 4);
  return 1;
}
