// Function: sub_1E9E980
// Address: 0x1e9e980
//
bool __fastcall sub_1E9E980(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  _BOOL4 v3; // eax
  __int64 v5; // rbx
  __int64 v6; // rdx
  _QWORD *v7; // rdi
  __int64 v8; // rax
  unsigned int v9; // eax
  unsigned int *v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // r13
  int v13; // r12d
  char *v14; // r13
  __int64 v15; // rcx
  unsigned __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r15
  __int64 v19; // rax
  __int16 *v20; // rax
  __int16 v21; // dx
  __int64 v22; // rax
  char *v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rdx
  unsigned __int64 v27; // r8
  int v28; // eax
  unsigned int n; // eax
  __int64 v30; // r12
  int v31; // eax
  char v32; // cl
  int v33; // ecx
  __int64 v34; // r9
  int v35; // esi
  _DWORD *v36; // r12
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rdx
  unsigned int j; // eax
  _DWORD *v40; // rdx
  int v41; // r10d
  unsigned int v42; // esi
  unsigned int v43; // r12d
  unsigned int v44; // r12d
  unsigned int v45; // eax
  __int64 v46; // rdx
  unsigned int v47; // edi
  __int64 v48; // rcx
  __int64 v49; // rcx
  __int64 v50; // rdx
  __int64 v51; // rdi
  unsigned int v52; // esi
  unsigned int v53; // eax
  unsigned int i; // edx
  __int64 v55; // rcx
  int v56; // ecx
  __int64 v57; // rax
  __int64 v58; // rax
  unsigned int *v59; // rax
  unsigned int v60; // edx
  char *v61; // rdi
  int v62; // eax
  int v63; // eax
  bool v64; // zf
  unsigned int v65; // eax
  __int64 v66; // rdx
  __int64 v67; // rax
  char v68; // al
  __int64 v69; // rdx
  unsigned __int64 v70; // rcx
  __int64 v71; // rdi
  __int64 (*v72)(); // rax
  __int64 v73; // rax
  __int64 v74; // rdi
  int v75; // edx
  int v76; // r10d
  unsigned __int64 v77; // r8
  unsigned __int64 v78; // r8
  unsigned int k; // eax
  unsigned int v80; // eax
  __int64 v81; // rdi
  int v82; // edx
  int v83; // r10d
  unsigned __int64 v84; // r8
  unsigned __int64 v85; // r8
  unsigned int m; // eax
  unsigned int v87; // eax
  unsigned int v88; // eax
  int v89; // edx
  __int64 v90; // rax
  unsigned int v91; // r15d
  __int64 v92; // r8
  int v93; // edx
  char *v94; // rax
  __int64 v95; // rdx
  __int64 v96; // rcx
  __int64 v97; // rsi
  __int64 v98; // rax
  __int64 v99; // rax
  unsigned int v100; // eax
  unsigned int v101; // eax
  __int64 v102; // r14
  __int64 v103; // rbx
  int v104; // eax
  unsigned __int64 v105; // r12
  __int64 v106; // rax
  __int64 v108; // [rsp+18h] [rbp-178h]
  int v109; // [rsp+20h] [rbp-170h]
  __int64 v110; // [rsp+20h] [rbp-170h]
  int v111; // [rsp+28h] [rbp-168h]
  unsigned int v112; // [rsp+2Ch] [rbp-164h]
  unsigned int v113; // [rsp+30h] [rbp-160h]
  __int64 v114; // [rsp+30h] [rbp-160h]
  unsigned __int64 v115; // [rsp+38h] [rbp-158h]
  __int64 v116; // [rsp+40h] [rbp-150h]
  bool v118; // [rsp+58h] [rbp-138h]
  __int64 v119; // [rsp+58h] [rbp-138h]
  unsigned __int64 v120; // [rsp+60h] [rbp-130h] BYREF
  __int64 v121; // [rsp+68h] [rbp-128h]
  _BYTE v122[16]; // [rsp+70h] [rbp-120h] BYREF
  __int64 v123; // [rsp+80h] [rbp-110h]
  char *v124; // [rsp+90h] [rbp-100h] BYREF
  __int128 v125; // [rsp+98h] [rbp-F8h] BYREF
  __int128 v126; // [rsp+A8h] [rbp-E8h]
  _QWORD *v127; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v128; // [rsp+C8h] [rbp-C8h]
  _QWORD v129[4]; // [rsp+D0h] [rbp-C0h] BYREF
  char *v130; // [rsp+F0h] [rbp-A0h] BYREF
  unsigned __int64 v131; // [rsp+F8h] [rbp-98h] BYREF
  __int64 v132; // [rsp+100h] [rbp-90h] BYREF
  char v133; // [rsp+108h] [rbp-88h] BYREF
  __int64 v134; // [rsp+110h] [rbp-80h]
  __int64 v135; // [rsp+118h] [rbp-78h]

  LOBYTE(v3) = 0;
  if ( (int)a2 > 0 )
    return v3;
  v5 = a1;
  v6 = *(_QWORD *)(a1 + 248);
  v7 = v129;
  v111 = 0;
  v8 = *(_QWORD *)(*(_QWORD *)(v6 + 24) + 16 * (a2 & 0x7FFFFFFF));
  v127 = v129;
  v129[0] = a2;
  v115 = v8 & 0xFFFFFFFFFFFFFFF8LL;
  v128 = 0x400000001LL;
  v9 = 1;
LABEL_4:
  v10 = (unsigned int *)&v7[v9 - 1];
  v11 = v10[1];
  v12 = *v10;
  LODWORD(v128) = v9 - 1;
  v13 = v12;
  v112 = v11;
  v14 = (char *)((v11 << 32) | v12);
  if ( v13 > 0 )
    goto LABEL_58;
  v108 = *(_QWORD *)(v5 + 232);
  v116 = *(_QWORD *)(v5 + 248);
  v18 = sub_1E69D00(v116, v13);
  if ( v13 )
    v19 = *(_QWORD *)(*(_QWORD *)(v116 + 24) + 16LL * (v13 & 0x7FFFFFFF) + 8);
  else
    v19 = **(_QWORD **)(v116 + 272);
  if ( v19 )
  {
    if ( (*(_BYTE *)(v19 + 3) & 0x10) == 0 )
    {
      v19 = *(_QWORD *)(v19 + 32);
      if ( v19 )
      {
        if ( (*(_BYTE *)(v19 + 3) & 0x10) == 0 )
          BUG();
      }
    }
  }
  v113 = -858993459 * ((v19 - *(_QWORD *)(*(_QWORD *)(v19 + 16) + 32LL)) >> 3);
LABEL_10:
  if ( !v18 )
  {
    v120 = (unsigned __int64)v122;
    goto LABEL_57;
  }
  v20 = *(__int16 **)(v18 + 16);
  v21 = *v20;
  if ( *v20 == 15 )
  {
    v24 = *(_QWORD *)(v18 + 32);
    if ( v112 != ((*(_DWORD *)(v24 + 40LL * v113) >> 8) & 0xFFF) || (*(_BYTE *)(v24 + 44) & 1) != 0 )
      goto LABEL_27;
    v65 = *(_DWORD *)(v24 + 40);
    v66 = *(unsigned int *)(v24 + 48);
    *((_QWORD *)&v126 + 1) = 0;
    v124 = (char *)&v125 + 8;
    *((_QWORD *)&v125 + 1) = v66 | ((unsigned __int64)((v65 >> 8) & 0xFFF) << 32);
    *(_QWORD *)&v125 = 0x200000001LL;
    goto LABEL_133;
  }
  v22 = *((_QWORD *)v20 + 1);
  if ( (v22 & 0x1000) != 0 )
  {
    if ( sub_1E17880(v18) )
      goto LABEL_27;
    if ( *(_BYTE *)(*(_QWORD *)(v18 + 16) + 4LL) != 1 )
      goto LABEL_27;
    v51 = *(_QWORD *)(v18 + 32);
    v16 = v51 + 40LL * v113;
    if ( v112 != ((*(_DWORD *)v16 >> 8) & 0xFFF) )
      goto LABEL_27;
    v52 = *(_DWORD *)(v18 + 40);
    v53 = v113 + 1;
    for ( i = v52; v52 != v53; ++v53 )
    {
      v55 = v51 + 40LL * v53;
      if ( !*(_BYTE *)v55 )
      {
        LODWORD(v17) = *(_DWORD *)(v55 + 8);
        if ( (_DWORD)v17 )
        {
          v56 = *(unsigned __int8 *)(v55 + 3);
          if ( (v56 & 0x20) == 0
            || (LODWORD(v17) = v56,
                LOBYTE(v17) = (unsigned __int8)v56 >> 6,
                (((v56 & 0x10) != 0) & ((unsigned __int8)v56 >> 6)) == 0) )
          {
            if ( v52 != i )
              goto LABEL_27;
            i = v53;
          }
        }
      }
    }
    v57 = *(unsigned int *)(v16 + 8);
    if ( (int)v57 < 0 )
      v58 = *(_QWORD *)(*(_QWORD *)(v116 + 24) + 16 * (v57 & 0x7FFFFFFF) + 8);
    else
      v58 = *(_QWORD *)(*(_QWORD *)(v116 + 272) + 8 * v57);
    while ( v58 )
    {
      if ( (*(_BYTE *)(v58 + 3) & 0x10) == 0 && (*(_BYTE *)(v58 + 4) & 8) == 0 )
      {
        v97 = *(_QWORD *)(v58 + 16);
LABEL_212:
        if ( **(_WORD **)(v97 + 16) == 10 )
          goto LABEL_27;
        while ( 1 )
        {
          v58 = *(_QWORD *)(v58 + 32);
          if ( !v58 )
            goto LABEL_113;
          if ( (*(_BYTE *)(v58 + 3) & 0x10) == 0 && (*(_BYTE *)(v58 + 4) & 8) == 0 && *(_QWORD *)(v58 + 16) != v97 )
          {
            v97 = *(_QWORD *)(v58 + 16);
            goto LABEL_212;
          }
        }
      }
      v58 = *(_QWORD *)(v58 + 32);
    }
LABEL_113:
    v59 = (unsigned int *)(v51 + 40LL * i);
    if ( (v59[1] & 1) != 0 )
      goto LABEL_27;
    v60 = *v59;
    v15 = v59[2];
    *((_QWORD *)&v126 + 1) = 0;
    v124 = (char *)&v125 + 8;
    *((_QWORD *)&v125 + 1) = (unsigned int)v15 | ((unsigned __int64)((v60 >> 8) & 0xFFF) << 32);
    *(_QWORD *)&v125 = 0x200000001LL;
LABEL_133:
    v23 = (char *)&v125 + 8;
    v63 = DWORD2(v125);
    *((_QWORD *)&v126 + 1) = v18;
    v64 = DWORD2(v125) == 0;
    if ( SDWORD2(v125) > 0 )
      goto LABEL_130;
    goto LABEL_134;
  }
  if ( byte_4FC8820 )
    goto LABEL_27;
  if ( v21 != 14 && (v22 & 0x20000000) == 0 )
  {
    if ( v21 == 8 || (int)v22 < 0 )
    {
      LODWORD(v17) = 40 * v113;
      if ( (*(_DWORD *)(*(_QWORD *)(v18 + 32) + 40LL * v113) & 0xFFF00) == 0 )
      {
        if ( v108 )
        {
          v120 = 0;
          v130 = 0;
          LODWORD(v131) = 0;
          v68 = sub_1F3BEE0(v108, v18, v113, &v120, &v130);
          v17 = 40LL * v113;
          if ( v68 )
          {
            if ( v112 == (_DWORD)v131 )
            {
              *((_QWORD *)&v126 + 1) = 0;
              v124 = (char *)&v125 + 8;
              *((_QWORD *)&v125 + 1) = v130;
              *(_QWORD *)&v125 = 0x200000001LL;
              goto LABEL_129;
            }
            v69 = *(_QWORD *)(v116 + 24);
            v70 = *(_QWORD *)(v69 + 16LL * (*(_DWORD *)(*(_QWORD *)(v18 + 32) + 40LL * v113 + 8) & 0x7FFFFFFF))
                & 0xFFFFFFFFFFFFFFF8LL;
            if ( v70 == (*(_QWORD *)(v69 + 16 * (v120 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL) && !HIDWORD(v120) )
            {
              v71 = *(_QWORD *)(*(_QWORD *)v116 + 16LL);
              v72 = *(__int64 (**)())(*(_QWORD *)v71 + 112LL);
              if ( v72 != sub_1D00B10 )
              {
                v73 = ((__int64 (__fastcall *)(__int64, __int64, __int64, unsigned __int64, unsigned __int64, __int64))v72)(
                        v71,
                        v18,
                        v69,
                        v70,
                        v16,
                        v17);
                if ( v73 )
                {
                  if ( (*(_DWORD *)(*(_QWORD *)(v73 + 248) + 4LL * v112)
                      & *(_DWORD *)(*(_QWORD *)(v73 + 248) + 4LL * (unsigned int)v131)) == 0 )
                  {
                    v15 = (__int64)&v125 + 8;
                    *((_QWORD *)&v126 + 1) = 0;
                    v124 = (char *)&v125 + 8;
                    *((_QWORD *)&v125 + 1) = (unsigned int)v120 | ((unsigned __int64)v112 << 32);
                    *(_QWORD *)&v125 = 0x200000001LL;
                    goto LABEL_129;
                  }
                }
              }
            }
          }
        }
      }
    }
    else if ( v21 == 7 || (v22 & 0x40000000) != 0 )
    {
      LODWORD(v17) = v112;
      if ( !v112 )
      {
        if ( v108 )
        {
          v130 = 0;
          LODWORD(v131) = 0;
          if ( (unsigned __int8)sub_1F3BE80(v108, v18, v113, &v130) )
          {
            LODWORD(v16) = HIDWORD(v130);
            if ( !HIDWORD(v130) )
            {
              *((_QWORD *)&v126 + 1) = 0;
              v124 = (char *)&v125 + 8;
              *(_QWORD *)&v125 = 0x200000000LL;
              v120 = __PAIR64__(v131, (unsigned int)v130);
              sub_1E9E930((__int64)&v124, &v120, v95, v96, 0, v17);
              v62 = v125;
              goto LABEL_127;
            }
          }
        }
      }
    }
    else if ( v21 == 10 )
    {
      v99 = *(_QWORD *)(v18 + 32);
      if ( v112 == *(_QWORD *)(v99 + 144) && (*(_DWORD *)(v99 + 80) & 0xFFF00) == 0 )
      {
        v100 = *(_DWORD *)(v99 + 88);
        *(_QWORD *)&v125 = 0x200000000LL;
        v130 = (char *)__PAIR64__(v112, v100);
        v124 = (char *)&v125 + 8;
        *((_QWORD *)&v126 + 1) = 0;
        sub_1E9E930((__int64)&v124, &v130, v112, (__int64)&v125 + 8, v16, v17);
        v62 = v125;
        goto LABEL_127;
      }
    }
    else if ( v21 == 45 || !v21 )
    {
      v134 = 0;
      v130 = (char *)&v132;
      v131 = 0x200000000LL;
      v23 = *(char **)(v18 + 32);
      if ( v112 != ((*(_DWORD *)v23 >> 8) & 0xFFF) )
      {
        v121 = 0x200000000LL;
        v124 = (char *)&v125 + 8;
        v125 = 0;
        v120 = (unsigned __int64)v122;
        v126 = 0;
LABEL_28:
        v18 = 0;
        v123 = *((_QWORD *)&v126 + 1);
        goto LABEL_29;
      }
      v101 = *(_DWORD *)(v18 + 40);
      if ( v101 <= 1 )
      {
        v124 = (char *)&v125 + 8;
        *(_QWORD *)&v125 = 0x200000000LL;
      }
      else
      {
        LODWORD(v17) = 40;
        v119 = a3;
        v102 = v5;
        v103 = 40;
        v110 = 80LL * ((v101 - 2) >> 1) + 120;
        while ( 1 )
        {
          v23 += v103;
          if ( (v23[4] & 1) != 0 )
          {
            v5 = v102;
            a3 = v119;
            v125 = 0;
            v124 = (char *)&v125 + 8;
            DWORD1(v125) = 2;
            v126 = 0;
            goto LABEL_231;
          }
          v105 = ((unsigned __int64)((*(_DWORD *)v23 >> 8) & 0xFFF) << 32) | *((unsigned int *)v23 + 2);
          v106 = (unsigned int)v131;
          if ( (unsigned int)v131 >= HIDWORD(v131) )
          {
            sub_16CD150((__int64)&v130, &v132, 0, 8, v16, v17);
            v106 = (unsigned int)v131;
          }
          v23 = v130;
          v103 += 80;
          *(_QWORD *)&v130[8 * v106] = v105;
          v104 = v131 + 1;
          LODWORD(v131) = v131 + 1;
          if ( v103 == v110 )
            break;
          v23 = *(char **)(v18 + 32);
        }
        v15 = (__int64)&v125 + 8;
        v5 = v102;
        v124 = (char *)&v125 + 8;
        a3 = v119;
        *(_QWORD *)&v125 = 0x200000000LL;
        if ( v104 )
          sub_1E9C460((__int64)&v124, &v130, (__int64)v23, (__int64)&v125 + 8, v16, v17);
      }
      *((_QWORD *)&v126 + 1) = v134;
LABEL_231:
      v61 = v130;
      goto LABEL_124;
    }
LABEL_27:
    v124 = (char *)&v125 + 8;
    v125 = 0;
    v120 = (unsigned __int64)v122;
    v121 = 0x200000000LL;
    v126 = 0;
    goto LABEL_28;
  }
  if ( (*(_DWORD *)(*(_QWORD *)(v18 + 32) + 40LL * v113) & 0xFFF00) != 0 || !v108 )
    goto LABEL_27;
  v130 = (char *)&v132;
  v131 = 0x800000000LL;
  if ( (unsigned __int8)sub_1F3BD60(v108, v18, v113, &v130) )
  {
    v61 = v130;
    v23 = &v130[12 * (unsigned int)v131];
    if ( v130 == v23 )
      goto LABEL_190;
    v15 = v112;
    v94 = v130;
    while ( v112 != *((_DWORD *)v94 + 2) )
    {
      v94 += 12;
      if ( v23 == v94 )
        goto LABEL_190;
    }
    if ( *((_DWORD *)v94 + 1) )
    {
LABEL_190:
      v125 = 0;
      v124 = (char *)&v125 + 8;
      DWORD1(v125) = 2;
      v126 = 0;
    }
    else
    {
      v98 = *(unsigned int *)v94;
      v15 = (__int64)&v125 + 8;
      *((_QWORD *)&v126 + 1) = 0;
      *((_QWORD *)&v125 + 1) = v98;
      v124 = (char *)&v125 + 8;
      *(_QWORD *)&v125 = 0x200000001LL;
    }
  }
  else
  {
    v61 = v130;
    v125 = 0;
    v124 = (char *)&v125 + 8;
    DWORD1(v125) = 2;
    v126 = 0;
  }
LABEL_124:
  if ( v61 != (char *)&v132 )
    _libc_free((unsigned __int64)v61);
  v62 = v125;
LABEL_127:
  if ( v62 <= 0 )
  {
    v120 = (unsigned __int64)v122;
    v121 = 0x200000000LL;
    if ( !v62 )
      goto LABEL_28;
    goto LABEL_131;
  }
  if ( v62 != 1 )
  {
    *((_QWORD *)&v126 + 1) = v18;
    goto LABEL_130;
  }
LABEL_129:
  v23 = v124;
  v63 = *(_DWORD *)v124;
  *((_QWORD *)&v126 + 1) = v18;
  v64 = v63 == 0;
  if ( v63 > 0 )
  {
LABEL_130:
    v120 = (unsigned __int64)v122;
    v121 = 0x200000000LL;
LABEL_131:
    sub_1E9C460((__int64)&v120, &v124, (__int64)v23, v15, v16, v17);
    goto LABEL_28;
  }
LABEL_134:
  if ( v64 )
  {
    v67 = **(_QWORD **)(v116 + 272);
  }
  else
  {
    v15 = v116;
    v67 = *(_QWORD *)(*(_QWORD *)(v116 + 24) + 16LL * (v63 & 0x7FFFFFFF) + 8);
  }
  v18 = 0;
  if ( v67 )
  {
    if ( (*(_BYTE *)(v67 + 3) & 0x10) != 0 || (v67 = *(_QWORD *)(v67 + 32)) != 0 && (*(_BYTE *)(v67 + 3) & 0x10) != 0 )
    {
      v18 = *(_QWORD *)(v67 + 16);
      v113 = -858993459 * ((v67 - *(_QWORD *)(v18 + 32)) >> 3);
      v112 = *((_DWORD *)v23 + 1);
    }
  }
  v120 = (unsigned __int64)v122;
  v121 = 0x200000000LL;
  sub_1E9C460((__int64)&v120, &v124, (__int64)v23, v15, v16, v17);
  v123 = *((_QWORD *)&v126 + 1);
LABEL_29:
  if ( v124 != (char *)&v125 + 8 )
    _libc_free((unsigned __int64)v124);
  if ( (int)v121 <= 0 )
    goto LABEL_67;
  if ( (*(_BYTE *)(a3 + 8) & 1) != 0 )
  {
    v25 = a3 + 16;
    v26 = 3;
  }
  else
  {
    v26 = *(unsigned int *)(a3 + 24);
    v25 = *(_QWORD *)(a3 + 16);
    if ( !(_DWORD)v26 )
    {
LABEL_41:
      v125 = 0;
      v124 = (char *)&v125 + 8;
      DWORD1(v125) = 2;
      v126 = 0;
LABEL_42:
      v130 = v14;
      v131 = (unsigned __int64)&v133;
      v132 = 0x200000000LL;
      if ( (_DWORD)v121 )
        sub_1E9C380((__int64)&v131, (__int64)&v120, v26, (unsigned int)v121, v16, v17);
      v32 = *(_BYTE *)(a3 + 8);
      v135 = v123;
      v33 = v32 & 1;
      if ( v33 )
      {
        v34 = a3 + 16;
        v35 = 3;
      }
      else
      {
        v42 = *(_DWORD *)(a3 + 24);
        v34 = *(_QWORD *)(a3 + 16);
        if ( !v42 )
        {
          v45 = *(_DWORD *)(a3 + 8);
          ++*(_QWORD *)a3;
          v36 = 0;
          v46 = (v45 >> 1) + 1;
LABEL_87:
          v47 = 3 * v42;
          goto LABEL_88;
        }
        v35 = v42 - 1;
      }
      LODWORD(v16) = HIDWORD(v130);
      v36 = 0;
      v109 = 1;
      v37 = ((((unsigned int)(37 * HIDWORD(v130)) | ((unsigned __int64)(unsigned int)(37 * (_DWORD)v130) << 32))
            - 1
            - ((unsigned __int64)(unsigned int)(37 * HIDWORD(v130)) << 32)) >> 22)
          ^ (((unsigned int)(37 * HIDWORD(v130)) | ((unsigned __int64)(unsigned int)(37 * (_DWORD)v130) << 32))
           - 1
           - ((unsigned __int64)(unsigned int)(37 * HIDWORD(v130)) << 32));
      v38 = ((9 * (((v37 - 1 - (v37 << 13)) >> 8) ^ (v37 - 1 - (v37 << 13)))) >> 15)
          ^ (9 * (((v37 - 1 - (v37 << 13)) >> 8) ^ (v37 - 1 - (v37 << 13))));
      for ( j = v35 & (((v38 - 1 - (v38 << 27)) >> 31) ^ (v38 - 1 - ((_DWORD)v38 << 27))); ; j = v35 & v88 )
      {
        v40 = (_DWORD *)(v34 + 48LL * j);
        v41 = *v40;
        if ( v130 == *(char **)v40 )
          goto LABEL_69;
        if ( v41 == -1 )
        {
          if ( v40[1] == -1 )
          {
            v45 = *(_DWORD *)(a3 + 8);
            if ( !v36 )
              v36 = v40;
            ++*(_QWORD *)a3;
            v46 = (v45 >> 1) + 1;
            if ( !(_BYTE)v33 )
            {
              v42 = *(_DWORD *)(a3 + 24);
              goto LABEL_87;
            }
            v47 = 12;
            v42 = 4;
LABEL_88:
            v48 = (unsigned int)(4 * v46);
            if ( v47 <= (unsigned int)v48 )
            {
              sub_1E9CCA0(a3, 2 * v42, v46, v48, v16, v34);
              if ( (*(_BYTE *)(a3 + 8) & 1) != 0 )
              {
                v74 = a3 + 16;
                v75 = 3;
              }
              else
              {
                v89 = *(_DWORD *)(a3 + 24);
                v74 = *(_QWORD *)(a3 + 16);
                if ( !v89 )
                  goto LABEL_238;
                v75 = v89 - 1;
              }
              v49 = (unsigned int)v130;
              v76 = 1;
              v34 = 0;
              v77 = ((((unsigned int)(37 * HIDWORD(v130)) | ((unsigned __int64)(unsigned int)(37 * (_DWORD)v130) << 32))
                    - 1
                    - ((unsigned __int64)(unsigned int)(37 * HIDWORD(v130)) << 32)) >> 22)
                  ^ (((unsigned int)(37 * HIDWORD(v130)) | ((unsigned __int64)(unsigned int)(37 * (_DWORD)v130) << 32))
                   - 1
                   - ((unsigned __int64)(unsigned int)(37 * HIDWORD(v130)) << 32));
              v78 = ((9 * (((v77 - 1 - (v77 << 13)) >> 8) ^ (v77 - 1 - (v77 << 13)))) >> 15)
                  ^ (9 * (((v77 - 1 - (v77 << 13)) >> 8) ^ (v77 - 1 - (v77 << 13))));
              for ( k = v75 & (((v78 - 1 - (v78 << 27)) >> 31) ^ (v78 - 1 - ((_DWORD)v78 << 27))); ; k = v75 & v80 )
              {
                v36 = (_DWORD *)(v74 + 48LL * k);
                LODWORD(v16) = *v36;
                if ( v130 == *(char **)v36 )
                  break;
                if ( (_DWORD)v16 == -1 )
                {
                  if ( v36[1] == -1 )
                  {
LABEL_197:
                    if ( v34 )
                      v36 = (_DWORD *)v34;
                    goto LABEL_199;
                  }
                }
                else if ( (_DWORD)v16 == -2 && v36[1] == -2 && !v34 )
                {
                  v34 = v74 + 48LL * k;
                }
                v80 = v76 + k;
                ++v76;
              }
              goto LABEL_199;
            }
            v49 = v42 - *(_DWORD *)(a3 + 12) - (unsigned int)v46;
            v50 = v42 >> 3;
            if ( (unsigned int)v49 > (unsigned int)v50 )
              goto LABEL_90;
            sub_1E9CCA0(a3, v42, v50, v49, v16, v34);
            if ( (*(_BYTE *)(a3 + 8) & 1) == 0 )
            {
              v93 = *(_DWORD *)(a3 + 24);
              v81 = *(_QWORD *)(a3 + 16);
              if ( v93 )
              {
                v82 = v93 - 1;
                goto LABEL_165;
              }
LABEL_238:
              *(_DWORD *)(a3 + 8) = (2 * (*(_DWORD *)(a3 + 8) >> 1) + 2) | *(_DWORD *)(a3 + 8) & 1;
              BUG();
            }
            v81 = a3 + 16;
            v82 = 3;
LABEL_165:
            v49 = (unsigned int)v130;
            v83 = 1;
            v34 = 0;
            v84 = ((((unsigned int)(37 * HIDWORD(v130)) | ((unsigned __int64)(unsigned int)(37 * (_DWORD)v130) << 32))
                  - 1
                  - ((unsigned __int64)(unsigned int)(37 * HIDWORD(v130)) << 32)) >> 22)
                ^ (((unsigned int)(37 * HIDWORD(v130)) | ((unsigned __int64)(unsigned int)(37 * (_DWORD)v130) << 32))
                 - 1
                 - ((unsigned __int64)(unsigned int)(37 * HIDWORD(v130)) << 32));
            v85 = ((9 * (((v84 - 1 - (v84 << 13)) >> 8) ^ (v84 - 1 - (v84 << 13)))) >> 15)
                ^ (9 * (((v84 - 1 - (v84 << 13)) >> 8) ^ (v84 - 1 - (v84 << 13))));
            for ( m = v82 & (((v85 - 1 - (v85 << 27)) >> 31) ^ (v85 - 1 - ((_DWORD)v85 << 27))); ; m = v82 & v87 )
            {
              v36 = (_DWORD *)(v81 + 48LL * m);
              LODWORD(v16) = *v36;
              if ( v130 == *(char **)v36 )
                break;
              if ( (_DWORD)v16 == -1 )
              {
                if ( v36[1] == -1 )
                  goto LABEL_197;
              }
              else if ( (_DWORD)v16 == -2 && v36[1] == -2 && !v34 )
              {
                v34 = v81 + 48LL * m;
              }
              v87 = v83 + m;
              ++v83;
            }
LABEL_199:
            v45 = *(_DWORD *)(a3 + 8);
LABEL_90:
            *(_DWORD *)(a3 + 8) = (2 * (v45 >> 1) + 2) | v45 & 1;
            if ( *v36 != -1 || v36[1] != -1 )
              --*(_DWORD *)(a3 + 12);
            *(_QWORD *)v36 = v130;
            *((_QWORD *)v36 + 1) = v36 + 6;
            *((_QWORD *)v36 + 2) = 0x200000000LL;
            if ( (_DWORD)v132 )
              sub_1E9C460((__int64)(v36 + 2), (char **)&v131, (unsigned int)v132, v49, v16, v34);
            *((_QWORD *)v36 + 5) = v135;
LABEL_69:
            if ( (char *)v131 != &v133 )
              _libc_free(v131);
            v43 = v121;
            if ( (unsigned int)v121 <= 1 )
            {
              v44 = *(_DWORD *)(v120 + 4);
              v14 = *(char **)v120;
              if ( *(int *)v120 > 0 )
                goto LABEL_65;
              if ( (*(unsigned __int8 (__fastcall **)(_QWORD, unsigned __int64, unsigned __int64, unsigned __int64, _QWORD))(**(_QWORD **)(v5 + 240) + 104LL))(
                     *(_QWORD *)(v5 + 240),
                     v115,
                     HIDWORD(a2),
                     *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v5 + 248) + 24LL) + 16LL * (*(_DWORD *)v120 & 0x7FFFFFFF))
                   & 0xFFFFFFFFFFFFFFF8LL,
                     v44)
                && (!v111 || !v44) )
              {
                goto LABEL_76;
              }
              if ( v124 != (char *)&v125 + 8 )
                _libc_free((unsigned __int64)v124);
              if ( (_BYTE *)v120 != v122 )
                _libc_free(v120);
              goto LABEL_10;
            }
            if ( ++v111 >= (unsigned int)dword_4FC8660 )
              goto LABEL_65;
            v90 = (unsigned int)v128;
            v91 = 0;
            do
            {
              v92 = *(_QWORD *)(v120 + 8LL * (int)v91);
              if ( (unsigned int)v90 >= HIDWORD(v128) )
              {
                v114 = *(_QWORD *)(v120 + 8LL * (int)v91);
                sub_16CD150((__int64)&v127, v129, 0, 8, v92, v34);
                v90 = (unsigned int)v128;
                v92 = v114;
              }
              ++v91;
              v127[v90] = v92;
              v90 = (unsigned int)(v128 + 1);
              LODWORD(v128) = v128 + 1;
            }
            while ( v91 < v43 );
LABEL_76:
            if ( v124 != (char *)&v125 + 8 )
              _libc_free((unsigned __int64)v124);
            if ( (_BYTE *)v120 != v122 )
              _libc_free(v120);
            v9 = v128;
            v7 = v127;
            if ( !(_DWORD)v128 )
            {
              v3 = a2 != (_DWORD)v14;
              goto LABEL_59;
            }
            goto LABEL_4;
          }
        }
        else if ( v41 == -2 && v40[1] == -2 && !v36 )
        {
          v36 = (_DWORD *)(v34 + 48LL * j);
        }
        v88 = v109 + j;
        ++v109;
      }
    }
    v26 = (unsigned int)(v26 - 1);
  }
  v27 = ((((unsigned int)(37 * HIDWORD(v14)) | ((unsigned __int64)(unsigned int)(37 * (_DWORD)v14) << 32))
        - 1
        - ((unsigned __int64)(unsigned int)(37 * HIDWORD(v14)) << 32)) >> 22)
      ^ (((unsigned int)(37 * HIDWORD(v14)) | ((unsigned __int64)(unsigned int)(37 * (_DWORD)v14) << 32))
       - 1
       - ((unsigned __int64)(unsigned int)(37 * HIDWORD(v14)) << 32));
  v16 = ((9 * (((v27 - 1 - (v27 << 13)) >> 8) ^ (v27 - 1 - (v27 << 13)))) >> 15)
      ^ (9 * (((v27 - 1 - (v27 << 13)) >> 8) ^ (v27 - 1 - (v27 << 13))));
  v28 = ((v16 - 1 - (v16 << 27)) >> 31) ^ (v16 - 1 - ((_DWORD)v16 << 27));
  LODWORD(v16) = 1;
  for ( n = v26 & v28; ; n = v26 & v31 )
  {
    v30 = v25 + 48LL * n;
    LODWORD(v17) = *(_DWORD *)v30;
    if ( v14 == *(char **)v30 )
      break;
    if ( (_DWORD)v17 == -1 && *(_DWORD *)(v30 + 4) == -1 )
      goto LABEL_41;
    v31 = v16 + n;
    LODWORD(v16) = v16 + 1;
  }
  v124 = (char *)&v125 + 8;
  *(_QWORD *)&v125 = 0x200000000LL;
  if ( !*(_DWORD *)(v30 + 16) )
  {
    *((_QWORD *)&v126 + 1) = *(_QWORD *)(v30 + 40);
    goto LABEL_42;
  }
  sub_1E9C380((__int64)&v124, v30 + 8, v26, v25, v16, v17);
  v26 = *(_QWORD *)(v30 + 40);
  *((_QWORD *)&v126 + 1) = v26;
  if ( (int)v125 <= 0 )
    goto LABEL_42;
  if ( (_DWORD)v125 == 1 )
    goto LABEL_76;
LABEL_65:
  if ( v124 != (char *)&v125 + 8 )
    _libc_free((unsigned __int64)v124);
LABEL_67:
  if ( (_BYTE *)v120 != v122 )
    _libc_free(v120);
LABEL_57:
  v7 = v127;
LABEL_58:
  LOBYTE(v3) = 0;
LABEL_59:
  if ( v7 != v129 )
  {
    v118 = v3;
    _libc_free((unsigned __int64)v7);
    LOBYTE(v3) = v118;
  }
  return v3;
}
