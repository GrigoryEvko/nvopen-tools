// Function: sub_10CB120
// Address: 0x10cb120
//
__int64 __fastcall sub_10CB120(_QWORD *a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v5; // r12
  _QWORD *v6; // rbx
  __int64 v7; // r15
  int v8; // edx
  __int64 v9; // rsi
  int v10; // edx
  __int64 v11; // rdi
  __int64 v12; // r14
  __int64 v13; // rcx
  int v14; // edx
  int v15; // r14d
  unsigned int v16; // eax
  unsigned int **v17; // r13
  _QWORD **v18; // rdx
  int v19; // ecx
  __int64 *v20; // rax
  __int64 v21; // rdx
  bool v22; // zf
  char v23; // al
  char v25; // al
  __int64 v26; // rsi
  _BYTE *v27; // rax
  _BYTE *v28; // rax
  __int64 v29; // rdi
  unsigned int v30; // r13d
  __int64 v31; // r12
  bool v32; // al
  __int64 v33; // rdx
  char v34; // r8
  __int64 v35; // r9
  unsigned int v36; // r14d
  __int64 v37; // rdi
  int v38; // eax
  bool v39; // al
  bool v40; // al
  __int64 v41; // rdx
  __int64 v42; // r9
  char v43; // r8
  bool v44; // r14
  unsigned int v45; // r14d
  bool v46; // al
  _QWORD *v47; // rdi
  int v48; // edx
  int v49; // eax
  __int64 *v50; // rax
  __int64 **v51; // rsi
  _BYTE *v52; // rax
  unsigned int **v53; // rdi
  __int64 v54; // r14
  _BYTE *v55; // rax
  bool v56; // r8
  unsigned int v57; // r12d
  __int64 v58; // r14
  _BYTE *v59; // rax
  __int64 v60; // rdi
  __int64 v61; // rax
  _BYTE *v62; // rax
  __int64 v63; // rax
  unsigned int v64; // r14d
  __int64 v65; // rdi
  int v66; // eax
  bool v67; // al
  unsigned int v68; // r14d
  int v69; // eax
  bool v70; // al
  int v71; // r14d
  __int64 v72; // rbx
  int v73; // edx
  int v74; // eax
  __int64 *v75; // rax
  __int64 v76; // r13
  __int64 v77; // r14
  unsigned int v78; // r15d
  unsigned int v79; // eax
  unsigned int v80; // ebx
  bool v81; // r14
  __int64 v82; // rax
  unsigned int v83; // r14d
  __int64 v84; // r14
  _BYTE *v85; // rax
  bool v86; // r14
  __int64 v87; // rsi
  __int64 v88; // rax
  unsigned int v89; // r14d
  int v90; // eax
  _BYTE *v91; // rax
  unsigned __int8 *v92; // r9
  unsigned int v93; // r14d
  int v94; // eax
  bool v95; // r14
  __int64 v96; // rax
  int v97; // edx
  unsigned int v98; // r14d
  int v99; // eax
  unsigned int v100; // r12d
  unsigned __int8 *v101; // rbx
  unsigned __int8 *v102; // rax
  unsigned int v103; // r14d
  _QWORD *v104; // rax
  __int64 v105; // r12
  unsigned int *v106; // rbx
  unsigned int *v107; // r12
  __int64 v108; // rdx
  unsigned int v109; // esi
  __int64 v110; // rdx
  int v111; // [rsp+0h] [rbp-E0h]
  int v112; // [rsp+0h] [rbp-E0h]
  _QWORD *v113; // [rsp+0h] [rbp-E0h]
  __int64 v114; // [rsp+8h] [rbp-D8h]
  __int64 v115; // [rsp+10h] [rbp-D0h]
  char v116; // [rsp+10h] [rbp-D0h]
  char v117; // [rsp+10h] [rbp-D0h]
  unsigned __int8 *v118; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v119; // [rsp+20h] [rbp-C0h]
  __int64 v120; // [rsp+28h] [rbp-B8h]
  char v121; // [rsp+28h] [rbp-B8h]
  char v122; // [rsp+28h] [rbp-B8h]
  _QWORD *v123; // [rsp+28h] [rbp-B8h]
  char v124; // [rsp+28h] [rbp-B8h]
  __int64 v125; // [rsp+28h] [rbp-B8h]
  __int64 v126; // [rsp+28h] [rbp-B8h]
  __int64 v127; // [rsp+30h] [rbp-B0h]
  __int64 v128; // [rsp+30h] [rbp-B0h]
  bool v129; // [rsp+30h] [rbp-B0h]
  __int64 v130; // [rsp+30h] [rbp-B0h]
  int v131; // [rsp+30h] [rbp-B0h]
  __int64 v132; // [rsp+30h] [rbp-B0h]
  __int64 v133; // [rsp+30h] [rbp-B0h]
  int v134; // [rsp+30h] [rbp-B0h]
  int v135; // [rsp+30h] [rbp-B0h]
  bool v136; // [rsp+3Bh] [rbp-A5h]
  int v137; // [rsp+3Ch] [rbp-A4h]
  __int64 v138; // [rsp+40h] [rbp-A0h]
  _BYTE *v139; // [rsp+48h] [rbp-98h] BYREF
  __int64 v140[4]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v141; // [rsp+70h] [rbp-70h]
  _QWORD *v142; // [rsp+80h] [rbp-60h] BYREF
  __int64 *v143; // [rsp+88h] [rbp-58h]
  __int16 v144; // [rsp+A0h] [rbp-40h]

  v5 = a2;
  v6 = a1;
  v7 = *(_QWORD *)(a2 + 8);
  v8 = *(unsigned __int8 *)(v7 + 8);
  if ( (unsigned int)(v8 - 17) <= 1 )
    LOBYTE(v8) = *(_BYTE *)(**(_QWORD **)(v7 + 16) + 8LL);
  if ( (_BYTE)v8 != 12 )
    return 0;
  v9 = *(_QWORD *)(a3 + 8);
  v10 = *(unsigned __int8 *)(v9 + 8);
  if ( (unsigned int)(v10 - 17) <= 1 )
    LOBYTE(v10) = *(_BYTE *)(**(_QWORD **)(v9 + 16) + 8LL);
  if ( (_BYTE)v10 != 12 )
    return 0;
  if ( !a4 )
  {
    v22 = *(_BYTE *)a3 == 59;
    v142 = 0;
    v143 = (__int64 *)v5;
    if ( v22 )
    {
      v25 = sub_995B10(&v142, *(_QWORD *)(a3 - 64));
      v26 = *(_QWORD *)(a3 - 32);
      if ( v25 )
      {
        if ( (__int64 *)v26 == v143 )
          goto LABEL_9;
      }
      if ( (unsigned __int8)sub_995B10(&v142, v26) && *(__int64 **)(a3 - 64) == v143 )
        goto LABEL_9;
    }
    v23 = *(_BYTE *)v5;
    if ( *(_BYTE *)v5 <= 0x15u )
    {
      if ( *(_BYTE *)a3 > 0x15u )
      {
LABEL_28:
        if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 > 1 )
          return 0;
        if ( *(_BYTE *)v5 != 59 )
          return 0;
        v27 = *(_BYTE **)(v5 - 64);
        if ( *v27 != 69 )
          return 0;
        v114 = *((_QWORD *)v27 - 4);
        if ( !v114 )
          return 0;
        v119 = *(_QWORD *)(v5 - 32);
        if ( *(_BYTE *)v119 > 0x15u )
          return 0;
        if ( *(_BYTE *)a3 != 59 )
          return 0;
        v28 = *(_BYTE **)(a3 - 64);
        if ( *v28 != 69 )
          return 0;
        if ( v114 != *((_QWORD *)v28 - 4) )
          return 0;
        v118 = *(unsigned __int8 **)(a3 - 32);
        if ( *v118 > 0x15u )
          return 0;
        v29 = *(_QWORD *)(v114 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v29 + 8) - 17 <= 1 )
          v29 = **(_QWORD **)(v29 + 16);
        v136 = sub_BCAC40(v29, 1);
        if ( !v136 )
          return 0;
        if ( !*(_DWORD *)(*(_QWORD *)(v119 + 8) + 32LL) )
        {
LABEL_63:
          v47 = *(_QWORD **)v7;
          v48 = *(unsigned __int8 *)(v7 + 8);
          if ( (unsigned int)(v48 - 17) > 1 )
          {
            v51 = (__int64 **)sub_BCB2A0(v47);
          }
          else
          {
            v49 = *(_DWORD *)(v7 + 32);
            BYTE4(v140[0]) = (_BYTE)v48 == 18;
            LODWORD(v140[0]) = v49;
            v50 = (__int64 *)sub_BCB2A0(v47);
            v51 = (__int64 **)sub_BCE1B0(v50, v140[0]);
          }
          v52 = (_BYTE *)sub_AD4C30(v119, v51, 0);
          v53 = (unsigned int **)v6[4];
          v144 = 257;
          return sub_A825B0(v53, (_BYTE *)v114, v52, (__int64)&v142);
        }
        v137 = *(_DWORD *)(*(_QWORD *)(v119 + 8) + 32LL);
        v30 = 0;
        while ( 1 )
        {
          v31 = sub_AD69F0((unsigned __int8 *)v119, v30);
          v120 = sub_AD69F0(v118, v30);
          if ( v120 == 0 || v31 == 0 )
            return 0;
          v32 = sub_AC30F0(v31);
          v34 = 0;
          v35 = v120;
          if ( !v32 )
            break;
LABEL_50:
          if ( *(_BYTE *)v35 == 17 )
          {
            v36 = *(_DWORD *)(v35 + 32);
            if ( !v36 )
              goto LABEL_62;
            if ( v36 > 0x40 )
            {
              v121 = v34;
              v37 = v35 + 24;
              v127 = v35;
              goto LABEL_54;
            }
            v39 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v36) == *(_QWORD *)(v35 + 24);
LABEL_55:
            if ( v39 )
              goto LABEL_62;
            goto LABEL_56;
          }
          v58 = *(_QWORD *)(v35 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v58 + 8) - 17 <= 1 )
          {
            v121 = v34;
            v127 = v35;
            v59 = sub_AD7630(v35, 0, v33);
            v35 = v127;
            v34 = v121;
            if ( v59 && *v59 == 17 )
            {
              v36 = *((_DWORD *)v59 + 8);
              if ( !v36 )
                goto LABEL_62;
              if ( v36 <= 0x40 )
              {
                v39 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v36) == *((_QWORD *)v59 + 3);
              }
              else
              {
                v37 = (__int64)(v59 + 24);
LABEL_54:
                v38 = sub_C445E0(v37);
                v35 = v127;
                v34 = v121;
                v39 = v36 == v38;
              }
              goto LABEL_55;
            }
            if ( *(_BYTE *)(v58 + 8) == 17 )
            {
              v111 = *(_DWORD *)(v58 + 32);
              if ( v111 )
              {
                v86 = 0;
                v87 = 0;
                while ( 1 )
                {
                  v124 = v34;
                  v132 = v35;
                  v88 = sub_AD69F0((unsigned __int8 *)v35, v87);
                  v35 = v132;
                  v34 = v124;
                  if ( !v88 )
                    break;
                  if ( *(_BYTE *)v88 != 13 )
                  {
                    if ( *(_BYTE *)v88 != 17 )
                      break;
                    v89 = *(_DWORD *)(v88 + 32);
                    if ( v89 )
                    {
                      if ( v89 <= 0x40 )
                      {
                        v86 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v89) == *(_QWORD *)(v88 + 24);
                      }
                      else
                      {
                        v90 = sub_C445E0(v88 + 24);
                        v35 = v132;
                        v34 = v124;
                        v86 = v89 == v90;
                      }
                      if ( !v86 )
                        break;
                    }
                    else
                    {
                      v86 = v136;
                    }
                  }
                  v87 = (unsigned int)(v87 + 1);
                  if ( v111 == (_DWORD)v87 )
                  {
                    if ( !v86 )
                      break;
                    goto LABEL_62;
                  }
                }
              }
            }
          }
LABEL_56:
          v122 = v34;
          v128 = v35;
          v40 = sub_AC30F0(v35);
          v42 = v128;
          v43 = v122;
          v44 = v40;
          if ( !v40 )
          {
            if ( *(_BYTE *)v128 == 17 )
            {
              v68 = *(_DWORD *)(v128 + 32);
              if ( v68 <= 0x40 )
              {
                v70 = *(_QWORD *)(v128 + 24) == 0;
              }
              else
              {
                v69 = sub_C444A0(v128 + 24);
                v43 = v122;
                v70 = v68 == v69;
              }
            }
            else
            {
              v133 = *(_QWORD *)(v128 + 8);
              if ( (unsigned int)*(unsigned __int8 *)(v133 + 8) - 17 > 1 )
                return 0;
              v115 = v42;
              v91 = sub_AD7630(v42, 0, v41);
              v92 = (unsigned __int8 *)v115;
              v43 = v122;
              if ( !v91 || *v91 != 17 )
              {
                if ( *(_BYTE *)(v133 + 8) == 17 )
                {
                  v135 = *(_DWORD *)(v133 + 32);
                  if ( v135 )
                  {
                    v117 = v122;
                    v126 = v31;
                    v100 = 0;
                    v113 = v6;
                    v101 = v92;
                    while ( 1 )
                    {
                      v102 = (unsigned __int8 *)sub_AD69F0(v101, v100);
                      if ( !v102 )
                        break;
                      v41 = *v102;
                      if ( (_BYTE)v41 != 13 )
                      {
                        if ( (_BYTE)v41 != 17 )
                          break;
                        v103 = *((_DWORD *)v102 + 8);
                        v44 = v103 <= 0x40
                            ? *((_QWORD *)v102 + 3) == 0
                            : v103 == (unsigned int)sub_C444A0((__int64)(v102 + 24));
                        if ( !v44 )
                          break;
                      }
                      if ( v135 == ++v100 )
                      {
                        v31 = v126;
                        v43 = v117;
                        v6 = v113;
                        if ( v44 )
                          goto LABEL_57;
                        return 0;
                      }
                    }
                  }
                }
                return 0;
              }
              v93 = *((_DWORD *)v91 + 8);
              if ( v93 <= 0x40 )
              {
                v70 = *((_QWORD *)v91 + 3) == 0;
              }
              else
              {
                v94 = sub_C444A0((__int64)(v91 + 24));
                v43 = v122;
                v70 = v93 == v94;
              }
            }
            if ( !v70 )
              return 0;
          }
LABEL_57:
          if ( *(_BYTE *)v31 == 17 )
          {
            v45 = *(_DWORD *)(v31 + 32);
            if ( v45 )
            {
              if ( v45 <= 0x40 )
                v46 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v45) == *(_QWORD *)(v31 + 24);
              else
                v46 = v45 == (unsigned int)sub_C445E0(v31 + 24);
              goto LABEL_61;
            }
          }
          else
          {
            v54 = *(_QWORD *)(v31 + 8);
            v129 = v43;
            if ( (unsigned int)*(unsigned __int8 *)(v54 + 8) - 17 > 1 )
              return 0;
            v55 = sub_AD7630(v31, 0, v41);
            v56 = v129;
            if ( !v55 || *v55 != 17 )
            {
              if ( *(_BYTE *)(v54 + 8) == 17 )
              {
                v131 = *(_DWORD *)(v54 + 32);
                if ( v131 )
                {
                  v123 = v6;
                  v80 = 0;
                  v81 = v56;
                  while ( 1 )
                  {
                    v82 = sub_AD69F0((unsigned __int8 *)v31, v80);
                    if ( !v82 )
                      break;
                    if ( *(_BYTE *)v82 != 13 )
                    {
                      if ( *(_BYTE *)v82 != 17 )
                        return 0;
                      v83 = *(_DWORD *)(v82 + 32);
                      if ( v83 )
                      {
                        if ( v83 <= 0x40 )
                          v81 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v83) == *(_QWORD *)(v82 + 24);
                        else
                          v81 = v83 == (unsigned int)sub_C445E0(v82 + 24);
                        if ( !v81 )
                          return 0;
                      }
                      else
                      {
                        v81 = v136;
                      }
                    }
                    if ( v131 == ++v80 )
                    {
                      v6 = v123;
                      if ( !v81 )
                        return 0;
                      goto LABEL_62;
                    }
                  }
                }
              }
              return 0;
            }
            v57 = *((_DWORD *)v55 + 8);
            if ( v57 )
            {
              if ( v57 <= 0x40 )
                v46 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v57) == *((_QWORD *)v55 + 3);
              else
                v46 = v57 == (unsigned int)sub_C445E0((__int64)(v55 + 24));
LABEL_61:
              if ( !v46 )
                return 0;
            }
          }
LABEL_62:
          if ( v137 == ++v30 )
            goto LABEL_63;
        }
        if ( *(_BYTE *)v31 == 17 )
        {
          v64 = *(_DWORD *)(v31 + 32);
          if ( v64 > 0x40 )
          {
            v65 = v31 + 24;
            v130 = v120;
LABEL_97:
            v66 = sub_C444A0(v65);
            v35 = v130;
            v34 = 0;
            v67 = v64 == v66;
            goto LABEL_98;
          }
          v67 = *(_QWORD *)(v31 + 24) == 0;
        }
        else
        {
          v84 = *(_QWORD *)(v31 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v84 + 8) - 17 > 1 )
            goto LABEL_56;
          v85 = sub_AD7630(v31, 0, v33);
          v35 = v120;
          v34 = 0;
          if ( !v85 || *v85 != 17 )
          {
            if ( *(_BYTE *)(v84 + 8) == 17 )
            {
              v112 = *(_DWORD *)(v84 + 32);
              if ( v112 )
              {
                LODWORD(v33) = 0;
                v95 = 0;
                while ( 1 )
                {
                  v116 = v34;
                  v125 = v35;
                  v134 = v33;
                  v96 = sub_AD69F0((unsigned __int8 *)v31, (unsigned int)v33);
                  v35 = v125;
                  v34 = v116;
                  if ( !v96 )
                    break;
                  v97 = v134;
                  if ( *(_BYTE *)v96 != 13 )
                  {
                    if ( *(_BYTE *)v96 != 17 )
                      break;
                    v98 = *(_DWORD *)(v96 + 32);
                    if ( v98 <= 0x40 )
                    {
                      v95 = *(_QWORD *)(v96 + 24) == 0;
                    }
                    else
                    {
                      v99 = sub_C444A0(v96 + 24);
                      v97 = v134;
                      v35 = v125;
                      v34 = v116;
                      v95 = v98 == v99;
                    }
                    if ( !v95 )
                      break;
                  }
                  v33 = (unsigned int)(v97 + 1);
                  if ( v112 == (_DWORD)v33 )
                  {
                    if ( v95 )
                      goto LABEL_50;
                    goto LABEL_56;
                  }
                }
              }
            }
            goto LABEL_56;
          }
          v64 = *((_DWORD *)v85 + 8);
          if ( v64 > 0x40 )
          {
            v65 = (__int64)(v85 + 24);
            v130 = v120;
            goto LABEL_97;
          }
          v67 = *((_QWORD *)v85 + 3) == 0;
        }
LABEL_98:
        if ( !v67 )
          goto LABEL_56;
        goto LABEL_50;
      }
      if ( v5 == sub_AD63D0(a3) )
      {
        v71 = sub_9AF8B0(v5, a1[11], 0, a1[8], 0, a1[10], 1);
        if ( (unsigned int)sub_BCB060(v7) == v71 )
        {
          v72 = a1[4];
          v141 = 257;
          v73 = *(unsigned __int8 *)(v7 + 8);
          if ( (unsigned int)(v73 - 17) > 1 )
          {
            v76 = sub_BCB2A0(*(_QWORD **)v7);
          }
          else
          {
            v74 = *(_DWORD *)(v7 + 32);
            BYTE4(v139) = (_BYTE)v73 == 18;
            LODWORD(v139) = v74;
            v75 = (__int64 *)sub_BCB2A0(*(_QWORD **)v7);
            v76 = sub_BCE1B0(v75, (__int64)v139);
          }
          v77 = *(_QWORD *)(v5 + 8);
          v78 = sub_BCB060(v77);
          v79 = sub_BCB060(v76);
          if ( v78 < v79 )
          {
            if ( v77 == v76 )
            {
              return v5;
            }
            else
            {
              v12 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v72 + 80) + 120LL))(
                      *(_QWORD *)(v72 + 80),
                      39,
                      v5,
                      v76);
              if ( !v12 )
              {
                v144 = 257;
                v104 = sub_BD2C40(72, unk_3F10A14);
                v12 = (__int64)v104;
                if ( v104 )
                  sub_B515B0((__int64)v104, v5, v76, (__int64)&v142, 0, 0);
                (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v72 + 88) + 16LL))(
                  *(_QWORD *)(v72 + 88),
                  v12,
                  v140,
                  *(_QWORD *)(v72 + 56),
                  *(_QWORD *)(v72 + 64));
                v105 = 4LL * *(unsigned int *)(v72 + 8);
                v106 = *(unsigned int **)v72;
                v107 = &v106[v105];
                while ( v107 != v106 )
                {
                  v108 = *((_QWORD *)v106 + 1);
                  v109 = *v106;
                  v106 += 4;
                  sub_B99FD0(v12, v109, v108);
                }
              }
            }
          }
          else
          {
            v12 = v5;
            if ( v78 > v79 )
              return sub_A82DA0((unsigned int **)v72, v5, v76, (__int64)v140, 0, 0);
          }
          return v12;
        }
      }
      v23 = *(_BYTE *)v5;
    }
    if ( v23 == 69 )
    {
      v12 = *(_QWORD *)(v5 - 32);
      if ( v12 )
      {
        v60 = *(_QWORD *)(v12 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v60 + 8) - 17 <= 1 )
          v60 = **(_QWORD **)(v60 + 16);
        if ( sub_BCAC40(v60, 1) )
        {
          v22 = *(_BYTE *)a3 == 69;
          v142 = 0;
          v143 = (__int64 *)v12;
          if ( v22 && sub_9987C0((__int64)&v142, 30, *(unsigned __int8 **)(a3 - 32)) )
            return v12;
          v142 = 0;
          v143 = (__int64 *)&v139;
          v61 = *(_QWORD *)(a3 + 16);
          if ( v61 )
          {
            if ( !*(_QWORD *)(v61 + 8) && (unsigned __int8)sub_996420(&v142, 30, (unsigned __int8 *)a3) )
            {
              v62 = v139;
              if ( *v139 == 78 )
              {
                v110 = *((_QWORD *)v139 + 2);
                if ( v110 )
                {
                  if ( !*(_QWORD *)(v110 + 8) )
                    v62 = (_BYTE *)*((_QWORD *)v139 - 4);
                }
              }
              v139 = v62;
              if ( *v62 == 69 )
              {
                v63 = *((_QWORD *)v62 - 4);
                if ( v63 )
                {
                  if ( v12 == v63 )
                    return v12;
                }
              }
            }
          }
        }
      }
    }
    goto LABEL_28;
  }
  if ( v5 != a3 )
    return 0;
LABEL_9:
  v11 = v7;
  if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
    v11 = **(_QWORD **)(v7 + 16);
  v12 = v5;
  if ( !sub_BCAC40(v11, 1) )
  {
    if ( *(_BYTE *)v5 == 78 )
      v5 = *(_QWORD *)(v5 - 32);
    v13 = *(_QWORD *)(v5 + 8);
    v14 = *(unsigned __int8 *)(v13 + 8);
    if ( (unsigned int)(v14 - 17) <= 1 )
      LOBYTE(v14) = *(_BYTE *)(**(_QWORD **)(v13 + 16) + 8LL);
    if ( (_BYTE)v14 == 12 )
    {
      v15 = sub_9AF8B0(v5, v6[11], 0, v6[8], 0, v6[10], 1);
      v16 = sub_BCB060(*(_QWORD *)(v5 + 8));
      if ( v16 == v15 && (unsigned int)sub_BCB060(v7) >= v16 )
      {
        v17 = (unsigned int **)v6[4];
        v144 = 257;
        v18 = *(_QWORD ***)(v5 + 8);
        v19 = *((unsigned __int8 *)v18 + 8);
        if ( (unsigned int)(v19 - 17) > 1 )
        {
          v21 = sub_BCB2A0(*v18);
        }
        else
        {
          BYTE4(v138) = (_BYTE)v19 == 18;
          LODWORD(v138) = *((_DWORD *)v18 + 8);
          v20 = (__int64 *)sub_BCB2A0(*v18);
          v21 = sub_BCE1B0(v20, v138);
        }
        return sub_A82DA0(v17, v5, v21, (__int64)&v142, 0, 0);
      }
    }
    return 0;
  }
  return v12;
}
