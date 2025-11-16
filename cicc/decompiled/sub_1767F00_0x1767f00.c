// Function: sub_1767F00
// Address: 0x1767f00
//
_QWORD *__fastcall sub_1767F00(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  int v15; // eax
  _BYTE *v16; // rdi
  unsigned __int8 v17; // al
  __int64 v18; // r12
  unsigned int v19; // ebx
  int v20; // eax
  int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // rcx
  int v24; // eax
  int v25; // eax
  __int64 v26; // rbx
  unsigned __int64 v27; // r15
  unsigned int v28; // ecx
  unsigned int v29; // edx
  unsigned int v31; // eax
  __int64 v33; // rdx
  unsigned int v34; // r15d
  unsigned __int8 *v35; // rax
  unsigned __int8 *v36; // rsi
  _QWORD *v37; // rax
  __int64 *v38; // rbx
  __int64 v39; // r15
  __int64 v40; // rdi
  unsigned __int8 *v41; // rax
  __int64 v42; // r14
  _QWORD *v43; // rax
  _QWORD *v44; // r13
  __int64 v45; // rdx
  char v46; // al
  _BYTE *v48; // rdi
  unsigned __int8 v49; // al
  __int64 v50; // r8
  __int64 v51; // rax
  __int64 v52; // r12
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rbx
  __int64 v56; // r15
  int v57; // r12d
  unsigned int v58; // r13d
  unsigned __int64 v59; // rax
  int v60; // r13d
  __int64 v61; // rsi
  __int64 v62; // r15
  _QWORD **v63; // rax
  _QWORD *v64; // r14
  __int64 *v65; // rax
  __int64 v66; // rsi
  _QWORD *v67; // rax
  char v68; // bl
  unsigned int v69; // r13d
  unsigned __int64 v70; // rax
  __int64 v71; // r13
  __int64 v72; // rbx
  __int64 v73; // rdx
  unsigned __int8 *v74; // r15
  __int64 v75; // r12
  _QWORD *v76; // rax
  __int64 v77; // rax
  int v78; // eax
  int v79; // eax
  __int64 v80; // rax
  __int64 v81; // r12
  _QWORD *v82; // rax
  __int64 v83; // rax
  __int64 v84; // r12
  _QWORD *v85; // rax
  int v86; // ebx
  __int64 v87; // rax
  _BYTE *v88; // r12
  bool v89; // al
  int v90; // eax
  __int64 v91; // rax
  unsigned int v92; // r13d
  __int64 v93; // r15
  _QWORD *v94; // rax
  int v95; // eax
  unsigned int v96; // eax
  __int64 v97; // rsi
  unsigned __int64 v98; // rax
  __int64 v99; // r14
  _QWORD *v100; // rax
  _QWORD *v101; // rax
  unsigned int v102; // ebx
  __int64 v103; // r12
  __int64 v104; // r12
  __int64 v105; // r13
  __int64 v106; // rdx
  unsigned __int8 *v107; // r12
  __int64 v108; // rdx
  __int64 v109; // rcx
  __int64 v110; // rax
  __int64 v111; // r14
  _QWORD *v112; // rax
  __int64 v113; // rax
  unsigned int v114; // ebx
  bool v115; // al
  _QWORD *v116; // rax
  unsigned int v117; // ebx
  __int64 v118; // rax
  char v119; // dl
  char v120; // [rsp+4h] [rbp-CCh]
  char v121; // [rsp+4h] [rbp-CCh]
  unsigned int v122; // [rsp+8h] [rbp-C8h]
  unsigned int v123; // [rsp+8h] [rbp-C8h]
  unsigned int v124; // [rsp+10h] [rbp-C0h]
  unsigned __int8 *v125; // [rsp+18h] [rbp-B8h]
  __int64 v126; // [rsp+20h] [rbp-B0h]
  int v127; // [rsp+20h] [rbp-B0h]
  bool v128; // [rsp+20h] [rbp-B0h]
  int v129; // [rsp+20h] [rbp-B0h]
  unsigned int v131; // [rsp+28h] [rbp-A8h]
  int v132; // [rsp+28h] [rbp-A8h]
  int v133; // [rsp+28h] [rbp-A8h]
  int v134; // [rsp+3Ch] [rbp-94h] BYREF
  __int64 *v135; // [rsp+40h] [rbp-90h] BYREF
  int v136; // [rsp+48h] [rbp-88h]
  const char *v137; // [rsp+50h] [rbp-80h] BYREF
  int v138; // [rsp+58h] [rbp-78h]
  const char *v139; // [rsp+60h] [rbp-70h] BYREF
  __int64 v140; // [rsp+68h] [rbp-68h]
  __int16 v141; // [rsp+70h] [rbp-60h]
  unsigned __int64 v142; // [rsp+80h] [rbp-50h] BYREF
  char *v143; // [rsp+88h] [rbp-48h]
  __int16 v144; // [rsp+90h] [rbp-40h]

  v15 = *(unsigned __int16 *)(a2 + 18);
  BYTE1(v15) &= ~0x80u;
  if ( (unsigned int)(v15 - 32) > 1 )
  {
LABEL_2:
    v16 = *(_BYTE **)(a3 - 24);
    v17 = v16[16];
    v18 = (__int64)(v16 + 24);
    if ( v17 != 13 )
    {
      v45 = *(_QWORD *)v16;
      if ( *(_BYTE *)(*(_QWORD *)v16 + 8LL) != 16
        || v17 > 0x10u
        || (v54 = sub_15A1020(v16, a2, v45, a4)) == 0
        || (v18 = v54 + 24, *(_BYTE *)(v54 + 16) != 13) )
      {
        v46 = *(_BYTE *)(a3 + 16);
        if ( v46 == 47 )
        {
          if ( !(unsigned __int8)sub_17573B0(*(_BYTE **)(a3 - 48), a2, v45, a4) )
            return 0;
          v55 = *(_QWORD *)(a3 - 24);
          if ( !v55 )
            return 0;
        }
        else
        {
          if ( v46 != 5 || *(_WORD *)(a3 + 18) != 23 )
            return 0;
          v86 = *(_DWORD *)(a3 + 20);
          v87 = v86 & 0xFFFFFFF;
          v88 = *(_BYTE **)(a3 - 24 * v87);
          if ( v88[16] == 13 )
          {
            if ( *((_DWORD *)v88 + 8) <= 0x40u )
            {
              v89 = *((_QWORD *)v88 + 3) == 1;
            }
            else
            {
              v132 = *((_DWORD *)v88 + 8);
              v89 = v132 - 1 == (unsigned int)sub_16A57B0((__int64)(v88 + 24));
            }
            if ( !v89 )
              return 0;
          }
          else
          {
            if ( *(_BYTE *)(*(_QWORD *)v88 + 8LL) != 16 )
              return 0;
            v113 = sub_15A1020(v88, a2, 4 * v87, a4);
            if ( v113 && *(_BYTE *)(v113 + 16) == 13 )
            {
              v114 = *(_DWORD *)(v113 + 32);
              if ( v114 <= 0x40 )
                v115 = *(_QWORD *)(v113 + 24) == 1;
              else
                v115 = v114 - 1 == (unsigned int)sub_16A57B0(v113 + 24);
              if ( !v115 )
                return 0;
            }
            else
            {
              v117 = 0;
              v133 = *(_QWORD *)(*(_QWORD *)v88 + 32LL);
              if ( v133 )
              {
                do
                {
                  v118 = sub_15A0A60((__int64)v88, v117);
                  if ( !v118 )
                    return 0;
                  v119 = *(_BYTE *)(v118 + 16);
                  if ( v119 != 9 )
                  {
                    if ( v119 != 13 )
                      return 0;
                    if ( *(_DWORD *)(v118 + 32) <= 0x40u )
                    {
                      if ( *(_QWORD *)(v118 + 24) != 1 )
                        return 0;
                    }
                    else
                    {
                      v129 = *(_DWORD *)(v118 + 32);
                      if ( (unsigned int)sub_16A57B0(v118 + 24) != v129 - 1 )
                        return 0;
                    }
                  }
                }
                while ( v133 != ++v117 );
              }
            }
            v86 = *(_DWORD *)(a3 + 20);
          }
          v55 = *(_QWORD *)(a3 + 24 * (1LL - (v86 & 0xFFFFFFF)));
          if ( !v55 )
            return 0;
        }
        v56 = *(_QWORD *)a3;
        v131 = *(_DWORD *)(a4 + 8);
        if ( v131 > 0x40 )
        {
          v127 = sub_16A5940(a4);
          v57 = *(_WORD *)(a2 + 18) & 0x7FFF;
          if ( !sub_15FF7E0(v57) )
          {
            v128 = v127 == 1;
LABEL_132:
            if ( !sub_15FF7F0(*(_WORD *)(a2 + 18) & 0x7FFF) )
            {
              v95 = *(unsigned __int16 *)(a2 + 18);
              BYTE1(v95) &= ~0x80u;
              if ( (unsigned int)(v95 - 32) <= 1 && v128 )
              {
                v96 = *(_DWORD *)(a4 + 8);
                if ( v96 > 0x40 )
                {
                  v97 = v96 - 1 - (unsigned int)sub_16A57B0(a4);
                }
                else
                {
                  v97 = 0xFFFFFFFFLL;
                  if ( *(_QWORD *)a4 )
                  {
                    _BitScanReverse64(&v98, *(_QWORD *)a4);
                    v97 = 63 - ((unsigned int)v98 ^ 0x3F);
                  }
                }
                v144 = 257;
                v99 = sub_15A0680(v56, v97, 0);
                v100 = sub_1648A60(56, 2u);
                v44 = v100;
                if ( v100 )
                  sub_17582E0((__int64)v100, v57, v55, v99, (__int64)&v142);
                return v44;
              }
              return 0;
            }
            goto LABEL_119;
          }
          if ( v127 == 1 )
            goto LABEL_68;
        }
        else
        {
          v57 = *(_WORD *)(a2 + 18) & 0x7FFF;
          if ( *(_QWORD *)a4 && (*(_QWORD *)a4 & (*(_QWORD *)a4 - 1LL)) == 0 )
          {
            if ( sub_15FF7E0(v57) )
            {
LABEL_68:
              v58 = *(_DWORD *)(a4 + 8);
              if ( v58 > 0x40 )
              {
                v60 = v58 - sub_16A57B0(a4);
                v61 = (unsigned int)(v60 - 1);
              }
              else if ( *(_QWORD *)a4 )
              {
                _BitScanReverse64(&v59, *(_QWORD *)a4);
                LODWORD(v59) = v59 ^ 0x3F;
                v60 = 64 - v59;
                v61 = (unsigned int)(63 - v59);
              }
              else
              {
                v61 = 0xFFFFFFFFLL;
                v60 = 0;
              }
              if ( v131 == v60 )
              {
                if ( v57 == 35 )
                {
                  LOWORD(v57) = 32;
                }
                else if ( v57 == 36 )
                {
                  LOWORD(v57) = 33;
                }
              }
              v144 = 257;
              v62 = sub_15A0680(v56, v61, 0);
              v44 = sub_1648A60(56, 2u);
              if ( v44 )
              {
                v63 = *(_QWORD ***)v55;
                if ( *(_BYTE *)(*(_QWORD *)v55 + 8LL) == 16 )
                {
                  v64 = v63[4];
                  v65 = (__int64 *)sub_1643320(*v63);
                  v66 = (__int64)sub_16463B0(v65, (unsigned int)v64);
                }
                else
                {
                  v66 = sub_1643320(*v63);
                }
                sub_15FEC10((__int64)v44, v66, 51, v57, v55, v62, (__int64)&v142, 0);
              }
              return v44;
            }
            v128 = 1;
            goto LABEL_132;
          }
          if ( !sub_15FF7E0(v57) )
          {
            if ( !sub_15FF7F0(*(_WORD *)(a2 + 18) & 0x7FFF) )
              return 0;
LABEL_119:
            v91 = sub_15A0680(v56, v131 - 1, 0);
            v92 = *(_DWORD *)(a4 + 8);
            v93 = v91;
            if ( v92 <= 0x40 )
            {
              if ( *(_QWORD *)a4 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v92) )
              {
LABEL_121:
                if ( v57 != 41 )
                {
                  if ( v57 == 38 )
                    goto LABEL_123;
                  return 0;
                }
                goto LABEL_176;
              }
            }
            else if ( v92 == (unsigned int)sub_16A58F0(a4) )
            {
              goto LABEL_121;
            }
            if ( !sub_13D01C0(a4) )
              return 0;
            if ( (unsigned int)(v57 - 40) > 1 )
            {
              if ( (unsigned int)(v57 - 38) <= 1 )
              {
LABEL_123:
                v144 = 257;
                v94 = sub_1648A60(56, 2u);
                v44 = v94;
                if ( v94 )
                  sub_17582E0((__int64)v94, 33, v55, v93, (__int64)&v142);
                return v44;
              }
              return 0;
            }
LABEL_176:
            v144 = 257;
            v116 = sub_1648A60(56, 2u);
            v44 = v116;
            if ( v116 )
              sub_17582E0((__int64)v116, 32, v55, v93, (__int64)&v142);
            return v44;
          }
        }
        if ( v57 == 36 )
        {
          v57 = 37;
        }
        else if ( v57 == 35 )
        {
          v57 = 34;
        }
        goto LABEL_68;
      }
    }
    v19 = *(_DWORD *)(v18 + 8);
    if ( v19 > 0x40 )
    {
      if ( v19 - (unsigned int)sub_16A57B0(v18) > 0x40 )
        return 0;
      v124 = *(_DWORD *)(a4 + 8);
      if ( (unsigned __int64)v124 <= **(_QWORD **)v18 )
        return 0;
    }
    else
    {
      v124 = *(_DWORD *)(a4 + 8);
      if ( (unsigned __int64)v124 <= *(_QWORD *)v18 )
        return 0;
    }
    v20 = *(unsigned __int16 *)(a2 + 18);
    BYTE1(v20) &= ~0x80u;
    v134 = v20;
    v125 = *(unsigned __int8 **)(a3 - 48);
    v126 = *(_QWORD *)a3;
    if ( sub_15F2380(a3) )
    {
      v21 = v134;
      if ( v134 == 38 )
      {
        sub_13A38D0((__int64)&v139, a4);
        sub_16A6020((__int64)&v139, v18);
        goto LABEL_89;
      }
      if ( (unsigned int)(v134 - 32) <= 1 )
      {
        sub_13A38D0((__int64)&v139, a4);
        sub_16A6020((__int64)&v139, v18);
        sub_13A38D0((__int64)&v142, (__int64)&v139);
        sub_16A7E20((__int64)&v142, v18);
        v120 = sub_1455820((__int64)&v142, (_QWORD *)a4);
        sub_135E100((__int64 *)&v142);
        sub_135E100((__int64 *)&v139);
        if ( v120 )
        {
          sub_13A38D0((__int64)&v139, a4);
          sub_16A6020((__int64)&v139, v18);
          goto LABEL_50;
        }
        v21 = v134;
      }
      if ( v21 == 40 )
      {
        sub_13A38D0((__int64)&v137, a4);
        sub_16A7800((__int64)&v137, 1u);
        v79 = v138;
        v138 = 0;
        LODWORD(v140) = v79;
        v139 = v137;
        sub_13A38D0((__int64)&v142, (__int64)&v139);
        sub_16A6020((__int64)&v142, v18);
        goto LABEL_96;
      }
      if ( (unsigned __int8)sub_1757250(&v134, a4) )
      {
        v83 = sub_15A06D0((__int64 **)v126, a4, v22, v23);
        v144 = 257;
        v84 = v83;
        v85 = sub_1648A60(56, 2u);
        v44 = v85;
        if ( v85 )
          sub_17582E0((__int64)v85, v134, (__int64)v125, v84, (__int64)&v142);
        return v44;
      }
    }
    if ( !sub_15F2370(a3) )
      goto LABEL_14;
    v24 = v134;
    if ( v134 != 34 )
    {
      if ( (unsigned int)(v134 - 32) > 1 )
      {
LABEL_13:
        if ( v24 != 36 )
        {
LABEL_14:
          v25 = *(unsigned __int16 *)(a2 + 18);
          v26 = *(_QWORD *)(a3 + 8);
          BYTE1(v25) &= ~0x80u;
          if ( (unsigned int)(v25 - 32) <= 1 )
          {
            if ( v26 && !*(_QWORD *)(v26 + 8) )
            {
              v67 = *(_QWORD **)v18;
              if ( *(_DWORD *)(v18 + 8) > 0x40u )
                v67 = (_QWORD *)*v67;
              v68 = (char)v67;
              LODWORD(v143) = v124;
              v69 = v124 - (_DWORD)v67;
              if ( v124 > 0x40 )
                sub_16A4EF0((__int64)&v142, 0, 0);
              else
                v142 = 0;
              if ( v69 )
              {
                if ( v69 > 0x40 )
                {
                  sub_16A5260(&v142, 0, v69);
                }
                else
                {
                  v70 = 0xFFFFFFFFFFFFFFFFLL >> (v68 - (unsigned __int8)v124 + 64);
                  if ( (unsigned int)v143 > 0x40 )
                    *(_QWORD *)v142 |= v70;
                  else
                    v142 |= v70;
                }
              }
              v71 = sub_15A1070(v126, (__int64)&v142);
              sub_135E100((__int64 *)&v142);
              v72 = a1[1];
              v139 = sub_1649960(a3);
              v140 = v73;
              v144 = 773;
              v142 = (unsigned __int64)&v139;
              v143 = ".mask";
              v74 = sub_1729500(v72, v125, v71, (__int64 *)&v142, *(double *)a5.m128_u64, a6, a7);
              sub_13A38D0((__int64)&v142, a4);
              sub_16A81B0((__int64)&v142, v18);
              v75 = sub_15A1070(v126, (__int64)&v142);
              sub_135E100((__int64 *)&v142);
              v144 = 257;
              v76 = sub_1648A60(56, 2u);
              v44 = v76;
              if ( v76 )
                sub_17582E0((__int64)v76, v134, (__int64)v74, v75, (__int64)&v142);
              return v44;
            }
            LOBYTE(v137) = 0;
            goto LABEL_17;
          }
          LOBYTE(v137) = 0;
          if ( !v26 || *(_QWORD *)(v26 + 8) )
          {
LABEL_17:
            v27 = v124 - 1;
            v122 = *(_DWORD *)(v18 + 8);
            v28 = v124 - 1;
            if ( v122 > 0x40 )
            {
              v78 = sub_16A57B0(v18);
              v28 = v124 - 1;
              if ( v122 - v78 <= 0x40 && v27 >= **(_QWORD **)v18 )
                v28 = **(_QWORD **)v18;
            }
            else if ( v27 >= *(_QWORD *)v18 )
            {
              v28 = *(_QWORD *)v18;
            }
            if ( v26 && !*(_QWORD *)(v26 + 8) && v28 )
            {
              v29 = *(_DWORD *)(a4 + 8);
              if ( v29 > 0x40 )
              {
                v123 = v28;
                v31 = sub_16A58A0(a4);
                v28 = v123;
              }
              else
              {
                _RSI = *(_QWORD *)a4;
                v31 = 64;
                __asm { tzcnt   rdi, rsi }
                if ( *(_QWORD *)a4 )
                  v31 = _RDI;
                if ( v31 > v29 )
                  v31 = *(_DWORD *)(a4 + 8);
              }
              if ( v28 <= v31 )
              {
                v33 = a1[333];
                v34 = v124 - v28;
                v35 = *(unsigned __int8 **)(v33 + 24);
                v36 = &v35[*(unsigned int *)(v33 + 32)];
                if ( v35 != v36 )
                {
                  while ( v34 != *v35 )
                  {
                    if ( v36 == ++v35 )
                      return 0;
                  }
                  v37 = (_QWORD *)sub_16498A0(a2);
                  v38 = (__int64 *)sub_1644900(v37, v34);
                  if ( *(_BYTE *)(v126 + 8) == 16 )
                    v38 = sub_16463B0(v38, *(_QWORD *)(v126 + 32));
                  sub_13A38D0((__int64)&v139, a4);
                  sub_16A6020((__int64)&v139, v18);
                  sub_16A5A50((__int64)&v142, (__int64 *)&v139, v34);
                  v39 = sub_15A1070((__int64)v38, (__int64)&v142);
                  sub_135E100((__int64 *)&v142);
                  sub_135E100((__int64 *)&v139);
                  v40 = a1[1];
                  v141 = 257;
                  v41 = sub_1708970(v40, 36, (__int64)v125, (__int64 **)v38, (__int64 *)&v139);
                  v144 = 257;
                  v42 = (__int64)v41;
                  v43 = sub_1648A60(56, 2u);
                  v44 = v43;
                  if ( v43 )
                    sub_17582E0((__int64)v43, v134, v42, v39, (__int64)&v142);
                  return v44;
                }
              }
            }
            return 0;
          }
          if ( !sub_1757FA0(v134, a4, &v137) )
          {
            v26 = *(_QWORD *)(a3 + 8);
            goto LABEL_17;
          }
          v101 = *(_QWORD **)v18;
          if ( *(_DWORD *)(v18 + 8) > 0x40u )
            v101 = (_QWORD *)*v101;
          LODWORD(v143) = v124;
          v102 = v124 - 1 - (_DWORD)v101;
          v103 = 1LL << ((unsigned __int8)v124 - 1 - (unsigned __int8)v101);
          if ( v124 > 0x40 )
          {
            sub_16A4EF0((__int64)&v142, 0, 0);
            if ( (unsigned int)v143 > 0x40 )
            {
              *(_QWORD *)(v142 + 8LL * (v102 >> 6)) |= v103;
LABEL_151:
              v104 = sub_15A1070(v126, (__int64)&v142);
              sub_135E100((__int64 *)&v142);
              v105 = a1[1];
              v139 = sub_1649960(a3);
              v140 = v106;
              v144 = 773;
              v142 = (unsigned __int64)&v139;
              v143 = ".mask";
              v107 = sub_1729500(v105, v125, v104, (__int64 *)&v142, *(double *)a5.m128_u64, a6, a7);
              v110 = sub_15A06D0((__int64 **)v126, (__int64)v125, v108, v109);
              v144 = 257;
              v111 = v110;
              v112 = sub_1648A60(56, 2u);
              v44 = v112;
              if ( v112 )
                sub_17582E0((__int64)v112, 32 - (((_BYTE)v137 == 0) - 1), (__int64)v107, v111, (__int64)&v142);
              return v44;
            }
          }
          else
          {
            v142 = 0;
          }
          v142 |= v103;
          goto LABEL_151;
        }
        sub_13A38D0((__int64)&v137, a4);
        sub_16A7800((__int64)&v137, 1u);
        v90 = v138;
        v138 = 0;
        LODWORD(v140) = v90;
        v139 = v137;
        sub_13A38D0((__int64)&v142, (__int64)&v139);
        sub_16A81B0((__int64)&v142, v18);
LABEL_96:
        sub_16A7490((__int64)&v142, 1);
        v136 = (int)v143;
        v135 = (__int64 *)v142;
        sub_135E100((__int64 *)&v139);
        sub_135E100((__int64 *)&v137);
        v80 = sub_15A1070(v126, (__int64)&v135);
        v144 = 257;
        v81 = v80;
        v82 = sub_1648A60(56, 2u);
        v44 = v82;
        if ( v82 )
          sub_17582E0((__int64)v82, v134, (__int64)v125, v81, (__int64)&v142);
        sub_135E100((__int64 *)&v135);
        return v44;
      }
      sub_13A38D0((__int64)&v139, a4);
      sub_16A81B0((__int64)&v139, v18);
      sub_13A38D0((__int64)&v142, (__int64)&v139);
      sub_16A7E20((__int64)&v142, v18);
      v121 = sub_1455820((__int64)&v142, (_QWORD *)a4);
      sub_135E100((__int64 *)&v142);
      sub_135E100((__int64 *)&v139);
      if ( !v121 )
      {
        v24 = v134;
        goto LABEL_13;
      }
      sub_13A38D0((__int64)&v139, a4);
      sub_16A81B0((__int64)&v139, v18);
LABEL_50:
      v51 = sub_15A1070(v126, (__int64)&v139);
      v144 = 257;
      v52 = v51;
      v44 = sub_1648A60(56, 2u);
      if ( !v44 )
      {
LABEL_52:
        sub_135E100((__int64 *)&v139);
        return v44;
      }
LABEL_51:
      sub_17582E0((__int64)v44, v134, (__int64)v125, v52, (__int64)&v142);
      goto LABEL_52;
    }
    sub_13A38D0((__int64)&v139, a4);
    sub_16A81B0((__int64)&v139, v18);
LABEL_89:
    v77 = sub_15A1070(v126, (__int64)&v139);
    v144 = 257;
    v52 = v77;
    v44 = sub_1648A60(56, 2u);
    if ( !v44 )
      goto LABEL_52;
    goto LABEL_51;
  }
  v48 = *(_BYTE **)(a3 - 48);
  v49 = v48[16];
  v50 = (__int64)(v48 + 24);
  if ( v49 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v48 + 8LL) != 16 )
      goto LABEL_2;
    if ( v49 > 0x10u )
      goto LABEL_2;
    v53 = sub_15A1020(v48, a2, *(_QWORD *)v48, a4);
    if ( !v53 || *(_BYTE *)(v53 + 16) != 13 )
      goto LABEL_2;
    v50 = v53 + 24;
  }
  return sub_1767AD0(a1, a2, *(__int64 **)(a3 - 24), a4, v50, a5, a6, a7, a8, a9, a10, a11, a12);
}
