// Function: sub_1763090
// Address: 0x1763090
//
__int64 __fastcall sub_1763090(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v12; // r14
  __int16 v13; // ax
  __int64 v14; // rsi
  char v15; // al
  __int64 v16; // rcx
  unsigned int v17; // ebx
  __int64 v18; // r15
  _BYTE *v19; // rdi
  unsigned __int8 v20; // al
  __int64 v21; // r15
  int v22; // eax
  char v23; // al
  char v24; // al
  __int64 v25; // rdi
  __int64 v26; // rax
  double v27; // xmm4_8
  double v28; // xmm5_8
  __int64 v30; // rax
  unsigned int v31; // eax
  unsigned int v33; // r15d
  unsigned int v35; // edx
  __int64 v36; // rax
  __int64 v37; // rax
  unsigned __int64 v40; // rax
  __int64 v41; // r12
  _QWORD *v42; // rax
  __int16 v43; // r12
  _QWORD *v44; // rax
  int v45; // eax
  char v46; // r14
  unsigned int v47; // eax
  const void **v48; // r14
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // rax
  __int64 v52; // r12
  _QWORD *v53; // rax
  int v54; // eax
  char v55; // r14
  int v56; // eax
  int v57; // eax
  char v58; // r14
  int v59; // eax
  unsigned __int64 v60; // rax
  __int64 v61; // r12
  _QWORD *v62; // rax
  _QWORD *v63; // rax
  unsigned int v64; // r13d
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // rdx
  __int64 v68; // rcx
  unsigned int v69; // r12d
  __int16 v70; // r13
  __int64 v71; // rax
  __int64 v72; // r12
  _QWORD *v73; // rax
  int v74; // eax
  char v75; // r14
  unsigned int v76; // eax
  const void **v77; // rsi
  unsigned int v78; // r13d
  __int64 v79; // rdx
  __int64 v80; // rcx
  int v81; // r14d
  unsigned __int64 v82; // rax
  int v83; // eax
  unsigned int v84; // r14d
  __int64 v85; // rax
  __int64 v86; // r12
  _QWORD *v87; // rax
  __int64 v88; // r15
  unsigned int v89; // eax
  __int16 v90; // r12
  _QWORD *v91; // rax
  __int64 v92; // rax
  unsigned int v93; // eax
  char v94; // [rsp+20h] [rbp-140h]
  char v95; // [rsp+20h] [rbp-140h]
  __int16 v96; // [rsp+2Ch] [rbp-134h]
  int v97; // [rsp+2Ch] [rbp-134h]
  __int64 ***v98; // [rsp+30h] [rbp-130h]
  __int64 v99; // [rsp+38h] [rbp-128h]
  __int64 v100; // [rsp+40h] [rbp-120h]
  __int64 v101; // [rsp+58h] [rbp-108h] BYREF
  __int64 v102; // [rsp+60h] [rbp-100h] BYREF
  const void **v103; // [rsp+68h] [rbp-F8h] BYREF
  __int64 v104; // [rsp+70h] [rbp-F0h] BYREF
  unsigned int v105; // [rsp+78h] [rbp-E8h]
  __int64 v106; // [rsp+80h] [rbp-E0h] BYREF
  unsigned int v107; // [rsp+88h] [rbp-D8h]
  __int64 v108; // [rsp+90h] [rbp-D0h] BYREF
  unsigned int v109; // [rsp+98h] [rbp-C8h]
  __int64 v110; // [rsp+A0h] [rbp-C0h] BYREF
  unsigned int v111; // [rsp+A8h] [rbp-B8h]
  __int64 *v112; // [rsp+B0h] [rbp-B0h] BYREF
  int v113; // [rsp+B8h] [rbp-A8h]
  __int64 *v114; // [rsp+C0h] [rbp-A0h] BYREF
  int v115; // [rsp+C8h] [rbp-98h]
  unsigned __int64 v116; // [rsp+D0h] [rbp-90h] BYREF
  __int64 *v117; // [rsp+D8h] [rbp-88h]
  __int16 v118; // [rsp+E0h] [rbp-80h]
  __int64 v119; // [rsp+F0h] [rbp-70h] BYREF
  unsigned int v120; // [rsp+F8h] [rbp-68h]
  __int64 v121; // [rsp+100h] [rbp-60h] BYREF
  unsigned int v122; // [rsp+108h] [rbp-58h]
  __int64 v123; // [rsp+110h] [rbp-50h] BYREF
  unsigned int v124; // [rsp+118h] [rbp-48h]
  __int64 v125; // [rsp+120h] [rbp-40h] BYREF
  unsigned int v126; // [rsp+128h] [rbp-38h]

  v12 = **(_QWORD **)(a2 - 48);
  v99 = *(_QWORD *)(a2 - 48);
  v13 = *(_WORD *)(a2 + 18);
  v98 = *(__int64 ****)(a2 - 24);
  v14 = v12;
  v96 = v13;
  v15 = *(_BYTE *)(v12 + 8);
  if ( v15 == 16 )
  {
    v14 = **(_QWORD **)(v12 + 16);
    v15 = *(_BYTE *)(v14 + 8);
  }
  if ( v15 == 11 )
    v17 = sub_16431D0(v12);
  else
    v17 = sub_15A95F0(a1[333], v14);
  v18 = 0;
  if ( v17 )
  {
    v120 = v17;
    if ( v17 <= 0x40 )
    {
      v119 = 0;
      v122 = v17;
      v121 = 0;
      v124 = v17;
      v123 = 0;
      v126 = v17;
      v125 = 0;
    }
    else
    {
      sub_16A4EF0((__int64)&v119, 0, 0);
      v122 = v17;
      sub_16A4EF0((__int64)&v121, 0, 0);
      v124 = v17;
      sub_16A4EF0((__int64)&v123, 0, 0);
      v14 = 0;
      v126 = v17;
      sub_16A4EF0((__int64)&v125, 0, 0);
    }
    v19 = *(_BYTE **)(a2 - 24);
    v20 = v19[16];
    v21 = (__int64)(v19 + 24);
    if ( v20 != 13 )
    {
      if ( *(_BYTE *)(*(_QWORD *)v19 + 8LL) != 16 )
        goto LABEL_12;
      if ( v20 > 0x10u )
        goto LABEL_12;
      v30 = sub_15A1020(v19, v14, *(_QWORD *)v19, v16);
      if ( !v30 )
        goto LABEL_12;
      v21 = v30 + 24;
      if ( *(_BYTE *)(v30 + 16) != 13 )
        goto LABEL_12;
    }
    if ( sub_1757FA0(*(_WORD *)(a2 + 18) & 0x7FFF, v21, &v114) )
    {
      LODWORD(v117) = v17;
      v36 = 1LL << ((unsigned __int8)v17 - 1);
      if ( v17 > 0x40 )
      {
        sub_16A4EF0((__int64)&v116, 0, 0);
        v36 = 1LL << ((unsigned __int8)v17 - 1);
        if ( (unsigned int)v117 > 0x40 )
        {
          *(_QWORD *)(v116 + 8LL * ((v17 - 1) >> 6)) |= 1LL << ((unsigned __int8)v17 - 1);
          goto LABEL_14;
        }
      }
      else
      {
        v116 = 0;
      }
    }
    else
    {
      v22 = *(unsigned __int16 *)(a2 + 18);
      BYTE1(v22) &= ~0x80u;
      if ( v22 == 34 )
      {
        if ( *(_DWORD *)(v21 + 8) <= 0x40u )
        {
          v37 = *(_QWORD *)v21;
          v33 = 64;
          _RAX = ~v37;
          __asm { tzcnt   rdx, rax }
          if ( _RAX )
            v33 = _RDX;
        }
        else
        {
          v33 = sub_16A58F0(v21);
        }
      }
      else
      {
        if ( v22 != 36 )
        {
LABEL_12:
          LODWORD(v117) = v17;
          if ( v17 > 0x40 )
            sub_16A4EF0((__int64)&v116, -1, 1);
          else
            v116 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v17);
          goto LABEL_14;
        }
        v31 = *(_DWORD *)(v21 + 8);
        if ( v31 > 0x40 )
        {
          v33 = sub_16A58A0(v21);
        }
        else
        {
          _RDX = *(_QWORD *)v21;
          v33 = 64;
          __asm { tzcnt   rcx, rdx }
          if ( _RDX )
            v33 = _RCX;
          if ( v33 > v31 )
            v33 = v31;
        }
      }
      LODWORD(v117) = v17;
      if ( v17 > 0x40 )
      {
        sub_16A4EF0((__int64)&v116, 0, 0);
        v35 = (unsigned int)v117;
      }
      else
      {
        v35 = v17;
        v116 = 0;
      }
      if ( v33 == v35 )
      {
LABEL_14:
        v18 = a2;
        v23 = sub_17ADA40(a1, a2, 0, &v116, &v119, 0);
        if ( (unsigned int)v117 > 0x40 && v116 )
        {
          v94 = v23;
          j_j___libc_free_0_0(v116);
          v23 = v94;
        }
        if ( v23 )
          goto LABEL_39;
        LODWORD(v117) = v17;
        if ( v17 <= 0x40 )
          v116 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v17);
        else
          sub_16A4EF0((__int64)&v116, -1, 1);
        v24 = sub_17ADA40(a1, a2, 1, &v116, &v123, 0);
        if ( (unsigned int)v117 > 0x40 && v116 )
        {
          v95 = v24;
          j_j___libc_free_0_0(v116);
          v24 = v95;
        }
        if ( v24 )
          goto LABEL_39;
        v105 = v17;
        if ( v17 > 0x40 )
        {
          sub_16A4EF0((__int64)&v104, 0, 0);
          v107 = v17;
          sub_16A4EF0((__int64)&v106, 0, 0);
          v109 = v17;
          sub_16A4EF0((__int64)&v108, 0, 0);
          v111 = v17;
          sub_16A4EF0((__int64)&v110, 0, 0);
        }
        else
        {
          v104 = 0;
          v107 = v17;
          v106 = 0;
          v109 = v17;
          v108 = 0;
          v111 = v17;
          v110 = 0;
        }
        if ( sub_15FF7F0(*(_WORD *)(a2 + 18) & 0x7FFF) )
        {
          sub_17579D0((__int64)&v119, (__int64)&v104, &v106);
          sub_17579D0((__int64)&v123, (__int64)&v108, &v110);
        }
        else
        {
          sub_1757690((__int64)&v119, (__int64)&v104, (__int64)&v106);
          sub_1757690((__int64)&v123, (__int64)&v108, (__int64)&v110);
        }
        v97 = v96 & 0x7FFF;
        if ( *(_BYTE *)(v99 + 16) > 0x10u && sub_1455820((__int64)&v104, &v106) )
        {
          v60 = sub_15A3C50(v12, (__int64)&v104);
          v118 = 257;
          v61 = v60;
          v62 = sub_1648A60(56, 2u);
          v18 = (__int64)v62;
          if ( v62 )
            sub_17582E0((__int64)v62, v97, v61, (__int64)v98, (__int64)&v116);
          goto LABEL_38;
        }
        if ( *((_BYTE *)v98 + 16) > 0x10u && sub_1455820((__int64)&v108, &v110) )
        {
          v40 = sub_15A3C50(v12, (__int64)&v108);
          v118 = 257;
          v41 = v40;
          v42 = sub_1648A60(56, 2u);
          v18 = (__int64)v42;
          if ( v42 )
            sub_17582E0((__int64)v42, v97, v99, v41, (__int64)&v116);
          goto LABEL_38;
        }
        switch ( v97 )
        {
          case ' ':
          case '!':
            if ( (int)sub_16A9900((__int64)&v106, (unsigned __int64 *)&v108) < 0
              || (int)sub_16A9900((__int64)&v104, (unsigned __int64 *)&v110) > 0 )
            {
              v25 = *(_QWORD *)a2;
              if ( v97 == 32 )
                goto LABEL_105;
LABEL_36:
              v26 = sub_15A0600(v25);
LABEL_37:
              v18 = sub_170E100(a1, a2, v26, a3, a4, a5, a6, v27, v28, a9, a10);
LABEL_38:
              sub_135E100(&v110);
              sub_135E100(&v108);
              sub_135E100(&v106);
              sub_135E100(&v104);
LABEL_39:
              if ( v126 > 0x40 && v125 )
                j_j___libc_free_0_0(v125);
              if ( v124 > 0x40 && v123 )
                j_j___libc_free_0_0(v123);
              if ( v122 > 0x40 && v121 )
                j_j___libc_free_0_0(v121);
              if ( v120 > 0x40 && v119 )
                j_j___libc_free_0_0(v119);
              return v18;
            }
            sub_13A38D0((__int64)&v116, (__int64)&v119);
            sub_13D0570((__int64)&v116);
            v64 = v124;
            v113 = (int)v117;
            v112 = (__int64 *)v116;
            if ( v124 <= 0x40 )
            {
              if ( v123 != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v124) )
                goto LABEL_151;
            }
            else if ( v64 != (unsigned int)sub_16A58F0((__int64)&v123) )
            {
LABEL_151:
              sub_135E100((__int64 *)&v112);
              goto LABEL_92;
            }
            v101 = 0;
            v116 = (unsigned __int64)&v101;
            v117 = &v102;
            if ( !(unsigned __int8)sub_13D5F90((_QWORD **)&v116, v99) || !sub_1455820(v102, &v112) )
              v101 = v99;
            v117 = (__int64 *)&v103;
            if ( !(unsigned __int8)sub_175D710((__int64)&v116, v101, v65, v66) )
              goto LABEL_157;
            sub_13A38D0((__int64)&v114, (__int64)&v112);
            v88 = (__int64)*v103;
            if ( sub_14A9C60((__int64)&v114) )
            {
              v89 = sub_1455870((__int64 *)&v114);
              v100 = sub_15A0680(v88, v89, 0);
              v90 = sub_15FF0F0(v97);
            }
            else
            {
              v92 = sub_16A7400((__int64)&v114);
              if ( !sub_14A9C60(v92) )
              {
                sub_135E100((__int64 *)&v114);
LABEL_157:
                if ( !sub_1455000((__int64)&v112) )
                  goto LABEL_151;
                v117 = (__int64 *)&v103;
                v116 = (unsigned __int64)&v114;
                if ( !(unsigned __int8)sub_175DA00((__int64 **)&v116, v101, v67, v68) )
                  goto LABEL_151;
                v69 = sub_1455870(v114);
                v70 = sub_15FF0F0(v97);
                v71 = sub_15A0680((__int64)*v103, v69, 0);
                v118 = 257;
                v72 = v71;
                v73 = sub_1648A60(56, 2u);
                v18 = (__int64)v73;
                if ( v73 )
                  sub_17582E0((__int64)v73, v70, (__int64)v103, v72, (__int64)&v116);
LABEL_131:
                sub_135E100((__int64 *)&v112);
                goto LABEL_38;
              }
              v93 = sub_1455870((__int64 *)&v114);
              v100 = sub_15A0680(v88, v93, 0);
              v90 = (v97 != 32) + 35;
            }
            v118 = 257;
            v91 = sub_1648A60(56, 2u);
            v18 = (__int64)v91;
            if ( v91 )
              sub_17582E0((__int64)v91, v90, (__int64)v103, v100, (__int64)&v116);
LABEL_130:
            sub_135E100((__int64 *)&v114);
            goto LABEL_131;
          case '"':
            if ( (int)sub_16A9900((__int64)&v104, (unsigned __int64 *)&v110) > 0 )
              goto LABEL_141;
            if ( (int)sub_16A9900((__int64)&v106, (unsigned __int64 *)&v108) <= 0 )
              goto LABEL_104;
            if ( sub_1455820((__int64)&v110, &v104) )
              goto LABEL_144;
            v116 = (unsigned __int64)&v103;
            if ( !(unsigned __int8)sub_13D2630((_QWORD **)&v116, v98) )
              goto LABEL_92;
            sub_13A38D0((__int64)&v114, (__int64)&v106);
            sub_16A7800((__int64)&v114, 1u);
            v45 = v115;
            v115 = 0;
            LODWORD(v117) = v45;
            v116 = (unsigned __int64)v114;
            v46 = sub_1455820((__int64)v103, &v116);
            sub_135E100((__int64 *)&v116);
            sub_135E100((__int64 *)&v114);
            if ( !v46 )
            {
              v47 = sub_14A9E10((__int64)&v119);
              v48 = v103;
              if ( *((_DWORD *)v48 + 2) - (unsigned int)sub_1455840((__int64)v103) <= v47 )
              {
                v51 = sub_15A06D0(*v98, (__int64)&v116, v49, v50);
                v118 = 257;
                v52 = v51;
                v53 = sub_1648A60(56, 2u);
                v18 = (__int64)v53;
                if ( v53 )
                  sub_17582E0((__int64)v53, 33, v99, v52, (__int64)&v116);
                goto LABEL_38;
              }
LABEL_92:
              v18 = 0;
              if ( sub_15FF7F0(*(_WORD *)(a2 + 18) & 0x7FFF) )
              {
                if ( sub_13D0200(&v119, v120 - 1) && sub_13D0200(&v123, v124 - 1)
                  || (v18 = 0, sub_13D0200(&v121, v122 - 1)) && sub_13D0200(&v125, v126 - 1) )
                {
                  v43 = sub_15FF470(*(_WORD *)(a2 + 18) & 0x7FFF);
                  v118 = 257;
                  v44 = sub_1648A60(56, 2u);
                  v18 = (__int64)v44;
                  if ( v44 )
                    sub_17582E0((__int64)v44, v43, v99, (__int64)v98, (__int64)&v116);
                }
              }
              goto LABEL_38;
            }
LABEL_126:
            sub_13A38D0((__int64)&v112, (__int64)v103);
            sub_16A7490((__int64)&v112, 1);
            goto LABEL_127;
          case '#':
            if ( (int)sub_16A9900((__int64)&v104, (unsigned __int64 *)&v110) >= 0 )
              goto LABEL_141;
            if ( (int)sub_16A9900((__int64)&v106, (unsigned __int64 *)&v108) < 0 )
              goto LABEL_104;
            goto LABEL_99;
          case '$':
            if ( (int)sub_16A9900((__int64)&v106, (unsigned __int64 *)&v108) < 0 )
              goto LABEL_141;
            if ( (int)sub_16A9900((__int64)&v104, (unsigned __int64 *)&v110) >= 0 )
              goto LABEL_104;
            if ( sub_1455820((__int64)&v108, &v106) )
              goto LABEL_144;
            v116 = (unsigned __int64)&v103;
            if ( !(unsigned __int8)sub_13D2630((_QWORD **)&v116, v98) )
              goto LABEL_92;
            sub_13A38D0((__int64)&v114, (__int64)&v104);
            sub_16A7490((__int64)&v114, 1);
            v57 = v115;
            v115 = 0;
            LODWORD(v117) = v57;
            v116 = (unsigned __int64)v114;
            v58 = sub_1455820((__int64)v103, &v116);
            sub_135E100((__int64 *)&v116);
            sub_135E100((__int64 *)&v114);
            if ( v58 )
            {
              sub_13A38D0((__int64)&v112, (__int64)v103);
              sub_16A7800((__int64)&v112, 1u);
              v59 = v113;
              v113 = 0;
              v115 = v59;
              v114 = v112;
              sub_15A1070((__int64)*v98, (__int64)&v114);
              v118 = 257;
              v18 = (__int64)sub_1648A60(56, 2u);
              if ( v18 )
                JUMPOUT(0x1763E12);
              goto LABEL_130;
            }
            v76 = sub_14A9E10((__int64)&v119);
            v77 = v103;
            v78 = v76;
            LODWORD(v117) = *((_DWORD *)v103 + 2);
            if ( (unsigned int)v117 > 0x40 )
              sub_16A4FD0((__int64)&v116, v103);
            else
              v116 = (unsigned __int64)*v103;
            sub_16A7770((__int64)&v116);
            v81 = (int)v117;
            if ( (unsigned int)v117 > 0x40 )
            {
              v84 = v81 - sub_16A57B0((__int64)&v116);
              if ( v116 )
                j_j___libc_free_0_0(v116);
            }
            else
            {
              if ( v116 )
              {
                _BitScanReverse64(&v82, v116);
                v83 = v82 ^ 0x3F;
              }
              else
              {
                v83 = 64;
              }
              v84 = 64 - v83;
            }
            if ( v78 < v84 )
              goto LABEL_92;
            v85 = sub_15A06D0(*v98, (__int64)v77, v79, v80);
            v118 = 257;
            v86 = v85;
            v87 = sub_1648A60(56, 2u);
            v18 = (__int64)v87;
            if ( v87 )
              sub_17582E0((__int64)v87, 32, v99, v86, (__int64)&v116);
            goto LABEL_38;
          case '%':
            if ( (int)sub_16A9900((__int64)&v106, (unsigned __int64 *)&v108) <= 0 )
              goto LABEL_141;
            if ( (int)sub_16A9900((__int64)&v104, (unsigned __int64 *)&v110) > 0 )
              goto LABEL_104;
            if ( sub_1455820((__int64)&v110, &v104) )
              goto LABEL_100;
            goto LABEL_92;
          case '&':
            if ( (int)sub_16AEA10((__int64)&v104, (__int64)&v110) > 0 )
              goto LABEL_141;
            if ( (int)sub_16AEA10((__int64)&v106, (__int64)&v108) <= 0 )
              goto LABEL_104;
            if ( sub_1455820((__int64)&v110, &v104) )
              goto LABEL_144;
            v116 = (unsigned __int64)&v103;
            if ( (unsigned __int8)sub_13D2630((_QWORD **)&v116, v98) )
            {
              sub_13A38D0((__int64)&v114, (__int64)&v106);
              sub_16A7800((__int64)&v114, 1u);
              v54 = v115;
              v115 = 0;
              LODWORD(v117) = v54;
              v116 = (unsigned __int64)v114;
              v55 = sub_1455820((__int64)v103, &v116);
              sub_135E100((__int64 *)&v116);
              sub_135E100((__int64 *)&v114);
              if ( v55 )
                goto LABEL_126;
            }
            goto LABEL_92;
          case '\'':
            if ( (int)sub_16AEA10((__int64)&v104, (__int64)&v110) >= 0 )
              goto LABEL_141;
            if ( (int)sub_16AEA10((__int64)&v106, (__int64)&v108) < 0 )
              goto LABEL_104;
LABEL_99:
            if ( sub_1455820((__int64)&v108, &v106) )
            {
LABEL_100:
              v118 = 257;
              sub_1648A60(56, 2u);
              JUMPOUT(0x17639FF);
            }
            goto LABEL_92;
          case '(':
            if ( (int)sub_16AEA10((__int64)&v106, (__int64)&v108) < 0 )
              goto LABEL_141;
            if ( (int)sub_16AEA10((__int64)&v104, (__int64)&v110) >= 0 )
            {
LABEL_104:
              v25 = *(_QWORD *)a2;
LABEL_105:
              v26 = sub_15A0640(v25);
              goto LABEL_37;
            }
            if ( sub_1455820((__int64)&v108, &v106) )
            {
LABEL_144:
              v118 = 257;
              v63 = sub_1648A60(56, 2u);
              v18 = (__int64)v63;
              if ( v63 )
                sub_17582E0((__int64)v63, 33, v99, (__int64)v98, (__int64)&v116);
              goto LABEL_38;
            }
            v116 = (unsigned __int64)&v103;
            if ( !(unsigned __int8)sub_13D2630((_QWORD **)&v116, v98) )
              goto LABEL_92;
            sub_13A38D0((__int64)&v114, (__int64)&v104);
            sub_16A7490((__int64)&v114, 1);
            v74 = v115;
            v115 = 0;
            LODWORD(v117) = v74;
            v116 = (unsigned __int64)v114;
            v75 = sub_1455820((__int64)v103, &v116);
            sub_135E100((__int64 *)&v116);
            sub_135E100((__int64 *)&v114);
            if ( !v75 )
              goto LABEL_92;
            sub_13A38D0((__int64)&v112, (__int64)v103);
            sub_16A7800((__int64)&v112, 1u);
LABEL_127:
            v56 = v113;
            v113 = 0;
            v115 = v56;
            JUMPOUT(0x1763DCE);
          case ')':
            if ( (int)sub_16AEA10((__int64)&v106, (__int64)&v108) > 0 )
              JUMPOUT(0x1763AA9);
LABEL_141:
            v25 = *(_QWORD *)a2;
            goto LABEL_36;
        }
      }
      if ( v33 > 0x3F || v35 > 0x40 )
      {
        sub_16A5260(&v116, v33, v35);
        goto LABEL_14;
      }
      v36 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v33 + 64 - (unsigned __int8)v35) << v33;
    }
    v116 |= v36;
    goto LABEL_14;
  }
  return v18;
}
