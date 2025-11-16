// Function: sub_1FB5FC0
// Address: 0x1fb5fc0
//
__int64 __fastcall sub_1FB5FC0(
        __int64 a1,
        __int64 a2,
        double a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  _QWORD *v9; // r13
  __int64 v10; // rax
  unsigned __int8 *v11; // rdx
  __m128i v12; // xmm0
  __int64 v13; // r14
  __int64 v14; // rcx
  int v15; // ebx
  __int128 v16; // xmm1
  const void **v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // r9
  __int64 v25; // rsi
  __int64 *v26; // r12
  __int64 *v27; // rax
  __int64 v28; // rsi
  __int64 *v30; // rdi
  _QWORD *v31; // rax
  unsigned int v32; // eax
  __int64 v33; // r9
  __int16 v34; // ax
  int v35; // eax
  const __m128i *v36; // rax
  int v37; // eax
  __int64 v38; // r10
  __int64 v39; // r8
  char v40; // al
  __int64 v41; // r9
  unsigned __int8 v42; // cl
  __int64 *v43; // r12
  __int64 v44; // rsi
  unsigned int v45; // r14d
  __int64 v46; // rax
  _QWORD *v47; // rcx
  __int64 v48; // rax
  char v49; // bl
  __int64 v50; // rax
  const __m128i *v51; // rax
  unsigned int v52; // eax
  __int64 v53; // rcx
  __int64 v54; // rax
  __int64 v55; // rsi
  __int64 *v56; // r12
  _OWORD *v57; // rcx
  __int64 v58; // r9
  unsigned int v59; // eax
  _OWORD *v60; // rcx
  __int64 v61; // rsi
  __int64 v62; // rcx
  unsigned int v63; // edx
  __int64 v64; // rsi
  __int64 *v65; // r12
  bool v66; // al
  __int64 v67; // rdx
  unsigned int v68; // eax
  __int64 v69; // rax
  __int64 v70; // rcx
  __int64 v71; // rax
  unsigned __int8 v72; // al
  __int64 v73; // r15
  __int64 v74; // rdx
  __int64 *v75; // r10
  __int64 v76; // r15
  __int64 v77; // rdx
  __int64 *v78; // rdx
  __int64 v79; // r15
  int v80; // eax
  __int64 v81; // rcx
  _QWORD *v82; // rax
  unsigned __int64 v83; // rdx
  __int64 v84; // rax
  __int64 v85; // rcx
  __int64 v86; // r14
  __int64 *v87; // r12
  __int64 *v88; // rax
  __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // r15
  __int64 v92; // rdx
  __int64 *v93; // r10
  __int64 v94; // r15
  __int64 v95; // rdx
  __int64 *v96; // r14
  unsigned __int64 v97; // rdx
  unsigned __int64 v98; // r15
  __int64 *v99; // r12
  __m128i v100; // [rsp-20h] [rbp-E0h]
  __int128 v101; // [rsp-10h] [rbp-D0h]
  __m128i v102; // [rsp+10h] [rbp-B0h]
  char v103; // [rsp+10h] [rbp-B0h]
  __int64 v104; // [rsp+10h] [rbp-B0h]
  __int64 v105; // [rsp+10h] [rbp-B0h]
  _OWORD *v106; // [rsp+20h] [rbp-A0h]
  _OWORD *v107; // [rsp+20h] [rbp-A0h]
  __int64 v108; // [rsp+28h] [rbp-98h]
  __int64 v109; // [rsp+28h] [rbp-98h]
  unsigned int v110; // [rsp+28h] [rbp-98h]
  unsigned int v111; // [rsp+30h] [rbp-90h]
  int v112; // [rsp+38h] [rbp-88h]
  unsigned int v113; // [rsp+3Ch] [rbp-84h]
  __int64 v114; // [rsp+40h] [rbp-80h]
  __int64 v115; // [rsp+40h] [rbp-80h]
  __int64 v116; // [rsp+40h] [rbp-80h]
  _OWORD *v117; // [rsp+50h] [rbp-70h]
  __int64 *v118; // [rsp+50h] [rbp-70h]
  __int64 v119; // [rsp+50h] [rbp-70h]
  __int64 *v120; // [rsp+50h] [rbp-70h]
  __int64 v121; // [rsp+50h] [rbp-70h]
  __int64 v122; // [rsp+60h] [rbp-60h] BYREF
  const void **v123; // [rsp+68h] [rbp-58h]
  __int64 v124; // [rsp+70h] [rbp-50h] BYREF
  __int64 v125; // [rsp+78h] [rbp-48h]
  __int64 v126; // [rsp+80h] [rbp-40h] BYREF
  __int64 v127; // [rsp+88h] [rbp-38h]

  v9 = (_QWORD *)a2;
  v10 = *(_QWORD *)(a2 + 32);
  v11 = *(unsigned __int8 **)(a2 + 40);
  v12 = _mm_loadu_si128((const __m128i *)v10);
  v13 = *(_QWORD *)v10;
  v14 = *v11;
  v15 = *(_DWORD *)(v10 + 8);
  v16 = (__int128)_mm_loadu_si128((const __m128i *)(v10 + 40));
  v17 = (const void **)*((_QWORD *)v11 + 1);
  v18 = *(_QWORD *)(v10 + 40);
  LOBYTE(v122) = v14;
  v123 = v17;
  v19 = *(_QWORD *)(v18 + 96);
  LOBYTE(v18) = *(_BYTE *)(v18 + 88);
  v112 = v15;
  v125 = v19;
  LOBYTE(v124) = v18;
  v111 = sub_1D159C0((__int64)&v122, a2, v19, v14, a8, a9);
  v113 = sub_1D159C0((__int64)&v124, a2, v20, v21, v22, v23);
  if ( *(_WORD *)(v13 + 24) == 48 )
  {
    v30 = *(__int64 **)a1;
    v126 = 0;
    LODWORD(v127) = 0;
    v31 = sub_1D2B300(v30, 0x30u, (__int64)&v126, v122, (__int64)v123, v24);
    v28 = v126;
    v9 = v31;
    if ( !v126 )
      return (__int64)v9;
    goto LABEL_8;
  }
  if ( sub_1D23600(*(_QWORD *)a1, v12.m128i_i64[0]) )
  {
    v25 = *(_QWORD *)(a2 + 72);
    v26 = *(__int64 **)a1;
    v126 = v25;
    if ( v25 )
      sub_1623A60((__int64)&v126, v25, 2);
    v101 = v16;
    LODWORD(v127) = *((_DWORD *)v9 + 16);
    v100 = v12;
LABEL_6:
    v27 = sub_1D332F0(
            v26,
            148,
            (__int64)&v126,
            (unsigned int)v122,
            v123,
            0,
            *(double *)v12.m128i_i64,
            *(double *)&v16,
            a5,
            v100.m128i_i64[0],
            v100.m128i_u64[1],
            v101);
    goto LABEL_7;
  }
  v32 = sub_1D23330(*(_QWORD *)a1, v12.m128i_i64[0], v12.m128i_i64[1], 0);
  v33 = 0;
  if ( v32 >= v111 + 1 - v113 )
    return v12.m128i_i64[0];
  v34 = *(_WORD *)(v13 + 24);
  if ( v34 == 148 )
  {
    v47 = *(_QWORD **)(v13 + 32);
    v48 = v47[5];
    v49 = *(_BYTE *)(v48 + 88);
    v50 = *(_QWORD *)(v48 + 96);
    LOBYTE(v126) = v49;
    v127 = v50;
    if ( v49 == (_BYTE)v124 )
    {
      if ( v49 || v50 == v125 )
        goto LABEL_21;
    }
    else if ( (_BYTE)v124 )
    {
      v110 = sub_1F6C8D0(v124);
      goto LABEL_55;
    }
    v107 = v47;
    v68 = sub_1F58D40((__int64)&v124);
    v57 = v107;
    v58 = 0;
    v110 = v68;
LABEL_55:
    if ( v49 )
    {
      v59 = sub_1F6C8D0(v49);
    }
    else
    {
      v106 = v57;
      v105 = v58;
      v59 = sub_1F58D40((__int64)&v126);
      v60 = v106;
      v33 = v105;
    }
    if ( v59 > v110 )
    {
      v61 = *(_QWORD *)(a2 + 72);
      v26 = *(__int64 **)a1;
      v126 = v61;
      if ( v61 )
      {
        v117 = v60;
        sub_1623A60((__int64)&v126, v61, 2);
        v60 = v117;
      }
      v101 = v16;
      LODWORD(v127) = *((_DWORD *)v9 + 16);
      v100 = *(__m128i *)v60;
      goto LABEL_6;
    }
    goto LABEL_21;
  }
  if ( ((v34 - 142) & 0xFFFD) == 0 )
  {
    v51 = *(const __m128i **)(v13 + 32);
    a5 = _mm_loadu_si128(v51);
    v102 = a5;
    v52 = sub_1F701D0(v51->m128i_i64[0], v51->m128i_u32[2]);
    v33 = 0;
    if ( v52 <= v113 )
    {
      if ( !*(_BYTE *)(a1 + 24) )
        goto LABEL_50;
      v53 = *(_QWORD *)(a1 + 8);
      v54 = 1;
      if ( (_BYTE)v122 == 1
        || (_BYTE)v122 && (v54 = (unsigned __int8)v122, *(_QWORD *)(v53 + 8LL * (unsigned __int8)v122 + 120)) )
      {
        if ( !*(_BYTE *)(v53 + 259 * v54 + 2564) )
          goto LABEL_50;
      }
    }
    v34 = *(_WORD *)(v13 + 24);
  }
  if ( (unsigned __int16)(v34 - 149) <= 2u )
  {
    v35 = sub_1F701D0(**(_QWORD **)(v13 + 32), *(_DWORD *)(*(_QWORD *)(v13 + 32) + 8LL));
    v33 = 0;
    if ( v35 == v113 )
    {
      if ( !*(_BYTE *)(a1 + 24)
        || ((v62 = *(_QWORD *)(a1 + 8), v63 = 1, (_BYTE)v122 == 1)
         || (_BYTE)v122 && (v63 = (unsigned __int8)v122, *(_QWORD *)(v62 + 8LL * (unsigned __int8)v122 + 120)))
        && !*(_BYTE *)(v62 + 259LL * v63 + 2572) )
      {
        v64 = *(_QWORD *)(a2 + 72);
        v65 = *(__int64 **)a1;
        v126 = v64;
        if ( v64 )
          sub_1623A60((__int64)&v126, v64, 2);
        LODWORD(v127) = *((_DWORD *)v9 + 16);
        v46 = sub_1D327E0(
                v65,
                **(_QWORD **)(v13 + 32),
                *(_QWORD *)(*(_QWORD *)(v13 + 32) + 8LL),
                (__int64)&v126,
                v122,
                v123,
                *(double *)v12.m128i_i64,
                *(double *)&v16,
                *(double *)a5.m128i_i64);
        goto LABEL_34;
      }
    }
    v34 = *(_WORD *)(v13 + 24);
  }
  if ( v34 != 143
    || (v36 = *(const __m128i **)(v13 + 32),
        v102 = _mm_loadu_si128(v36),
        v37 = sub_1F701D0(v36->m128i_i64[0], v36->m128i_u32[2]),
        v33 = 0,
        v37 != v113)
    || *(_BYTE *)(a1 + 24)
    && ((v70 = *(_QWORD *)(a1 + 8), v71 = 1, (_BYTE)v122 != 1)
     && (!(_BYTE)v122 || (v71 = (unsigned __int8)v122, !*(_QWORD *)(v70 + 8LL * (unsigned __int8)v122 + 120)))
     || *(_BYTE *)(v70 + 259 * v71 + 2564)) )
  {
LABEL_21:
    v38 = *(_QWORD *)a1;
    v39 = 1LL << ((unsigned __int8)v113 - 1);
    LODWORD(v127) = v111;
    if ( v111 > 0x40 )
    {
      v104 = v38;
      v109 = v33;
      sub_16A4EF0((__int64)&v126, 0, 0);
      v33 = v109;
      v38 = v104;
      v39 = 1LL << ((unsigned __int8)v113 - 1);
      if ( (unsigned int)v127 > 0x40 )
      {
        *(_QWORD *)(v126 + 8LL * ((v113 - 1) >> 6)) |= 1LL << ((unsigned __int8)v113 - 1);
        goto LABEL_24;
      }
    }
    else
    {
      v126 = 0;
    }
    v126 |= v39;
LABEL_24:
    v108 = v33;
    v40 = sub_1D1F940(v38, v12.m128i_i64[0], v12.m128i_i64[1], (__int64)&v126, 0);
    v41 = v108;
    if ( (unsigned int)v127 > 0x40 && v126 )
    {
      v103 = v40;
      j_j___libc_free_0_0(v126);
      v40 = v103;
      v41 = v108;
    }
    if ( v40 )
    {
      v42 = v124;
      v43 = *(__int64 **)a1;
      if ( (_BYTE)v124 )
      {
        if ( (unsigned __int8)(v124 - 14) <= 0x5Fu )
        {
          switch ( (char)v124 )
          {
            case 24:
            case 25:
            case 26:
            case 27:
            case 28:
            case 29:
            case 30:
            case 31:
            case 32:
            case 62:
            case 63:
            case 64:
            case 65:
            case 66:
            case 67:
              v42 = 3;
              break;
            case 33:
            case 34:
            case 35:
            case 36:
            case 37:
            case 38:
            case 39:
            case 40:
            case 68:
            case 69:
            case 70:
            case 71:
            case 72:
            case 73:
              v42 = 4;
              break;
            case 41:
            case 42:
            case 43:
            case 44:
            case 45:
            case 46:
            case 47:
            case 48:
            case 74:
            case 75:
            case 76:
            case 77:
            case 78:
            case 79:
              v42 = 5;
              break;
            case 49:
            case 50:
            case 51:
            case 52:
            case 53:
            case 54:
            case 80:
            case 81:
            case 82:
            case 83:
            case 84:
            case 85:
              v42 = 6;
              break;
            case 55:
              v42 = 7;
              break;
            case 86:
            case 87:
            case 88:
            case 98:
            case 99:
            case 100:
              v42 = 8;
              break;
            case 89:
            case 90:
            case 91:
            case 92:
            case 93:
            case 101:
            case 102:
            case 103:
            case 104:
            case 105:
              v42 = 9;
              break;
            case 94:
            case 95:
            case 96:
            case 97:
            case 106:
            case 107:
            case 108:
            case 109:
              v42 = 10;
              break;
            default:
              v42 = 2;
              break;
          }
          goto LABEL_31;
        }
      }
      else
      {
        v66 = sub_1F58D20((__int64)&v124);
        v42 = 0;
        if ( v66 )
        {
          v42 = sub_1F596B0((__int64)&v124);
          v41 = v67;
          goto LABEL_31;
        }
      }
      v41 = v125;
LABEL_31:
      v44 = *(_QWORD *)(a2 + 72);
      v45 = v42;
      v126 = v44;
      if ( v44 )
      {
        v114 = v41;
        sub_1623A60((__int64)&v126, v44, 2);
        v41 = v114;
      }
      LODWORD(v127) = *((_DWORD *)v9 + 16);
      v46 = (__int64)sub_1D3BC50(
                       v43,
                       v12.m128i_i64[0],
                       v12.m128i_u64[1],
                       (__int64)&v126,
                       v45,
                       v41,
                       v12,
                       *(double *)&v16,
                       a5);
LABEL_34:
      v28 = v126;
      v9 = (_QWORD *)v46;
      if ( !v126 )
        return (__int64)v9;
      goto LABEL_8;
    }
    if ( (unsigned __int8)sub_1FB1D70(a1, a2, 0) )
      return (__int64)v9;
    v69 = sub_1F84730((_QWORD *)a1, a2, *(double *)v12.m128i_i64, (__m128i)v16, a5);
    if ( v69 )
      return v69;
    if ( *(_WORD *)(v13 + 24) == 124 )
    {
      v78 = *(__int64 **)(v13 + 32);
      v79 = v78[5];
      v80 = *(unsigned __int16 *)(v79 + 24);
      if ( v80 != 10 && v80 != 32 )
        goto LABEL_93;
      v81 = *(_QWORD *)(v79 + 88);
      v82 = *(_QWORD **)(v81 + 24);
      if ( *(_DWORD *)(v81 + 32) > 0x40u )
        v82 = (_QWORD *)*v82;
      if ( (unsigned __int64)v82 + v113 > v111 )
        goto LABEL_93;
      v83 = (unsigned int)sub_1D23330(*(_QWORD *)a1, *v78, v78[1], 0);
      v84 = *(_QWORD *)(v79 + 88);
      if ( *(_DWORD *)(v84 + 32) <= 0x40u )
        v85 = *(_QWORD *)(v84 + 24);
      else
        v85 = **(_QWORD **)(v84 + 24);
      if ( v111 - (unsigned __int64)v113 - v85 < v83 )
      {
        v86 = *(_QWORD *)(v13 + 32);
        v87 = *(__int64 **)a1;
        v126 = *(_QWORD *)(a2 + 72);
        if ( v126 )
          sub_1F6CA20(&v126);
        LODWORD(v127) = *(_DWORD *)(a2 + 64);
        v88 = sub_1D332F0(
                v87,
                123,
                (__int64)&v126,
                (unsigned int)v122,
                v123,
                0,
                *(double *)v12.m128i_i64,
                *(double *)&v16,
                a5,
                *(_QWORD *)v86,
                *(_QWORD *)(v86 + 8),
                *(_OWORD *)(v86 + 40));
        goto LABEL_118;
      }
    }
    if ( *(_WORD *)(v13 + 24) != 185 )
      goto LABEL_93;
    v72 = *(_BYTE *)(v13 + 27);
    if ( ((v72 >> 2) & 3) != 1 )
    {
LABEL_97:
      if ( ((v72 ^ 0xC) & 0xC) != 0 || (*(_WORD *)(v13 + 26) & 0x380) != 0 )
        return 0;
      if ( sub_1D18C00(v13, 1, v112) && *(_BYTE *)(v13 + 88) == (_BYTE)v124 )
      {
        if ( (_BYTE)v124 )
        {
          if ( !*(_BYTE *)(a1 + 24) && (*(_BYTE *)(v13 + 26) & 8) == 0
            || (_BYTE)v122
            && (*(_BYTE *)(*(_QWORD *)(a1 + 8) + 2 * ((unsigned __int8)v124 + 115LL * (unsigned __int8)v122 + 16104) + 1)
              & 0xF) == 0 )
          {
            goto LABEL_104;
          }
        }
        else if ( *(_QWORD *)(v13 + 96) == v125 && !*(_BYTE *)(a1 + 24) && (*(_BYTE *)(v13 + 26) & 8) == 0 )
        {
LABEL_104:
          v73 = *(_QWORD *)(v13 + 104);
          v74 = *(_QWORD *)(v13 + 32);
          v75 = *(__int64 **)a1;
          v126 = *(_QWORD *)(a2 + 72);
          if ( v126 )
          {
            v115 = v74;
            v118 = v75;
            sub_1F6CA20(&v126);
            v74 = v115;
            v75 = v118;
          }
          LODWORD(v127) = *(_DWORD *)(a2 + 64);
          v76 = sub_1D2B590(
                  v75,
                  2,
                  (__int64)&v126,
                  v122,
                  (__int64)v123,
                  v73,
                  *(_OWORD *)v74,
                  *(_QWORD *)(v74 + 40),
                  *(_QWORD *)(v74 + 48),
                  v124,
                  v125);
          v119 = v77;
          sub_17CD270(&v126);
          v126 = v76;
          v127 = v119;
          sub_1F994A0(a1, a2, &v126, 1, 1);
          sub_1F9A400(a1, v13, v76, v119, v76, 1, 1);
          return (__int64)v9;
        }
      }
LABEL_93:
      if ( v113 <= 0x10 && *(_WORD *)(v13 + 24) == 119 )
      {
        v96 = sub_1F7E190(
                a1,
                v13,
                **(_QWORD **)(v13 + 32),
                *(_QWORD *)(*(_QWORD *)(v13 + 32) + 40LL),
                0,
                *(double *)v12.m128i_i64,
                *(double *)&v16,
                a5);
        v98 = v97;
        if ( v96 )
        {
          v99 = *(__int64 **)a1;
          v126 = *(_QWORD *)(a2 + 72);
          if ( v126 )
            sub_1F6CA20(&v126);
          LODWORD(v127) = *(_DWORD *)(a2 + 64);
          v88 = sub_1D332F0(
                  v99,
                  148,
                  (__int64)&v126,
                  (unsigned int)v122,
                  v123,
                  0,
                  *(double *)v12.m128i_i64,
                  *(double *)&v16,
                  a5,
                  (__int64)v96,
                  v98,
                  v16);
LABEL_118:
          v9 = v88;
          sub_17CD270(&v126);
          return (__int64)v9;
        }
      }
      return 0;
    }
    if ( (*(_WORD *)(v13 + 26) & 0x380) != 0 )
      return 0;
    v89 = (unsigned __int8)v124;
    if ( *(_BYTE *)(v13 + 88) != (_BYTE)v124 )
      return 0;
    if ( (_BYTE)v124 )
    {
      if ( *(_BYTE *)(a1 + 24) )
        goto LABEL_125;
    }
    else if ( *(_QWORD *)(v13 + 96) != v125 || *(_BYTE *)(a1 + 24) )
    {
      return 0;
    }
    if ( (*(_BYTE *)(v13 + 26) & 8) == 0 )
    {
      if ( sub_1D18C00(v13, 1, v112) )
        goto LABEL_128;
      v90 = (unsigned __int8)v122;
      v89 = (unsigned __int8)v124;
      if ( !(_BYTE)v122 || !(_BYTE)v124 )
      {
LABEL_141:
        if ( *(_WORD *)(v13 + 24) != 185 )
          goto LABEL_93;
        v72 = *(_BYTE *)(v13 + 27);
        goto LABEL_97;
      }
LABEL_127:
      if ( (*(_BYTE *)(*(_QWORD *)(a1 + 8) + 2 * (v89 + 115 * v90 + 16104) + 1) & 0xF) == 0 )
      {
LABEL_128:
        v91 = *(_QWORD *)(v13 + 104);
        v92 = *(_QWORD *)(v13 + 32);
        v93 = *(__int64 **)a1;
        v126 = *(_QWORD *)(a2 + 72);
        if ( v126 )
        {
          v116 = v92;
          v120 = v93;
          sub_1F6CA20(&v126);
          v92 = v116;
          v93 = v120;
        }
        LODWORD(v127) = *(_DWORD *)(a2 + 64);
        v94 = sub_1D2B590(
                v93,
                2,
                (__int64)&v126,
                v122,
                (__int64)v123,
                v91,
                *(_OWORD *)v92,
                *(_QWORD *)(v92 + 40),
                *(_QWORD *)(v92 + 48),
                v124,
                v125);
        v121 = v95;
        sub_17CD270(&v126);
        v126 = v94;
        v127 = v121;
        sub_1F994A0(a1, a2, &v126, 1, 1);
        sub_1F9A400(a1, v13, v94, v121, v94, 1, 1);
        sub_1F81BC0(a1, v94);
        return (__int64)v9;
      }
      goto LABEL_141;
    }
LABEL_125:
    v90 = (unsigned __int8)v122;
    if ( !(_BYTE)v122 || !(_BYTE)v124 )
      return 0;
    goto LABEL_127;
  }
LABEL_50:
  v55 = *(_QWORD *)(a2 + 72);
  v56 = *(__int64 **)a1;
  v126 = v55;
  if ( v55 )
    sub_1623A60((__int64)&v126, v55, 2);
  LODWORD(v127) = *((_DWORD *)v9 + 16);
  v27 = sub_1D332F0(
          v56,
          142,
          (__int64)&v126,
          (unsigned int)v122,
          v123,
          0,
          *(double *)v12.m128i_i64,
          *(double *)&v16,
          a5,
          v102.m128i_i64[0],
          v102.m128i_u64[1],
          v16);
LABEL_7:
  v28 = v126;
  v9 = v27;
  if ( v126 )
LABEL_8:
    sub_161E7C0((__int64)&v126, v28);
  return (__int64)v9;
}
