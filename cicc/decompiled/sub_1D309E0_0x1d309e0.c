// Function: sub_1D309E0
// Address: 0x1d309e0
//
__int64 __fastcall sub_1D309E0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        const void **a5,
        __int64 a6,
        double a7,
        double a8,
        double a9,
        __int128 a10)
{
  unsigned int v10; // r14d
  int v12; // r12d
  __int64 v13; // rdx
  unsigned __int16 v14; // ax
  __int64 v15; // rcx
  __int64 v16; // rcx
  int v17; // r15d
  __int64 v18; // r14
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r9
  __int64 v23; // rax
  int v24; // edx
  _QWORD *v25; // rax
  __int16 v26; // si
  int v27; // eax
  __int64 v28; // rsi
  int v29; // ebx
  __int64 v30; // r13
  unsigned __int8 *v31; // rsi
  _BOOL4 v32; // ebx
  _BOOL4 v33; // r14d
  unsigned int v34; // edx
  __int64 v35; // r8
  __int64 v36; // rax
  __int64 v37; // rdi
  _QWORD *v38; // rax
  _BOOL4 v39; // ebx
  unsigned int v40; // eax
  _BOOL4 v41; // r14d
  unsigned int v43; // esi
  _BOOL4 v46; // r14d
  unsigned int v47; // eax
  _BOOL4 v48; // ebx
  unsigned __int64 v49; // rcx
  unsigned int v50; // edx
  unsigned __int64 v51; // rax
  unsigned int v52; // eax
  __int64 v53; // r8
  void *v54; // rax
  void *v55; // rbx
  __int64 v56; // r8
  bool v57; // dl
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // rdx
  unsigned int v61; // eax
  unsigned __int8 *v62; // rdx
  unsigned int v63; // edx
  __int64 *v64; // r8
  void *v65; // rax
  void *v66; // r14
  _BOOL4 v67; // ebx
  unsigned int v68; // edi
  __int64 v69; // rsi
  _BOOL4 v70; // ebx
  _BOOL4 v71; // r14d
  __int64 v72; // rax
  __int64 v73; // rax
  void *v74; // r14
  __int16 *v75; // rsi
  __int16 *v76; // rax
  __int64 v77; // rax
  __int64 v78; // rsi
  __int64 v79; // rax
  int v80; // edx
  _QWORD *v81; // rdx
  unsigned __int8 *v82; // rdx
  __int64 v83; // rax
  __int64 v84; // rax
  char v85; // dl
  __int64 v86; // rax
  int v87; // eax
  __int64 v88; // rdx
  __m128i v89; // rax
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r9
  char v93; // r14
  unsigned int *v94; // rbx
  char v95; // r15
  __int64 v96; // rdx
  __int64 v97; // rax
  char v98; // di
  __int64 v99; // rax
  __int64 v100; // rax
  __int64 *v101; // rax
  __int64 v102; // r15
  __int64 v103; // rsi
  __int64 v104; // rax
  __int64 *v105; // rax
  __int32 v106; // ecx
  int v107; // edx
  __int32 v108; // r8d
  int v109; // esi
  __int64 v111; // rax
  __int64 v112; // rax
  __int64 v113; // rax
  int v114; // edx
  __int64 v115; // rsi
  int v116; // ecx
  unsigned __int8 *v117; // rsi
  unsigned int v118; // eax
  unsigned int v119; // eax
  __int64 v120; // rax
  __int64 v121; // rdx
  __int64 v122; // rcx
  __int64 v123; // r8
  __int64 v124; // r9
  unsigned int v125; // eax
  __int64 v126; // rax
  __int64 *v127; // rax
  __int64 v128; // rdx
  __int64 v129; // rcx
  __int64 v130; // r8
  __int64 v131; // r9
  unsigned int v132; // r14d
  unsigned int v133; // eax
  __int64 v134; // [rsp-10h] [rbp-140h]
  __int64 v135; // [rsp-8h] [rbp-138h]
  int v136; // [rsp+8h] [rbp-128h]
  void *v137; // [rsp+10h] [rbp-120h]
  __int64 v138; // [rsp+10h] [rbp-120h]
  bool v139; // [rsp+18h] [rbp-118h]
  __int64 *v140; // [rsp+18h] [rbp-118h]
  __int64 v141; // [rsp+20h] [rbp-110h]
  int v142; // [rsp+20h] [rbp-110h]
  int v143; // [rsp+20h] [rbp-110h]
  unsigned __int16 v144; // [rsp+28h] [rbp-108h]
  _QWORD *v145; // [rsp+28h] [rbp-108h]
  void *v146; // [rsp+28h] [rbp-108h]
  __int16 v147; // [rsp+28h] [rbp-108h]
  __int64 v148; // [rsp+28h] [rbp-108h]
  unsigned int v149; // [rsp+28h] [rbp-108h]
  __m128i v150; // [rsp+30h] [rbp-100h] BYREF
  __int64 *v151; // [rsp+48h] [rbp-E8h] BYREF
  _QWORD v152[2]; // [rsp+50h] [rbp-E0h] BYREF
  __int128 v153; // [rsp+60h] [rbp-D0h] BYREF
  __m128i v154; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v155; // [rsp+80h] [rbp-B0h] BYREF
  char v156; // [rsp+8Ah] [rbp-A6h]

  v10 = a2;
  v12 = (int)a1;
  v150.m128i_i64[0] = a4;
  v150.m128i_i64[1] = (__int64)a5;
  v13 = *(unsigned __int16 *)(a10 + 24);
  v144 = a6;
  v14 = v13;
  if ( (_WORD)v13 == 10 || (_DWORD)v13 == 32 )
  {
    a2 = *(_QWORD *)(a10 + 88);
    v16 = v10 - 121;
    a5 = (const void **)(a2 + 24);
    switch ( v10 )
    {
      case 0x79u:
        v68 = *(_DWORD *)(a2 + 32);
        v69 = *(_QWORD *)(a2 + 24);
        v70 = (*(_BYTE *)(a10 + 26) & 8) != 0;
        v71 = (__int16)v13 > 258;
        v72 = 1LL << ((unsigned __int8)v68 - 1);
        if ( v68 > 0x40 )
        {
          if ( (*(_QWORD *)(v69 + 8LL * ((v68 - 1) >> 6)) & v72) != 0 )
          {
            v154.m128i_i32[2] = v68;
            sub_16A4FD0((__int64)&v154, a5);
            LOBYTE(v68) = v154.m128i_i8[8];
            if ( v154.m128i_i32[2] > 0x40u )
            {
              sub_16A8F40(v154.m128i_i64);
              goto LABEL_101;
            }
            v69 = v154.m128i_i64[0];
LABEL_100:
            v154.m128i_i64[0] = ~v69 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v68);
LABEL_101:
            sub_16A7400((__int64)&v154);
            DWORD2(v153) = v154.m128i_i32[2];
            *(_QWORD *)&v153 = v154.m128i_i64[0];
            v73 = sub_1D38970(v12, (unsigned int)&v153, a3, v150.m128i_i32[0], v150.m128i_i32[2], v71, v70);
            goto LABEL_102;
          }
          DWORD2(v153) = v68;
          sub_16A4FD0((__int64)&v153, a5);
          v73 = sub_1D38970(v12, (unsigned int)&v153, a3, v150.m128i_i32[0], v150.m128i_i32[2], v71, v70);
        }
        else
        {
          if ( (v69 & v72) != 0 )
          {
            v154.m128i_i32[2] = v68;
            goto LABEL_100;
          }
          DWORD2(v153) = v68;
          *(_QWORD *)&v153 = v69;
          v73 = sub_1D38970(v12, (unsigned int)&v153, a3, v150.m128i_i32[0], v150.m128i_i32[2], v71, v70);
        }
LABEL_102:
        v18 = v73;
        if ( DWORD2(v153) > 0x40 )
        {
          v37 = v153;
          if ( (_QWORD)v153 )
LABEL_44:
            j_j___libc_free_0_0(v37);
        }
        return v18;
      case 0x7Fu:
        v147 = *(_WORD *)(a10 + 24);
        v67 = (*(_BYTE *)(a10 + 26) & 8) != 0;
        sub_16A85B0((__int64)&v154, a2 + 24);
        goto LABEL_96;
      case 0x80u:
      case 0x84u:
        v39 = (*(_BYTE *)(a10 + 26) & 8) != 0;
        v40 = *(_DWORD *)(a2 + 32);
        v41 = (__int16)v13 > 258;
        if ( v40 > 0x40 )
        {
          v40 = sub_16A58A0(a2 + 24);
        }
        else
        {
          _RDX = *(_QWORD *)(a2 + 24);
          v43 = 64;
          __asm { tzcnt   rcx, rdx }
          if ( _RDX )
            v43 = _RCX;
          if ( v40 > v43 )
            v40 = v43;
        }
        return sub_1D38BB0((_DWORD)a1, v40, a3, v150.m128i_i32[0], v150.m128i_i32[2], v41, v39);
      case 0x81u:
      case 0x85u:
        v46 = (*(_BYTE *)(a10 + 26) & 8) != 0;
        v47 = *(_DWORD *)(a2 + 32);
        v48 = (__int16)v13 > 258;
        if ( v47 > 0x40 )
        {
          v47 = sub_16A57B0(a2 + 24);
        }
        else
        {
          v49 = *(_QWORD *)(a2 + 24);
          v50 = v47 - 64;
          if ( v49 )
          {
            _BitScanReverse64(&v51, v49);
            v47 = v50 + (v51 ^ 0x3F);
          }
        }
        return sub_1D38BB0((_DWORD)a1, v47, a3, v150.m128i_i32[0], v150.m128i_i32[2], v48, v46);
      case 0x82u:
        v39 = (*(_BYTE *)(a10 + 26) & 8) != 0;
        v41 = (__int16)v13 > 258;
        if ( *(_DWORD *)(a2 + 32) > 0x40u )
          v40 = sub_16A5940(a2 + 24);
        else
          v40 = sub_39FAC40(*(_QWORD *)(a2 + 24));
        return sub_1D38BB0((_DWORD)a1, v40, a3, v150.m128i_i32[0], v150.m128i_i32[2], v41, v39);
      case 0x83u:
        v147 = *(_WORD *)(a10 + 24);
        v67 = (*(_BYTE *)(a10 + 26) & 8) != 0;
        sub_16A8270((__int64)&v154, (unsigned __int8 *)(a2 + 24));
LABEL_96:
        v36 = sub_1D38970((_DWORD)a1, (unsigned int)&v154, a3, v150.m128i_i32[0], v150.m128i_i32[2], v147 > 258, v67);
        goto LABEL_42;
      case 0x8Eu:
        v32 = (*(_BYTE *)(a10 + 26) & 8) != 0;
        v33 = (__int16)v13 > 258;
        if ( v150.m128i_i8[0] )
        {
          v63 = sub_1D13440(v150.m128i_i8[0]);
        }
        else
        {
          v119 = sub_1F58D40(&v150, a2, v13, v16, a5, a6);
          v64 = (__int64 *)(a2 + 24);
          v63 = v119;
        }
        sub_16A5D70((__int64)&v154, v64, v63);
        goto LABEL_41;
      case 0x8Fu:
      case 0x90u:
      case 0x91u:
        v32 = (*(_BYTE *)(a10 + 26) & 8) != 0;
        v33 = (__int16)v13 > 258;
        if ( v150.m128i_i8[0] )
        {
          v34 = sub_1D13440(v150.m128i_i8[0]);
        }
        else
        {
          v118 = sub_1F58D40(&v150, a2, v13, v16, a5, a6);
          v35 = a2 + 24;
          v34 = v118;
        }
        sub_16A5D10((__int64)&v154, v35, v34);
LABEL_41:
        v36 = sub_1D38970((_DWORD)a1, (unsigned int)&v154, a3, v150.m128i_i32[0], v150.m128i_i32[2], v33, v32);
LABEL_42:
        v18 = v36;
        if ( v154.m128i_i32[2] > 0x40u )
        {
          v37 = v154.m128i_i64[0];
          if ( v154.m128i_i64[0] )
            goto LABEL_44;
        }
        return v18;
      case 0x92u:
      case 0x93u:
        if ( v150.m128i_i8[0] )
        {
          v52 = sub_1D13440(v150.m128i_i8[0]);
        }
        else
        {
          v52 = sub_1F58D40(&v150, a2, v13, v16, a5, a6);
          v53 = a2 + 24;
        }
        DWORD2(v153) = v52;
        if ( v52 > 0x40 )
        {
          v148 = v53;
          sub_16A4EF0((__int64)&v153, 0, 0);
          v53 = v148;
        }
        else
        {
          *(_QWORD *)&v153 = 0;
        }
        v141 = v53;
        v146 = sub_1D15FA0(v150.m128i_u32[0], v150.m128i_i64[1]);
        v54 = sub_16982C0();
        v55 = v54;
        if ( v146 == v54 )
          sub_169D060(&v154.m128i_i64[1], (__int64)v54, (__int64 *)&v153);
        else
          sub_169D050((__int64)&v154.m128i_i64[1], v146, (__int64 *)&v153);
        v56 = v141;
        if ( DWORD2(v153) > 0x40 && (_QWORD)v153 )
        {
          j_j___libc_free_0_0(v153);
          v56 = v141;
        }
        v57 = v10 == 146;
        if ( (void *)v154.m128i_i64[1] == v55 )
          sub_169E6C0(&v154.m128i_i64[1], v56, v57, 0);
        else
          sub_169A290((__int64)&v154.m128i_i64[1], v56, v57, 0);
        goto LABEL_69;
      case 0x9Eu:
        v140 = (__int64 *)(a2 + 24);
        switch ( v150.m128i_i8[0] )
        {
          case 8:
            if ( **(_BYTE **)(a10 + 40) != 4 )
              goto LABEL_3;
            v65 = sub_1698260();
            break;
          case 9:
            if ( **(_BYTE **)(a10 + 40) != 5 )
              goto LABEL_3;
            v65 = sub_1698270();
            break;
          case 0xA:
            if ( **(_BYTE **)(a10 + 40) != 6 )
              goto LABEL_3;
            v65 = sub_1698280();
            break;
          default:
            if ( v150.m128i_i8[0] != 12 || **(_BYTE **)(a10 + 40) != 7 )
              goto LABEL_3;
            v65 = sub_1698290();
            break;
        }
        v66 = v65;
        if ( v65 == sub_16982C0() )
          sub_169D060(&v154.m128i_i64[1], (__int64)v66, v140);
        else
          sub_169D050((__int64)&v154.m128i_i64[1], v66, v140);
LABEL_69:
        v58 = v150.m128i_u32[0];
        v59 = v150.m128i_i64[1];
        v60 = a3;
LABEL_70:
        v18 = sub_1D36490(a1, &v154, v60, v58, v59, 0);
        goto LABEL_71;
      case 0xA0u:
        if ( *(_DWORD *)(a2 + 32) == 16 )
        {
          DWORD2(v153) = 16;
          *(_QWORD *)&v153 = *(_QWORD *)(a2 + 24);
        }
        else
        {
          sub_16A5A50((__int64)&v153, (__int64 *)(a2 + 24), 0x10u);
        }
        v74 = sub_1698260();
        if ( v74 == sub_16982C0() )
          sub_169D060(&v154.m128i_i64[1], (__int64)v74, (__int64 *)&v153);
        else
          sub_169D050((__int64)&v154.m128i_i64[1], v74, (__int64 *)&v153);
        if ( DWORD2(v153) > 0x40 && (_QWORD)v153 )
          j_j___libc_free_0_0(v153);
        v75 = (__int16 *)sub_1D15FA0(v150.m128i_u32[0], v150.m128i_i64[1]);
LABEL_114:
        sub_16A3360((__int64)&v154, v75, 0, (bool *)&v153);
        v60 = a3;
        v58 = v150.m128i_u32[0];
        v59 = v150.m128i_i64[1];
        goto LABEL_70;
      default:
        break;
    }
  }
LABEL_3:
  v15 = (unsigned __int16)a6;
  LOWORD(v15) = (unsigned __int16)a6 >> 6;
  v139 = (a6 & 0x40) != 0;
  if ( (_DWORD)v13 == 11 || (_DWORD)v13 == 33 )
  {
    v137 = sub_16982C0();
    a2 = *(_QWORD *)(a10 + 88) + 32LL;
    if ( *(void **)a2 == v137 )
      sub_169C6E0(&v154.m128i_i64[1], a2);
    else
      sub_16986C0(&v154.m128i_i64[1], (__int64 *)a2);
    switch ( v10 )
    {
      case 0x98u:
      case 0x99u:
        if ( v150.m128i_i8[0] )
          v61 = sub_1D13440(v150.m128i_i8[0]);
        else
          v61 = sub_1F58D40(&v150, a2, v20, v21, v137, v22);
        DWORD2(v153) = v61;
        if ( v61 > 0x40 )
          sub_16A4EF0((__int64)&v153, 0, 0);
        else
          *(_QWORD *)&v153 = 0;
        a2 = (__int64)&v153;
        BYTE12(v153) = v10 == 153;
        if ( (unsigned int)sub_169E1A0((__int64)&v154, (__int64)&v153, 3u, v152) != 1 )
          goto LABEL_77;
        sub_135E100((__int64 *)&v153);
        goto LABEL_28;
      case 0x9Du:
        v75 = (__int16 *)sub_1D15FA0(v150.m128i_u32[0], v150.m128i_i64[1]);
        goto LABEL_114;
      case 0x9Eu:
        if ( v150.m128i_i8[0] == 4 )
        {
          if ( **(_BYTE **)(a10 + 40) == 8 )
          {
            if ( v137 == (void *)v154.m128i_i64[1] )
              sub_169D930((__int64)&v153, (__int64)&v154.m128i_i64[1]);
            else
              sub_169D7E0((__int64)&v153, &v154.m128i_i64[1]);
            LOWORD(v77) = v153;
            if ( DWORD2(v153) > 0x40 )
              v77 = *(_QWORD *)v153;
            LODWORD(v78) = (unsigned __int16)v77;
LABEL_130:
            v18 = sub_1D38BB0((_DWORD)a1, v78, a3, v150.m128i_i32[0], v150.m128i_i32[2], 0, 0);
            sub_135E100((__int64 *)&v153);
LABEL_71:
            sub_127D120(&v154.m128i_i64[1]);
            return v18;
          }
        }
        else if ( v150.m128i_i8[0] == 5 )
        {
          if ( **(_BYTE **)(a10 + 40) == 9 )
          {
            if ( v137 == (void *)v154.m128i_i64[1] )
              sub_169D930((__int64)&v153, (__int64)&v154.m128i_i64[1]);
            else
              sub_169D7E0((__int64)&v153, &v154.m128i_i64[1]);
            LODWORD(v120) = v153;
            if ( DWORD2(v153) > 0x40 )
              v120 = *(_QWORD *)v153;
            LODWORD(v78) = v120;
            goto LABEL_130;
          }
        }
        else if ( v150.m128i_i8[0] == 6 && **(_BYTE **)(a10 + 40) == 10 )
        {
          if ( v137 == (void *)v154.m128i_i64[1] )
            sub_169D930((__int64)&v153, (__int64)&v154.m128i_i64[1]);
          else
            sub_169D7E0((__int64)&v153, &v154.m128i_i64[1]);
          LODWORD(v78) = v153;
          if ( DWORD2(v153) > 0x40 )
            v78 = *(_QWORD *)v153;
          goto LABEL_130;
        }
LABEL_28:
        sub_127D120(&v154.m128i_i64[1]);
        v14 = *(_WORD *)(a10 + 24);
        break;
      case 0xA1u:
        v76 = (__int16 *)sub_1698260();
        sub_16A3360((__int64)&v154, v76, 0, (bool *)v152);
        if ( v137 == (void *)v154.m128i_i64[1] )
          sub_169D930((__int64)&v153, (__int64)&v154.m128i_i64[1]);
        else
          sub_169D7E0((__int64)&v153, &v154.m128i_i64[1]);
LABEL_77:
        v18 = sub_1D38970((_DWORD)a1, (unsigned int)&v153, a3, v150.m128i_i32[0], v150.m128i_i32[2], 0, 0);
        if ( DWORD2(v153) > 0x40 && (_QWORD)v153 )
          j_j___libc_free_0_0(v153);
        goto LABEL_71;
      case 0xA2u:
        if ( v137 != (void *)v154.m128i_i64[1] )
          goto LABEL_120;
        goto LABEL_230;
      case 0xA3u:
        if ( v137 == (void *)v154.m128i_i64[1] )
        {
          if ( (*(_BYTE *)(v155 + 26) & 8) != 0 )
LABEL_230:
            sub_169C8D0((__int64)&v154.m128i_i64[1], a7, a8, a9);
        }
        else if ( (v156 & 8) != 0 )
        {
LABEL_120:
          sub_1699490((__int64)&v154.m128i_i64[1]);
        }
        goto LABEL_118;
      case 0xAEu:
        a2 = 1;
        if ( v137 == (void *)v154.m128i_i64[1] )
          goto LABEL_132;
        goto LABEL_26;
      case 0xAFu:
        a2 = 3;
        if ( v137 == (void *)v154.m128i_i64[1] )
          goto LABEL_132;
        goto LABEL_26;
      case 0xB3u:
        a2 = 2;
        if ( v137 == (void *)v154.m128i_i64[1] )
LABEL_132:
          v27 = sub_169EBA0(&v154.m128i_i64[1], a2);
        else
LABEL_26:
          v27 = sub_169D440((__int64)&v154.m128i_i64[1], a2);
        if ( (v27 & 0xFFFFFFEF) != 0 )
          goto LABEL_28;
LABEL_118:
        v18 = sub_1D36490(a1, &v154, a3, v150.m128i_u32[0], v150.m128i_i64[1], 0);
        goto LABEL_71;
      default:
        goto LABEL_28;
    }
  }
  if ( v14 == 104 )
  {
    if ( (unsigned __int8)sub_1D23510(a10) )
    {
      switch ( v10 )
      {
        case 0x79u:
        case 0x7Fu:
        case 0x80u:
        case 0x81u:
        case 0x82u:
        case 0x83u:
        case 0x84u:
        case 0x85u:
        case 0x8Eu:
        case 0x8Fu:
        case 0x90u:
        case 0x91u:
        case 0x92u:
        case 0x93u:
        case 0x98u:
        case 0x99u:
        case 0x9Du:
        case 0xA2u:
        case 0xA3u:
        case 0xAEu:
        case 0xAFu:
        case 0xB3u:
          a2 = v10;
          v154 = (__m128i)a10;
          v38 = (_QWORD *)sub_1D39800((_DWORD)a1, v10, a3, v150.m128i_i32[0], v150.m128i_i32[2], 0, (__int64)&v154, 1);
          if ( v38 )
            goto LABEL_46;
          break;
        default:
          break;
      }
    }
    v14 = *(_WORD *)(a10 + 24);
  }
  v17 = v14;
  if ( v10 > 0xA3 )
    goto LABEL_21;
  if ( v10 <= 0x6A )
  {
    if ( v10 == 2 || v10 == 51 )
      return a10;
LABEL_21:
    v23 = sub_1D29190((__int64)a1, v150.m128i_u32[0], v150.m128i_i64[1], v15, (__int64)a5, a6);
    v153 = a10;
    v138 = v23;
    v136 = v24;
    if ( v150.m128i_i8[0] == 111 )
    {
      v28 = *(_QWORD *)a3;
      v29 = *(_DWORD *)(a3 + 8);
      v154.m128i_i64[0] = v28;
      if ( v28 )
        sub_1623A60((__int64)&v154, v28, 2);
      v30 = a1[26];
      if ( v30 )
        a1[26] = *(_QWORD *)v30;
      else
        v30 = sub_145CBF0(a1 + 27, 112, 8);
      *(_QWORD *)v30 = 0;
      *(_QWORD *)(v30 + 8) = 0;
      *(_QWORD *)(v30 + 40) = v138;
      *(_QWORD *)(v30 + 16) = 0;
      *(_WORD *)(v30 + 24) = v10;
      *(_DWORD *)(v30 + 28) = -1;
      *(_QWORD *)(v30 + 32) = 0;
      *(_QWORD *)(v30 + 48) = 0;
      *(_DWORD *)(v30 + 56) = 0;
      *(_DWORD *)(v30 + 60) = v136;
      *(_DWORD *)(v30 + 64) = v29;
      v31 = (unsigned __int8 *)v154.m128i_i64[0];
      *(_QWORD *)(v30 + 72) = v154.m128i_i64[0];
      if ( v31 )
        sub_1623210((__int64)&v154, v31, v30 + 72);
      *(_WORD *)(v30 + 80) &= 0xF000u;
      *(_WORD *)(v30 + 26) = 0;
      sub_1D23B60((__int64)a1, v30, (__int64)&v153, 1);
    }
    else
    {
      v154.m128i_i64[1] = 0x2000000000LL;
      v154.m128i_i64[0] = (__int64)&v155;
      sub_16BD430((__int64)&v154, (unsigned __int16)v10);
      sub_16BD4C0((__int64)&v154, v138);
      sub_16BD4C0((__int64)&v154, v153);
      sub_16BD430((__int64)&v154, SDWORD2(v153));
      v151 = 0;
      v25 = sub_1D17920((__int64)a1, (__int64)&v154, a3, (__int64 *)&v151);
      if ( v25 )
      {
        v26 = v144;
        v145 = v25;
        sub_1D19330((__int64)v25, v26);
        v18 = (__int64)v145;
        if ( (__int64 *)v154.m128i_i64[0] != &v155 )
          _libc_free(v154.m128i_u64[0]);
        return v18;
      }
      v115 = *(_QWORD *)a3;
      v116 = *(_DWORD *)(a3 + 8);
      v152[0] = v115;
      if ( v115 )
      {
        v142 = v116;
        sub_1623A60((__int64)v152, v115, 2);
        v116 = v142;
      }
      v30 = a1[26];
      if ( v30 )
      {
        a1[26] = *(_QWORD *)v30;
      }
      else
      {
        v143 = v116;
        v126 = sub_145CBF0(a1 + 27, 112, 8);
        v116 = v143;
        v30 = v126;
      }
      *(_QWORD *)v30 = 0;
      *(_QWORD *)(v30 + 8) = 0;
      *(_QWORD *)(v30 + 40) = v138;
      *(_QWORD *)(v30 + 16) = 0;
      *(_WORD *)(v30 + 24) = v10;
      *(_DWORD *)(v30 + 28) = -1;
      *(_QWORD *)(v30 + 32) = 0;
      *(_QWORD *)(v30 + 48) = 0;
      *(_DWORD *)(v30 + 56) = 0;
      *(_DWORD *)(v30 + 60) = v136;
      *(_DWORD *)(v30 + 64) = v116;
      v117 = (unsigned __int8 *)v152[0];
      *(_QWORD *)(v30 + 72) = v152[0];
      if ( v117 )
        sub_1623210((__int64)v152, v117, v30 + 72);
      *(_WORD *)(v30 + 26) = 0;
      *(_WORD *)(v30 + 80) = (v139 << 6) | v144 & 0xFFBF;
      sub_1D23B60((__int64)a1, v30, (__int64)&v153, 1);
      sub_16BDA20(a1 + 40, (__int64 *)v30, v151);
      if ( (__int64 *)v154.m128i_i64[0] != &v155 )
        _libc_free(v154.m128i_u64[0]);
    }
    v18 = v30;
    sub_1D172A0((__int64)a1, v30);
    return v18;
  }
  switch ( v10 )
  {
    case 0x6Bu:
      return a10;
    case 0x6Fu:
      if ( v14 == 48 )
        goto LABEL_84;
      if ( v14 == 106 )
      {
        v79 = *(_QWORD *)(a10 + 32);
        v15 = *(_QWORD *)(v79 + 40);
        v80 = *(unsigned __int16 *)(v15 + 24);
        if ( v80 == 32 || v80 == 10 )
        {
          v15 = *(_QWORD *)(v15 + 88);
          v81 = *(_QWORD **)(v15 + 24);
          if ( *(_DWORD *)(v15 + 32) > 0x40u )
            v81 = (_QWORD *)*v81;
          if ( !v81 )
          {
            v82 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v79 + 40LL) + 16LL * *(unsigned int *)(v79 + 8));
            v15 = *v82;
            if ( (_BYTE)v15 == v150.m128i_i8[0] && (*((_QWORD *)v82 + 1) == v150.m128i_i64[1] || (_BYTE)v15) )
              return *(_QWORD *)v79;
          }
        }
      }
      goto LABEL_21;
    case 0x79u:
    case 0x7Fu:
    case 0x83u:
      goto LABEL_20;
    case 0x8Eu:
      v113 = *(_QWORD *)(a10 + 40) + 16LL * DWORD2(a10);
      v15 = *(_QWORD *)(v113 + 8);
      if ( *(_BYTE *)v113 == v150.m128i_i8[0] && (v15 == v150.m128i_i64[1] || *(_BYTE *)v113) )
        return a10;
      if ( (unsigned int)(v17 - 142) > 1 )
        goto LABEL_184;
      v18 = sub_1D309E0(
              (_DWORD)a1,
              v17,
              a3,
              v150.m128i_i32[0],
              v150.m128i_i32[2],
              0,
              **(_QWORD **)(a10 + 32),
              *(_QWORD *)(*(_QWORD *)(a10 + 32) + 8LL));
      sub_1D306C0((__int64)a1, a10, SDWORD2(a10), v18, v114, 0, 0, 1);
      return v18;
    case 0x8Fu:
      v112 = *(_QWORD *)(a10 + 40) + 16LL * DWORD2(a10);
      v15 = *(_QWORD *)(v112 + 8);
      if ( *(_BYTE *)v112 == v150.m128i_i8[0] && (v15 == v150.m128i_i64[1] || *(_BYTE *)v112) )
        return a10;
      if ( v17 == 143 )
        return sub_1D309E0(
                 (_DWORD)a1,
                 143,
                 a3,
                 v150.m128i_i32[0],
                 v150.m128i_i32[2],
                 0,
                 **(_QWORD **)(a10 + 32),
                 *(_QWORD *)(*(_QWORD *)(a10 + 32) + 8LL));
LABEL_184:
      if ( v17 != 48 )
        goto LABEL_21;
      return sub_1D38BB0((_DWORD)a1, 0, a3, v150.m128i_i32[0], v150.m128i_i32[2], 0, 0);
    case 0x90u:
      v15 = v150.m128i_i64[1];
      v100 = *(_QWORD *)(a10 + 40) + 16LL * DWORD2(a10);
      if ( *(_BYTE *)v100 == v150.m128i_i8[0] && (*(_QWORD *)(v100 + 8) == v150.m128i_i64[1] || v150.m128i_i8[0]) )
        return a10;
      if ( (unsigned int)(v17 - 142) <= 2 )
        return sub_1D309E0(
                 (_DWORD)a1,
                 v17,
                 a3,
                 v150.m128i_i32[0],
                 v150.m128i_i32[2],
                 0,
                 **(_QWORD **)(a10 + 32),
                 *(_QWORD *)(*(_QWORD *)(a10 + 32) + 8LL));
      if ( v17 == 48 )
        goto LABEL_84;
      if ( v17 != 145 )
        goto LABEL_21;
      v101 = *(__int64 **)(a10 + 32);
      v102 = *v101;
      v103 = v101[1];
      v104 = *(_QWORD *)(*v101 + 40) + 16LL * *((unsigned int *)v101 + 2);
      if ( *(_BYTE *)v104 != v150.m128i_i8[0] || *(_QWORD *)(v104 + 8) != v150.m128i_i64[1] && !v150.m128i_i8[0] )
        goto LABEL_21;
      v18 = v102;
      sub_1D306C0((__int64)a1, a10, SDWORD2(a10), v102, v103, 0, 0, 1);
      return v18;
    case 0x91u:
      v83 = *(_QWORD *)(a10 + 40) + 16LL * DWORD2(a10);
      v15 = *(_QWORD *)(v83 + 8);
      if ( *(_BYTE *)v83 == v150.m128i_i8[0] && (v15 == v150.m128i_i64[1] || *(_BYTE *)v83) )
        return a10;
      if ( v17 == 145 )
      {
        v127 = *(__int64 **)(a10 + 32);
        v135 = v127[1];
        v134 = *v127;
        goto LABEL_255;
      }
      if ( (unsigned int)(v17 - 142) <= 2 )
      {
        v84 = *(_QWORD *)(**(_QWORD **)(a10 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a10 + 32) + 8LL);
        v85 = *(_BYTE *)v84;
        v86 = *(_QWORD *)(v84 + 8);
        LOBYTE(v152[0]) = v85;
        v152[1] = v86;
        LOBYTE(v87) = sub_1D15870((char *)v152);
        LODWORD(v153) = v87;
        *((_QWORD *)&v153 + 1) = v88;
        v89.m128i_i8[0] = sub_1D15870(v150.m128i_i8);
        v154 = v89;
        v93 = v89.m128i_i8[0];
        if ( v89.m128i_i8[0] == (_BYTE)v153 )
        {
          if ( v89.m128i_i8[0] || v89.m128i_i64[1] == *((_QWORD *)&v153 + 1) )
            goto LABEL_153;
        }
        else if ( (_BYTE)v153 )
        {
          v149 = sub_1D13440(v153);
          goto LABEL_247;
        }
        v149 = sub_1F58D40(&v153, a2, v89.m128i_i64[1], v90, v91, v92);
LABEL_247:
        if ( v93 )
          v125 = sub_1D13440(v93);
        else
          v125 = sub_1F58D40(&v154, a2, v121, v122, v123, v124);
        if ( v125 > v149 )
          return sub_1D309E0(
                   (_DWORD)a1,
                   v17,
                   a3,
                   v150.m128i_i32[0],
                   v150.m128i_i32[2],
                   0,
                   **(_QWORD **)(a10 + 32),
                   *(_QWORD *)(*(_QWORD *)(a10 + 32) + 8LL));
LABEL_153:
        v94 = *(unsigned int **)(a10 + 32);
        v95 = v150.m128i_i8[0];
        v96 = *(_QWORD *)v94;
        v97 = *(_QWORD *)(*(_QWORD *)v94 + 40LL) + 16LL * v94[2];
        v98 = *(_BYTE *)v97;
        v99 = *(_QWORD *)(v97 + 8);
        v154 = _mm_load_si128(&v150);
        LOBYTE(v153) = v98;
        *((_QWORD *)&v153 + 1) = v99;
        if ( v98 == v150.m128i_i8[0] )
        {
          v96 = v154.m128i_i64[1];
          if ( v150.m128i_i8[0] || v99 == v154.m128i_i64[1] )
            return *(_QWORD *)v94;
        }
        else if ( v98 )
        {
          v132 = sub_1D13440(v98);
          goto LABEL_261;
        }
        v132 = sub_1F58D40(&v153, a2, v96, v90, v91, v92);
LABEL_261:
        if ( v95 )
          v133 = sub_1D13440(v95);
        else
          v133 = sub_1F58D40(&v154, a2, v128, v129, v130, v131);
        if ( v133 < v132 )
        {
          v135 = *((_QWORD *)v94 + 1);
          v134 = *(_QWORD *)v94;
LABEL_255:
          v106 = v150.m128i_i32[0];
          v107 = a3;
          v109 = 145;
          v108 = v150.m128i_i32[2];
          return sub_1D309E0(v12, v109, v107, v106, v108, 0, v134, v135);
        }
        return *(_QWORD *)v94;
      }
LABEL_20:
      if ( v17 != 48 )
        goto LABEL_21;
LABEL_84:
      v38 = sub_1D2B530(a1, v150.m128i_u32[0], v150.m128i_i64[1], v15, (__int64)a5, a6);
LABEL_46:
      v18 = (__int64)v38;
      break;
    case 0x9Du:
      v62 = (unsigned __int8 *)(*(_QWORD *)(a10 + 40) + 16LL * DWORD2(a10));
      v15 = *v62;
      if ( (_BYTE)v15 == v150.m128i_i8[0] && (*((_QWORD *)v62 + 1) == v150.m128i_i64[1] || (_BYTE)v15) )
        return a10;
      if ( v14 == 48 )
        goto LABEL_84;
      goto LABEL_21;
    case 0x9Eu:
      v111 = *(_QWORD *)(a10 + 40) + 16LL * DWORD2(a10);
      if ( v150.m128i_i8[0] == *(_BYTE *)v111 && (v150.m128i_i8[0] || v150.m128i_i64[1] == *(_QWORD *)(v111 + 8)) )
        return a10;
      if ( v17 != 158 )
        goto LABEL_20;
      return sub_1D309E0(
               (_DWORD)a1,
               158,
               a3,
               v150.m128i_i32[0],
               v150.m128i_i32[2],
               0,
               **(_QWORD **)(a10 + 32),
               *(_QWORD *)(*(_QWORD *)(a10 + 32) + 8LL));
    case 0xA2u:
      if ( ((*(_BYTE *)(*a1 + 792) & 2) != 0 || v139) && v14 == 77 )
        return sub_1D332F0(
                 (_DWORD)a1,
                 77,
                 a3,
                 v150.m128i_i32[0],
                 v150.m128i_i32[2],
                 v144,
                 *(_QWORD *)(*(_QWORD *)(a10 + 32) + 40LL),
                 *(_QWORD *)(*(_QWORD *)(a10 + 32) + 48LL),
                 *(_OWORD *)*(_QWORD *)(a10 + 32));
      if ( v14 != 162 )
        goto LABEL_21;
      v79 = *(_QWORD *)(a10 + 32);
      return *(_QWORD *)v79;
    case 0xA3u:
      if ( v14 != 162 )
        goto LABEL_21;
      v105 = *(__int64 **)(a10 + 32);
      v106 = v150.m128i_i32[0];
      v107 = a3;
      v108 = v150.m128i_i32[2];
      v109 = 163;
      v135 = v105[1];
      v134 = *v105;
      return sub_1D309E0(v12, v109, v107, v106, v108, 0, v134, v135);
    default:
      goto LABEL_21;
  }
  return v18;
}
