// Function: sub_128A450
// Address: 0x128a450
//
__int64 __fastcall sub_128A450(
        __int64 *a1,
        _QWORD *a2,
        unsigned __int8 a3,
        __int64 a4,
        unsigned __int8 a5,
        char a6,
        _DWORD *a7)
{
  __int64 v7; // r14
  __int64 v10; // r14
  _QWORD *v11; // r12
  unsigned __int8 v13; // dl
  char v14; // al
  bool v15; // cc
  __int64 *v16; // rbx
  __int64 v17; // rax
  unsigned int v19; // eax
  __int64 v20; // rax
  __int64 *v21; // r15
  __int64 v22; // rax
  char v23; // al
  __int64 v24; // rax
  int v25; // r13d
  __int64 v26; // rbx
  int v27; // eax
  __int64 v28; // rax
  int v29; // r13d
  __int64 *v30; // rbx
  __int64 v31; // rax
  __int64 v32; // r15
  _QWORD *v33; // rax
  __int64 v34; // rax
  int v35; // esi
  __int64 v36; // rdi
  __int64 *v37; // r12
  __int64 v38; // rax
  __int64 v39; // rcx
  __int64 v40; // rsi
  __int64 v41; // rsi
  __int64 v42; // rax
  __int64 v43; // r15
  _QWORD *v44; // rax
  __int64 v45; // rax
  int v46; // esi
  __int64 v47; // rdx
  unsigned int v48; // r12d
  __int64 v49; // rax
  __int64 *v50; // r12
  __int64 v51; // rax
  __int64 v52; // rcx
  __int64 v53; // rsi
  __int64 v54; // rsi
  char v55; // al
  int v56; // r10d
  const char *v57; // r9
  _BOOL4 v58; // r8d
  __int64 v59; // rdx
  int v60; // r8d
  unsigned int v61; // eax
  __int64 v62; // rcx
  char *v63; // rdi
  unsigned int v64; // r8d
  char *v65; // rdx
  unsigned int v66; // eax
  __int64 v67; // rcx
  char *v68; // r9
  __int64 v69; // rax
  unsigned __int64 v70; // r14
  _QWORD *v71; // rax
  _WORD *v72; // rdi
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rdi
  __int64 *v76; // r12
  __int64 v77; // rax
  __int64 v78; // rcx
  __int64 v79; // rsi
  __int64 v80; // rsi
  __int64 v81; // rax
  __int64 v82; // rdi
  __int64 *v83; // rbx
  __int64 v84; // rdi
  __int64 v85; // rcx
  __int64 v86; // rax
  __int64 v87; // rdx
  __int64 v88; // rdi
  __int64 v89; // rax
  __int64 *v90; // r12
  __int64 v91; // rax
  __int64 v92; // rcx
  __int64 v93; // r13
  __int64 v94; // rdx
  __int64 v95; // rcx
  __m128i *v96; // rax
  __m128i si128; // xmm0
  __int64 v98; // rdi
  __int64 v99; // rax
  __int64 v100; // rax
  unsigned __int64 v101; // rcx
  char *v102; // rax
  char *v103; // r9
  unsigned int v104; // edx
  unsigned int v105; // eax
  __int64 v106; // rsi
  __int64 v107; // rax
  _QWORD *v108; // [rsp-8h] [rbp-A8h]
  int v109; // [rsp+4h] [rbp-9Ch]
  char *v110; // [rsp+8h] [rbp-98h]
  __int64 v112; // [rsp+18h] [rbp-88h]
  __int64 v113; // [rsp+18h] [rbp-88h]
  __int64 v114; // [rsp+18h] [rbp-88h]
  __int64 v115; // [rsp+28h] [rbp-78h] BYREF
  _QWORD v116[2]; // [rsp+30h] [rbp-70h] BYREF
  char v117; // [rsp+40h] [rbp-60h]
  char v118; // [rsp+41h] [rbp-5Fh]
  _WORD *v119; // [rsp+50h] [rbp-50h] BYREF
  __int64 v120; // [rsp+58h] [rbp-48h]
  _WORD v121[32]; // [rsp+60h] [rbp-40h] BYREF

  v7 = 0;
  if ( a6 )
    return v7;
  v10 = *a2;
  v11 = a2;
  if ( a4 == sub_1643320(a1[2]) )
  {
    v23 = *(_BYTE *)(*a2 + 8LL);
    if ( (unsigned __int8)(v23 - 1) > 5u )
    {
      if ( (v23 & 0xFB) != 0xB )
        sub_127B550("unexpected type when converting to boolean!", a7, 1);
      if ( *((_BYTE *)a2 + 16) == 61 && (v93 = *(_QWORD *)*(a2 - 3), v93 == sub_1643320(*(_QWORD *)(*a1 + 40))) )
      {
        v7 = *(a2 - 3);
        if ( !a2[1] )
          sub_15F20C0(a2, a2, v94, v95);
      }
      else
      {
        v28 = sub_15A06D0(*a2);
        v15 = *((_BYTE *)a2 + 16) <= 0x10u;
        v118 = 1;
        v29 = v28;
        v117 = 3;
        v30 = (__int64 *)a1[1];
        v116[0] = "tobool";
        if ( v15 && *(_BYTE *)(v28 + 16) <= 0x10u )
          return sub_15A37B0(33, a2, v28, 0);
        v121[0] = 257;
        v31 = sub_1648A60(56, 2);
        v7 = v31;
        if ( v31 )
        {
          v32 = v31;
          v33 = (_QWORD *)*a2;
          if ( *(_BYTE *)(*a2 + 8LL) == 16 )
          {
            v112 = v33[4];
            v34 = sub_1643320(*v33);
            v35 = sub_16463B0(v34, v112);
          }
          else
          {
            v35 = sub_1643320(*v33);
          }
          sub_15FEC10(v7, v35, 51, 33, (_DWORD)v11, v29, (__int64)&v119, 0);
        }
        else
        {
          v32 = 0;
        }
        v36 = v30[1];
        if ( v36 )
        {
          v37 = (__int64 *)v30[2];
          sub_157E9D0(v36 + 40, v7);
          v38 = *(_QWORD *)(v7 + 24);
          v39 = *v37;
          *(_QWORD *)(v7 + 32) = v37;
          v39 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v7 + 24) = v39 | v38 & 7;
          *(_QWORD *)(v39 + 8) = v7 + 24;
          *v37 = *v37 & 7 | (v7 + 24);
        }
        sub_164B780(v32, v116);
        v40 = *v30;
        if ( *v30 )
          goto LABEL_62;
      }
    }
    else
    {
      v24 = sub_15A06D0(*a2);
      v15 = *((_BYTE *)a2 + 16) <= 0x10u;
      v118 = 1;
      v25 = v24;
      v117 = 3;
      v26 = a1[1];
      v116[0] = "tobool";
      if ( v15 && *(_BYTE *)(v24 + 16) <= 0x10u )
      {
        v7 = sub_15A37B0(14, a2, v24, 0);
      }
      else
      {
        v121[0] = 257;
        v42 = sub_1648A60(56, 2);
        v7 = v42;
        if ( v42 )
        {
          v43 = v42;
          v44 = (_QWORD *)*a2;
          if ( *(_BYTE *)(*a2 + 8LL) == 16 )
          {
            v113 = v44[4];
            v45 = sub_1643320(*v44);
            v46 = sub_16463B0(v45, v113);
          }
          else
          {
            v46 = sub_1643320(*v44);
          }
          sub_15FEC10(v7, v46, 52, 14, (_DWORD)v11, v25, (__int64)&v119, 0);
        }
        else
        {
          v43 = 0;
        }
        v47 = *(_QWORD *)(v26 + 32);
        v48 = *(_DWORD *)(v26 + 40);
        if ( v47 )
          sub_1625C10(v7, 3, v47);
        sub_15F2440(v7, v48);
        v49 = *(_QWORD *)(v26 + 8);
        if ( v49 )
        {
          v50 = *(__int64 **)(v26 + 16);
          sub_157E9D0(v49 + 40, v7);
          v51 = *(_QWORD *)(v7 + 24);
          v52 = *v50;
          *(_QWORD *)(v7 + 32) = v50;
          v52 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v7 + 24) = v52 | v51 & 7;
          *(_QWORD *)(v52 + 8) = v7 + 24;
          *v50 = *v50 & 7 | (v7 + 24);
        }
        sub_164B780(v43, v116);
        v53 = *(_QWORD *)v26;
        if ( *(_QWORD *)v26 )
        {
          v115 = *(_QWORD *)v26;
          sub_1623A60(&v115, v53, 2);
          if ( *(_QWORD *)(v7 + 48) )
            sub_161E7C0(v7 + 48);
          v54 = v115;
          *(_QWORD *)(v7 + 48) = v115;
          if ( v54 )
            sub_1623210(&v115, v54, v7 + 48);
        }
      }
      if ( unk_4D04700 && *(_BYTE *)(v7 + 16) > 0x17u )
      {
        v27 = sub_15F24E0(v7);
        sub_15F2440(v7, v27 | 1u);
      }
    }
    return v7;
  }
  if ( v10 == a4 )
    return (__int64)a2;
  v13 = *(_BYTE *)(a4 + 8);
  v14 = *(_BYTE *)(v10 + 8);
  if ( v13 == 15 )
  {
    if ( v14 == 15 )
    {
      v118 = 1;
      v16 = (__int64 *)a1[1];
      v116[0] = "conv";
      v117 = 3;
      if ( a4 != *a2 )
      {
        if ( *((_BYTE *)a2 + 16) <= 0x10u )
          return sub_15A46C0(47, a2, a4, 0);
        v87 = a4;
        v121[0] = 257;
        v88 = 47;
        goto LABEL_127;
      }
      return (__int64)a2;
    }
    if ( v14 != 11 )
      sub_127B550("unexpected destination type for cast from pointer type", a7, 1);
    v19 = sub_127B390();
    v20 = sub_1644900(a1[2], v19);
    v118 = 1;
    v21 = (__int64 *)a1[1];
    v116[0] = "conv";
    v117 = 3;
    if ( v20 == *a2 )
    {
      v7 = (__int64)a2;
    }
    else if ( *((_BYTE *)a2 + 16) > 0x10u )
    {
      v121[0] = 257;
      v74 = sub_15FE0A0(a2, v20, a3, &v119, 0);
      v75 = v21[1];
      v7 = v74;
      if ( v75 )
      {
        v76 = (__int64 *)v21[2];
        sub_157E9D0(v75 + 40, v74);
        v77 = *(_QWORD *)(v7 + 24);
        v78 = *v76;
        *(_QWORD *)(v7 + 32) = v76;
        v78 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v7 + 24) = v78 | v77 & 7;
        *(_QWORD *)(v78 + 8) = v7 + 24;
        *v76 = *v76 & 7 | (v7 + 24);
      }
      sub_164B780(v7, v116);
      v79 = *v21;
      if ( *v21 )
      {
        v115 = *v21;
        sub_1623A60(&v115, v79, 2);
        if ( *(_QWORD *)(v7 + 48) )
          sub_161E7C0(v7 + 48);
        v80 = v115;
        *(_QWORD *)(v7 + 48) = v115;
        if ( v80 )
          sub_1623210(&v115, v80, v7 + 48);
      }
      v21 = (__int64 *)a1[1];
    }
    else
    {
      v22 = sub_15A4750(a2, v20, a3);
      v21 = (__int64 *)a1[1];
      v7 = v22;
    }
    v118 = 1;
    v116[0] = "conv";
    v117 = 3;
    if ( a4 == *(_QWORD *)v7 )
      return v7;
    if ( *(_BYTE *)(v7 + 16) <= 0x10u )
      return sub_15A46C0(46, v7, a4, 0);
    v121[0] = 257;
    v81 = sub_15FDBD0(46, v7, a4, &v119, 0);
    v82 = v21[1];
    v7 = v81;
    if ( v82 )
    {
      v83 = (__int64 *)v21[2];
      v84 = v82 + 40;
LABEL_121:
      sub_157E9D0(v84, v7);
      v85 = *v83;
      v86 = *(_QWORD *)(v7 + 24);
      *(_QWORD *)(v7 + 32) = v83;
      v85 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v7 + 24) = v85 | v86 & 7;
      *(_QWORD *)(v85 + 8) = v7 + 24;
      *v83 = *v83 & 7 | (v7 + 24);
    }
LABEL_122:
    sub_164B780(v7, v116);
    v40 = *v21;
    if ( !*v21 )
      return v7;
    goto LABEL_62;
  }
  if ( v14 == 15 )
  {
    if ( v13 != 11 )
      sub_127B550("unexpected non-integer type for cast from pointer type!", a7, 1);
    v118 = 1;
    v16 = (__int64 *)a1[1];
    v116[0] = "conv";
    v117 = 3;
    if ( a4 != *a2 )
    {
      if ( *((_BYTE *)a2 + 16) <= 0x10u )
        return sub_15A46C0(45, a2, a4, 0);
      v87 = a4;
      v121[0] = 257;
      v88 = 45;
      goto LABEL_127;
    }
    return (__int64)a2;
  }
  if ( v14 == 11 )
  {
    if ( v13 != 11 )
    {
      if ( a3 )
      {
        v118 = 1;
        v16 = (__int64 *)a1[1];
        v116[0] = "conv";
        v117 = 3;
        if ( a4 != *a2 )
        {
          if ( *((_BYTE *)a2 + 16) <= 0x10u )
            return sub_15A46C0(42, a2, a4, 0);
          v87 = a4;
          v121[0] = 257;
          v88 = 42;
          goto LABEL_127;
        }
        return (__int64)a2;
      }
      if ( unk_4D04630 || v13 != 2 || *(_DWORD *)(v10 + 8) >> 8 != 64 )
      {
        v118 = 1;
        v16 = (__int64 *)a1[1];
        v116[0] = "conv";
        v117 = 3;
        if ( a4 != *a2 )
        {
          if ( *((_BYTE *)a2 + 16) <= 0x10u )
            return sub_15A46C0(41, a2, a4, 0);
          v87 = a4;
          v121[0] = 257;
          v88 = 41;
          goto LABEL_127;
        }
        return (__int64)a2;
      }
      v120 = 0x1000000000LL;
      v119 = v121;
      sub_16CD150(&v119, v121, 17, 1);
      v96 = (__m128i *)((char *)v119 + (unsigned int)v120);
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F10FE0);
      v96[1].m128i_i8[0] = 110;
      *v96 = si128;
      v98 = *a1;
      LODWORD(v120) = v120 + 17;
      v99 = sub_128A3C0(v98, a2, a4, (__int64)&v119);
      v72 = v119;
      v7 = v99;
      if ( v119 == v121 )
        return v7;
LABEL_102:
      _libc_free(v72, a2);
      return v7;
    }
    v118 = 1;
    v21 = (__int64 *)a1[1];
    v116[0] = "conv";
    v117 = 3;
    if ( a4 == *a2 )
      return (__int64)a2;
    if ( *((_BYTE *)a2 + 16) <= 0x10u )
      return sub_15A4750(a2, a4, a3);
    v121[0] = 257;
    v7 = sub_15FE0A0(a2, a4, a3, &v119, 0);
    v107 = v21[1];
    if ( v107 )
    {
      v83 = (__int64 *)v21[2];
      v84 = v107 + 40;
      goto LABEL_121;
    }
    goto LABEL_122;
  }
  if ( (unsigned __int8)(v14 - 1) > 5u )
    sub_127B550("expected floating point source type in cast!", a7, 1);
  if ( v13 != 11 )
  {
    if ( (unsigned __int8)(v13 - 1) > 5u )
      sub_127B550("expected floating point destination type in cast!", a7, 1);
    v15 = *(_BYTE *)(v10 + 8) <= v13;
    v16 = (__int64 *)a1[1];
    v118 = 1;
    v116[0] = "conv";
    v17 = *a2;
    v117 = 3;
    if ( v15 )
    {
      if ( a4 != v17 )
      {
        if ( *((_BYTE *)a2 + 16) <= 0x10u )
          return sub_15A46C0(44, a2, a4, 0);
        v87 = a4;
        v121[0] = 257;
        v88 = 44;
        goto LABEL_127;
      }
    }
    else if ( a4 != v17 )
    {
      if ( *((_BYTE *)a2 + 16) <= 0x10u )
        return sub_15A46C0(43, a2, a4, 0);
      v121[0] = 257;
      v88 = 43;
      v87 = a4;
LABEL_127:
      v7 = sub_15FDBD0(v88, a2, v87, &v119, 0);
      v89 = v16[1];
      if ( v89 )
      {
        v90 = (__int64 *)v16[2];
        sub_157E9D0(v89 + 40, v7);
        v91 = *(_QWORD *)(v7 + 24);
        v92 = *v90;
        *(_QWORD *)(v7 + 32) = v90;
        v92 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v7 + 24) = v92 | v91 & 7;
        *(_QWORD *)(v92 + 8) = v7 + 24;
        *v90 = *v90 & 7 | (v7 + 24);
      }
      sub_164B780(v7, v116);
      v40 = *v16;
      if ( !*v16 )
        return v7;
LABEL_62:
      v115 = v40;
      sub_1623A60(&v115, v40, 2);
      if ( *(_QWORD *)(v7 + 48) )
        sub_161E7C0(v7 + 48);
      v41 = v115;
      *(_QWORD *)(v7 + 48) = v115;
      if ( v41 )
        sub_1623210(&v115, v41, v7 + 48);
      return v7;
    }
    return (__int64)a2;
  }
  if ( unk_4D04630 || (v55 = *(_BYTE *)(v10 + 8), v56 = *(_DWORD *)(a4 + 8) >> 8, v56 == 128) || v55 == 5 )
  {
    v16 = (__int64 *)a1[1];
    v73 = *a2;
    v118 = 1;
    v116[0] = "conv";
    v117 = 3;
    if ( a5 )
    {
      if ( a4 != v73 )
      {
        if ( *((_BYTE *)a2 + 16) <= 0x10u )
          return sub_15A46C0(40, a2, a4, 0);
        v87 = a4;
        v121[0] = 257;
        v88 = 40;
        goto LABEL_127;
      }
    }
    else if ( a4 != v73 )
    {
      if ( *((_BYTE *)a2 + 16) <= 0x10u )
        return sub_15A46C0(39, a2, a4, 0);
      v87 = a4;
      v121[0] = 257;
      v88 = 39;
      goto LABEL_127;
    }
    return (__int64)a2;
  }
  v57 = "__nv_double";
  HIDWORD(v120) = 16;
  v119 = v121;
  v58 = v55 == 3;
  v59 = v58 + 10LL;
  v60 = v58 + 10;
  if ( v55 != 3 )
    v57 = "__nv_float";
  *(_QWORD *)((char *)&v121[-4] + v59) = *(_QWORD *)&v57[v59 - 8];
  v61 = 0;
  do
  {
    v62 = v61;
    v61 += 8;
    *(_QWORD *)((char *)v121 + v62) = *(_QWORD *)&v57[v62];
  }
  while ( v61 < (((_DWORD)v59 - 1) & 0xFFFFFFF8) );
  LODWORD(v120) = v60;
  v63 = (char *)"2";
  v64 = (a5 == 0) + 1;
  if ( !a5 )
    v63 = "2u";
  v65 = (char *)v121 + v59;
  v66 = 0;
  if ( (a5 == 0) != -1 )
  {
    do
    {
      v67 = v66++;
      v65[v67] = v63[v67];
    }
    while ( v66 < v64 );
  }
  v68 = "ll_rz";
  LODWORD(v120) = v64 + v120;
  v69 = (unsigned int)v120;
  v70 = (v56 != 64) + 5LL;
  if ( v56 != 64 )
    v68 = "int_rz";
  if ( HIDWORD(v120) - (unsigned __int64)(unsigned int)v120 < v70 )
  {
    v109 = v56;
    v110 = v68;
    sub_16CD150(&v119, v121, (unsigned int)v120 + v70, 1);
    v69 = (unsigned int)v120;
    v68 = v110;
    v56 = v109;
  }
  v71 = (_QWORD *)((char *)v119 + v69);
  if ( (unsigned int)v70 >= 8 )
  {
    v101 = (unsigned __int64)(v71 + 1) & 0xFFFFFFFFFFFFFFF8LL;
    *v71 = *(_QWORD *)v68;
    *(_QWORD *)((char *)v71 + v70 - 8) = *(_QWORD *)&v68[v70 - 8];
    v102 = (char *)v71 - v101;
    v103 = (char *)(v68 - v102);
    if ( (((_DWORD)v70 + (_DWORD)v102) & 0xFFFFFFF8) >= 8 )
    {
      v104 = (v70 + (_DWORD)v102) & 0xFFFFFFF8;
      v105 = 0;
      do
      {
        v106 = v105;
        v105 += 8;
        *(_QWORD *)(v101 + v106) = *(_QWORD *)&v103[v106];
      }
      while ( v105 < v104 );
    }
  }
  else if ( (v70 & 4) != 0 )
  {
    *(_DWORD *)v71 = *(_DWORD *)v68;
    *(_DWORD *)((char *)v71 + (unsigned int)v70 - 4) = *(_DWORD *)&v68[(unsigned int)v70 - 4];
  }
  else if ( (_DWORD)v70 )
  {
    *(_BYTE *)v71 = *v68;
    if ( (v70 & 2) != 0 )
      *(_WORD *)((char *)v71 + (unsigned int)v70 - 2) = *(_WORD *)&v68[(unsigned int)v70 - 2];
  }
  LODWORD(v120) = v70 + v120;
  if ( v56 == 64 )
  {
    a2 = v11;
    v7 = sub_128A3C0(*a1, v11, a4, (__int64)&v119);
  }
  else
  {
    a2 = v11;
    v114 = sub_1643350(a1[2]);
    v100 = sub_128A3C0(*a1, v11, v114, (__int64)&v119);
    v7 = v100;
    if ( v114 != a4 )
    {
      v7 = sub_128A450((_DWORD)a1, v100, a5, a4, a5, 0, (__int64)a7);
      a2 = v108;
    }
  }
  v72 = v119;
  if ( v119 != v121 )
    goto LABEL_102;
  return v7;
}
