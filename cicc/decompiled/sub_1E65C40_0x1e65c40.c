// Function: sub_1E65C40
// Address: 0x1e65c40
//
unsigned __int64 __fastcall sub_1E65C40(__int64 *a1, __int64 a2, char a3, unsigned int a4, unsigned int a5)
{
  __int64 v7; // rdi
  _BYTE *v8; // rax
  __int64 v9; // rax
  _WORD *v10; // rdx
  __int64 v11; // r8
  unsigned __int64 result; // rax
  __int64 *v13; // r15
  __int64 *v14; // r13
  __int64 v15; // rdi
  _BYTE *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  _WORD *v20; // rdx
  __int64 *v21; // rdi
  _BYTE *v22; // rsi
  _BYTE *v23; // r8
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // rdx
  _BYTE *v28; // rax
  char v29; // di
  _BYTE *v30; // rcx
  __int64 v31; // rax
  __int64 v32; // rdx
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rdx
  _BYTE *v35; // rax
  char v36; // r8
  unsigned __int64 v37; // rax
  unsigned __int64 v38; // rcx
  unsigned __int64 v39; // rdx
  const char *v40; // rax
  size_t v41; // rdx
  _WORD *v42; // rdi
  char *v43; // rsi
  unsigned __int64 v44; // rax
  __int64 v45; // r15
  unsigned __int64 v46; // r15
  __int64 v47; // rbx
  __int64 *v48; // rax
  char v49; // dl
  __int64 v50; // r14
  __int64 *v51; // rax
  __int64 *v52; // rcx
  __int64 *v53; // rsi
  unsigned __int64 v54; // rsi
  char v55; // al
  char v56; // r8
  bool v57; // al
  unsigned __int64 v58; // rdi
  __int64 v59; // rcx
  unsigned __int64 v60; // r15
  __int64 v61; // rax
  __int64 v62; // rcx
  char v63; // si
  __int64 v64; // rax
  __int64 v65; // rcx
  unsigned __int64 v66; // r15
  __int64 v67; // rax
  __int64 v68; // rdi
  __int64 v69; // rdx
  __int64 v70; // rax
  char v71; // si
  __int64 v72; // r14
  _WORD *v73; // rdi
  unsigned __int64 v74; // rax
  __int64 v75; // rdx
  __int64 *v76; // rsi
  const char *v77; // rax
  size_t v78; // rdx
  char *v79; // rsi
  size_t v80; // r15
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // rsi
  char v84; // al
  char v85; // r8
  bool v86; // al
  __int64 v87; // rax
  unsigned __int64 v88; // [rsp+0h] [rbp-460h]
  __int64 v89; // [rsp+0h] [rbp-460h]
  size_t v90; // [rsp+0h] [rbp-460h]
  __int64 v91; // [rsp+8h] [rbp-458h]
  unsigned int v92; // [rsp+8h] [rbp-458h]
  unsigned int v95; // [rsp+28h] [rbp-438h]
  __int64 v97; // [rsp+30h] [rbp-430h] BYREF
  __int64 *v98; // [rsp+38h] [rbp-428h]
  __int64 *v99; // [rsp+40h] [rbp-420h]
  unsigned int v100; // [rsp+48h] [rbp-418h]
  unsigned int v101; // [rsp+4Ch] [rbp-414h]
  int v102; // [rsp+50h] [rbp-410h]
  char v103[64]; // [rsp+58h] [rbp-408h] BYREF
  unsigned __int64 v104; // [rsp+98h] [rbp-3C8h] BYREF
  unsigned __int64 v105; // [rsp+A0h] [rbp-3C0h]
  unsigned __int64 v106; // [rsp+A8h] [rbp-3B8h]
  _QWORD v107[2]; // [rsp+B0h] [rbp-3B0h] BYREF
  unsigned __int64 v108; // [rsp+C0h] [rbp-3A0h]
  char v109[64]; // [rsp+D8h] [rbp-388h] BYREF
  unsigned __int64 v110; // [rsp+118h] [rbp-348h]
  unsigned __int64 v111; // [rsp+120h] [rbp-340h]
  __int64 v112; // [rsp+128h] [rbp-338h]
  _QWORD v113[2]; // [rsp+130h] [rbp-330h] BYREF
  unsigned __int64 v114; // [rsp+140h] [rbp-320h]
  _BYTE v115[64]; // [rsp+158h] [rbp-308h] BYREF
  __int64 v116; // [rsp+198h] [rbp-2C8h]
  __int64 v117; // [rsp+1A0h] [rbp-2C0h]
  unsigned __int64 v118; // [rsp+1A8h] [rbp-2B8h]
  _QWORD v119[2]; // [rsp+1B0h] [rbp-2B0h] BYREF
  unsigned __int64 v120; // [rsp+1C0h] [rbp-2A0h]
  _BYTE v121[64]; // [rsp+1D8h] [rbp-288h] BYREF
  __int64 v122; // [rsp+218h] [rbp-248h]
  __int64 i; // [rsp+220h] [rbp-240h]
  unsigned __int64 v124; // [rsp+228h] [rbp-238h]
  __m128i v125; // [rsp+230h] [rbp-230h] BYREF
  unsigned __int64 v126; // [rsp+240h] [rbp-220h] BYREF
  _BYTE *v127; // [rsp+298h] [rbp-1C8h]
  _BYTE *v128; // [rsp+2A0h] [rbp-1C0h]
  __int64 v129; // [rsp+2A8h] [rbp-1B8h]
  char v130[8]; // [rsp+2B0h] [rbp-1B0h] BYREF
  __int64 v131; // [rsp+2B8h] [rbp-1A8h]
  unsigned __int64 v132; // [rsp+2C0h] [rbp-1A0h]
  _BYTE *v133; // [rsp+318h] [rbp-148h]
  _BYTE *v134; // [rsp+320h] [rbp-140h]
  __int64 v135; // [rsp+328h] [rbp-138h]
  __m128i v136; // [rsp+330h] [rbp-130h] BYREF
  unsigned __int64 v137; // [rsp+340h] [rbp-120h] BYREF
  unsigned __int64 v138; // [rsp+398h] [rbp-C8h]
  __int64 v139; // [rsp+3A0h] [rbp-C0h]
  __int64 v140; // [rsp+3A8h] [rbp-B8h]
  char v141[8]; // [rsp+3B0h] [rbp-B0h] BYREF
  __int64 v142; // [rsp+3B8h] [rbp-A8h]
  unsigned __int64 v143; // [rsp+3C0h] [rbp-A0h]
  __int64 v144; // [rsp+418h] [rbp-48h]
  __int64 v145; // [rsp+420h] [rbp-40h]
  __int64 v146; // [rsp+428h] [rbp-38h]

  v95 = 2 * a4;
  if ( a3 )
  {
    v7 = sub_16E8750(a2, 2 * a4);
    v8 = *(_BYTE **)(v7 + 24);
    if ( (unsigned __int64)v8 >= *(_QWORD *)(v7 + 16) )
    {
      v7 = sub_16E7DE0(v7, 91);
    }
    else
    {
      *(_QWORD *)(v7 + 24) = v8 + 1;
      *v8 = 91;
    }
    v9 = sub_16E7A90(v7, a4);
    v10 = *(_WORD **)(v9 + 24);
    v11 = v9;
    if ( *(_QWORD *)(v9 + 16) - (_QWORD)v10 <= 1u )
    {
      v11 = sub_16E7EE0(v9, "] ", 2u);
    }
    else
    {
      *v10 = 8285;
      *(_QWORD *)(v9 + 24) += 2LL;
    }
  }
  else
  {
    v11 = sub_16E8750(a2, v95);
  }
  v91 = v11;
  sub_1E62530(&v136, a1);
  sub_16E7EE0(v91, (char *)v136.m128i_i64[0], v136.m128i_u64[1]);
  if ( (unsigned __int64 *)v136.m128i_i64[0] != &v137 )
    j_j___libc_free_0(v136.m128i_i64[0], v137 + 1);
  result = *(_QWORD *)(a2 + 24);
  if ( result >= *(_QWORD *)(a2 + 16) )
  {
    result = sub_16E7DE0(a2, 10);
  }
  else
  {
    *(_QWORD *)(a2 + 24) = result + 1;
    *(_BYTE *)result = 10;
  }
  if ( a5 )
  {
    v19 = sub_16E8750(a2, v95);
    v20 = *(_WORD **)(v19 + 24);
    if ( *(_QWORD *)(v19 + 16) - (_QWORD)v20 <= 1u )
    {
      sub_16E7EE0(v19, "{\n", 2u);
    }
    else
    {
      *v20 = 2683;
      *(_QWORD *)(v19 + 24) += 2LL;
    }
    v92 = a4 + 1;
    sub_16E8750(a2, v95 + 2);
    if ( a5 != 1 )
    {
      if ( a5 != 2 )
        goto LABEL_21;
      sub_1E65070(&v136, a1);
      v22 = v115;
      v21 = v113;
      sub_16CCCB0(v113, (__int64)v115, (__int64)&v136);
      v59 = v139;
      v24 = v138;
      v116 = 0;
      v117 = 0;
      v118 = 0;
      v60 = v139 - v138;
      if ( v139 == v138 )
      {
        v61 = 0;
      }
      else
      {
        if ( v60 > 0x7FFFFFFFFFFFFFE0LL )
          goto LABEL_176;
        v61 = sub_22077B0(v139 - v138);
        v59 = v139;
        v24 = v138;
      }
      v116 = v61;
      v117 = v61;
      v118 = v61 + v60;
      if ( v59 == v24 )
      {
        v62 = v61;
      }
      else
      {
        v62 = v61 + v59 - v24;
        do
        {
          if ( v61 )
          {
            *(_QWORD *)v61 = *(_QWORD *)v24;
            v63 = *(_BYTE *)(v24 + 24);
            *(_BYTE *)(v61 + 24) = v63;
            if ( v63 )
              *(__m128i *)(v61 + 8) = _mm_loadu_si128((const __m128i *)(v24 + 8));
          }
          v61 += 32;
          v24 += 32LL;
        }
        while ( v62 != v61 );
      }
      v21 = v119;
      v117 = v62;
      v22 = v121;
      sub_16CCCB0(v119, (__int64)v121, (__int64)v141);
      v64 = v145;
      v65 = v144;
      v122 = 0;
      i = 0;
      v124 = 0;
      v66 = v145 - v144;
      if ( v145 == v144 )
      {
        v68 = 0;
        goto LABEL_113;
      }
      if ( v66 <= 0x7FFFFFFFFFFFFFE0LL )
      {
        v67 = sub_22077B0(v145 - v144);
        v65 = v144;
        v68 = v67;
        v64 = v145;
LABEL_113:
        v122 = v68;
        i = v68;
        v124 = v68 + v66;
        if ( v64 == v65 )
        {
          v70 = v68;
        }
        else
        {
          v69 = v68;
          v70 = v68 + v64 - v65;
          do
          {
            if ( v69 )
            {
              *(_QWORD *)v69 = *(_QWORD *)v65;
              v71 = *(_BYTE *)(v65 + 24);
              *(_BYTE *)(v69 + 24) = v71;
              if ( v71 )
                *(__m128i *)(v69 + 8) = _mm_loadu_si128((const __m128i *)(v65 + 8));
            }
            v69 += 32;
            v65 += 32;
          }
          while ( v70 != v69 );
        }
        for ( i = v70; ; v70 = i )
        {
          v75 = v116;
          if ( v117 - v116 == v70 - v68 )
          {
            if ( v116 == v117 )
            {
LABEL_150:
              if ( v68 )
                j_j___libc_free_0(v68, v124 - v68);
              if ( v120 != v119[1] )
                _libc_free(v120);
              if ( v116 )
                j_j___libc_free_0(v116, v118 - v116);
              if ( v114 != v113[1] )
                _libc_free(v114);
              if ( v144 )
                j_j___libc_free_0(v144, v146 - v144);
              if ( v143 != v142 )
                _libc_free(v143);
              if ( v138 )
                j_j___libc_free_0(v138, v140 - v138);
              v58 = v137;
              if ( v137 == v136.m128i_i64[1] )
                goto LABEL_21;
LABEL_100:
              _libc_free(v58);
              goto LABEL_21;
            }
            v83 = v68;
            while ( *(_QWORD *)v75 == *(_QWORD *)v83 )
            {
              v84 = *(_BYTE *)(v75 + 24);
              v85 = *(_BYTE *)(v83 + 24);
              if ( v84 && v85 )
                v86 = ((*(__int64 *)(v75 + 8) >> 1) & 3) != 0
                    ? ((*(__int64 *)(v83 + 8) >> 1) & 3) == ((*(__int64 *)(v75 + 8) >> 1) & 3)
                    : *(_QWORD *)(v75 + 16) == *(_QWORD *)(v83 + 16);
              else
                v86 = v85 == v84;
              if ( !v86 )
                break;
              v75 += 32;
              v83 += 32;
              if ( v117 == v75 )
                goto LABEL_150;
            }
          }
          v76 = *(__int64 **)(v117 - 32);
          if ( (*v76 & 4) != 0 )
            break;
          v77 = sub_1DD6290(*v76 & 0xFFFFFFFFFFFFFFF8LL);
          v73 = *(_WORD **)(a2 + 24);
          v79 = (char *)v77;
          v80 = v78;
          v74 = *(_QWORD *)(a2 + 16) - (_QWORD)v73;
          if ( v78 > v74 )
          {
            v72 = sub_16E7EE0(a2, v79, v78);
LABEL_122:
            v73 = *(_WORD **)(v72 + 24);
            v74 = *(_QWORD *)(v72 + 16) - (_QWORD)v73;
            goto LABEL_123;
          }
          v72 = a2;
          if ( v78 )
          {
            memcpy(v73, v79, v78);
            v81 = *(_QWORD *)(a2 + 16);
            v73 = (_WORD *)(v80 + *(_QWORD *)(a2 + 24));
            *(_QWORD *)(a2 + 24) = v73;
            v74 = v81 - (_QWORD)v73;
          }
LABEL_123:
          if ( v74 <= 1 )
          {
            sub_16E7EE0(v72, ", ", 2u);
          }
          else
          {
            *v73 = 8236;
            *(_QWORD *)(v72 + 24) += 2LL;
          }
          sub_1E64D90((__int64)v113);
          v68 = v122;
        }
        sub_1E62530(&v125, v76);
        v72 = sub_16E7EE0(a2, (char *)v125.m128i_i64[0], v125.m128i_u64[1]);
        if ( (unsigned __int64 *)v125.m128i_i64[0] != &v126 )
          j_j___libc_free_0(v125.m128i_i64[0], v126 + 1);
        goto LABEL_122;
      }
LABEL_176:
      sub_4261EA(v21, v22, v24);
    }
    sub_1E65980(&v125, a1);
    v21 = &v97;
    sub_16CCCB0(&v97, (__int64)v103, (__int64)&v125);
    v22 = v128;
    v23 = v127;
    v104 = 0;
    v105 = 0;
    v106 = 0;
    v24 = v128 - v127;
    if ( v128 == v127 )
    {
      v26 = 0;
    }
    else
    {
      if ( v24 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_176;
      v88 = v128 - v127;
      v25 = sub_22077B0(v128 - v127);
      v22 = v128;
      v23 = v127;
      v24 = v88;
      v26 = v25;
    }
    v104 = v26;
    v105 = v26;
    v106 = v26 + v24;
    if ( v22 != v23 )
    {
      v27 = v26;
      v28 = v23;
      do
      {
        if ( v27 )
        {
          *(_QWORD *)v27 = *(_QWORD *)v28;
          v29 = v28[16];
          *(_BYTE *)(v27 + 16) = v29;
          if ( v29 )
            *(_QWORD *)(v27 + 8) = *((_QWORD *)v28 + 1);
        }
        v28 += 24;
        v27 += 24LL;
      }
      while ( v28 != v22 );
      v26 += 8 * ((unsigned __int64)(v28 - 24 - v23) >> 3) + 24;
    }
    v105 = v26;
    v21 = v107;
    sub_16CCCB0(v107, (__int64)v109, (__int64)v130);
    v30 = v134;
    v22 = v133;
    v110 = 0;
    v111 = 0;
    v112 = 0;
    v24 = v134 - v133;
    if ( v134 == v133 )
    {
      v32 = 0;
      v33 = 0;
    }
    else
    {
      if ( v24 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_176;
      v89 = v134 - v133;
      v31 = sub_22077B0(v134 - v133);
      v30 = v134;
      v22 = v133;
      v32 = v89;
      v33 = v31;
    }
    v110 = v33;
    v111 = v33;
    v112 = v33 + v32;
    if ( v30 == v22 )
    {
      v37 = v33;
    }
    else
    {
      v34 = v33;
      v35 = v22;
      do
      {
        if ( v34 )
        {
          *(_QWORD *)v34 = *(_QWORD *)v35;
          v36 = v35[16];
          *(_BYTE *)(v34 + 16) = v36;
          if ( v36 )
            *(_QWORD *)(v34 + 8) = *((_QWORD *)v35 + 1);
        }
        v35 += 24;
        v34 += 24LL;
      }
      while ( v35 != v30 );
      v37 = v33 + 8 * ((unsigned __int64)(v35 - 24 - v22) >> 3) + 24;
    }
    v38 = v105;
    v39 = v104;
    v111 = v37;
    if ( v105 - v104 != v37 - v33 )
      goto LABEL_56;
LABEL_77:
    if ( v39 == v38 )
    {
LABEL_85:
      if ( v33 )
        j_j___libc_free_0(v33, v112 - v33);
      if ( v108 != v107[1] )
        _libc_free(v108);
      if ( v104 )
        j_j___libc_free_0(v104, v106 - v104);
      if ( v99 != v98 )
        _libc_free((unsigned __int64)v99);
      if ( v133 )
        j_j___libc_free_0(v133, v135 - (_QWORD)v133);
      if ( v132 != v131 )
        _libc_free(v132);
      if ( v127 )
        j_j___libc_free_0(v127, v129 - (_QWORD)v127);
      v58 = v126;
      if ( v126 != v125.m128i_i64[1] )
        goto LABEL_100;
LABEL_21:
      v16 = *(_BYTE **)(a2 + 24);
      if ( (unsigned __int64)v16 >= *(_QWORD *)(a2 + 16) )
      {
        sub_16E7DE0(a2, 10);
      }
      else
      {
        *(_QWORD *)(a2 + 24) = v16 + 1;
        *v16 = 10;
      }
      if ( !a3 )
        goto LABEL_24;
      result = a1[5];
      v13 = (__int64 *)a1[6];
      if ( v13 == (__int64 *)result )
        goto LABEL_24;
      goto LABEL_14;
    }
    v54 = v33;
    while ( *(_QWORD *)v39 == *(_QWORD *)v54 )
    {
      v55 = *(_BYTE *)(v39 + 16);
      v56 = *(_BYTE *)(v54 + 16);
      if ( v55 && v56 )
        v57 = *(_QWORD *)(v39 + 8) == *(_QWORD *)(v54 + 8);
      else
        v57 = v56 == v55;
      if ( !v57 )
        break;
      v39 += 24LL;
      v54 += 24LL;
      if ( v39 == v38 )
        goto LABEL_85;
    }
LABEL_56:
    v40 = sub_1DD6290(*(_QWORD *)(v38 - 24));
    v42 = *(_WORD **)(a2 + 24);
    v43 = (char *)v40;
    v44 = *(_QWORD *)(a2 + 16) - (_QWORD)v42;
    if ( v44 < v41 )
    {
      v82 = sub_16E7EE0(a2, v43, v41);
      v42 = *(_WORD **)(v82 + 24);
      v45 = v82;
      v44 = *(_QWORD *)(v82 + 16) - (_QWORD)v42;
    }
    else
    {
      v45 = a2;
      if ( v41 )
      {
        v90 = v41;
        memcpy(v42, v43, v41);
        v87 = *(_QWORD *)(a2 + 16);
        v42 = (_WORD *)(v90 + *(_QWORD *)(a2 + 24));
        *(_QWORD *)(a2 + 24) = v42;
        v44 = v87 - (_QWORD)v42;
      }
    }
    if ( v44 <= 1 )
    {
      sub_16E7EE0(v45, ", ", 2u);
    }
    else
    {
      *v42 = 8236;
      *(_QWORD *)(v45 + 24) += 2LL;
    }
    v46 = v105;
LABEL_62:
    v47 = *(_QWORD *)(v46 - 24);
    if ( !*(_BYTE *)(v46 - 8) )
    {
      v48 = *(__int64 **)(v47 + 88);
      *(_BYTE *)(v46 - 8) = 1;
      *(_QWORD *)(v46 - 16) = v48;
      goto LABEL_66;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v48 = *(__int64 **)(v46 - 16);
LABEL_66:
        if ( *(__int64 **)(v47 + 96) == v48 )
        {
          v105 -= 24LL;
          v39 = v104;
          v46 = v105;
          if ( v105 == v104 )
          {
            v38 = v104;
LABEL_76:
            v33 = v110;
            if ( v38 - v39 == v111 - v110 )
              goto LABEL_77;
            goto LABEL_56;
          }
          goto LABEL_62;
        }
        *(_QWORD *)(v46 - 16) = v48 + 1;
        v50 = *v48;
        v51 = v98;
        if ( v99 == v98 )
          break;
LABEL_64:
        sub_16CCBA0((__int64)&v97, v50);
        if ( v49 )
          goto LABEL_75;
      }
      v52 = &v98[v101];
      if ( v98 == v52 )
      {
LABEL_135:
        if ( v101 < v100 )
        {
          ++v101;
          *v52 = v50;
          ++v97;
LABEL_75:
          v136.m128i_i64[0] = v50;
          LOBYTE(v137) = 0;
          sub_1E65930(&v104, (__int64)&v136);
          v39 = v104;
          v38 = v105;
          goto LABEL_76;
        }
        goto LABEL_64;
      }
      v53 = 0;
      while ( v50 != *v51 )
      {
        if ( *v51 == -2 )
        {
          v53 = v51;
          if ( v52 == v51 + 1 )
            goto LABEL_74;
          ++v51;
        }
        else if ( v52 == ++v51 )
        {
          if ( !v53 )
            goto LABEL_135;
LABEL_74:
          *v53 = v50;
          --v102;
          ++v97;
          goto LABEL_75;
        }
      }
    }
  }
  if ( a3 )
  {
    result = a1[5];
    v13 = (__int64 *)a1[6];
    v92 = a4 + 1;
    if ( v13 != (__int64 *)result )
    {
LABEL_14:
      v14 = (__int64 *)result;
      do
      {
        v15 = *v14++;
        result = sub_1E65C40(v15, a2, 1, v92, a5);
      }
      while ( v13 != v14 );
      if ( !a5 )
        return result;
LABEL_24:
      v17 = sub_16E8750(a2, v95);
      v18 = *(_QWORD *)(v17 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v17 + 16) - v18) <= 2 )
      {
        return sub_16E7EE0(v17, "} \n", 3u);
      }
      else
      {
        *(_BYTE *)(v18 + 2) = 10;
        *(_WORD *)v18 = 8317;
        *(_QWORD *)(v17 + 24) += 3LL;
        return 8317;
      }
    }
  }
  return result;
}
