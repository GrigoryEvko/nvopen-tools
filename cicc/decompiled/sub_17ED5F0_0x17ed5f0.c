// Function: sub_17ED5F0
// Address: 0x17ed5f0
//
__m128i *__fastcall sub_17ED5F0(__m128i *a1, __int64 **a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  void **v7; // rdi
  _BYTE *v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rdx
  char *v11; // rsi
  _QWORD *v12; // rdx
  __int64 *p_p_src; // r13
  _QWORD *v14; // rax
  __m128i *v15; // rdx
  __int64 v16; // rdi
  __m128i si128; // xmm0
  __int64 v18; // rax
  char *v19; // rsi
  char **v21; // rsi
  __int64 v22; // rax
  __int64 *v23; // r13
  size_t v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 *v30; // rbx
  __int64 v31; // rdx
  __int64 v32; // rdx
  char *v33; // rsi
  size_t v34; // rdx
  char *v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // r12
  __int64 v40; // rsi
  unsigned int v41; // ecx
  __int64 *v42; // rdx
  __int64 v43; // r8
  __int64 *v44; // rdx
  __int64 v45; // r14
  __int64 v46; // rbx
  __m128i *v47; // rdx
  __m128i *v48; // rdx
  __m128i v49; // xmm0
  unsigned __int64 v50; // rdi
  int v51; // eax
  __int64 v52; // rbx
  char v53; // r15
  char **v54; // rdi
  __int64 v55; // r15
  _BYTE *v56; // rax
  _WORD *v57; // rdx
  unsigned __int64 v58; // rax
  __int64 v59; // r13
  int v60; // eax
  unsigned int v61; // ebx
  __int64 *v62; // rdi
  __int64 *v63; // rdi
  __int64 v64; // r12
  size_t v65; // r14
  bool v66; // zf
  int v67; // r14d
  _BYTE *v68; // rdx
  __int64 *v69; // rdi
  _QWORD *v70; // rax
  _QWORD *v71; // rdx
  size_t v72; // rax
  __int64 *v73; // rdi
  __int64 v74; // rdi
  _BYTE *v75; // rax
  __int64 v76; // rax
  __int64 v77; // rdx
  __int64 v78; // rdi
  __int64 v79; // rax
  _DWORD *v80; // rdx
  __int64 v81; // r12
  __int64 v82; // rax
  __int64 v83; // r14
  __int64 v84; // r14
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 *v87; // rdi
  __int64 v88; // rax
  __int64 *v89; // rdi
  __int64 v90; // r12
  size_t v91; // r15
  _BYTE *v92; // rdx
  int v93; // r15d
  __int64 v94; // rax
  __int64 v95; // rax
  int v96; // edx
  size_t v97; // rdx
  char *v98; // rsi
  __int64 v99; // rdi
  __int64 v100; // rax
  __int64 v101; // r12
  int v102; // r9d
  __int64 v104; // [rsp+18h] [rbp-248h]
  int v105; // [rsp+24h] [rbp-23Ch]
  __int64 v107; // [rsp+50h] [rbp-210h]
  __int64 v108; // [rsp+60h] [rbp-200h]
  __int64 v109; // [rsp+78h] [rbp-1E8h]
  int v110; // [rsp+78h] [rbp-1E8h]
  unsigned int v111; // [rsp+9Ch] [rbp-1C4h] BYREF
  void *dest; // [rsp+A0h] [rbp-1C0h] BYREF
  size_t v113; // [rsp+A8h] [rbp-1B8h]
  _QWORD v114[2]; // [rsp+B0h] [rbp-1B0h] BYREF
  __m128i *v115; // [rsp+C0h] [rbp-1A0h] BYREF
  size_t v116; // [rsp+C8h] [rbp-198h]
  __m128i v117; // [rsp+D0h] [rbp-190h] BYREF
  __int64 *v118; // [rsp+E0h] [rbp-180h] BYREF
  __int64 v119; // [rsp+E8h] [rbp-178h]
  __int64 v120; // [rsp+F0h] [rbp-170h] BYREF
  char *v121; // [rsp+100h] [rbp-160h] BYREF
  size_t v122; // [rsp+108h] [rbp-158h]
  _QWORD v123[2]; // [rsp+110h] [rbp-150h] BYREF
  _QWORD v124[2]; // [rsp+120h] [rbp-140h] BYREF
  _QWORD v125[2]; // [rsp+130h] [rbp-130h] BYREF
  char *v126; // [rsp+140h] [rbp-120h] BYREF
  size_t v127; // [rsp+148h] [rbp-118h]
  _QWORD v128[2]; // [rsp+150h] [rbp-110h] BYREF
  _QWORD *v129; // [rsp+160h] [rbp-100h] BYREF
  __int64 v130; // [rsp+168h] [rbp-F8h]
  _QWORD v131[2]; // [rsp+170h] [rbp-F0h] BYREF
  char *v132; // [rsp+180h] [rbp-E0h] BYREF
  size_t v133; // [rsp+188h] [rbp-D8h]
  __int64 v134; // [rsp+190h] [rbp-D0h] BYREF
  __m128i *v135; // [rsp+198h] [rbp-C8h]
  int v136; // [rsp+1A0h] [rbp-C0h]
  _QWORD *v137; // [rsp+1A8h] [rbp-B8h]
  char *v138; // [rsp+1B0h] [rbp-B0h] BYREF
  _BYTE *v139; // [rsp+1B8h] [rbp-A8h]
  _BYTE *v140; // [rsp+1C0h] [rbp-A0h] BYREF
  _BYTE *v141; // [rsp+1C8h] [rbp-98h]
  int v142; // [rsp+1D0h] [rbp-90h]
  char **v143; // [rsp+1D8h] [rbp-88h]
  void **p_src; // [rsp+1E0h] [rbp-80h] BYREF
  size_t n; // [rsp+1E8h] [rbp-78h]
  _BYTE *src; // [rsp+1F0h] [rbp-70h] BYREF
  _BYTE *v147; // [rsp+1F8h] [rbp-68h]

  sub_16E2FC0((__int64 *)&dest, a3);
  v6 = v113;
  p_src = (void **)&src;
  if ( v113 > 0x8C )
    v6 = 140;
  sub_17E2210((__int64 *)&p_src, dest, (__int64)dest + v6);
  v7 = (void **)dest;
  if ( p_src == (void **)&src )
  {
    v24 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = (_BYTE)src;
      else
        memcpy(dest, &src, n);
      v24 = n;
      v7 = (void **)dest;
    }
    v113 = v24;
    *((_BYTE *)v7 + v24) = 0;
    v7 = p_src;
  }
  else
  {
    if ( dest == v114 )
    {
      dest = p_src;
      v113 = n;
      v114[0] = src;
    }
    else
    {
      v8 = (_BYTE *)v114[0];
      dest = p_src;
      v113 = n;
      v114[0] = src;
      if ( v7 )
      {
        p_src = v7;
        src = v8;
        goto LABEL_7;
      }
    }
    p_src = (void **)&src;
    v7 = (void **)&src;
  }
LABEL_7:
  n = 0;
  *(_BYTE *)v7 = 0;
  if ( p_src != (void **)&src )
    j_j___libc_free_0(p_src, src + 1);
  p_src = &dest;
  LOWORD(src) = 260;
  sub_16BEB10((__int64)&v115, (__int64)&p_src, &v111);
  sub_16E8970((__int64)&p_src, v111, 1, 0, v9);
  if ( v111 == -1 )
  {
    v14 = sub_16E8CB0();
    v15 = (__m128i *)v14[3];
    v16 = (__int64)v14;
    if ( v14[2] - (_QWORD)v15 <= 0x13u )
    {
      v16 = sub_16E7EE0((__int64)v14, "error opening file '", 0x14u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CBF0);
      v15[1].m128i_i32[0] = 656434540;
      *v15 = si128;
      v14[3] += 20LL;
    }
    v18 = sub_16E7EE0(v16, v115->m128i_i8, v116);
    sub_1263B40(v18, "' for writing!\n");
    v19 = (char *)byte_3F871B3;
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    sub_17E2210(a1->m128i_i64, byte_3F871B3, (__int64)byte_3F871B3);
    goto LABEL_20;
  }
  sub_16E2FC0((__int64 *)&v118, a5);
  v11 = (char *)sub_1649960(**a2);
  if ( !v11 )
  {
    v133 = 0;
    v132 = (char *)&v134;
    LOBYTE(v134) = 0;
    if ( !v119 )
      goto LABEL_12;
LABEL_26:
    v21 = (char **)&v118;
    p_p_src = (__int64 *)sub_1263B40((__int64)&p_src, "digraph \"");
    goto LABEL_27;
  }
  v132 = (char *)&v134;
  sub_17E2210((__int64 *)&v132, v11, (__int64)&v11[v10]);
  if ( v119 )
    goto LABEL_26;
LABEL_12:
  if ( !v133 )
  {
    sub_1263B40((__int64)&p_src, "digraph unnamed {\n");
LABEL_37:
    if ( v119 )
      goto LABEL_29;
    goto LABEL_38;
  }
  v12 = v147;
  if ( (unsigned __int64)(src - v147) <= 8 )
  {
    p_p_src = (__int64 *)sub_16E7EE0((__int64)&p_src, "digraph \"", 9u);
  }
  else
  {
    v147[8] = 34;
    p_p_src = (__int64 *)&p_src;
    *v12 = 0x2068706172676964LL;
    v147 += 9;
  }
  v21 = &v132;
LABEL_27:
  sub_16BE9B0((__int64 *)&v138, (__int64)v21);
  v22 = sub_16E7EE0((__int64)p_p_src, v138, (size_t)v139);
  sub_1263B40(v22, "\" {\n");
  if ( v138 == (char *)&v140 )
    goto LABEL_37;
  j_j___libc_free_0(v138, v140 + 1);
  if ( v119 )
  {
LABEL_29:
    if ( (unsigned __int64)(src - v147) <= 7 )
    {
      v23 = (__int64 *)sub_16E7EE0((__int64)&p_src, "\tlabel=\"", 8u);
    }
    else
    {
      v23 = (__int64 *)&p_src;
      *(_QWORD *)v147 = 0x223D6C6562616C09LL;
      v147 += 8;
    }
    sub_16BE9B0((__int64 *)&v138, (__int64)&v118);
    v97 = (size_t)v139;
    v98 = v138;
    v99 = (__int64)v23;
    goto LABEL_218;
  }
LABEL_38:
  if ( !v133 )
    goto LABEL_39;
  v101 = sub_1263B40((__int64)&p_src, "\tlabel=\"");
  sub_16BE9B0((__int64 *)&v138, (__int64)&v132);
  v97 = (size_t)v139;
  v98 = v138;
  v99 = v101;
LABEL_218:
  v100 = sub_16E7EE0(v99, v98, v97);
  sub_1263B40(v100, "\";\n");
  if ( v138 != (char *)&v140 )
    j_j___libc_free_0(v138, v140 + 1);
LABEL_39:
  v138 = (char *)&v140;
  sub_17E2210((__int64 *)&v138, byte_3F871B3, (__int64)byte_3F871B3);
  v19 = v138;
  sub_16E7EE0((__int64)&p_src, v138, (size_t)v139);
  if ( v138 != (char *)&v140 )
  {
    v19 = v140 + 1;
    j_j___libc_free_0(v138, v140 + 1);
  }
  if ( src == v147 )
  {
    v19 = "\n";
    sub_16E7EE0((__int64)&p_src, "\n", 1u);
  }
  else
  {
    *v147++ = 10;
  }
  if ( v132 != (char *)&v134 )
  {
    v19 = (char *)(v134 + 1);
    j_j___libc_free_0(v132, v134 + 1);
  }
  v25 = **a2;
  v104 = v25 + 72;
  v107 = *(_QWORD *)(v25 + 80);
  if ( v25 + 72 == v107 )
    goto LABEL_141;
  do
  {
    v26 = v107 - 24;
    if ( !v107 )
      v26 = 0;
    v108 = v26;
    v27 = v26;
    v121 = (char *)v123;
    sub_17E2210((__int64 *)&v121, byte_3F871B3, (__int64)byte_3F871B3);
    v28 = sub_1263B40((__int64)&p_src, "\tNode");
    v29 = sub_16E7B40(v28, v27);
    sub_1263B40(v29, " [shape=record,");
    if ( v122 )
    {
      v85 = sub_16E7EE0((__int64)&p_src, v121, v122);
      sub_1263B40(v85, ",");
    }
    sub_1263B40((__int64)&p_src, "label=\"{");
    v124[1] = 0;
    LOBYTE(v125[0]) = 0;
    v136 = 1;
    v30 = *a2;
    v135 = 0;
    v134 = 0;
    v124[0] = v125;
    v133 = 0;
    v132 = (char *)&unk_49EFBE0;
    v137 = v124;
    sub_1649960(v108);
    if ( v31 )
    {
      v33 = (char *)sub_1649960(v108);
      if ( v33 )
      {
        v126 = (char *)v128;
        sub_17E2210((__int64 *)&v126, v33, (__int64)&v33[v32]);
        v34 = v127;
        v35 = v126;
      }
      else
      {
        LOBYTE(v128[0]) = 0;
        v34 = 0;
        v127 = 0;
        v126 = (char *)v128;
        v35 = (char *)v128;
      }
    }
    else
    {
      v130 = 0;
      v143 = (char **)&v129;
      v129 = v131;
      LOBYTE(v131[0]) = 0;
      v142 = 1;
      v141 = 0;
      v140 = 0;
      v139 = 0;
      v138 = (char *)&unk_49EFBE0;
      sub_15537D0(v108, (__int64)&v138, 0, 0);
      if ( v141 != v139 )
        sub_16E7BA0((__int64 *)&v138);
      v126 = (char *)v128;
      sub_17E2330((__int64 *)&v126, *v143, (__int64)&v143[1][(_QWORD)*v143]);
      sub_16E7BC0((__int64 *)&v138);
      if ( v129 != v131 )
        j_j___libc_free_0(v129, v131[0] + 1LL);
      v34 = v127;
      v35 = v126;
    }
    v36 = sub_16E7EE0((__int64)&v132, v35, v34);
    v37 = *(_QWORD *)(v36 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v36 + 16) - v37) <= 2 )
    {
      sub_16E7EE0(v36, ":\\l", 3u);
    }
    else
    {
      *(_BYTE *)(v37 + 2) = 108;
      *(_WORD *)v37 = 23610;
      *(_QWORD *)(v36 + 24) += 3LL;
    }
    if ( v126 != (char *)v128 )
      j_j___libc_free_0(v126, v128[0] + 1LL);
    v38 = *((unsigned int *)v30 + 74);
    v39 = 0;
    if ( !(_DWORD)v38 )
      goto LABEL_61;
    v40 = v30[35];
    v41 = (v38 - 1) & (((unsigned int)v108 >> 9) ^ ((unsigned int)v108 >> 4));
    v42 = (__int64 *)(v40 + 16LL * v41);
    v43 = *v42;
    if ( v108 != *v42 )
    {
      v96 = 1;
      while ( v43 != -8 )
      {
        v102 = v96 + 1;
        v41 = (v38 - 1) & (v96 + v41);
        v42 = (__int64 *)(v40 + 16LL * v41);
        v43 = *v42;
        if ( v108 == *v42 )
          goto LABEL_59;
        v96 = v102;
      }
LABEL_210:
      v39 = 0;
      goto LABEL_61;
    }
LABEL_59:
    if ( v42 == (__int64 *)(v40 + 16 * v38) )
      goto LABEL_210;
    v39 = v42[1];
LABEL_61:
    if ( (unsigned __int64)(v134 - (_QWORD)v135) <= 7 )
    {
      sub_16E7EE0((__int64)&v132, "Count : ", 8u);
    }
    else
    {
      v135->m128i_i64[0] = 0x203A20746E756F43LL;
      v135 = (__m128i *)((char *)v135 + 8);
    }
    if ( v39 && *(_BYTE *)(v39 + 24) )
    {
      v86 = sub_16E7A90((__int64)&v132, *(_QWORD *)(v39 + 16));
      sub_1263B40(v86, "\\l");
    }
    else
    {
      v44 = (__int64 *)v135;
      if ( (unsigned __int64)(v134 - (_QWORD)v135) <= 8 )
      {
        sub_16E7EE0((__int64)&v132, "Unknown\\l", 9u);
      }
      else
      {
        v135->m128i_i8[8] = 108;
        *v44 = 0x5C6E776F6E6B6E55LL;
        v135 = (__m128i *)((char *)v135 + 9);
      }
    }
    if ( byte_4FA52E0 )
    {
      v45 = *(_QWORD *)(v108 + 48);
      v46 = v108 + 40;
      if ( v45 != v108 + 40 )
      {
        while ( 1 )
        {
          if ( !v45 )
            BUG();
          if ( *(_BYTE *)(v45 - 8) != 79 )
            goto LABEL_70;
          v47 = v135;
          if ( (unsigned __int64)(v134 - (_QWORD)v135) <= 0xE )
          {
            sub_16E7EE0((__int64)&v132, "SELECT : { T = ", 0xFu);
          }
          else
          {
            v135->m128i_i64[0] = 0x3A205443454C4553LL;
            v47->m128i_i32[2] = 1411414816;
            v47->m128i_i16[6] = 15648;
            v47->m128i_i8[14] = 32;
            v135 = (__m128i *)((char *)v135 + 15);
          }
          if ( (unsigned __int8)sub_1625AE0(v45 - 24, &v129, &v138) )
            break;
          v48 = v135;
          if ( (unsigned __int64)(v134 - (_QWORD)v135) <= 0x17 )
          {
            sub_16E7EE0((__int64)&v132, "Unknown, F = Unknown }\\l", 0x18u);
LABEL_70:
            v45 = *(_QWORD *)(v45 + 8);
            if ( v45 == v46 )
              goto LABEL_78;
          }
          else
          {
            v49 = _mm_load_si128((const __m128i *)&xmmword_42B6820);
            v135[1].m128i_i64[0] = 0x6C5C7D206E776F6ELL;
            *v48 = v49;
            v135 = (__m128i *)((char *)v135 + 24);
            v45 = *(_QWORD *)(v45 + 8);
            if ( v45 == v46 )
              goto LABEL_78;
          }
        }
        v76 = sub_16E7A90((__int64)&v132, (__int64)v129);
        v77 = *(_QWORD *)(v76 + 24);
        v78 = v76;
        if ( (unsigned __int64)(*(_QWORD *)(v76 + 16) - v77) <= 5 )
        {
          v78 = sub_16E7EE0(v76, ", F = ", 6u);
        }
        else
        {
          *(_DWORD *)v77 = 541466668;
          *(_WORD *)(v77 + 4) = 8253;
          *(_QWORD *)(v76 + 24) += 6LL;
        }
        v79 = sub_16E7A90(v78, (__int64)v138);
        v80 = *(_DWORD **)(v79 + 24);
        if ( *(_QWORD *)(v79 + 16) - (_QWORD)v80 <= 3u )
        {
          sub_16E7EE0(v79, " }\\l", 4u);
        }
        else
        {
          *v80 = 1818000672;
          *(_QWORD *)(v79 + 24) += 4LL;
        }
        goto LABEL_70;
      }
    }
LABEL_78:
    sub_16E7BC0((__int64 *)&v132);
    sub_16BE9B0((__int64 *)&v138, (__int64)v124);
    sub_16E7EE0((__int64)&p_src, v138, (size_t)v139);
    if ( v138 != (char *)&v140 )
      j_j___libc_free_0(v138, v140 + 1);
    if ( (_QWORD *)v124[0] != v125 )
      j_j___libc_free_0(v124[0], v125[0] + 1LL);
    v129 = v131;
    sub_17E2210((__int64 *)&v129, byte_3F871B3, (__int64)byte_3F871B3);
    if ( v130 )
    {
      v84 = sub_1263B40((__int64)&p_src, "|");
      sub_16BE9B0((__int64 *)&v138, (__int64)&v129);
      sub_16E7EE0(v84, v138, (size_t)v139);
      if ( v138 != (char *)&v140 )
        j_j___libc_free_0(v138, v140 + 1);
    }
    v132 = (char *)&v134;
    sub_17E2210((__int64 *)&v132, byte_3F871B3, (__int64)byte_3F871B3);
    if ( v133 )
    {
      v83 = sub_1263B40((__int64)&p_src, "|");
      sub_16BE9B0((__int64 *)&v138, (__int64)&v132);
      sub_16E7EE0(v83, v138, (size_t)v139);
      if ( v138 != (char *)&v140 )
        j_j___libc_free_0(v138, v140 + 1);
    }
    if ( v132 != (char *)&v134 )
      j_j___libc_free_0(v132, v134 + 1);
    if ( v129 != v131 )
      j_j___libc_free_0(v129, v131[0] + 1LL);
    v127 = 0;
    LOBYTE(v128[0]) = 0;
    v126 = (char *)v128;
    v142 = 1;
    v141 = 0;
    v140 = 0;
    v139 = 0;
    v138 = (char *)&unk_49EFBE0;
    v143 = &v126;
    v50 = sub_157EBA0(v108);
    if ( !v50 )
      goto LABEL_111;
    v51 = sub_15F4D60(v50);
    v105 = v51;
    if ( !v51 )
      goto LABEL_111;
    v52 = 0;
    v53 = 0;
    v109 = (unsigned int)(v51 - 1);
    do
    {
      v129 = v131;
      sub_17E2210((__int64 *)&v129, byte_3F871B3, (__int64)byte_3F871B3);
      if ( v130 )
      {
        v57 = v141;
        if ( (_DWORD)v52 )
        {
          if ( v141 != v140 )
          {
            *v141 = 124;
            v57 = v141 + 1;
            v141 = v57;
            if ( (unsigned __int64)(v140 - (_BYTE *)v57) <= 1 )
              goto LABEL_106;
            goto LABEL_92;
          }
          sub_16E7EE0((__int64)&v138, "|", 1u);
          v57 = v141;
        }
        if ( (unsigned __int64)(v140 - (_BYTE *)v57) <= 1 )
        {
LABEL_106:
          v54 = (char **)sub_16E7EE0((__int64)&v138, "<s", 2u);
          goto LABEL_93;
        }
LABEL_92:
        v54 = &v138;
        *v57 = 29500;
        v141 += 2;
LABEL_93:
        v55 = sub_16E7A90((__int64)v54, v52);
        v56 = *(_BYTE **)(v55 + 24);
        if ( *(_BYTE **)(v55 + 16) == v56 )
        {
          v55 = sub_16E7EE0(v55, ">", 1u);
        }
        else
        {
          *v56 = 62;
          ++*(_QWORD *)(v55 + 24);
        }
        sub_16BE9B0((__int64 *)&v132, (__int64)&v129);
        sub_16E7EE0(v55, v132, v133);
        if ( v132 != (char *)&v134 )
          j_j___libc_free_0(v132, v134 + 1);
        if ( v129 != v131 )
          j_j___libc_free_0(v129, v131[0] + 1LL);
        v53 = 1;
LABEL_100:
        if ( v52 == v109 )
          goto LABEL_110;
        goto LABEL_101;
      }
      if ( v129 == v131 )
        goto LABEL_100;
      j_j___libc_free_0(v129, v131[0] + 1LL);
      if ( v52 == v109 )
        goto LABEL_110;
LABEL_101:
      ++v52;
    }
    while ( v52 != 64 );
    if ( v105 == 64 )
    {
LABEL_110:
      if ( !v53 )
        goto LABEL_111;
      goto LABEL_163;
    }
    if ( !v53 )
      goto LABEL_111;
    sub_1263B40((__int64)&v138, "|<s64>truncated...");
LABEL_163:
    sub_1263B40((__int64)&p_src, "|");
    v81 = sub_1263B40((__int64)&p_src, "{");
    if ( v141 != v139 )
      sub_16E7BA0((__int64 *)&v138);
    v82 = sub_16E7EE0(v81, *v143, (size_t)v143[1]);
    sub_1263B40(v82, "}");
LABEL_111:
    v19 = "}\"];\n";
    sub_1263B40((__int64)&p_src, "}\"];\n");
    v58 = sub_157EBA0(v108);
    v59 = v58;
    if ( v58 )
    {
      v60 = sub_15F4D60(v58);
      if ( v60 )
      {
        v110 = v60;
        v61 = 0;
        while ( 1 )
        {
          sub_15F4DF0(v59, v61);
          v19 = (char *)v61;
          v64 = sub_15F4DF0(v59, v61);
          if ( v64 )
          {
            v132 = (char *)&v134;
            sub_17E2210((__int64 *)&v132, byte_3F871B3, (__int64)byte_3F871B3);
            v65 = v133;
            if ( v132 != (char *)&v134 )
              j_j___libc_free_0(v132, v134 + 1);
            v66 = v65 == 0;
            v67 = -1;
            if ( !v66 )
              v67 = v61;
            v132 = (char *)&v134;
            sub_17E2210((__int64 *)&v132, byte_3F871B3, (__int64)byte_3F871B3);
            v68 = v147;
            if ( (unsigned __int64)(src - v147) > 4 )
            {
              *(_DWORD *)v147 = 1685016073;
              v62 = (__int64 *)&p_src;
              v68[4] = 101;
              v147 += 5;
            }
            else
            {
              v62 = (__int64 *)sub_16E7EE0((__int64)&p_src, "\tNode", 5u);
            }
            sub_16E7B40((__int64)v62, v108);
            if ( v67 != -1 )
            {
              if ( (unsigned __int64)(src - v147) <= 1 )
              {
                v69 = (__int64 *)sub_16E7EE0((__int64)&p_src, ":s", 2u);
              }
              else
              {
                *(_WORD *)v147 = 29498;
                v69 = (__int64 *)&p_src;
                v147 += 2;
              }
              sub_16E7AB0((__int64)v69, v67);
            }
            if ( (unsigned __int64)(src - v147) <= 7 )
            {
              v63 = (__int64 *)sub_16E7EE0((__int64)&p_src, " -> Node", 8u);
            }
            else
            {
              v63 = (__int64 *)&p_src;
              *(_QWORD *)v147 = 0x65646F4E203E2D20LL;
              v147 += 8;
            }
            sub_16E7B40((__int64)v63, v64);
            if ( v133 )
            {
              if ( src == v147 )
              {
                v73 = (__int64 *)sub_16E7EE0((__int64)&p_src, "[", 1u);
              }
              else
              {
                *v147 = 91;
                v73 = (__int64 *)&p_src;
                ++v147;
              }
              v74 = sub_16E7EE0((__int64)v73, v132, v133);
              v75 = *(_BYTE **)(v74 + 24);
              if ( *(_BYTE **)(v74 + 16) == v75 )
              {
                sub_16E7EE0(v74, "]", 1u);
              }
              else
              {
                *v75 = 93;
                ++*(_QWORD *)(v74 + 24);
              }
            }
            if ( (unsigned __int64)(src - v147) <= 1 )
            {
              v19 = ";\n";
              sub_16E7EE0((__int64)&p_src, ";\n", 2u);
            }
            else
            {
              v19 = (char *)2619;
              *(_WORD *)v147 = 2619;
              v147 += 2;
            }
            if ( v132 != (char *)&v134 )
            {
              v19 = (char *)(v134 + 1);
              j_j___libc_free_0(v132, v134 + 1);
            }
          }
          if ( ++v61 == v110 )
            break;
          if ( v61 == 64 )
          {
            if ( v110 != 64 )
            {
              do
              {
                sub_15F4DF0(v59, v61);
                v19 = (char *)v61;
                v90 = sub_15F4DF0(v59, v61);
                if ( v90 )
                {
                  v132 = (char *)&v134;
                  sub_17E2210((__int64 *)&v132, byte_3F871B3, (__int64)byte_3F871B3);
                  v91 = v133;
                  if ( v132 != (char *)&v134 )
                    j_j___libc_free_0(v132, v134 + 1);
                  v132 = (char *)&v134;
                  sub_17E2210((__int64 *)&v132, byte_3F871B3, (__int64)byte_3F871B3);
                  v92 = v147;
                  v93 = v91 == 0 ? -1 : 0x40;
                  if ( (unsigned __int64)(src - v147) > 4 )
                  {
                    *(_DWORD *)v147 = 1685016073;
                    v87 = (__int64 *)&p_src;
                    v92[4] = 101;
                    v147 += 5;
                  }
                  else
                  {
                    v87 = (__int64 *)sub_16E7EE0((__int64)&p_src, "\tNode", 5u);
                  }
                  sub_16E7B40((__int64)v87, v108);
                  if ( v93 != -1 )
                  {
                    v88 = sub_1263B40((__int64)&p_src, ":s");
                    sub_16E7AB0(v88, 64);
                  }
                  if ( (unsigned __int64)(src - v147) <= 7 )
                  {
                    v89 = (__int64 *)sub_16E7EE0((__int64)&p_src, " -> Node", 8u);
                  }
                  else
                  {
                    v89 = (__int64 *)&p_src;
                    *(_QWORD *)v147 = 0x65646F4E203E2D20LL;
                    v147 += 8;
                  }
                  v19 = (char *)v90;
                  sub_16E7B40((__int64)v89, v90);
                  if ( v133 )
                  {
                    v94 = sub_1263B40((__int64)&p_src, "[");
                    v95 = sub_16E7EE0(v94, v132, v133);
                    v19 = "]";
                    sub_1263B40(v95, "]");
                  }
                  if ( (unsigned __int64)(src - v147) <= 1 )
                  {
                    v19 = ";\n";
                    sub_16E7EE0((__int64)&p_src, ";\n", 2u);
                  }
                  else
                  {
                    *(_WORD *)v147 = 2619;
                    v147 += 2;
                  }
                  if ( v132 != (char *)&v134 )
                  {
                    v19 = (char *)(v134 + 1);
                    j_j___libc_free_0(v132, v134 + 1);
                  }
                }
                ++v61;
              }
              while ( v61 != v110 );
            }
            break;
          }
        }
      }
    }
    sub_16E7BC0((__int64 *)&v138);
    if ( v126 != (char *)v128 )
    {
      v19 = (char *)(v128[0] + 1LL);
      j_j___libc_free_0(v126, v128[0] + 1LL);
    }
    if ( v121 != (char *)v123 )
    {
      v19 = (char *)(v123[0] + 1LL);
      j_j___libc_free_0(v121, v123[0] + 1LL);
    }
    v107 = *(_QWORD *)(v107 + 8);
  }
  while ( v104 != v107 );
LABEL_141:
  if ( (unsigned __int64)(src - v147) <= 1 )
  {
    v19 = "}\n";
    sub_16E7EE0((__int64)&p_src, "}\n", 2u);
  }
  else
  {
    *(_WORD *)v147 = 2685;
    v147 += 2;
  }
  if ( v118 != &v120 )
  {
    v19 = (char *)(v120 + 1);
    j_j___libc_free_0(v118, v120 + 1);
  }
  v70 = sub_16E8CB0();
  v71 = (_QWORD *)v70[3];
  if ( v70[2] - (_QWORD)v71 <= 7u )
  {
    v19 = " done. \n";
    sub_16E7EE0((__int64)v70, " done. \n", 8u);
  }
  else
  {
    *v71 = 0xA202E656E6F6420LL;
    v70[3] += 8LL;
  }
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( v115 == &v117 )
  {
    a1[1] = _mm_load_si128(&v117);
  }
  else
  {
    a1->m128i_i64[0] = (__int64)v115;
    a1[1].m128i_i64[0] = v117.m128i_i64[0];
  }
  v72 = v116;
  v115 = &v117;
  v116 = 0;
  a1->m128i_i64[1] = v72;
  v117.m128i_i8[0] = 0;
LABEL_20:
  sub_16E7C30((int *)&p_src, (__int64)v19);
  if ( v115 != &v117 )
    j_j___libc_free_0(v115, v117.m128i_i64[0] + 1);
  if ( dest != v114 )
    j_j___libc_free_0(dest, v114[0] + 1LL);
  return a1;
}
