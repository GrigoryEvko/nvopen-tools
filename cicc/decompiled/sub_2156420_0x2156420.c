// Function: sub_2156420
// Address: 0x2156420
//
char __fastcall sub_2156420(__int64 a1, _BYTE *a2, __int64 a3, char a4)
{
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rdx
  __int64 v11; // r15
  __int64 v12; // r14
  _BYTE *v13; // rdi
  __int64 v14; // r12
  const char *v15; // rax
  size_t v16; // rdx
  __int64 v17; // rax
  _QWORD *v18; // rax
  __int64 v19; // r8
  _QWORD *v20; // rdi
  unsigned __int64 v21; // rdx
  __int64 v22; // r12
  const char *v23; // rax
  size_t v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  char v27; // al
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // r14
  int v31; // eax
  unsigned __int64 v32; // rcx
  __int8 *v33; // r8
  const char *v34; // rax
  void *v35; // rdx
  const void *v36; // r8
  size_t v37; // r15
  _QWORD *v38; // rdi
  __m128i *v39; // rax
  __int64 v40; // rcx
  __m128i *v41; // rax
  __int64 v42; // rsi
  __m128i *v43; // r9
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // rcx
  __int64 v46; // rax
  void *v47; // rcx
  __int64 v48; // rdx
  __m128i *v49; // rax
  __int64 v50; // rcx
  __int64 v51; // rbx
  int v52; // eax
  unsigned __int64 v53; // rcx
  int v54; // eax
  __int64 v55; // rax
  __int64 v56; // r13
  const char *v57; // rax
  size_t v58; // rdx
  __int64 v59; // rax
  _QWORD *v60; // r13
  unsigned int v61; // r15d
  __int64 v62; // rbx
  int v63; // r13d
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  bool v68; // zf
  __int64 v69; // rcx
  __int64 v70; // rdx
  __int64 v71; // r15
  char *v72; // rbx
  char v73; // al
  size_t v74; // rcx
  char v75; // r14
  char v76; // al
  __int64 v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rdx
  _QWORD *v83; // r12
  char *v84; // r15
  char *v85; // r13
  _BYTE *v86; // rdi
  size_t v87; // r14
  _BYTE *v88; // r8
  size_t v89; // rdx
  _BYTE *v90; // rsi
  unsigned __int64 v91; // rax
  _QWORD *v92; // r13
  char *v93; // rsi
  char *v94; // rbx
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // rax
  __int64 v98; // r14
  __int64 v99; // rax
  __int64 v100; // rax
  size_t v102; // [rsp+0h] [rbp-1D0h]
  size_t n; // [rsp+18h] [rbp-1B8h]
  size_t na; // [rsp+18h] [rbp-1B8h]
  __int64 v105; // [rsp+20h] [rbp-1B0h]
  __int64 v106; // [rsp+20h] [rbp-1B0h]
  _BYTE *v107[2]; // [rsp+28h] [rbp-1A8h] BYREF
  void *v108; // [rsp+38h] [rbp-198h] BYREF
  _QWORD v109[2]; // [rsp+40h] [rbp-190h] BYREF
  __m128i v110; // [rsp+50h] [rbp-180h] BYREF
  _QWORD *v111; // [rsp+60h] [rbp-170h] BYREF
  void *v112; // [rsp+68h] [rbp-168h]
  _QWORD dest[2]; // [rsp+70h] [rbp-160h] BYREF
  __m128i *v114; // [rsp+80h] [rbp-150h] BYREF
  __int64 v115; // [rsp+88h] [rbp-148h]
  __m128i v116; // [rsp+90h] [rbp-140h] BYREF
  __m128i *v117; // [rsp+A0h] [rbp-130h] BYREF
  __int64 v118; // [rsp+A8h] [rbp-128h]
  __m128i v119; // [rsp+B0h] [rbp-120h] BYREF
  const char *v120; // [rsp+C0h] [rbp-110h] BYREF
  __int64 v121; // [rsp+C8h] [rbp-108h]
  _QWORD v122[2]; // [rsp+D0h] [rbp-100h] BYREF
  void *src; // [rsp+E0h] [rbp-F0h] BYREF
  char *v124; // [rsp+E8h] [rbp-E8h]
  __m128i v125; // [rsp+F0h] [rbp-E0h] BYREF
  _BYTE *v126; // [rsp+100h] [rbp-D0h]
  __int64 v127; // [rsp+108h] [rbp-C8h]
  _BYTE v128[16]; // [rsp+110h] [rbp-C0h] BYREF
  _BYTE *v129; // [rsp+120h] [rbp-B0h]
  __int64 v130; // [rsp+128h] [rbp-A8h]
  _BYTE v131[32]; // [rsp+130h] [rbp-A0h] BYREF
  _BYTE *v132; // [rsp+150h] [rbp-80h]
  __int64 v133; // [rsp+158h] [rbp-78h]
  _BYTE v134[32]; // [rsp+160h] [rbp-70h] BYREF
  int v135; // [rsp+180h] [rbp-50h]
  __int64 v136; // [rsp+188h] [rbp-48h]
  __int64 v137; // [rsp+190h] [rbp-40h]
  char v138; // [rsp+198h] [rbp-38h]
  char v139; // [rsp+199h] [rbp-37h]

  v107[0] = a2;
  if ( (a2[34] & 0x20) == 0 )
  {
    v7 = (unsigned __int64)sub_1649960((__int64)a2);
    if ( v21 <= 4 || *(_DWORD *)v7 != 1836477548 )
      goto LABEL_5;
    goto LABEL_35;
  }
  v7 = sub_15E61A0((__int64)a2);
  if ( v8 == 13
    && *(_QWORD *)v7 == 0x74656D2E6D766C6CLL
    && *(_DWORD *)(v7 + 8) == 1952539745
    && *(_BYTE *)(v7 + 12) == 97 )
  {
    return v7;
  }
  v7 = (unsigned __int64)sub_1649960((__int64)v107[0]);
  if ( v9 > 4 && *(_DWORD *)v7 == 1836477548 )
  {
LABEL_35:
    if ( *(_BYTE *)(v7 + 4) == 46 )
      return v7;
  }
LABEL_5:
  v7 = (unsigned __int64)sub_1649960((__int64)v107[0]);
  if ( v10 > 4 && *(_DWORD *)v7 == 1836480110 && *(_BYTE *)(v7 + 4) == 46 )
    return v7;
  v105 = sub_396DDB0(a1);
  v11 = *(_QWORD *)v107[0];
  v12 = *((_QWORD *)v107[0] + 3);
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 232) + 952LL) != 1 )
  {
    if ( !(unsigned __int8)sub_1C2E830((__int64)v107[0]) )
      goto LABEL_9;
LABEL_28:
    v22 = sub_1263B40(a3, ".global .texref ");
    v23 = sub_1CCA9F0((__int64)v107[0]);
LABEL_29:
    v25 = sub_1549FF0(v22, v23, v24);
    LOBYTE(v7) = sub_1263B40(v25, ";\n");
    return v7;
  }
  sub_214CAD0(v107[0], a3);
  if ( (unsigned __int8)sub_1C2E830((__int64)v107[0]) )
    goto LABEL_28;
LABEL_9:
  if ( (unsigned __int8)sub_1C2E860((__int64)v107[0]) )
  {
    v22 = sub_1263B40(a3, ".global .surfref ");
    v23 = sub_1CCAA00((__int64)v107[0]);
    goto LABEL_29;
  }
  if ( sub_15E4F60((__int64)v107[0]) )
  {
    sub_214FEE0(a1, (__int64)v107[0], a3);
    LOBYTE(v7) = sub_1263B40(a3, ";\n");
    return v7;
  }
  if ( (unsigned __int8)sub_1C2E890((__int64)v107[0]) )
  {
    v56 = sub_1263B40(a3, ".global .samplerref ");
    v57 = sub_1CCAA10((__int64)v107[0]);
    sub_1549FF0(v56, v57, v58);
    if ( !sub_15E4F60((__int64)v107[0]) )
    {
      v59 = *((_QWORD *)v107[0] - 3);
      if ( v59 )
      {
        if ( *(_BYTE *)(v59 + 16) == 13 )
        {
          v60 = *(_QWORD **)(v59 + 24);
          if ( *(_DWORD *)(v59 + 32) > 0x40u )
            v60 = (_QWORD *)*v60;
          v61 = (unsigned int)v60;
          v62 = 0;
          sub_1263B40(a3, " = { ");
          v63 = (unsigned __int8)v60 & 7;
          do
          {
            v64 = sub_1263B40(a3, "addr_mode_");
            v65 = sub_16E7AB0(v64, v62);
            sub_1263B40(v65, " = ");
            switch ( v63 )
            {
              case 0:
              case 3:
                sub_1263B40(a3, "wrap");
                break;
              case 1:
                sub_1263B40(a3, "clamp_to_border");
                break;
              case 2:
                sub_1263B40(a3, "clamp_to_edge");
                break;
              case 4:
                sub_1263B40(a3, "mirror");
                break;
              default:
                break;
            }
            ++v62;
            sub_1263B40(a3, ", ");
          }
          while ( v62 != 3 );
          sub_1263B40(a3, "filter_mode = ");
          if ( ((v61 >> 4) & 3) == 1 )
            sub_1263B40(a3, "linear");
          else
            sub_1263B40(a3, "nearest");
          if ( (v61 & 8) == 0 )
            sub_1263B40(a3, ", force_unnormalized_coords = 1");
          sub_1263B40(a3, " }");
        }
      }
    }
    goto LABEL_78;
  }
  v13 = v107[0];
  if ( (v107[0][32] & 0xF) == 8 )
  {
    v68 = memcmp(sub_1649960((__int64)v107[0]), "unrollpragma", 0xCu) == 0;
    LOBYTE(v7) = !v68;
    if ( v68 )
      return v7;
    v68 = memcmp(sub_1649960((__int64)v107[0]), "filename", 8u) == 0;
    LOBYTE(v7) = !v68;
    if ( v68 )
      return v7;
    v13 = v107[0];
    if ( !*((_QWORD *)v107[0] + 1) )
      return v7;
  }
  v108 = 0;
  if ( a4
    || (v13[32] & 0xF) != 7
    || *(_DWORD *)(*(_QWORD *)v13 + 8LL) >> 8 != 3
    || (src = 0, !(unsigned __int8)sub_214ACB0((__int64)v13, &src))
    || !src )
  {
    sub_1263B40(a3, ".");
    sub_214FA80(a1, *(_DWORD *)(v11 + 8) >> 8, a3);
    if ( (unsigned __int8)sub_1C2E7D0((__int64)v107[0]) )
      sub_1263B40(a3, " .attribute(.managed)");
    if ( (unsigned __int8)sub_1C2E800((__int64)v107[0]) )
    {
      v124 = 0;
      src = &v125;
      v125.m128i_i8[0] = 0;
      if ( (unsigned __int8)sub_1C2FA30((__int64)v107[0], &v120) )
      {
        v66 = sub_1263B40(a3, " .attribute(.unified(");
        v67 = sub_16E7A90(v66, (__int64)v120);
        sub_1263B40(v67, "))");
      }
      else if ( (unsigned __int8)sub_1C2F120((__int64)v107[0], &src) )
      {
        v80 = sub_1263B40(a3, " .attribute(.unified(");
        v81 = sub_16E7EE0(v80, (char *)src, (size_t)v124);
        sub_1263B40(v81, "))");
      }
      else
      {
        sub_1263B40(a3, " .attribute(.unified)");
      }
      sub_2240A30(&src);
    }
    if ( (unsigned int)(1 << (*((_DWORD *)v107[0] + 8) >> 15)) >> 1 )
    {
      v26 = sub_1263B40(a3, " .align ");
      sub_16E7A90(v26, (unsigned int)(1 << (*((_DWORD *)v107[0] + 8) >> 15)) >> 1);
    }
    else
    {
      v51 = sub_1263B40(a3, " .align ");
      v52 = sub_15AAE50(v105, v12);
      sub_16E7AB0(v51, v52);
    }
    v27 = *(_BYTE *)(v12 + 8);
    if ( (unsigned __int8)(v27 - 1) > 5u && v27 != 15 && (v27 != 11 || (unsigned int)sub_16431D0(v12) > 0x40) )
    {
      v53 = sub_127FA20(v105, v12) + 7;
      v106 = v53 >> 3;
      v54 = *(_DWORD *)(v11 + 8) >> 8;
      if ( v54 != 4 && v54 != 1
        || (n = v53, sub_15E4F60((__int64)v107[0]))
        || (v71 = *((_QWORD *)v107[0] - 3), *(_BYTE *)(v71 + 16) == 9)
        || sub_1593BB0(*((_QWORD *)v107[0] - 3), v12, v70, n) )
      {
        sub_1263B40(a3, " .b8 ");
        v55 = sub_396EAF0(a1, v107[0]);
        sub_38E2490(v55, a3, *(_QWORD *)(a1 + 240));
        if ( v106 )
        {
          sub_1263B40(a3, "[");
          sub_16E7A90(a3, v106);
          sub_1263B40(a3, "]");
        }
      }
      else
      {
        v72 = 0;
        v73 = sub_1BF95F0(v12);
        v74 = n;
        v124 = 0;
        v125 = 0u;
        v75 = v73;
        HIDWORD(src) = v106;
        if ( (_DWORD)v106 )
        {
          v124 = (char *)sub_22077B0((unsigned int)v106);
          v72 = &v124[(unsigned int)v106];
          v125.m128i_i64[1] = (__int64)v72;
          memset(v124, 0, (unsigned int)v106);
          v74 = n;
        }
        v102 = v74;
        v126 = v128;
        v125.m128i_i64[0] = (__int64)v72;
        v127 = 0x400000000LL;
        v130 = 0x400000000LL;
        v133 = 0x400000000LL;
        v76 = *(_BYTE *)(a1 + 896);
        v129 = v131;
        v138 = v76;
        v132 = v134;
        v136 = a3;
        v137 = a1;
        v135 = 0;
        LODWORD(src) = 0;
        v139 = v75;
        sub_2152590(a1, v71, (__int64)&src);
        if ( (_DWORD)src )
        {
          if ( v75 )
          {
            v98 = sub_1263B40(a3, " .u8 ");
            v99 = sub_396EAF0(a1, v107[0]);
            sub_38E2490(v99, v98, 0);
            v100 = sub_1263B40(v98, "[");
            sub_16E7A90(v100, v106);
          }
          else if ( *(_BYTE *)(*(_QWORD *)(a1 + 232) + 936LL) )
          {
            sub_1263B40(a3, " .u64 ");
            v77 = sub_396EAF0(a1, v107[0]);
            sub_38E2490(v77, a3, *(_QWORD *)(a1 + 240));
            sub_1263B40(a3, "[");
            sub_16E7A90(a3, v102 >> 6);
          }
          else
          {
            sub_1263B40(a3, " .u32 ");
            v97 = sub_396EAF0(a1, v107[0]);
            sub_38E2490(v97, a3, *(_QWORD *)(a1 + 240));
            sub_1263B40(a3, "[");
            sub_16E7A90(a3, v102 >> 5);
          }
        }
        else
        {
          sub_1263B40(a3, " .b8 ");
          v95 = sub_396EAF0(a1, v107[0]);
          sub_38E2490(v95, a3, *(_QWORD *)(a1 + 240));
          sub_1263B40(a3, "[");
          sub_16E7A90(a3, v106);
        }
        sub_1263B40(a3, "]");
        sub_1263B40(a3, " = {");
        sub_2153AE0((__int64)&src, (__int64)" = {", v78, v79);
        sub_1263B40(a3, "}");
        if ( v132 != v134 )
          _libc_free((unsigned __int64)v132);
        if ( v129 != v131 )
          _libc_free((unsigned __int64)v129);
        if ( v126 != v128 )
          _libc_free((unsigned __int64)v126);
        if ( v124 )
          j_j___libc_free_0(v124, v125.m128i_i64[1] - (_QWORD)v124);
      }
      goto LABEL_78;
    }
    sub_1263B40(a3, " .");
    if ( sub_1642F90(v12, 1) )
    {
      sub_1263B40(a3, "u8");
    }
    else
    {
      sub_214FBF0((__int64)&src, a1, v12, 0);
      sub_16E7EE0(a3, (char *)src, (size_t)v124);
      sub_2240A30(&src);
    }
    sub_1263B40(a3, " ");
    v28 = sub_396EAF0(a1, v107[0]);
    sub_38E2490(v28, a3, *(_QWORD *)(a1 + 240));
    if ( sub_15E4F60((__int64)v107[0]) )
      goto LABEL_78;
    v30 = *((_QWORD *)v107[0] - 3);
    v31 = *(_DWORD *)(v11 + 8) >> 8;
    if ( v31 == 1 || v31 == 4 )
    {
      if ( !sub_1593BB0(v30, a3, (__int64)v107[0], v29) && *(_BYTE *)(v30 + 16) != 9 )
      {
        sub_1263B40(a3, " = ");
        sub_2154240(a1, v30, a3, v69);
      }
      goto LABEL_78;
    }
    if ( sub_1593BB0(v30, a3, (__int64)v107[0], v29) || *(_BYTE *)(*((_QWORD *)v107[0] - 3) + 16LL) == 9 )
    {
LABEL_78:
      LOBYTE(v7) = sub_1263B40(a3, ";\n");
      return v7;
    }
    v32 = *(_DWORD *)(v11 + 8) >> 8;
    if ( *(_DWORD *)(v11 + 8) >> 8 )
    {
      v33 = &v125.m128i_i8[5];
      do
      {
        *--v33 = v32 % 0xA + 48;
        v91 = v32;
        v32 /= 0xAu;
      }
      while ( v91 > 9 );
    }
    else
    {
      v125.m128i_i8[4] = 48;
      v33 = &v125.m128i_i8[4];
    }
    v120 = (const char *)v122;
    sub_214ADD0((__int64 *)&v120, v33, (__int64)v125.m128i_i64 + 5);
    v34 = sub_1649960((__int64)v107[0]);
    v36 = v34;
    v37 = (size_t)v35;
    if ( !v34 )
    {
      LOBYTE(dest[0]) = 0;
      v111 = dest;
      v112 = 0;
LABEL_61:
      v39 = (__m128i *)sub_2241130(&v111, 0, 0, "initial value of '", 18);
      v114 = &v116;
      if ( (__m128i *)v39->m128i_i64[0] == &v39[1] )
      {
        v116 = _mm_loadu_si128(v39 + 1);
      }
      else
      {
        v114 = (__m128i *)v39->m128i_i64[0];
        v116.m128i_i64[0] = v39[1].m128i_i64[0];
      }
      v40 = v39->m128i_i64[1];
      v39[1].m128i_i8[0] = 0;
      v115 = v40;
      v39->m128i_i64[0] = (__int64)v39[1].m128i_i64;
      v39->m128i_i64[1] = 0;
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v115) <= 0x1D )
        goto LABEL_182;
      v41 = (__m128i *)sub_2241490(&v114, "' is not allowed in addrspace(", 30);
      v117 = &v119;
      if ( (__m128i *)v41->m128i_i64[0] == &v41[1] )
      {
        v119 = _mm_loadu_si128(v41 + 1);
      }
      else
      {
        v117 = (__m128i *)v41->m128i_i64[0];
        v119.m128i_i64[0] = v41[1].m128i_i64[0];
      }
      v42 = v41->m128i_i64[1];
      v41[1].m128i_i8[0] = 0;
      v118 = v42;
      v41->m128i_i64[0] = (__int64)v41[1].m128i_i64;
      v43 = v117;
      v41->m128i_i64[1] = 0;
      v44 = 15;
      v45 = 15;
      if ( v43 != &v119 )
        v45 = v119.m128i_i64[0];
      if ( v118 + v121 <= v45 )
        goto LABEL_72;
      if ( v120 != (const char *)v122 )
        v44 = v122[0];
      if ( v118 + v121 <= v44 )
      {
        v46 = sub_2241400(&v120, 0, 0, v43, v118);
        src = &v125;
        v47 = *(void **)v46;
        v48 = v46 + 16;
        if ( *(_QWORD *)v46 != v46 + 16 )
          goto LABEL_73;
      }
      else
      {
LABEL_72:
        v46 = sub_2241490(&v117, v120, v121, v45, v118);
        src = &v125;
        v47 = *(void **)v46;
        v48 = v46 + 16;
        if ( *(_QWORD *)v46 != v46 + 16 )
        {
LABEL_73:
          src = v47;
          v125.m128i_i64[0] = *(_QWORD *)(v46 + 16);
          goto LABEL_74;
        }
      }
      v125 = _mm_loadu_si128((const __m128i *)(v46 + 16));
LABEL_74:
      v124 = *(char **)(v46 + 8);
      *(_QWORD *)v46 = v48;
      *(_QWORD *)(v46 + 8) = 0;
      *(_BYTE *)(v46 + 16) = 0;
      if ( v124 != (char *)0x3FFFFFFFFFFFFFFFLL )
      {
        v49 = (__m128i *)sub_2241490(&src, ")", 1);
        v109[0] = &v110;
        if ( (__m128i *)v49->m128i_i64[0] == &v49[1] )
        {
          v110 = _mm_loadu_si128(v49 + 1);
        }
        else
        {
          v109[0] = v49->m128i_i64[0];
          v110.m128i_i64[0] = v49[1].m128i_i64[0];
        }
        v50 = v49->m128i_i64[1];
        v49[1].m128i_i8[0] = 0;
        v109[1] = v50;
        v49->m128i_i64[0] = (__int64)v49[1].m128i_i64;
        v49->m128i_i64[1] = 0;
        sub_2240A30(&src);
        sub_2240A30(&v117);
        sub_2240A30(&v114);
        sub_2240A30(&v111);
        sub_2240A30(&v120);
        sub_1C3F040((__int64)v109);
        sub_2240A30(v109);
        goto LABEL_78;
      }
LABEL_182:
      sub_4262D8((__int64)"basic_string::append");
    }
    v38 = dest;
    src = v35;
    v111 = dest;
    if ( (unsigned __int64)v35 > 0xF )
    {
      na = (size_t)v34;
      v96 = sub_22409D0(&v111, &src, 0);
      v36 = (const void *)na;
      v111 = (_QWORD *)v96;
      v38 = (_QWORD *)v96;
      dest[0] = src;
    }
    else
    {
      if ( v35 == (void *)1 )
      {
        LOBYTE(dest[0]) = *v34;
LABEL_60:
        v112 = src;
        *((_BYTE *)src + (_QWORD)v111) = 0;
        goto LABEL_61;
      }
      if ( !v35 )
        goto LABEL_60;
    }
    memcpy(v38, v36, v37);
    goto LABEL_60;
  }
  v108 = src;
  v14 = sub_1263B40(a3, "// ");
  v15 = sub_1649960((__int64)v107[0]);
  v17 = sub_1549FF0(v14, v15, v16);
  sub_1263B40(v17, " has been demoted\n");
  v18 = *(_QWORD **)(a1 + 864);
  v19 = a1 + 856;
  v20 = (_QWORD *)(a1 + 856);
  if ( !v18 )
  {
    v92 = (_QWORD *)(a1 + 848);
LABEL_137:
    src = 0;
    v124 = 0;
    v125.m128i_i64[0] = 0;
    sub_2155140((__int64)&src, 0, v107);
    v7 = (unsigned __int64)sub_21563A0(v92, (unsigned __int64 *)&v108);
    v83 = (_QWORD *)v7;
    if ( (void **)v7 != &src )
    {
      v84 = v124;
      v85 = (char *)src;
      v86 = *(_BYTE **)v7;
      v87 = v124 - (_BYTE *)src;
      v7 = *(_QWORD *)(v7 + 16) - *(_QWORD *)v7;
      if ( v124 - (_BYTE *)src > v7 )
      {
        if ( v87 )
        {
          if ( v87 > 0x7FFFFFFFFFFFFFF8LL )
            sub_4261EA(v86, &v108, v82);
          v94 = (char *)sub_22077B0(v124 - (_BYTE *)src);
        }
        else
        {
          v94 = 0;
        }
        if ( v84 != v85 )
          memcpy(v94, v85, v87);
        if ( *v83 )
          j_j___libc_free_0(*v83, v83[2] - *v83);
        LOBYTE(v7) = (_BYTE)v94 + v87;
        *v83 = v94;
        v86 = v94;
        v83[2] = &v94[v87];
      }
      else
      {
        v88 = (_BYTE *)v83[1];
        v89 = v88 - v86;
        if ( v87 > v88 - v86 )
        {
          if ( v89 )
          {
            LOBYTE(v7) = (unsigned __int8)memmove(v86, src, v89);
            v84 = v124;
            v88 = (_BYTE *)v83[1];
            v85 = (char *)src;
          }
          v93 = &v88[(_QWORD)v85 - *v83];
          if ( v84 != v93 )
            LOBYTE(v7) = (unsigned __int8)memmove(v88, v93, v84 - v93);
          v86 = (_BYTE *)*v83;
        }
        else if ( v124 != src )
        {
          LOBYTE(v7) = (unsigned __int8)memmove(v86, src, v124 - (_BYTE *)src);
          v86 = (_BYTE *)*v83;
        }
      }
      v83[1] = &v86[v87];
    }
    if ( src )
      LOBYTE(v7) = j_j___libc_free_0(src, v125.m128i_i64[0] - (_QWORD)src);
    return v7;
  }
  do
  {
    if ( v18[4] < (unsigned __int64)v108 )
    {
      v18 = (_QWORD *)v18[3];
    }
    else
    {
      v20 = v18;
      v18 = (_QWORD *)v18[2];
    }
  }
  while ( v18 );
  v92 = (_QWORD *)(a1 + 848);
  if ( (_QWORD *)v19 == v20 || v20[4] > (unsigned __int64)v108 )
    goto LABEL_137;
  v7 = (unsigned __int64)sub_21563A0(v92, (unsigned __int64 *)&v108);
  v90 = *(_BYTE **)(v7 + 8);
  if ( v90 == *(_BYTE **)(v7 + 16) )
  {
    LOBYTE(v7) = (unsigned __int8)sub_2155140(v7, v90, v107);
  }
  else
  {
    if ( v90 )
    {
      *(_BYTE **)v90 = v107[0];
      v90 = *(_BYTE **)(v7 + 8);
    }
    *(_QWORD *)(v7 + 8) = v90 + 8;
  }
  return v7;
}
