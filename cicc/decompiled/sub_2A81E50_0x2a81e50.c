// Function: sub_2A81E50
// Address: 0x2a81e50
//
__int64 __fastcall sub_2A81E50(__int64 a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r15
  _QWORD *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  char *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rax
  unsigned int v22; // r12d
  char *v24; // rax
  unsigned __int64 v25; // rdx
  const void *v26; // r9
  size_t v27; // r8
  _QWORD *v28; // rax
  _QWORD *v29; // rdi
  __int64 v30; // r8
  char *v31; // rax
  unsigned __int64 v32; // rdx
  unsigned __int64 *v33; // r8
  size_t v34; // r14
  _QWORD *v35; // rax
  _QWORD *v36; // rdi
  __int64 v37; // r8
  char *v38; // rax
  unsigned __int64 v39; // rdx
  char *v40; // r9
  size_t v41; // r8
  __int64 *v42; // rax
  __int64 *v43; // rdi
  __int64 v44; // r8
  size_t v45; // r15
  size_t v46; // r13
  _BYTE *v47; // r12
  __int64 v48; // rax
  _QWORD *v49; // rbx
  __m128i *v50; // r8
  _BYTE *v51; // rdi
  _BYTE *v52; // rdi
  _QWORD *v53; // rax
  __int64 v54; // rax
  _QWORD *v55; // rdi
  size_t v56; // rdx
  __int64 v57; // rax
  _QWORD *v58; // rdi
  size_t v59; // rdx
  char *v60; // rax
  unsigned __int64 v61; // rdx
  unsigned __int64 *v62; // r8
  size_t v63; // r14
  _QWORD *v64; // rax
  _QWORD *v65; // rdi
  __int64 v66; // r8
  __int64 v67; // rax
  _QWORD *v68; // rdi
  size_t v69; // rdx
  __int64 v70; // rax
  _QWORD *v71; // rdi
  size_t v72; // rdx
  _BYTE *v73; // r12
  __int64 v74; // rax
  __m128i *v75; // r8
  _BYTE *v76; // rdi
  size_t v77; // r9
  _BYTE *v78; // rdi
  __int64 v79; // rax
  __m128i *v80; // rax
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rcx
  __int64 v86; // r8
  __int64 v87; // rax
  char *src; // [rsp+18h] [rbp-268h]
  size_t n; // [rsp+20h] [rbp-260h]
  char na; // [rsp+20h] [rbp-260h]
  size_t nb; // [rsp+20h] [rbp-260h]
  size_t nc; // [rsp+20h] [rbp-260h]
  size_t nd; // [rsp+20h] [rbp-260h]
  size_t ne; // [rsp+20h] [rbp-260h]
  bool v96; // [rsp+38h] [rbp-248h]
  size_t v97; // [rsp+38h] [rbp-248h]
  size_t v98; // [rsp+38h] [rbp-248h]
  __int64 ***v99; // [rsp+40h] [rbp-240h]
  void *v100; // [rsp+40h] [rbp-240h]
  void *v101; // [rsp+40h] [rbp-240h]
  __m128i *v102; // [rsp+40h] [rbp-240h]
  __m128i *dest; // [rsp+60h] [rbp-220h]
  unsigned __int64 v104; // [rsp+68h] [rbp-218h]
  _QWORD v105[2]; // [rsp+70h] [rbp-210h] BYREF
  void *v106; // [rsp+80h] [rbp-200h]
  size_t v107; // [rsp+88h] [rbp-1F8h]
  _QWORD v108[2]; // [rsp+90h] [rbp-1F0h] BYREF
  void *v109; // [rsp+A0h] [rbp-1E0h]
  size_t v110; // [rsp+A8h] [rbp-1D8h]
  _QWORD v111[2]; // [rsp+B0h] [rbp-1D0h] BYREF
  unsigned __int64 v112[2]; // [rsp+C0h] [rbp-1C0h] BYREF
  _QWORD v113[2]; // [rsp+D0h] [rbp-1B0h] BYREF
  __m128i *v114; // [rsp+E0h] [rbp-1A0h] BYREF
  size_t v115; // [rsp+E8h] [rbp-198h]
  _QWORD v116[2]; // [rsp+F0h] [rbp-190h] BYREF
  _QWORD *v117; // [rsp+100h] [rbp-180h] BYREF
  size_t v118; // [rsp+108h] [rbp-178h]
  _QWORD v119[2]; // [rsp+110h] [rbp-170h] BYREF
  _QWORD *v120; // [rsp+120h] [rbp-160h] BYREF
  size_t v121; // [rsp+128h] [rbp-158h]
  _QWORD v122[2]; // [rsp+130h] [rbp-150h] BYREF
  void *v123; // [rsp+140h] [rbp-140h] BYREF
  size_t v124; // [rsp+148h] [rbp-138h]
  _QWORD v125[2]; // [rsp+150h] [rbp-130h] BYREF
  _QWORD *v126; // [rsp+160h] [rbp-120h] BYREF
  size_t v127; // [rsp+168h] [rbp-118h]
  _QWORD v128[2]; // [rsp+170h] [rbp-110h] BYREF
  _QWORD *v129; // [rsp+180h] [rbp-100h] BYREF
  size_t v130; // [rsp+188h] [rbp-F8h]
  _BYTE v131[16]; // [rsp+190h] [rbp-F0h] BYREF
  __m128i *v132; // [rsp+1A0h] [rbp-E0h] BYREF
  size_t v133; // [rsp+1A8h] [rbp-D8h]
  __m128i v134; // [rsp+1B0h] [rbp-D0h] BYREF
  __int16 v135; // [rsp+1C0h] [rbp-C0h]
  unsigned __int64 v136[3]; // [rsp+1D0h] [rbp-B0h] BYREF
  _BYTE v137[40]; // [rsp+1E8h] [rbp-98h] BYREF
  unsigned __int64 v138[3]; // [rsp+210h] [rbp-70h] BYREF
  _BYTE v139[88]; // [rsp+228h] [rbp-58h] BYREF

  *(_BYTE *)(a4 + 76) = 0;
  dest = (__m128i *)v105;
  v106 = v108;
  v99 = (__int64 ***)a2;
  v104 = 0;
  LOBYTE(v105[0]) = 0;
  v107 = 0;
  LOBYTE(v108[0]) = 0;
  v109 = v111;
  v110 = 0;
  LOBYTE(v111[0]) = 0;
  sub_CAEB90(a4, (unsigned __int64)a2);
  v9 = *(_QWORD *)(a4 + 80);
  v96 = 0;
  if ( !v9 )
  {
LABEL_77:
    v45 = v107;
    if ( (v107 == 0) == (v110 == 0) )
    {
      v139[9] = 1;
      v138[0] = (unsigned __int64)"exactly one of transform or target must be specified";
      v139[8] = 3;
      sub_CA89D0(v99, a4, (__int64)v138, 0);
      goto LABEL_14;
    }
    v46 = v104;
    if ( v107 )
    {
      v47 = v106;
      v48 = sub_22077B0(0x50u);
      v49 = (_QWORD *)v48;
      if ( !v48 )
      {
LABEL_92:
        v22 = 1;
        v53 = (_QWORD *)sub_22077B0(0x18u);
        v53[2] = v49;
        sub_2208C80(v53, a5);
        ++*(_QWORD *)(a5 + 16);
        goto LABEL_15;
      }
      *(_DWORD *)(v48 + 8) = 1;
      *(_QWORD *)v48 = off_49D3DB8;
      v50 = dest;
      if ( v96 )
      {
        if ( dest )
        {
          v129 = v131;
          sub_2A7FC90((__int64 *)&v129, dest, (__int64)dest->m128i_i64 + v104);
        }
        else
        {
          v131[0] = 0;
          v129 = v131;
          v130 = 0;
        }
        v80 = (__m128i *)sub_2241130((unsigned __int64 *)&v129, 0, 0, &unk_3F871B2, 1u);
        v132 = &v134;
        if ( (__m128i *)v80->m128i_i64[0] == &v80[1] )
        {
          v134 = _mm_loadu_si128(v80 + 1);
        }
        else
        {
          v132 = (__m128i *)v80->m128i_i64[0];
          v134.m128i_i64[0] = v80[1].m128i_i64[0];
        }
        v133 = v80->m128i_u64[1];
        v80->m128i_i64[0] = (__int64)v80[1].m128i_i64;
        v80->m128i_i64[1] = 0;
        v80[1].m128i_i8[0] = 0;
        v50 = v132;
        v46 = v133;
      }
      v51 = v49 + 4;
      v49[2] = v49 + 4;
      if ( &v50->m128i_i8[v46] && !v50 )
        goto LABEL_194;
      v138[0] = v46;
      if ( v46 > 0xF )
      {
        v102 = v50;
        v83 = sub_22409D0((__int64)(v49 + 2), v138, 0);
        v50 = v102;
        v49[2] = v83;
        v51 = (_BYTE *)v83;
        v49[4] = v138[0];
      }
      else
      {
        if ( v46 == 1 )
        {
          *((_BYTE *)v49 + 32) = v50->m128i_i8[0];
          goto LABEL_86;
        }
        if ( !v46 )
        {
LABEL_86:
          v49[3] = v46;
          v51[v46] = 0;
          if ( v96 )
          {
            if ( v132 != &v134 )
              j_j___libc_free_0((unsigned __int64)v132);
            if ( v129 != (_QWORD *)v131 )
              j_j___libc_free_0((unsigned __int64)v129);
          }
          v52 = v49 + 8;
          v49[6] = v49 + 8;
          if ( v47 )
          {
            v138[0] = v45;
            if ( v45 > 0xF )
            {
              v82 = sub_22409D0((__int64)(v49 + 6), v138, 0);
              v49[6] = v82;
              v52 = (_BYTE *)v82;
              v49[8] = v138[0];
            }
            else if ( v45 == 1 )
            {
              *((_BYTE *)v49 + 64) = *v47;
LABEL_91:
              v49[7] = v45;
              v52[v45] = 0;
              goto LABEL_92;
            }
            memcpy(v52, v47, v45);
            v45 = v138[0];
            v52 = (_BYTE *)v49[6];
            goto LABEL_91;
          }
LABEL_194:
          sub_426248((__int64)"basic_string::_M_construct null not valid");
        }
      }
      memcpy(v51, v50, v46);
      v46 = v138[0];
      v51 = (_BYTE *)v49[2];
      goto LABEL_86;
    }
    v98 = v110;
    v73 = v109;
    v74 = sub_22077B0(0x50u);
    v49 = (_QWORD *)v74;
    if ( !v74 )
      goto LABEL_92;
    v75 = dest;
    *(_DWORD *)(v74 + 8) = 1;
    v76 = (_BYTE *)(v74 + 32);
    *(_QWORD *)v74 = off_49D3DE0;
    v77 = v98;
    *(_QWORD *)(v74 + 16) = v74 + 32;
    if ( &dest->m128i_i8[v104] && !dest )
      goto LABEL_194;
    v138[0] = v104;
    if ( v104 > 0xF )
    {
      v79 = sub_22409D0(v74 + 16, v138, 0);
      v75 = dest;
      v77 = v98;
      v49[2] = v79;
      v76 = (_BYTE *)v79;
      v49[4] = v138[0];
    }
    else
    {
      if ( v104 == 1 )
      {
        *(_BYTE *)(v74 + 32) = dest->m128i_i8[0];
LABEL_153:
        v49[3] = v46;
        v76[v46] = 0;
        v78 = v49 + 8;
        v49[6] = v49 + 8;
        if ( &v73[v77] && !v73 )
          goto LABEL_194;
        v138[0] = v77;
        if ( v77 > 0xF )
        {
          v101 = (void *)v77;
          v81 = sub_22409D0((__int64)(v49 + 6), v138, 0);
          v77 = (size_t)v101;
          v49[6] = v81;
          v78 = (_BYTE *)v81;
          v49[8] = v138[0];
        }
        else
        {
          if ( v77 == 1 )
          {
            *((_BYTE *)v49 + 64) = *v73;
LABEL_158:
            v49[7] = v77;
            v78[v77] = 0;
            goto LABEL_92;
          }
          if ( !v77 )
          {
LABEL_167:
            v78 = (_BYTE *)v49[6];
            goto LABEL_158;
          }
        }
        memcpy(v78, v73, v77);
        v77 = v138[0];
        goto LABEL_167;
      }
      if ( !v104 )
        goto LABEL_153;
    }
    v100 = (void *)v77;
    memcpy(v76, v75, v104);
    v46 = v138[0];
    v76 = (_BYTE *)v49[2];
    v77 = (size_t)v100;
    goto LABEL_153;
  }
  while ( 1 )
  {
    v138[0] = (unsigned __int64)v139;
    v136[0] = (unsigned __int64)v137;
    v136[1] = 0;
    v136[2] = 32;
    v138[1] = 0;
    v138[2] = 32;
    v13 = sub_CAE820(v9, (unsigned __int64)a2, v6, v7, v8);
    if ( *(_DWORD *)(v13 + 32) != 1 )
    {
      v132 = (__m128i *)"descriptor key must be a scalar";
      v135 = 259;
      v21 = sub_CAE820(v9, (unsigned __int64)a2, v10, v11, v12);
      goto LABEL_9;
    }
    v14 = sub_CAE940(v9, (unsigned __int64)a2, v10, v11, v12);
    if ( *((_DWORD *)v14 + 8) != 1 )
    {
      v132 = (__m128i *)"descriptor value must be a scalar";
      v135 = 259;
      v21 = (__int64)sub_CAE940(v9, (unsigned __int64)a2, v15, v16, v17);
      goto LABEL_9;
    }
    n = (size_t)v14;
    v18 = sub_CA8C30(v13, v136);
    if ( v19 != 6 )
    {
      if ( v19 != 9 )
      {
        if ( v19 != 5 || *(_DWORD *)v18 != 1701536110 || v18[4] != 100 )
          goto LABEL_7;
        v124 = 0;
        v123 = v125;
        LOBYTE(v125[0]) = 0;
        v24 = sub_CA8C30(n, v138);
        v26 = v24;
        v126 = v128;
        v27 = v25;
        if ( &v24[v25] && !v24 )
          goto LABEL_194;
        v132 = (__m128i *)v25;
        if ( v25 > 0xF )
        {
          nc = (size_t)v24;
          v97 = v25;
          v57 = sub_22409D0((__int64)&v126, (unsigned __int64 *)&v132, 0);
          v27 = v97;
          v26 = (const void *)nc;
          v126 = (_QWORD *)v57;
          v58 = (_QWORD *)v57;
          v128[0] = v132;
        }
        else
        {
          if ( v25 == 1 )
          {
            LOBYTE(v128[0]) = *v24;
            v28 = v128;
            goto LABEL_32;
          }
          if ( !v25 )
          {
            v28 = v128;
            goto LABEL_32;
          }
          v58 = v128;
        }
        memcpy(v58, v26, v27);
        v27 = (size_t)v132;
        v28 = v126;
LABEL_32:
        v127 = v27;
        *((_BYTE *)v28 + v27) = 0;
        v29 = v123;
        if ( v126 == v128 )
        {
          v59 = v127;
          if ( v127 )
          {
            if ( v127 == 1 )
              *(_BYTE *)v123 = v128[0];
            else
              memcpy(v123, v128, v127);
            v59 = v127;
            v29 = v123;
          }
          v124 = v59;
          *((_BYTE *)v29 + v59) = 0;
          v29 = v126;
          goto LABEL_36;
        }
        if ( v123 == v125 )
        {
          v123 = v126;
          v124 = v127;
          v125[0] = v128[0];
        }
        else
        {
          v30 = v125[0];
          v123 = v126;
          v124 = v127;
          v125[0] = v128[0];
          if ( v29 )
          {
            v126 = v29;
            v128[0] = v30;
LABEL_36:
            v127 = 0;
            *(_BYTE *)v29 = 0;
            if ( v126 != v128 )
              j_j___libc_free_0((unsigned __int64)v126);
            v129 = v123;
            v130 = v124;
            sub_C93130((__int64 *)&v132, (__int64)&v129);
            a2 = (unsigned __int64 *)"true";
            v96 = 1;
            if ( sub_2241AC0((__int64)&v132, "true") )
            {
              a2 = (unsigned __int64 *)"1";
              v96 = sub_2241AC0((__int64)&v123, "1") == 0;
            }
            if ( v132 != &v134 )
            {
              a2 = (unsigned __int64 *)(v134.m128i_i64[0] + 1);
              j_j___libc_free_0((unsigned __int64)v132);
            }
            if ( v123 != v125 )
            {
              a2 = (unsigned __int64 *)(v125[0] + 1LL);
              j_j___libc_free_0((unsigned __int64)v123);
            }
            goto LABEL_72;
          }
        }
        v126 = v128;
        v29 = v128;
        goto LABEL_36;
      }
      v20 = 0x726F66736E617274LL;
      if ( *(_QWORD *)v18 == 0x726F66736E617274LL && v18[8] == 109 )
      {
        a2 = v138;
        v31 = sub_CA8C30(n, v138);
        v33 = (unsigned __int64 *)v31;
        v120 = v122;
        v34 = v32;
        if ( &v31[v32] && !v31 )
          goto LABEL_194;
        v132 = (__m128i *)v32;
        if ( v32 > 0xF )
        {
          ne = (size_t)v31;
          v70 = sub_22409D0((__int64)&v120, (unsigned __int64 *)&v132, 0);
          v33 = (unsigned __int64 *)ne;
          v120 = (_QWORD *)v70;
          v71 = (_QWORD *)v70;
          v122[0] = v132;
        }
        else
        {
          if ( v32 == 1 )
          {
            LOBYTE(v122[0]) = *v31;
            v35 = v122;
            goto LABEL_51;
          }
          if ( !v32 )
          {
            v35 = v122;
            goto LABEL_51;
          }
          v71 = v122;
        }
        a2 = v33;
        memcpy(v71, v33, v34);
        v34 = (size_t)v132;
        v35 = v120;
LABEL_51:
        v121 = v34;
        *((_BYTE *)v35 + v34) = 0;
        v36 = v109;
        if ( v120 == v122 )
        {
          v72 = v121;
          if ( v121 )
          {
            if ( v121 == 1 )
            {
              *(_BYTE *)v109 = v122[0];
            }
            else
            {
              a2 = v122;
              memcpy(v109, v122, v121);
            }
            v72 = v121;
            v36 = v109;
          }
          v110 = v72;
          *((_BYTE *)v36 + v72) = 0;
          v36 = v120;
          goto LABEL_55;
        }
        a2 = (unsigned __int64 *)v121;
        if ( v109 == v111 )
        {
          v109 = v120;
          v110 = v121;
          v111[0] = v122[0];
        }
        else
        {
          v37 = v111[0];
          v109 = v120;
          v110 = v121;
          v111[0] = v122[0];
          if ( v36 )
          {
            v120 = v36;
            v122[0] = v37;
            goto LABEL_55;
          }
        }
        v120 = v122;
        v36 = v122;
LABEL_55:
        v121 = 0;
        *(_BYTE *)v36 = 0;
        if ( v120 != v122 )
        {
          a2 = (unsigned __int64 *)(v122[0] + 1LL);
          j_j___libc_free_0((unsigned __int64)v120);
        }
        goto LABEL_72;
      }
LABEL_7:
      v132 = (__m128i *)"unknown key for function";
      v135 = 259;
      v21 = sub_CAE820(v9, (unsigned __int64)v136, v19, v20, n);
LABEL_9:
      sub_CA89D0(v99, v21, (__int64)&v132, 0);
      goto LABEL_10;
    }
    if ( *(_DWORD *)v18 != 1920298867 || *((_WORD *)v18 + 2) != 25955 )
    {
      if ( *(_DWORD *)v18 != 1735549300 || *((_WORD *)v18 + 2) != 29797 )
        goto LABEL_7;
      a2 = v138;
      v60 = sub_CA8C30(n, v138);
      v62 = (unsigned __int64 *)v60;
      v117 = v119;
      v63 = v61;
      if ( &v60[v61] && !v60 )
        goto LABEL_194;
      v132 = (__m128i *)v61;
      if ( v61 > 0xF )
      {
        nd = (size_t)v60;
        v67 = sub_22409D0((__int64)&v117, (unsigned __int64 *)&v132, 0);
        v62 = (unsigned __int64 *)nd;
        v117 = (_QWORD *)v67;
        v68 = (_QWORD *)v67;
        v119[0] = v132;
      }
      else
      {
        if ( v61 == 1 )
        {
          LOBYTE(v119[0]) = *v60;
          v64 = v119;
          goto LABEL_117;
        }
        if ( !v61 )
        {
          v64 = v119;
          goto LABEL_117;
        }
        v68 = v119;
      }
      a2 = v62;
      memcpy(v68, v62, v63);
      v63 = (size_t)v132;
      v64 = v117;
LABEL_117:
      v118 = v63;
      *((_BYTE *)v64 + v63) = 0;
      v65 = v106;
      if ( v117 == v119 )
      {
        v69 = v118;
        if ( v118 )
        {
          if ( v118 == 1 )
          {
            *(_BYTE *)v106 = v119[0];
          }
          else
          {
            a2 = v119;
            memcpy(v106, v119, v118);
          }
          v69 = v118;
          v65 = v106;
        }
        v107 = v69;
        *((_BYTE *)v65 + v69) = 0;
        v65 = v117;
        goto LABEL_121;
      }
      a2 = (unsigned __int64 *)v118;
      if ( v106 == v108 )
      {
        v106 = v117;
        v107 = v118;
        v108[0] = v119[0];
      }
      else
      {
        v66 = v108[0];
        v106 = v117;
        v107 = v118;
        v108[0] = v119[0];
        if ( v65 )
        {
          v117 = v65;
          v119[0] = v66;
          goto LABEL_121;
        }
      }
      v117 = v119;
      v65 = v119;
LABEL_121:
      v118 = 0;
      *(_BYTE *)v65 = 0;
      if ( v117 != v119 )
      {
        a2 = (unsigned __int64 *)(v119[0] + 1LL);
        j_j___libc_free_0((unsigned __int64)v117);
      }
      goto LABEL_72;
    }
    v112[1] = 0;
    v112[0] = (unsigned __int64)v113;
    LOBYTE(v113[0]) = 0;
    v38 = sub_CA8C30(n, v138);
    v40 = v38;
    v114 = (__m128i *)v116;
    v41 = v39;
    if ( &v38[v39] && !v38 )
      goto LABEL_194;
    v132 = (__m128i *)v39;
    if ( v39 > 0xF )
    {
      src = v38;
      nb = v39;
      v54 = sub_22409D0((__int64)&v114, (unsigned __int64 *)&v132, 0);
      v41 = nb;
      v40 = src;
      v114 = (__m128i *)v54;
      v55 = (_QWORD *)v54;
      v116[0] = v132;
    }
    else
    {
      if ( v39 == 1 )
      {
        LOBYTE(v116[0]) = *v38;
        v42 = v116;
        goto LABEL_63;
      }
      if ( !v39 )
      {
        v42 = v116;
        goto LABEL_63;
      }
      v55 = v116;
    }
    memcpy(v55, v40, v41);
    v41 = (size_t)v132;
    v42 = (__int64 *)v114;
LABEL_63:
    v115 = v41;
    *((_BYTE *)v42 + v41) = 0;
    v43 = (__int64 *)dest;
    if ( v114 == (__m128i *)v116 )
    {
      v56 = v115;
      if ( v115 )
      {
        if ( v115 == 1 )
          dest->m128i_i8[0] = v116[0];
        else
          memcpy(dest, v116, v115);
        v56 = v115;
        v43 = (__int64 *)dest;
      }
      v104 = v56;
      *((_BYTE *)v43 + v56) = 0;
      v43 = (__int64 *)v114;
    }
    else
    {
      if ( dest == (__m128i *)v105 )
      {
        dest = v114;
        v104 = v115;
        v105[0] = v116[0];
      }
      else
      {
        v44 = v105[0];
        dest = v114;
        v104 = v115;
        v105[0] = v116[0];
        if ( v43 )
        {
          v114 = (__m128i *)v43;
          v116[0] = v44;
          goto LABEL_67;
        }
      }
      v114 = (__m128i *)v116;
      v43 = v116;
    }
LABEL_67:
    v115 = 0;
    *(_BYTE *)v43 = 0;
    if ( v114 != (__m128i *)v116 )
      j_j___libc_free_0((unsigned __int64)v114);
    sub_C88F40((__int64)&v132, (__int64)dest, v104, 0);
    a2 = v112;
    na = sub_C89030((__int64 *)&v132, v112);
    sub_C88FF0(&v132);
    if ( !na )
      break;
    if ( (_QWORD *)v112[0] != v113 )
    {
      a2 = (unsigned __int64 *)(v113[0] + 1LL);
      j_j___libc_free_0(v112[0]);
    }
LABEL_72:
    if ( (_BYTE *)v138[0] != v139 )
      _libc_free(v138[0]);
    if ( (_BYTE *)v136[0] != v137 )
      _libc_free(v136[0]);
    sub_CAEB90(a4, (unsigned __int64)a2);
    v9 = *(_QWORD *)(a4 + 80);
    if ( !v9 )
      goto LABEL_77;
  }
  sub_8FD6D0((__int64)&v129, "invalid regex: ", v112);
  v132 = (__m128i *)&v129;
  v135 = 260;
  v87 = sub_CAE820(v9, (unsigned __int64)"invalid regex: ", v84, v85, v86);
  sub_CA89D0(v99, v87, (__int64)&v132, 0);
  if ( v129 != (_QWORD *)v131 )
    j_j___libc_free_0((unsigned __int64)v129);
  if ( (_QWORD *)v112[0] != v113 )
    j_j___libc_free_0(v112[0]);
LABEL_10:
  if ( (_BYTE *)v138[0] != v139 )
    _libc_free(v138[0]);
  if ( (_BYTE *)v136[0] != v137 )
    _libc_free(v136[0]);
LABEL_14:
  v22 = 0;
LABEL_15:
  if ( v109 != v111 )
    j_j___libc_free_0((unsigned __int64)v109);
  if ( v106 != v108 )
    j_j___libc_free_0((unsigned __int64)v106);
  if ( dest != (__m128i *)v105 )
    j_j___libc_free_0((unsigned __int64)dest);
  return v22;
}
