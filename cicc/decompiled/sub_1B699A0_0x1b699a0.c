// Function: sub_1B699A0
// Address: 0x1b699a0
//
__int64 __fastcall sub_1B699A0(__int64 a1, char *p_dest, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r12
  _QWORD *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  const char *v18; // rax
  unsigned __int64 v19; // rdx
  const char *v20; // r15
  size_t v21; // r8
  size_t v22; // rax
  char *v23; // rdx
  _BYTE *v24; // rdi
  size_t v25; // rdx
  char *v26; // rax
  const char *v27; // rax
  __int64 v28; // rax
  unsigned int v29; // r12d
  _QWORD *v31; // rax
  void *v32; // rdx
  const char *v33; // r15
  size_t v34; // r8
  _BYTE *v35; // rax
  char *v36; // rdx
  _BYTE *v37; // rdi
  char *v38; // rax
  __int64 v39; // rdi
  _BYTE *v40; // r12
  __int64 v41; // rax
  __int64 v42; // r14
  char *v43; // r10
  size_t v44; // r8
  size_t v45; // r9
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  const char *v48; // r11
  size_t v49; // r8
  size_t v50; // rax
  char *v51; // rdx
  _BYTE *v52; // rdi
  char *v53; // rax
  __int64 v54; // rdi
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // rax
  void *v59; // rdx
  const char *v60; // r15
  size_t v61; // r8
  _BYTE *v62; // rax
  char *v63; // rdx
  _BYTE *v64; // rdi
  size_t v65; // rdx
  size_t v66; // rdx
  size_t v67; // rdx
  __int64 v68; // rax
  _QWORD *v69; // rdi
  _BYTE *v70; // r12
  __int64 v71; // rax
  const void *v72; // r10
  _BYTE *v73; // rdi
  size_t v74; // r11
  size_t v75; // r8
  _BYTE *v76; // rdi
  __int64 v77; // rax
  _QWORD *v78; // rdi
  __int64 v79; // rax
  _QWORD *v80; // rdi
  __int64 v81; // rax
  _QWORD *v82; // rdi
  __m128i *v83; // rax
  __int64 v84; // rax
  __int64 v85; // rax
  size_t n; // [rsp+10h] [rbp-1B0h]
  void *src; // [rsp+18h] [rbp-1A8h]
  char srca; // [rsp+18h] [rbp-1A8h]
  const char *srcb; // [rsp+18h] [rbp-1A8h]
  void *srcc; // [rsp+18h] [rbp-1A8h]
  void *srcd; // [rsp+18h] [rbp-1A8h]
  void *srce; // [rsp+18h] [rbp-1A8h]
  __int64 v93; // [rsp+30h] [rbp-190h]
  bool v95; // [rsp+40h] [rbp-180h]
  __int64 *v96; // [rsp+50h] [rbp-170h]
  size_t v97; // [rsp+50h] [rbp-170h]
  char *v98; // [rsp+60h] [rbp-160h]
  size_t v99; // [rsp+60h] [rbp-160h]
  size_t v100; // [rsp+68h] [rbp-158h]
  size_t v101; // [rsp+68h] [rbp-158h]
  _BYTE *v102; // [rsp+68h] [rbp-158h]
  size_t v103; // [rsp+68h] [rbp-158h]
  size_t v104; // [rsp+68h] [rbp-158h]
  size_t v105; // [rsp+68h] [rbp-158h]
  size_t v106; // [rsp+68h] [rbp-158h]
  size_t v107; // [rsp+68h] [rbp-158h]
  _QWORD v108[2]; // [rsp+70h] [rbp-150h] BYREF
  __int16 v109; // [rsp+80h] [rbp-140h]
  void *v110; // [rsp+90h] [rbp-130h]
  size_t v111; // [rsp+98h] [rbp-128h]
  _QWORD v112[2]; // [rsp+A0h] [rbp-120h] BYREF
  void *v113; // [rsp+B0h] [rbp-110h]
  size_t v114; // [rsp+B8h] [rbp-108h]
  _QWORD v115[2]; // [rsp+C0h] [rbp-100h] BYREF
  void *v116; // [rsp+D0h] [rbp-F0h]
  size_t v117; // [rsp+D8h] [rbp-E8h]
  _QWORD v118[2]; // [rsp+E0h] [rbp-E0h] BYREF
  void *dest; // [rsp+F0h] [rbp-D0h] BYREF
  size_t v120; // [rsp+F8h] [rbp-C8h]
  _QWORD v121[2]; // [rsp+100h] [rbp-C0h] BYREF
  char *v122; // [rsp+110h] [rbp-B0h] BYREF
  size_t v123; // [rsp+118h] [rbp-A8h]
  _QWORD v124[2]; // [rsp+120h] [rbp-A0h] BYREF
  _QWORD *v125; // [rsp+130h] [rbp-90h] BYREF
  __int64 v126; // [rsp+138h] [rbp-88h]
  _QWORD v127[4]; // [rsp+140h] [rbp-80h] BYREF
  const char *v128; // [rsp+160h] [rbp-60h] BYREF
  __int64 v129; // [rsp+168h] [rbp-58h]
  _OWORD v130[5]; // [rsp+170h] [rbp-50h] BYREF

  v5 = a4;
  *(_BYTE *)(a4 + 76) = 0;
  v96 = (__int64 *)p_dest;
  v110 = v112;
  v111 = 0;
  LOBYTE(v112[0]) = 0;
  v113 = v115;
  v114 = 0;
  LOBYTE(v115[0]) = 0;
  v116 = v118;
  v117 = 0;
  LOBYTE(v118[0]) = 0;
  sub_16FD380(a4, (unsigned __int64)p_dest);
  v9 = *(_QWORD *)(v5 + 80);
  v95 = 0;
  if ( !v9 )
  {
LABEL_62:
    if ( (v114 == 0) == (v117 == 0) )
    {
      LOWORD(v130[0]) = 259;
      v128 = "exactly one of transform or target must be specified";
      sub_16F8270(v96, v5, (__int64)&v128);
      goto LABEL_32;
    }
    v100 = v114;
    v97 = v111;
    if ( v114 )
    {
      v98 = (char *)v110;
      v40 = v113;
      v41 = sub_22077B0(80);
      v42 = v41;
      if ( !v41 )
      {
LABEL_71:
        v29 = 1;
        v46 = sub_22077B0(24);
        *(_QWORD *)(v46 + 16) = v42;
        sub_2208C80(v46, a5);
        ++*(_QWORD *)(a5 + 16);
        goto LABEL_33;
      }
      *(_DWORD *)(v41 + 8) = 1;
      *(_QWORD *)v41 = off_4985358;
      v43 = v98;
      v44 = v97;
      v45 = v100;
      if ( v95 )
      {
        if ( v98 )
        {
          v125 = v127;
          sub_1B678F0((__int64 *)&v125, v98, (__int64)&v98[v97]);
          v45 = v100;
        }
        else
        {
          LOBYTE(v127[0]) = 0;
          v125 = v127;
          v126 = 0;
        }
        v103 = v45;
        v83 = (__m128i *)sub_2241130(&v125, 0, 0, &unk_3F871B2, 1);
        v45 = v103;
        v128 = (const char *)v130;
        if ( (__m128i *)v83->m128i_i64[0] == &v83[1] )
        {
          v130[0] = _mm_loadu_si128(v83 + 1);
        }
        else
        {
          v128 = (const char *)v83->m128i_i64[0];
          *(_QWORD *)&v130[0] = v83[1].m128i_i64[0];
        }
        v129 = v83->m128i_i64[1];
        v83->m128i_i64[0] = (__int64)v83[1].m128i_i64;
        v83->m128i_i64[1] = 0;
        v83[1].m128i_i8[0] = 0;
        v43 = (char *)v128;
        v44 = v129;
      }
      *(_QWORD *)(v42 + 16) = v42 + 32;
      if ( v43 )
      {
        v101 = v45;
        sub_1B678F0((__int64 *)(v42 + 16), v43, (__int64)&v43[v44]);
        v45 = v101;
      }
      else
      {
        *(_QWORD *)(v42 + 24) = 0;
        *(_BYTE *)(v42 + 32) = 0;
      }
      if ( v95 )
      {
        if ( v128 != (const char *)v130 )
        {
          v104 = v45;
          j_j___libc_free_0(v128, *(_QWORD *)&v130[0] + 1LL);
          v45 = v104;
        }
        if ( v125 != v127 )
        {
          v105 = v45;
          j_j___libc_free_0(v125, v127[0] + 1LL);
          v45 = v105;
        }
      }
      *(_QWORD *)(v42 + 48) = v42 + 64;
      if ( v40 )
      {
        sub_1B678F0((__int64 *)(v42 + 48), v40, (__int64)&v40[v45]);
        goto LABEL_71;
      }
LABEL_123:
      *(_QWORD *)(v42 + 56) = 0;
      *(_BYTE *)(v42 + 64) = 0;
      goto LABEL_71;
    }
    v99 = v117;
    v70 = v116;
    v102 = v110;
    v71 = sub_22077B0(80);
    v42 = v71;
    if ( !v71 )
      goto LABEL_71;
    v72 = v102;
    v73 = (_BYTE *)(v71 + 32);
    *(_DWORD *)(v71 + 8) = 1;
    v74 = v99;
    v75 = v97;
    *(_QWORD *)v71 = off_4985380;
    *(_QWORD *)(v71 + 16) = v71 + 32;
    if ( !v102 )
    {
      *(_QWORD *)(v71 + 24) = 0;
      *(_BYTE *)(v71 + 32) = 0;
LABEL_118:
      v76 = (_BYTE *)(v42 + 64);
      *(_QWORD *)(v42 + 48) = v42 + 64;
      if ( !v70 )
        goto LABEL_123;
      v128 = (const char *)v74;
      if ( v74 > 0xF )
      {
        v106 = v74;
        v84 = sub_22409D0(v42 + 48, &v128, 0);
        v74 = v106;
        *(_QWORD *)(v42 + 48) = v84;
        v76 = (_BYTE *)v84;
        *(_QWORD *)(v42 + 64) = v128;
      }
      else
      {
        if ( v74 == 1 )
        {
          *(_BYTE *)(v42 + 64) = *v70;
LABEL_122:
          *(_QWORD *)(v42 + 56) = v74;
          v76[v74] = 0;
          goto LABEL_71;
        }
        if ( !v74 )
          goto LABEL_122;
      }
      memcpy(v76, v70, v74);
      v74 = (size_t)v128;
      v76 = *(_BYTE **)(v42 + 48);
      goto LABEL_122;
    }
    v128 = (const char *)v97;
    if ( v97 > 0xF )
    {
      v85 = sub_22409D0(v71 + 16, &v128, 0);
      v72 = v102;
      v74 = v99;
      *(_QWORD *)(v42 + 16) = v85;
      v73 = (_BYTE *)v85;
      v75 = v97;
      *(_QWORD *)(v42 + 32) = v128;
    }
    else
    {
      if ( v97 == 1 )
      {
        *(_BYTE *)(v71 + 32) = *v102;
LABEL_117:
        *(_QWORD *)(v42 + 24) = v75;
        v73[v75] = 0;
        goto LABEL_118;
      }
      if ( !v97 )
        goto LABEL_117;
    }
    v107 = v74;
    memcpy(v73, v72, v75);
    v75 = (size_t)v128;
    v73 = *(_BYTE **)(v42 + 16);
    v74 = v107;
    goto LABEL_117;
  }
  v93 = v5;
  while ( 1 )
  {
    v126 = 0x2000000000LL;
    v129 = 0x2000000000LL;
    v125 = v127;
    v128 = (const char *)v130;
    v13 = sub_16FD110(v9, (unsigned __int64)p_dest, v6, v7, v8);
    if ( *(_DWORD *)(v13 + 32) != 1 )
    {
      BYTE1(v124[0]) = 1;
      v27 = "descriptor key must be a scalar";
      goto LABEL_27;
    }
    v14 = sub_16FD200(v9, (unsigned __int64)p_dest, v10, v11, v12);
    if ( *((_DWORD *)v14 + 8) != 1 )
    {
      v122 = "descriptor value must be a scalar";
      LOWORD(v124[0]) = 259;
      v31 = sub_16FD200(v9, (unsigned __int64)p_dest, v15, v16, v17);
      sub_16F8270(v96, (__int64)v31, (__int64)&v122);
      goto LABEL_28;
    }
    p_dest = (char *)&v125;
    src = v14;
    v18 = sub_16F8F10(v13, &v125);
    v12 = (__int64)src;
    if ( v10 == 6 )
      break;
    if ( v10 == 9 )
    {
      v11 = 0x726F66736E617274LL;
      if ( *(_QWORD *)v18 != 0x726F66736E617274LL || v18[8] != 109 )
        goto LABEL_42;
      p_dest = (char *)&v128;
      v60 = sub_16F8F10((__int64)src, &v128);
      v61 = (size_t)v59;
      if ( !v60 )
      {
        LOBYTE(v124[0]) = 0;
        v64 = v116;
        v67 = 0;
        v122 = (char *)v124;
LABEL_102:
        v117 = v67;
        v64[v67] = 0;
        v38 = v122;
LABEL_54:
        v123 = 0;
        *v38 = 0;
        if ( v122 != (char *)v124 )
        {
          p_dest = (char *)(v124[0] + 1LL);
          j_j___libc_free_0(v122, v124[0] + 1LL);
        }
        goto LABEL_56;
      }
      dest = v59;
      v62 = v59;
      v122 = (char *)v124;
      if ( (unsigned __int64)v59 > 0xF )
      {
        srcd = v59;
        v79 = sub_22409D0(&v122, &dest, 0);
        v61 = (size_t)srcd;
        v122 = (char *)v79;
        v80 = (_QWORD *)v79;
        v124[0] = dest;
      }
      else
      {
        if ( v59 == (void *)1 )
        {
          LOBYTE(v124[0]) = *v60;
          v63 = (char *)v124;
LABEL_93:
          v123 = (size_t)v62;
          v62[(_QWORD)v63] = 0;
          v64 = v116;
          v38 = (char *)v116;
          if ( v122 != (char *)v124 )
          {
            p_dest = (char *)v123;
            if ( v116 == v118 )
            {
              v116 = v122;
              v117 = v123;
              v118[0] = v124[0];
              goto LABEL_96;
            }
            v39 = v118[0];
            v116 = v122;
            v117 = v123;
            v118[0] = v124[0];
            if ( !v38 )
              goto LABEL_96;
            goto LABEL_53;
          }
          v67 = v123;
          if ( v123 )
          {
            if ( v123 == 1 )
            {
              *(_BYTE *)v116 = v124[0];
            }
            else
            {
              p_dest = (char *)v124;
              memcpy(v116, v124, v123);
            }
            v67 = v123;
            v64 = v116;
          }
          goto LABEL_102;
        }
        if ( !v59 )
        {
          v63 = (char *)v124;
          goto LABEL_93;
        }
        v80 = v124;
      }
      p_dest = (char *)v60;
      memcpy(v80, v60, v61);
      v62 = dest;
      v63 = v122;
      goto LABEL_93;
    }
    if ( v10 != 5 || *(_DWORD *)v18 != 1701536110 || v18[4] != 100 )
      goto LABEL_42;
    v120 = 0;
    LOBYTE(v121[0]) = 0;
    dest = v121;
    v20 = sub_16F8F10((__int64)src, &v128);
    v21 = v19;
    if ( !v20 )
    {
      LOBYTE(v124[0]) = 0;
      v24 = dest;
      v25 = 0;
      v122 = (char *)v124;
LABEL_104:
      v120 = v25;
      v24[v25] = 0;
      v26 = v122;
      goto LABEL_18;
    }
    v108[0] = v19;
    v22 = v19;
    v122 = (char *)v124;
    if ( v19 > 0xF )
    {
      srce = (void *)v19;
      v81 = sub_22409D0(&v122, v108, 0);
      v21 = (size_t)srce;
      v122 = (char *)v81;
      v82 = (_QWORD *)v81;
      v124[0] = v108[0];
    }
    else
    {
      if ( v19 == 1 )
      {
        LOBYTE(v124[0]) = *v20;
        v23 = (char *)v124;
        goto LABEL_14;
      }
      if ( !v19 )
      {
        v23 = (char *)v124;
        goto LABEL_14;
      }
      v82 = v124;
    }
    memcpy(v82, v20, v21);
    v22 = v108[0];
    v23 = v122;
LABEL_14:
    v123 = v22;
    v23[v22] = 0;
    v24 = dest;
    v25 = (size_t)v122;
    v26 = (char *)dest;
    if ( v122 == (char *)v124 )
    {
      v25 = v123;
      if ( v123 )
      {
        if ( v123 == 1 )
          *(_BYTE *)dest = v124[0];
        else
          memcpy(dest, v124, v123);
        v25 = v123;
        v24 = dest;
      }
      goto LABEL_104;
    }
    if ( dest == v121 )
    {
      dest = v122;
      v120 = v123;
      v121[0] = v124[0];
LABEL_134:
      v122 = (char *)v124;
      v26 = (char *)v124;
      goto LABEL_18;
    }
    v21 = v121[0];
    dest = v122;
    v120 = v123;
    v121[0] = v124[0];
    if ( !v26 )
      goto LABEL_134;
    v122 = v26;
    v124[0] = v21;
LABEL_18:
    v123 = 0;
    *v26 = 0;
    if ( v122 != (char *)v124 )
      j_j___libc_free_0(v122, v124[0] + 1LL);
    v108[0] = dest;
    v108[1] = v120;
    sub_16D2060(&v122, v108, v25, (__int64)v108, v21);
    p_dest = "true";
    v95 = 1;
    if ( (unsigned int)sub_2241AC0(&v122, "true") )
    {
      p_dest = "1";
      v95 = (unsigned int)sub_2241AC0(&dest, "1") == 0;
    }
    if ( v122 != (char *)v124 )
    {
      p_dest = (char *)(v124[0] + 1LL);
      j_j___libc_free_0(v122, v124[0] + 1LL);
    }
LABEL_24:
    if ( dest != v121 )
    {
      p_dest = (char *)(v121[0] + 1LL);
      j_j___libc_free_0(dest, v121[0] + 1LL);
    }
LABEL_56:
    if ( v128 != (const char *)v130 )
      _libc_free((unsigned __int64)v128);
    if ( v125 != v127 )
      _libc_free((unsigned __int64)v125);
    sub_16FD380(v93, (unsigned __int64)p_dest);
    v9 = *(_QWORD *)(v93 + 80);
    if ( !v9 )
    {
      v5 = v93;
      goto LABEL_62;
    }
  }
  if ( *(_DWORD *)v18 != 1920298867 || *((_WORD *)v18 + 2) != 25955 )
  {
    if ( *(_DWORD *)v18 == 1735549300 && *((_WORD *)v18 + 2) == 29797 )
    {
      p_dest = (char *)&v128;
      v33 = sub_16F8F10((__int64)src, &v128);
      v34 = (size_t)v32;
      if ( !v33 )
      {
        LOBYTE(v124[0]) = 0;
        v37 = v113;
        v66 = 0;
        v122 = (char *)v124;
LABEL_100:
        v114 = v66;
        v37[v66] = 0;
        v38 = v122;
        goto LABEL_54;
      }
      dest = v32;
      v35 = v32;
      v122 = (char *)v124;
      if ( (unsigned __int64)v32 > 0xF )
      {
        srcc = v32;
        v77 = sub_22409D0(&v122, &dest, 0);
        v34 = (size_t)srcc;
        v122 = (char *)v77;
        v78 = (_QWORD *)v77;
        v124[0] = dest;
      }
      else
      {
        if ( v32 == (void *)1 )
        {
          LOBYTE(v124[0]) = *v33;
          v36 = (char *)v124;
LABEL_50:
          v123 = (size_t)v35;
          v35[(_QWORD)v36] = 0;
          v37 = v113;
          v38 = (char *)v113;
          if ( v122 != (char *)v124 )
          {
            p_dest = (char *)v123;
            if ( v113 == v115 )
            {
              v113 = v122;
              v114 = v123;
              v115[0] = v124[0];
              goto LABEL_96;
            }
            v39 = v115[0];
            v113 = v122;
            v114 = v123;
            v115[0] = v124[0];
            if ( !v38 )
            {
LABEL_96:
              v122 = (char *)v124;
              v38 = (char *)v124;
              goto LABEL_54;
            }
LABEL_53:
            v122 = v38;
            v124[0] = v39;
            goto LABEL_54;
          }
          v66 = v123;
          if ( v123 )
          {
            if ( v123 == 1 )
            {
              *(_BYTE *)v113 = v124[0];
            }
            else
            {
              p_dest = (char *)v124;
              memcpy(v113, v124, v123);
            }
            v66 = v123;
            v37 = v113;
          }
          goto LABEL_100;
        }
        if ( !v32 )
        {
          v36 = (char *)v124;
          goto LABEL_50;
        }
        v78 = v124;
      }
      p_dest = (char *)v33;
      memcpy(v78, v33, v34);
      v35 = dest;
      v36 = v122;
      goto LABEL_50;
    }
LABEL_42:
    BYTE1(v124[0]) = 1;
    v27 = "unknown key for function";
LABEL_27:
    v122 = (char *)v27;
    LOBYTE(v124[0]) = 3;
    v28 = sub_16FD110(v9, (unsigned __int64)p_dest, v10, v11, v12);
    sub_16F8270(v96, v28, (__int64)&v122);
    goto LABEL_28;
  }
  v120 = 0;
  LOBYTE(v121[0]) = 0;
  dest = v121;
  v48 = sub_16F8F10((__int64)src, &v128);
  v49 = v47;
  if ( !v48 )
  {
    LOBYTE(v124[0]) = 0;
    v52 = v110;
    v65 = 0;
    v122 = (char *)v124;
LABEL_98:
    v111 = v65;
    v52[v65] = 0;
    v53 = v122;
    goto LABEL_81;
  }
  v108[0] = v47;
  v50 = v47;
  v122 = (char *)v124;
  if ( v47 > 0xF )
  {
    n = v47;
    srcb = v48;
    v68 = sub_22409D0(&v122, v108, 0);
    v48 = srcb;
    v49 = n;
    v122 = (char *)v68;
    v69 = (_QWORD *)v68;
    v124[0] = v108[0];
  }
  else
  {
    if ( v47 == 1 )
    {
      LOBYTE(v124[0]) = *v48;
      v51 = (char *)v124;
      goto LABEL_77;
    }
    if ( !v47 )
    {
      v51 = (char *)v124;
      goto LABEL_77;
    }
    v69 = v124;
  }
  memcpy(v69, v48, v49);
  v50 = v108[0];
  v51 = v122;
LABEL_77:
  v123 = v50;
  v51[v50] = 0;
  v52 = v110;
  v53 = (char *)v110;
  if ( v122 == (char *)v124 )
  {
    v65 = v123;
    if ( v123 )
    {
      if ( v123 == 1 )
        *(_BYTE *)v110 = v124[0];
      else
        memcpy(v110, v124, v123);
      v65 = v123;
      v52 = v110;
    }
    goto LABEL_98;
  }
  if ( v110 == v112 )
  {
    v110 = v122;
    v111 = v123;
    v112[0] = v124[0];
  }
  else
  {
    v54 = v112[0];
    v110 = v122;
    v111 = v123;
    v112[0] = v124[0];
    if ( v53 )
    {
      v122 = v53;
      v124[0] = v54;
      goto LABEL_81;
    }
  }
  v122 = (char *)v124;
  v53 = (char *)v124;
LABEL_81:
  v123 = 0;
  *v53 = 0;
  if ( v122 != (char *)v124 )
    j_j___libc_free_0(v122, v124[0] + 1LL);
  sub_16C9340((__int64)&v122, (__int64)v110, v111, 0);
  p_dest = (char *)&dest;
  srca = sub_16C9430(&v122, &dest);
  sub_16C93F0(&v122);
  if ( srca )
    goto LABEL_24;
  sub_8FD6D0((__int64)&v122, "invalid regex: ", &dest);
  v108[0] = &v122;
  v109 = 260;
  v58 = sub_16FD110(v9, (unsigned __int64)"invalid regex: ", v55, v56, v57);
  sub_16F8270(v96, v58, (__int64)v108);
  if ( v122 != (char *)v124 )
    j_j___libc_free_0(v122, v124[0] + 1LL);
  if ( dest != v121 )
    j_j___libc_free_0(dest, v121[0] + 1LL);
LABEL_28:
  if ( v128 != (const char *)v130 )
    _libc_free((unsigned __int64)v128);
  if ( v125 != v127 )
    _libc_free((unsigned __int64)v125);
LABEL_32:
  v29 = 0;
LABEL_33:
  if ( v116 != v118 )
    j_j___libc_free_0(v116, v118[0] + 1LL);
  if ( v113 != v115 )
    j_j___libc_free_0(v113, v115[0] + 1LL);
  if ( v110 != v112 )
    j_j___libc_free_0(v110, v112[0] + 1LL);
  return v29;
}
