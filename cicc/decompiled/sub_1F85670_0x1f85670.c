// Function: sub_1F85670
// Address: 0x1f85670
//
__int64 *__fastcall sub_1F85670(
        __int64 a1,
        unsigned __int8 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        unsigned int a6,
        __m128i a7,
        double a8,
        __m128i a9,
        __int64 a10)
{
  unsigned int v16; // r8d
  __int64 v17; // r9
  __int64 v18; // rax
  char v19; // dl
  __int64 v20; // rax
  bool v21; // zf
  char v22; // cl
  const void **v23; // rax
  __int64 v24; // rax
  const void **v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rax
  int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // r8
  __int64 v31; // rdi
  __int64 (*v32)(); // rax
  unsigned __int64 v33; // rax
  __int64 v34; // rdx
  signed int v35; // eax
  unsigned int v36; // r8d
  __int64 v37; // rdi
  unsigned int v38; // eax
  __int64 v39; // rax
  unsigned __int8 v40; // r8
  __int64 v41; // rax
  int v42; // eax
  bool v43; // cl
  __int128 v44; // rdi
  unsigned int v45; // eax
  int v46; // r11d
  _BOOL4 v47; // ecx
  __int64 *v48; // r15
  __int64 *v49; // rax
  __int16 *v50; // rdx
  unsigned __int64 v51; // r12
  __int16 *v52; // r13
  unsigned int v53; // eax
  __int64 v54; // rsi
  unsigned int v55; // eax
  bool v56; // al
  char v57; // r8
  bool v58; // al
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // r15
  __int64 v62; // r14
  __int128 v63; // rax
  __int64 *v64; // r12
  __int16 *v65; // rdx
  __int16 *v66; // r13
  char v67; // al
  bool v68; // al
  bool v69; // al
  __int64 *v70; // r12
  unsigned __int64 v71; // rdx
  unsigned __int64 v72; // r13
  __int64 *v73; // r14
  __int64 v74; // rdx
  __int64 v75; // r15
  __int64 *v76; // rax
  __int16 *v77; // rdx
  __int16 *v78; // r13
  unsigned __int64 v79; // r12
  __int128 v80; // rax
  char v81; // al
  int v82; // esi
  char v83; // di
  __int64 *v84; // r15
  bool v85; // al
  __int128 v86; // [rsp-10h] [rbp-130h]
  __int128 v87; // [rsp+0h] [rbp-120h]
  __int64 v88; // [rsp+18h] [rbp-108h]
  int v89; // [rsp+18h] [rbp-108h]
  unsigned __int8 v90; // [rsp+18h] [rbp-108h]
  __int64 (__fastcall *v91)(__int64, __int64, __int64, __int64, const void **); // [rsp+20h] [rbp-100h]
  unsigned __int8 v92; // [rsp+20h] [rbp-100h]
  bool v93; // [rsp+20h] [rbp-100h]
  __int64 v94; // [rsp+28h] [rbp-F8h]
  __int64 v95; // [rsp+28h] [rbp-F8h]
  unsigned __int8 v96; // [rsp+28h] [rbp-F8h]
  char v97; // [rsp+28h] [rbp-F8h]
  char v98; // [rsp+28h] [rbp-F8h]
  char v99; // [rsp+28h] [rbp-F8h]
  char v100; // [rsp+28h] [rbp-F8h]
  const void **v101; // [rsp+30h] [rbp-F0h]
  unsigned int v102; // [rsp+30h] [rbp-F0h]
  __int64 *v103; // [rsp+30h] [rbp-F0h]
  __int64 *v104; // [rsp+30h] [rbp-F0h]
  __int64 *v105; // [rsp+30h] [rbp-F0h]
  __int64 v106; // [rsp+38h] [rbp-E8h]
  unsigned int v107; // [rsp+38h] [rbp-E8h]
  __int128 v109; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v110; // [rsp+60h] [rbp-C0h] BYREF
  __int16 *v111; // [rsp+68h] [rbp-B8h]
  __int128 v112; // [rsp+70h] [rbp-B0h] BYREF
  __m128i v113; // [rsp+80h] [rbp-A0h] BYREF
  __int128 v114; // [rsp+90h] [rbp-90h] BYREF
  __int64 v115; // [rsp+A0h] [rbp-80h] BYREF
  int v116; // [rsp+A8h] [rbp-78h]
  __int64 v117; // [rsp+B0h] [rbp-70h] BYREF
  int v118; // [rsp+B8h] [rbp-68h]
  unsigned int v119; // [rsp+C0h] [rbp-60h] BYREF
  const void **v120; // [rsp+C8h] [rbp-58h]
  __int64 v121; // [rsp+D0h] [rbp-50h] BYREF
  const void **v122; // [rsp+D8h] [rbp-48h]
  __int64 v123[8]; // [rsp+E0h] [rbp-40h] BYREF

  v110 = 0;
  LODWORD(v111) = 0;
  *(_QWORD *)&v112 = 0;
  DWORD2(v112) = 0;
  v113.m128i_i64[0] = 0;
  v113.m128i_i32[2] = 0;
  *(_QWORD *)&v114 = 0;
  DWORD2(v114) = 0;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v118 = 0;
  if ( !(unsigned __int8)sub_1F6E520(a1, a3, a4, (__int64)&v110, (__int64)&v112, (__int64)&v115)
    || !(unsigned __int8)sub_1F6E520(a1, a5, a6, (__int64)&v113, (__int64)&v114, (__int64)&v117) )
  {
    return 0;
  }
  v18 = *(_QWORD *)(a3 + 40) + 16LL * a4;
  v19 = *(_BYTE *)v18;
  v120 = *(const void ***)(v18 + 8);
  LOBYTE(v119) = v19;
  v20 = *(_QWORD *)(v110 + 40) + 16LL * (unsigned int)v111;
  v21 = *(_BYTE *)(a1 + 24) == 0;
  v22 = *(_BYTE *)v20;
  v23 = *(const void ***)(v20 + 8);
  LOBYTE(v121) = v22;
  v122 = v23;
  if ( v21 )
  {
    if ( v19 )
    {
      if ( (unsigned __int8)(v19 - 14) <= 0x5Fu )
      {
        switch ( v19 )
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
          case 33:
          case 34:
          case 35:
          case 36:
          case 37:
          case 38:
          case 39:
          case 40:
          case 41:
          case 42:
          case 43:
          case 44:
          case 45:
          case 46:
          case 47:
          case 48:
          case 49:
          case 50:
          case 51:
          case 52:
          case 53:
          case 54:
          case 55:
          case 62:
          case 63:
          case 64:
          case 65:
          case 66:
          case 67:
          case 68:
          case 69:
          case 70:
          case 71:
          case 72:
          case 73:
          case 74:
          case 75:
          case 76:
          case 77:
          case 78:
          case 79:
          case 80:
          case 81:
          case 82:
          case 83:
          case 84:
          case 85:
          case 86:
          case 87:
          case 88:
          case 89:
          case 90:
          case 91:
          case 92:
          case 93:
          case 94:
          case 95:
          case 96:
          case 97:
          case 98:
          case 99:
          case 100:
          case 101:
          case 102:
          case 103:
          case 104:
          case 105:
          case 106:
          case 107:
          case 108:
          case 109:
            goto LABEL_11;
          default:
            goto LABEL_13;
        }
      }
LABEL_10:
      if ( v19 == 2 )
        goto LABEL_13;
      goto LABEL_11;
    }
    if ( sub_1F58D20((__int64)&v119) )
    {
      v19 = sub_1F596B0((__int64)&v119);
      goto LABEL_10;
    }
  }
LABEL_11:
  v88 = *(_QWORD *)(a1 + 8);
  v101 = v122;
  v106 = v121;
  v91 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, const void **))(*(_QWORD *)v88 + 264LL);
  v94 = *(_QWORD *)(*(_QWORD *)a1 + 48LL);
  v24 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)a1 + 32LL));
  v16 = v91(v88, v24, v94, v106, v101);
  if ( (_BYTE)v119 != (_BYTE)v16 || !(_BYTE)v119 && v120 != v25 )
    return 0;
LABEL_13:
  v26 = v113.m128i_i64[0];
  v27 = *(_QWORD *)(v113.m128i_i64[0] + 40) + 16LL * v113.m128i_u32[2];
  if ( (_BYTE)v121 != *(_BYTE *)v27 )
    return 0;
  if ( (_BYTE)v121 )
  {
    v102 = *(_DWORD *)(v115 + 84);
    v107 = *(_DWORD *)(v117 + 84);
    v28 = (unsigned __int8)v121 - 2;
    LOBYTE(v16) = (unsigned __int8)(v121 - 2) <= 5u;
    v29 = (unsigned int)(unsigned __int8)v121 - 14;
    LOBYTE(v28) = (unsigned __int8)(v121 - 14) <= 0x47u;
    v30 = v28 | v16;
  }
  else
  {
    if ( v122 != *(const void ***)(v27 + 8) )
      return 0;
    v95 = v113.m128i_i64[0];
    v102 = *(_DWORD *)(v115 + 84);
    v107 = *(_DWORD *)(v117 + 84);
    LOBYTE(v38) = sub_1F58CF0((__int64)&v121);
    v26 = v95;
    v30 = v38;
  }
  if ( (_QWORD)v112 != (_QWORD)v114 || DWORD2(v112) != DWORD2(v114) || v107 != v102 || !(_BYTE)v30 )
  {
    if ( !a2 )
      goto LABEL_19;
    if ( v26 != v110 )
      goto LABEL_19;
    if ( (_DWORD)v111 != v113.m128i_i32[2] )
      goto LABEL_19;
    v54 = v102;
    if ( v107 != v102 )
      goto LABEL_19;
    goto LABEL_53;
  }
  v92 = v30;
  v39 = sub_1D1ADA0(v112, DWORD2(v112), v29, v26, v30, v17);
  v40 = v92;
  v96 = a2 & (v107 == 17);
  if ( v39 )
  {
    v41 = *(_QWORD *)(v39 + 88);
    if ( *(_DWORD *)(v41 + 32) <= 0x40u )
    {
      v43 = *(_QWORD *)(v41 + 24) == 0;
    }
    else
    {
      v89 = *(_DWORD *)(v41 + 32);
      v42 = sub_16A57B0(v41 + 24);
      v40 = v92;
      v43 = v89 == v42;
    }
    LODWORD(v44) = v112;
    v90 = v40;
    v93 = v43;
    v45 = sub_1F709E0(v112, DWORD2(v112));
    v47 = v93;
    v30 = v90;
    v29 = v45;
    if ( v96 )
    {
      if ( v93 )
      {
LABEL_45:
        v48 = *(__int64 **)a1;
        sub_1F80610((__int64)v123, a3);
        v49 = sub_1D332F0(
                v48,
                119,
                (__int64)v123,
                (unsigned int)v121,
                v122,
                0,
                *(double *)a7.m128i_i64,
                a8,
                a9,
                v110,
                (unsigned __int64)v111,
                *(_OWORD *)&v113);
LABEL_46:
        v51 = (unsigned __int64)v49;
        v52 = v50;
        sub_17CD270(v123);
        sub_1F81BC0(a1, v51);
        return sub_1F81070(*(__int64 **)a1, a10, v119, v120, v51, v52, (__m128)a7, a8, a9, v112, v107);
      }
      LOBYTE(v45) = 0;
      DWORD2(v44) = 0;
      LODWORD(v44) = 0;
      goto LABEL_74;
    }
  }
  else
  {
    v44 = v112;
    v45 = sub_1F709E0(v44, DWORD2(v44));
    v30 = v92;
    v47 = 0;
    v29 = v45;
  }
  LOBYTE(v44) = v107 == 18;
  if ( ((v107 == 18) & a2) != 0 && (_BYTE)v29 )
    goto LABEL_45;
  DWORD2(v44) = a2 ^ 1;
  LOBYTE(v45) = (a2 ^ 1) & (v107 == 22);
  if ( (_BYTE)v45 )
  {
    if ( v47 )
      goto LABEL_45;
    DWORD2(v44) = v45;
  }
LABEL_74:
  if ( ((v107 == 20) & v47 & BYTE8(v44)) != 0 )
    goto LABEL_45;
  LOBYTE(v46) = v107 == 20;
  v26 = v46 & (unsigned int)v47;
  v17 = (unsigned __int8)(v29 & v96);
  LOBYTE(v26) = a2 & v26;
  if ( !(_BYTE)v26 && (v81 = v29 & v45) != 0 )
  {
    v83 = 0;
  }
  else
  {
    v82 = v44 & DWORD2(v44);
    v81 = 0;
    v83 = v26;
    v26 = (unsigned int)v29 & v82;
  }
  if ( ((unsigned __int8)v29 & v96) != 0 || v83 || (_BYTE)v26 || v81 )
  {
    v84 = *(__int64 **)a1;
    sub_1F80610((__int64)v123, a3);
    v49 = sub_1D332F0(
            v84,
            118,
            (__int64)v123,
            (unsigned int)v121,
            v122,
            0,
            *(double *)a7.m128i_i64,
            a8,
            a9,
            v110,
            (unsigned __int64)v111,
            *(_OWORD *)&v113);
    goto LABEL_46;
  }
  if ( !a2 )
    goto LABEL_20;
  if ( v110 != v113.m128i_i64[0] )
    goto LABEL_20;
  if ( v113.m128i_i32[2] != (_DWORD)v111 )
    goto LABEL_20;
  v54 = v102;
  if ( v107 != v102 )
    goto LABEL_20;
LABEL_53:
  v98 = v30;
  v55 = sub_1D159C0((__int64)&v121, v54, v29, v26, v30, v17);
  LOBYTE(v30) = v98;
  if ( v55 > 1 && v102 == 22 )
  {
    if ( !v98 )
      goto LABEL_21;
    v56 = sub_1D185B0(v112);
    v57 = v98;
    if ( v56 )
    {
      v58 = sub_1D188A0(v114);
      v57 = v98;
      if ( v58 )
        goto LABEL_58;
    }
    v100 = v57;
    v85 = sub_1D188A0(v112);
    LOBYTE(v30) = v100;
    if ( v85 )
    {
      if ( sub_1D185B0(v114) )
      {
LABEL_58:
        v59 = sub_1D38BB0(*(_QWORD *)a1, 1, a10, (unsigned int)v121, v122, 0, a7, a8, a9, 0);
        v61 = v60;
        v62 = v59;
        *(_QWORD *)&v63 = sub_1D38BB0(*(_QWORD *)a1, 2, a10, (unsigned int)v121, v122, 0, a7, a8, a9, 0);
        v109 = v63;
        v103 = *(__int64 **)a1;
        sub_1F80610((__int64)v123, a3);
        *((_QWORD *)&v86 + 1) = v61;
        *(_QWORD *)&v86 = v62;
        v64 = sub_1D332F0(
                v103,
                52,
                (__int64)v123,
                (unsigned int)v121,
                v122,
                0,
                *(double *)a7.m128i_i64,
                a8,
                a9,
                v110,
                (unsigned __int64)v111,
                v86);
        v66 = v65;
        sub_17CD270(v123);
        sub_1F81BC0(a1, (__int64)v64);
        return sub_1F81070(*(__int64 **)a1, a10, v119, v120, (unsigned __int64)v64, v66, (__m128)a7, a8, a9, v109, 0xBu);
      }
      LOBYTE(v30) = v100;
    }
    goto LABEL_20;
  }
LABEL_19:
  if ( !(_BYTE)v30 )
    goto LABEL_21;
LABEL_20:
  v31 = *(_QWORD *)(a1 + 8);
  v32 = *(__int64 (**)())(*(_QWORD *)v31 + 200LL);
  if ( v32 == sub_1F3CA10
    || (v99 = v30,
        v67 = ((__int64 (__fastcall *)(__int64, _QWORD, const void **))v32)(v31, (unsigned int)v121, v122),
        LOBYTE(v30) = v99,
        v107 != v102)
    || !v67
    || (v68 = sub_1D18C00(a3, 1, a4), LOBYTE(v30) = v99, !v68)
    || (v69 = sub_1D18C00(a5, 1, a6), LOBYTE(v30) = v99, !v69)
    || (v107 != 17 || !a2) && (a2 == 1 || v107 != 22) )
  {
LABEL_21:
    v33 = v110;
    v34 = v113.m128i_i64[0];
    if ( v110 == (_QWORD)v114
      && (_DWORD)v111 == DWORD2(v114)
      && v113.m128i_i64[0] == (_QWORD)v112
      && DWORD2(v112) == v113.m128i_i32[2] )
    {
      v97 = v30;
      v53 = sub_1D16ED0(v107);
      a7 = _mm_loadu_si128(&v113);
      v34 = v114;
      v107 = v53;
      LOBYTE(v30) = v97;
      v113.m128i_i32[2] = DWORD2(v114);
      v113.m128i_i64[0] = v114;
      DWORD2(v114) = a7.m128i_i32[2];
      v33 = v110;
      *(_QWORD *)&v114 = a7.m128i_i64[0];
    }
    if ( v34 == v33 && (_DWORD)v111 == v113.m128i_i32[2] && (_QWORD)v112 == (_QWORD)v114 && DWORD2(v112) == DWORD2(v114) )
    {
      v35 = a2 ? sub_1D16F90(v102, v107, v30) : sub_1D16F10(v102, v107, v30);
      v36 = v35;
      if ( v35 != 24 )
      {
        if ( !*(_BYTE *)(a1 + 24) )
          return sub_1F81070(*(__int64 **)a1, a10, v119, v120, v110, v111, (__m128)a7, a8, a9, v112, v36);
        v37 = *(_QWORD *)(a1 + 8);
        if ( ((*(_DWORD *)(v37
                         + 4
                         * (((*(_BYTE *)(*(_QWORD *)(v110 + 40) + 16LL * (unsigned int)v111) >> 3) & 0x1F)
                          + 15LL * v35
                          + 18112)
                         + 12) >> (4 * (*(_BYTE *)(*(_QWORD *)(v110 + 40) + 16LL * (unsigned int)v111) & 7)))
            & 0xF) == 0
          && sub_1F6C830(v37, 0x89u, v121) )
        {
          return sub_1F81070(*(__int64 **)a1, a10, v119, v120, v110, v111, (__m128)a7, a8, a9, v112, v36);
        }
      }
    }
    return 0;
  }
  v104 = *(__int64 **)a1;
  sub_1F80610((__int64)v123, a3);
  v70 = sub_1D332F0(
          v104,
          120,
          (__int64)v123,
          (unsigned int)v121,
          v122,
          0,
          *(double *)a7.m128i_i64,
          a8,
          a9,
          v110,
          (unsigned __int64)v111,
          v112);
  v72 = v71;
  sub_17CD270(v123);
  v105 = *(__int64 **)a1;
  sub_1F80610((__int64)v123, a5);
  v73 = sub_1D332F0(
          v105,
          120,
          (__int64)v123,
          (unsigned int)v121,
          v122,
          0,
          *(double *)a7.m128i_i64,
          a8,
          a9,
          v113.m128i_i64[0],
          v113.m128i_u64[1],
          v114);
  v75 = v74;
  sub_17CD270(v123);
  *((_QWORD *)&v87 + 1) = v75;
  *(_QWORD *)&v87 = v73;
  v76 = sub_1D332F0(
          *(__int64 **)a1,
          119,
          a10,
          (unsigned int)v121,
          v122,
          0,
          *(double *)a7.m128i_i64,
          a8,
          a9,
          (__int64)v70,
          v72,
          v87);
  v78 = v77;
  v79 = (unsigned __int64)v76;
  *(_QWORD *)&v80 = sub_1D38BB0(*(_QWORD *)a1, 0, a10, (unsigned int)v121, v122, 0, a7, a8, a9, 0);
  return sub_1F81070(*(__int64 **)a1, a10, v119, v120, v79, v78, (__m128)a7, a8, a9, v80, v107);
}
