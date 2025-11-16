// Function: sub_3465D80
// Address: 0x3465d80
//
unsigned __int8 *__fastcall sub_3465D80(
        __m128i a1,
        __int64 a2,
        _QWORD *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        unsigned __int16 a8,
        int a9,
        __int128 a10)
{
  __int64 v11; // rbx
  __int64 v12; // rsi
  unsigned __int16 *v13; // rax
  unsigned int v14; // edx
  __int64 v15; // rcx
  __int64 v16; // r8
  unsigned __int8 *v17; // r15
  unsigned int v18; // ebx
  unsigned __int8 *v19; // r14
  unsigned __int16 v20; // ax
  __int64 v21; // rdx
  unsigned __int64 v22; // rsi
  __int64 v23; // rdx
  unsigned __int16 v24; // dx
  unsigned __int64 v25; // r8
  unsigned int v26; // r11d
  __m128i v27; // xmm0
  __int64 v28; // rcx
  unsigned int v29; // r10d
  unsigned __int16 v30; // si
  __int64 v31; // rcx
  int v32; // eax
  __int64 v33; // r9
  __int64 v34; // rax
  __int64 v35; // rdx
  unsigned __int8 *v36; // rax
  unsigned int v37; // r11d
  unsigned int v38; // r10d
  unsigned __int8 *v39; // r14
  __int64 v40; // rdx
  __int64 v41; // r15
  unsigned int v42; // ebx
  __int128 v43; // rax
  __int64 v44; // r9
  __int128 v45; // rax
  __int64 v46; // r9
  unsigned __int8 *v47; // rax
  unsigned int v48; // edx
  __int64 v49; // rax
  __int64 v50; // rcx
  __int64 v51; // rax
  bool v52; // al
  __int64 v53; // rsi
  __int128 v54; // rax
  __int64 v55; // r9
  unsigned __int8 *v56; // rax
  unsigned int v57; // edx
  __int64 v58; // rbx
  __int64 v59; // rax
  unsigned __int16 v60; // r15
  __int64 v61; // rax
  __int128 v62; // rax
  __int64 v63; // r9
  unsigned int v64; // edx
  unsigned __int8 *v65; // r14
  __int64 v67; // rax
  __int64 v68; // rdx
  __int128 v69; // rax
  __int64 v70; // r9
  unsigned int v71; // edx
  unsigned __int64 v72; // rax
  __int64 v73; // rcx
  __int64 v74; // rcx
  unsigned int v75; // r14d
  unsigned int v76; // r14d
  unsigned int v77; // ebx
  __int64 v78; // rdx
  unsigned __int64 v79; // rdx
  char v80; // al
  __int128 v81; // rax
  __int64 v82; // r9
  unsigned __int8 *v83; // rax
  unsigned int v84; // edx
  unsigned __int64 v85; // rax
  __int64 v86; // rax
  __int128 v87; // [rsp-30h] [rbp-180h]
  unsigned __int16 v88; // [rsp+8h] [rbp-148h]
  char v89; // [rsp+Fh] [rbp-141h]
  unsigned int v90; // [rsp+18h] [rbp-138h]
  unsigned int v91; // [rsp+1Ch] [rbp-134h]
  __int64 v92; // [rsp+28h] [rbp-128h]
  __int64 v93; // [rsp+28h] [rbp-128h]
  unsigned int v94; // [rsp+28h] [rbp-128h]
  unsigned __int64 v95; // [rsp+30h] [rbp-120h]
  unsigned int v96; // [rsp+38h] [rbp-118h]
  unsigned int v97; // [rsp+38h] [rbp-118h]
  unsigned int v98; // [rsp+38h] [rbp-118h]
  unsigned int v99; // [rsp+38h] [rbp-118h]
  unsigned int v102; // [rsp+50h] [rbp-100h]
  unsigned int v103; // [rsp+50h] [rbp-100h]
  __m128i v104; // [rsp+70h] [rbp-E0h] BYREF
  unsigned __int64 v105; // [rsp+80h] [rbp-D0h]
  __int64 v106; // [rsp+88h] [rbp-C8h]
  __int64 v107; // [rsp+90h] [rbp-C0h] BYREF
  int v108; // [rsp+98h] [rbp-B8h]
  unsigned __int16 v109; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v110; // [rsp+A8h] [rbp-A8h]
  unsigned int v111; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v112; // [rsp+B8h] [rbp-98h]
  __m128i v113; // [rsp+C0h] [rbp-90h] BYREF
  unsigned __int64 v114; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v115; // [rsp+D8h] [rbp-78h]
  __int64 v116; // [rsp+E0h] [rbp-70h]
  __int64 v117; // [rsp+E8h] [rbp-68h]
  unsigned __int64 v118; // [rsp+F0h] [rbp-60h] BYREF
  __int64 v119; // [rsp+F8h] [rbp-58h]
  __int64 v120; // [rsp+100h] [rbp-50h]
  __int64 v121; // [rsp+108h] [rbp-48h]
  unsigned __int64 v122; // [rsp+110h] [rbp-40h] BYREF
  __int64 v123; // [rsp+118h] [rbp-38h]

  v11 = a10;
  v12 = *(_QWORD *)(a10 + 80);
  v104.m128i_i64[0] = a6;
  v104.m128i_i64[1] = a7;
  v107 = v12;
  if ( v12 )
    sub_B96E90((__int64)&v107, v12, 1);
  v108 = *(_DWORD *)(v11 + 72);
  v13 = (unsigned __int16 *)(*(_QWORD *)(a4 + 48) + 16LL * (unsigned int)a5);
  v17 = sub_33FB310((__int64)a3, a10, DWORD2(a10), (__int64)&v107, *v13, *((_QWORD *)v13 + 1), a1);
  v18 = v14;
  *(_QWORD *)&a10 = v17;
  v19 = v17;
  v96 = v14;
  *((_QWORD *)&a10 + 1) = v14 | *((_QWORD *)&a10 + 1) & 0xFFFFFFFF00000000LL;
  if ( v104.m128i_i16[0] )
  {
    v20 = word_4456580[v104.m128i_u16[0] - 1];
    v21 = 0;
  }
  else
  {
    v20 = sub_3009970((__int64)&v104, 0xFFFFFFFF00000000LL, v14, v15, v16);
  }
  v109 = v20;
  v110 = v21;
  if ( v20 )
  {
    if ( v20 == 1 || (unsigned __int16)(v20 - 504) <= 7u )
      goto LABEL_77;
    v22 = *(_QWORD *)&byte_444C4A0[16 * v20 - 16];
  }
  else
  {
    v116 = sub_3007260((__int64)&v109);
    v22 = v116;
    v117 = v23;
  }
  v24 = a8;
  v95 = v22 >> 3;
  if ( a8 )
  {
    LOBYTE(v25) = (unsigned __int16)(a8 - 176) <= 0x34u;
    v26 = word_4456340[a8 - 1];
  }
  else
  {
    v72 = sub_3007240((__int64)&a8);
    v24 = 0;
    v105 = v72;
    v26 = v72;
    v25 = HIDWORD(v72);
  }
  v27 = _mm_loadu_si128((const __m128i *)&a10);
  v92 = v18;
  v113 = _mm_load_si128(&v104);
  if ( v104.m128i_i16[0] )
  {
    v28 = *((_QWORD *)v17 + 6) + 16LL * v18;
    v29 = word_4456340[v104.m128i_u16[0] - 1];
    v30 = *(_WORD *)v28;
    v31 = *(_QWORD *)(v28 + 8);
    LOWORD(v114) = v30;
    v115 = v31;
    if ( (unsigned __int16)(v104.m128i_i16[0] - 176) > 0x34u )
      goto LABEL_23;
  }
  else
  {
    v88 = v24;
    v89 = v25;
    v90 = v26;
    v49 = sub_3007240((__int64)&v113);
    v50 = *((_QWORD *)v17 + 6) + 16LL * v18;
    v30 = *(_WORD *)v50;
    v106 = v49;
    v91 = v49;
    v51 = *(_QWORD *)(v50 + 8);
    LOWORD(v114) = v30;
    v115 = v51;
    v52 = sub_3007100((__int64)&v113);
    v29 = v91;
    v26 = v90;
    LOBYTE(v25) = v89;
    v24 = v88;
    if ( !v52 )
    {
LABEL_23:
      if ( !v29 )
        goto LABEL_24;
      if ( (v29 & (v29 - 1)) != 0 || v26 != 1 )
      {
        if ( v26 < v29 )
        {
          v53 = v29 - v26;
          goto LABEL_25;
        }
LABEL_24:
        v53 = 0;
LABEL_25:
        *(_QWORD *)&v54 = sub_3400BD0((__int64)a3, v53, (__int64)&v107, (unsigned int)v114, v115, 0, v27, 0);
        v56 = sub_3406EB0(
                a3,
                0xB6u,
                (__int64)&v107,
                (unsigned int)v114,
                v115,
                v55,
                __PAIR128__(v18 | v27.m128i_i64[1] & 0xFFFFFFFF00000000LL, (unsigned __int64)v17),
                v54);
        v96 = v57;
        v24 = a8;
        v19 = v56;
        goto LABEL_26;
      }
      _BitScanReverse(&v75, v29);
      v76 = v75 ^ 0x1F;
      v77 = 31 - v76;
      if ( v30 )
      {
        if ( v30 == 1 || (unsigned __int16)(v30 - 504) <= 7u )
          goto LABEL_77;
        v86 = 16LL * (v30 - 1);
        v79 = *(_QWORD *)&byte_444C4A0[v86];
        v80 = byte_444C4A0[v86 + 8];
      }
      else
      {
        v120 = sub_3007260((__int64)&v114);
        v121 = v78;
        v79 = v120;
        v80 = v121;
      }
      v122 = v79;
      LOBYTE(v123) = v80;
      LODWORD(v119) = sub_CA1930(&v122);
      if ( (unsigned int)v119 > 0x40 )
      {
        sub_C43690((__int64)&v118, 0, 0);
        if ( !v77 )
          goto LABEL_64;
        v85 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v76 + 33);
        if ( (unsigned int)v119 > 0x40 )
        {
          *(_QWORD *)v118 |= v85;
          goto LABEL_64;
        }
      }
      else
      {
        v118 = 0;
        if ( !v77 )
          goto LABEL_64;
        v85 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v76 + 33);
      }
      v118 |= v85;
LABEL_64:
      *(_QWORD *)&v81 = sub_34007B0((__int64)a3, (__int64)&v118, (__int64)&v107, v114, v115, 0, v27, 0);
      v83 = sub_3406EB0(
              a3,
              0xBAu,
              (__int64)&v107,
              (unsigned int)v114,
              v115,
              v82,
              __PAIR128__(v92 | v27.m128i_i64[1] & 0xFFFFFFFF00000000LL, (unsigned __int64)v17),
              v81);
      v96 = v84;
      v19 = v83;
      if ( (unsigned int)v119 > 0x40 && v118 )
        j_j___libc_free_0_0(v118);
      v24 = a8;
      goto LABEL_26;
    }
  }
  if ( (_BYTE)v25 )
    goto LABEL_23;
  v32 = *((_DWORD *)v17 + 6);
  v33 = v29;
  if ( v32 != 11 && v32 != 35
    || ((v73 = *((_QWORD *)v17 + 12), *(_DWORD *)(v73 + 32) <= 0x40u)
      ? (v74 = *(_QWORD *)(v73 + 24))
      : (v74 = **(_QWORD **)(v73 + 24)),
        v74 + (unsigned __int64)(v26 - 1) >= v29) )
  {
    if ( v30 )
    {
      if ( v30 == 1 || (unsigned __int16)(v30 - 504) <= 7u )
        goto LABEL_77;
      v34 = *(_QWORD *)&byte_444C4A0[16 * v30 - 16];
    }
    else
    {
      v93 = v29;
      v97 = v29;
      v102 = v26;
      v34 = sub_3007260((__int64)&v114);
      v33 = v93;
      v29 = v97;
      v118 = v34;
      v26 = v102;
      v119 = v35;
    }
    LODWORD(v123) = v34;
    if ( (unsigned int)v34 > 0x40 )
    {
      v94 = v29;
      v99 = v26;
      sub_C43690((__int64)&v122, v33, 0);
      v29 = v94;
      v26 = v99;
    }
    else
    {
      v122 = v33;
    }
    v98 = v29;
    v103 = v26;
    v36 = sub_3401900((__int64)a3, (__int64)&v107, v114, v115, (__int64)&v122, 1, v27);
    v37 = v103;
    v38 = v98;
    v39 = v36;
    v41 = v40;
    if ( (unsigned int)v123 > 0x40 && v122 )
    {
      j_j___libc_free_0_0(v122);
      v38 = v98;
      v37 = v103;
    }
    v42 = v38 < v37 ? 85 : 57;
    *(_QWORD *)&v43 = sub_3400BD0((__int64)a3, v37, (__int64)&v107, (unsigned int)v114, v115, 0, v27, 0);
    *((_QWORD *)&v87 + 1) = v41;
    *(_QWORD *)&v87 = v39;
    *(_QWORD *)&v45 = sub_3406EB0(a3, v42, (__int64)&v107, (unsigned int)v114, v115, v44, v87, v43);
    v47 = sub_3406EB0(a3, 0xB6u, (__int64)&v107, (unsigned int)v114, v115, v46, a10, v45);
    v96 = v48;
    v19 = v47;
    v24 = a8;
  }
LABEL_26:
  v58 = v96;
  v59 = *((_QWORD *)v19 + 6) + 16LL * v96;
  v60 = *(_WORD *)v59;
  v61 = *(_QWORD *)(v59 + 8);
  LOWORD(v111) = v60;
  v112 = v61;
  if ( v24 )
  {
    if ( (unsigned __int16)(v24 - 176) > 0x34u )
      goto LABEL_28;
  }
  else if ( !sub_3007100((__int64)&a8) )
  {
    goto LABEL_28;
  }
  if ( !v60 )
  {
    v67 = sub_3007260((__int64)&v111);
    v122 = v67;
    v123 = v68;
    goto LABEL_38;
  }
  if ( v60 == 1 || (unsigned __int16)(v60 - 504) <= 7u )
LABEL_77:
    BUG();
  v68 = 16LL * (v60 - 1);
  v67 = *(_QWORD *)&byte_444C4A0[v68];
  LOBYTE(v68) = byte_444C4A0[v68 + 8];
LABEL_38:
  v113.m128i_i64[0] = v67;
  v113.m128i_i8[8] = v68;
  LODWORD(v115) = sub_CA1930(&v113);
  if ( (unsigned int)v115 > 0x40 )
    sub_C43690((__int64)&v114, 1, 0);
  else
    v114 = 1;
  *(_QWORD *)&v69 = sub_3401900((__int64)a3, (__int64)&v107, v111, v112, (__int64)&v114, 1, v27);
  *(_QWORD *)&a10 = v19;
  *((_QWORD *)&a10 + 1) = v96 | *((_QWORD *)&a10 + 1) & 0xFFFFFFFF00000000LL;
  v19 = sub_3406EB0(
          a3,
          0x3Au,
          (__int64)&v107,
          v111,
          v112,
          v70,
          __PAIR128__(*((unsigned __int64 *)&a10 + 1), (unsigned __int64)v19),
          v69);
  v58 = v71;
  if ( (unsigned int)v115 > 0x40 && v114 )
    j_j___libc_free_0_0(v114);
LABEL_28:
  *(_QWORD *)&v62 = sub_3400BD0((__int64)a3, (unsigned int)v95, (__int64)&v107, v111, v112, 0, v27, 0);
  *(_QWORD *)&a10 = v19;
  *((_QWORD *)&a10 + 1) = v58 | *((_QWORD *)&a10 + 1) & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&a10 = sub_3406EB0(a3, 0x3Au, (__int64)&v107, v111, v112, v63, a10, v62);
  v65 = sub_34092D0(a3, a4, a5, a10, v64 | *((_QWORD *)&a10 + 1) & 0xFFFFFFFF00000000LL, (__int64)&v107, v27, 0);
  if ( v107 )
    sub_B91220((__int64)&v107, v107);
  return v65;
}
