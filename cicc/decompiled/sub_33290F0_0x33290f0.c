// Function: sub_33290F0
// Address: 0x33290f0
//
__int64 __fastcall sub_33290F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v10; // r12
  __int64 v12; // rax
  unsigned __int16 v13; // r13
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // rax
  char v17; // al
  unsigned __int16 v18; // r15
  __int64 v19; // r8
  unsigned __int16 v20; // si
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int16 *v23; // r12
  __int64 v24; // r15
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  unsigned int v31; // eax
  __int64 v32; // r8
  unsigned int v33; // r15d
  __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rdx
  char v37; // al
  __int64 v38; // rdx
  __int64 v39; // rdi
  unsigned int v40; // edx
  unsigned __int16 v41; // r14
  __int16 v42; // ax
  __int64 v43; // rdi
  __int64 v45; // rax
  unsigned __int64 v46; // rsi
  __int64 v47; // rdx
  char v48; // r9
  __int64 v49; // rax
  __int64 v50; // rdi
  __int64 v51; // rdx
  __int64 v52; // rax
  unsigned __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rsi
  unsigned __int64 v56; // r8
  __int64 v57; // rdx
  char v58; // r9
  __int64 v59; // rax
  __int64 v60; // rdi
  __int64 v61; // rdx
  __int64 v62; // rax
  unsigned __int64 v63; // rdx
  unsigned __int64 v64; // r8
  __int64 v65; // rdx
  char v66; // cl
  __int64 v67; // rax
  __int64 v68; // rsi
  __int64 v69; // rdx
  __int64 v70; // rax
  unsigned __int64 v71; // rdx
  unsigned int v72; // edx
  __int64 v73; // r15
  __int64 v74; // rdx
  char v75; // r13
  __int64 v76; // rcx
  __int64 v77; // rdx
  __int64 v78; // rax
  __int64 v79; // rdx
  int v80; // eax
  __int64 v81; // rdi
  __int64 v82; // [rsp+8h] [rbp-1C8h]
  unsigned __int64 v83; // [rsp+8h] [rbp-1C8h]
  unsigned __int64 v84; // [rsp+8h] [rbp-1C8h]
  __int64 v85; // [rsp+10h] [rbp-1C0h]
  char v86; // [rsp+10h] [rbp-1C0h]
  __int64 v87; // [rsp+18h] [rbp-1B8h]
  __int64 v88; // [rsp+20h] [rbp-1B0h]
  char v89; // [rsp+20h] [rbp-1B0h]
  int v90; // [rsp+28h] [rbp-1A8h]
  int v91; // [rsp+30h] [rbp-1A0h]
  int v92; // [rsp+38h] [rbp-198h]
  char v93; // [rsp+47h] [rbp-189h]
  int v94; // [rsp+48h] [rbp-188h]
  __int64 v95; // [rsp+50h] [rbp-180h]
  __int64 v96; // [rsp+50h] [rbp-180h]
  char v97; // [rsp+50h] [rbp-180h]
  __int64 v98; // [rsp+58h] [rbp-178h]
  __int64 v99; // [rsp+80h] [rbp-150h] BYREF
  __int64 v100; // [rsp+88h] [rbp-148h]
  unsigned __int16 v101; // [rsp+90h] [rbp-140h] BYREF
  __int64 v102; // [rsp+98h] [rbp-138h]
  unsigned __int64 v103; // [rsp+A0h] [rbp-130h]
  __int64 v104; // [rsp+A8h] [rbp-128h]
  __int16 v105; // [rsp+B0h] [rbp-120h] BYREF
  __int64 v106; // [rsp+B8h] [rbp-118h]
  __int64 v107; // [rsp+C0h] [rbp-110h]
  __int64 v108; // [rsp+C8h] [rbp-108h]
  __int64 v109; // [rsp+D0h] [rbp-100h]
  __int64 v110; // [rsp+D8h] [rbp-F8h]
  unsigned __int16 v111; // [rsp+E0h] [rbp-F0h] BYREF
  __int64 v112; // [rsp+E8h] [rbp-E8h]
  __int64 v113; // [rsp+F0h] [rbp-E0h]
  __int64 v114; // [rsp+F8h] [rbp-D8h]
  __int64 v115; // [rsp+100h] [rbp-D0h]
  __int64 v116; // [rsp+108h] [rbp-C8h]
  __int64 v117; // [rsp+110h] [rbp-C0h]
  __int64 v118; // [rsp+118h] [rbp-B8h]
  __int64 v119; // [rsp+120h] [rbp-B0h]
  __int64 v120; // [rsp+128h] [rbp-A8h]
  __int64 v121; // [rsp+130h] [rbp-A0h]
  __int64 v122; // [rsp+138h] [rbp-98h]
  __int64 v123; // [rsp+140h] [rbp-90h] BYREF
  __int64 v124; // [rsp+148h] [rbp-88h]
  __int64 v125; // [rsp+150h] [rbp-80h]
  __int64 v126; // [rsp+158h] [rbp-78h]
  __int128 v127; // [rsp+160h] [rbp-70h] BYREF
  __int64 v128; // [rsp+170h] [rbp-60h]
  __m128i v129; // [rsp+180h] [rbp-50h] BYREF
  __int64 v130; // [rsp+190h] [rbp-40h]
  __int64 v131; // [rsp+198h] [rbp-38h]

  v10 = 16LL * (unsigned int)a3;
  v100 = a5;
  v99 = a4;
  v91 = a8;
  v92 = a3;
  v90 = a9;
  v94 = a6;
  v12 = v10 + *(_QWORD *)(a2 + 48);
  v13 = *(_WORD *)v12;
  v88 = *(_QWORD *)(v12 + 8);
  v102 = v88;
  v14 = *(_QWORD *)(a1 + 16);
  v101 = v13;
  v15 = sub_3007410((__int64)&a7, *(__int64 **)(v14 + 64), a3, a4, a5, a6);
  v16 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 16) + 40LL));
  v17 = sub_AE5260(v16, v15);
  v18 = v99;
  v19 = v100;
  v93 = v17;
  if ( v13 == (_WORD)v99 )
  {
    if ( v13 )
      goto LABEL_3;
    if ( v88 == v100 )
    {
      v20 = a7;
      v21 = *((_QWORD *)&a7 + 1);
      if ( !(_WORD)a7 )
      {
LABEL_63:
        if ( v21 == v19 )
          goto LABEL_5;
        v111 = v20;
        v112 = v21;
        goto LABEL_28;
      }
LABEL_27:
      v111 = v20;
      v112 = v21;
      if ( v20 )
      {
        if ( v20 == 1 || (unsigned __int16)(v20 - 504) <= 7u )
          goto LABEL_89;
        v56 = *(_QWORD *)&byte_444C4A0[16 * v20 - 16];
        v58 = byte_444C4A0[16 * v20 - 8];
        if ( !(_WORD)v99 )
          goto LABEL_29;
        goto LABEL_56;
      }
LABEL_28:
      v115 = sub_3007260((__int64)&v111);
      v56 = v115;
      v116 = v57;
      v58 = v57;
      if ( !(_WORD)v99 )
      {
LABEL_29:
        v83 = v56;
        v86 = v58;
        v59 = sub_3007260((__int64)&v99);
        v56 = v83;
        v58 = v86;
        v60 = v59;
        v62 = v61;
        v113 = v60;
        v63 = v60;
        v114 = v62;
        goto LABEL_30;
      }
LABEL_56:
      if ( (_WORD)v99 == 1 || (unsigned __int16)(v99 - 504) <= 7u )
        goto LABEL_89;
      v63 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v99 - 16];
      LOBYTE(v62) = byte_444C4A0[16 * (unsigned __int16)v99 - 8];
LABEL_30:
      if ( (!(_BYTE)v62 || v58)
        && v56 > v63
        && (!v20
         || !v18
         || (((int)*(unsigned __int16 *)(*(_QWORD *)(a1 + 8) + 2 * (v18 + 274LL * v20 + 71704) + 6) >> 4) & 0xB) != 0) )
      {
        return 0;
      }
      goto LABEL_5;
    }
    v106 = v100;
    v105 = 0;
    goto LABEL_15;
  }
  v105 = v99;
  v106 = v100;
  if ( !(_WORD)v99 )
  {
LABEL_15:
    v45 = sub_3007260((__int64)&v105);
    v19 = v100;
    v109 = v45;
    v46 = v45;
    v110 = v47;
    v48 = v47;
    goto LABEL_16;
  }
  if ( (_WORD)v99 == 1 || (unsigned __int16)(v99 - 504) <= 7u )
    goto LABEL_89;
  v46 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v99 - 16];
  v48 = byte_444C4A0[16 * (unsigned __int16)v99 - 8];
LABEL_16:
  if ( v13 )
  {
    if ( v13 == 1 || (unsigned __int16)(v13 - 504) <= 7u )
      goto LABEL_89;
    v53 = *(_QWORD *)&byte_444C4A0[16 * v13 - 16];
    LOBYTE(v52) = byte_444C4A0[16 * v13 - 8];
  }
  else
  {
    v82 = v19;
    v97 = v48;
    v49 = sub_3007260((__int64)&v101);
    v19 = v82;
    v50 = v49;
    v52 = v51;
    v48 = v97;
    v107 = v50;
    v53 = v50;
    v108 = v52;
  }
  if ( !(_BYTE)v52 && v48 || v46 >= v53 )
  {
LABEL_3:
    v20 = a7;
    v21 = *((_QWORD *)&a7 + 1);
    if ( (_WORD)a7 == (_WORD)v99 )
    {
      if ( (_WORD)v99 )
        goto LABEL_5;
      goto LABEL_63;
    }
    goto LABEL_27;
  }
  v54 = *(unsigned __int16 *)(*(_QWORD *)(a2 + 48) + v10);
  if ( !(_WORD)v54 )
    return 0;
  v55 = *(_QWORD *)(a1 + 8);
  if ( !*(_QWORD *)(v55 + 8 * v54 + 112)
    || !(_WORD)v99
    || (*(_BYTE *)((unsigned __int16)v99 + v55 + 274LL * (unsigned __int16)v54 + 443718) & 0xFB) != 0 )
  {
    return 0;
  }
  v20 = a7;
  if ( (_WORD)a7 != (_WORD)v99 )
  {
    v21 = *((_QWORD *)&a7 + 1);
    goto LABEL_27;
  }
LABEL_5:
  v22 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 16) + 40LL));
  v23 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + v10);
  v24 = v22;
  v25 = *v23;
  v129.m128i_i64[1] = *((_QWORD *)v23 + 1);
  v26 = *(_QWORD *)(a1 + 16);
  v129.m128i_i16[0] = v25;
  v30 = sub_3007410((__int64)&v129, *(__int64 **)(v26 + 64), v25, v27, v28, v29);
  v31 = sub_AE5260(v24, v30);
  v32 = *(_QWORD *)(a1 + 16);
  v33 = v31;
  if ( (_WORD)v99 )
  {
    if ( (_WORD)v99 == 1 || (unsigned __int16)(v99 - 504) <= 7u )
      goto LABEL_89;
    v35 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v99 - 16];
    v37 = byte_444C4A0[16 * (unsigned __int16)v99 - 8];
  }
  else
  {
    v95 = *(_QWORD *)(a1 + 16);
    v34 = sub_3007260((__int64)&v99);
    v32 = v95;
    v117 = v34;
    v35 = v34;
    v118 = v36;
    v37 = v36;
  }
  LOBYTE(v104) = v37;
  v103 = (unsigned __int64)(v35 + 7) >> 3;
  v85 = sub_33EDE90(v32, v103, v104, v33);
  v87 = v38;
  sub_2EAC300((__int64)&v127, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 40LL), *(_DWORD *)(v85 + 96), 0);
  if ( v13 == (_WORD)v99 )
  {
    if ( v13 || v88 == v100 )
    {
      v39 = *(_QWORD *)(a1 + 16);
LABEL_10:
      v129 = 0u;
      v130 = 0;
      v131 = 0;
      v96 = sub_33F4560(v39, v91, v90, v94, a2, v92, v85, v87, v127, v128, v33, 0, (__int64)&v129);
      v98 = v40;
      goto LABEL_11;
    }
    v124 = v100;
    LOWORD(v123) = 0;
    goto LABEL_38;
  }
  LOWORD(v123) = v99;
  v124 = v100;
  if ( !(_WORD)v99 )
  {
LABEL_38:
    v121 = sub_3007260((__int64)&v123);
    v64 = v121;
    v122 = v65;
    v66 = v65;
    goto LABEL_39;
  }
  if ( (_WORD)v99 == 1 || (unsigned __int16)(v99 - 504) <= 7u )
    goto LABEL_89;
  v64 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v99 - 16];
  v66 = byte_444C4A0[16 * (unsigned __int16)v99 - 8];
LABEL_39:
  if ( v13 )
  {
    if ( v13 == 1 || (unsigned __int16)(v13 - 504) <= 7u )
      goto LABEL_89;
    v71 = *(_QWORD *)&byte_444C4A0[16 * v13 - 16];
    LOBYTE(v70) = byte_444C4A0[16 * v13 - 8];
  }
  else
  {
    v84 = v64;
    v89 = v66;
    v67 = sub_3007260((__int64)&v101);
    v64 = v84;
    v66 = v89;
    v68 = v67;
    v70 = v69;
    v119 = v68;
    v71 = v68;
    v120 = v70;
  }
  v39 = *(_QWORD *)(a1 + 16);
  if ( !(_BYTE)v70 && v66 || v71 <= v64 )
    goto LABEL_10;
  v129 = 0u;
  v130 = 0;
  v131 = 0;
  v96 = sub_33F5040(v39, v91, v90, v94, a2, v92, v85, v87, v127, v128, v99, v100, v33, 0, (__int64)&v129);
  v98 = v72;
LABEL_11:
  v41 = v99;
  v129 = _mm_loadu_si128((const __m128i *)&a7);
  if ( (_WORD)a7 == (_WORD)v99 )
  {
    if ( (_WORD)v99 || v129.m128i_i64[1] == v100 )
      goto LABEL_13;
    goto LABEL_68;
  }
  if ( !(_WORD)a7 )
  {
LABEL_68:
    v125 = sub_3007260((__int64)&v129);
    v73 = v125;
    v126 = v74;
    v75 = v74;
    goto LABEL_69;
  }
  if ( (_WORD)a7 == 1 || (unsigned __int16)(a7 - 504) <= 7u )
    goto LABEL_89;
  v73 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)a7 - 16];
  v75 = byte_444C4A0[16 * (unsigned __int16)a7 - 8];
LABEL_69:
  if ( !v41 )
  {
    v76 = sub_3007260((__int64)&v99);
    v78 = v77;
    v123 = v76;
    v79 = v76;
    v124 = v78;
    goto LABEL_71;
  }
  if ( v41 == 1 || (unsigned __int16)(v41 - 504) <= 7u )
LABEL_89:
    BUG();
  v79 = *(_QWORD *)&byte_444C4A0[16 * v41 - 16];
  LOBYTE(v78) = byte_444C4A0[16 * v41 - 8];
LABEL_71:
  if ( v73 == v79 && (_BYTE)v78 == v75 )
  {
LABEL_13:
    HIBYTE(v42) = 1;
    LOBYTE(v42) = v93;
    v43 = *(_QWORD *)(a1 + 16);
    v129 = 0u;
    v130 = 0;
    v131 = 0;
    return sub_33F1F00(v43, a7, DWORD2(a7), v94, v96, v98, v85, v87, v127, v128, v42, 0, (__int64)&v129, 0);
  }
  v80 = 256;
  LOBYTE(v80) = v93;
  v81 = *(_QWORD *)(a1 + 16);
  v129 = 0u;
  v130 = 0;
  v131 = 0;
  return sub_33F1DB0(v81, 1, v94, a7, DWORD2(a7), v80, v96, v98, v85, v87, v127, v128, v99, v100, 0, (__int64)&v129);
}
