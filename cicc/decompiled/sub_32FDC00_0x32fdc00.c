// Function: sub_32FDC00
// Address: 0x32fdc00
//
__int64 __fastcall sub_32FDC00(__int64 *a1, __int64 a2)
{
  __int64 v2; // r10
  const __m128i *v4; // rax
  unsigned __int64 v5; // r9
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // r8
  __int32 v9; // ebx
  unsigned __int64 v10; // r15
  int v11; // ecx
  unsigned __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // r13
  int v18; // ecx
  __int64 v19; // rax
  __int128 *v20; // rcx
  unsigned int v21; // r11d
  int v22; // eax
  int v23; // edi
  __int64 v24; // rsi
  char v25; // cl
  __int64 v26; // rax
  unsigned int v27; // edi
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // r14
  __int64 v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rcx
  __int64 v34; // rcx
  __int64 v35; // rsi
  __int128 *v36; // r15
  __int64 v37; // r14
  int v38; // r9d
  __int64 v39; // rax
  __int64 v40; // r12
  __int64 v41; // rdx
  __int64 v42; // r13
  __int64 v43; // r10
  __int64 v44; // r15
  __int64 v45; // rsi
  int v46; // r9d
  __int64 v47; // rsi
  unsigned __int16 *v48; // rax
  unsigned __int16 v49; // r14
  __int64 v50; // r8
  unsigned __int64 v51; // r15
  __int128 v52; // rax
  int v53; // r9d
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // r10
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // rsi
  __int64 v60; // rax
  unsigned __int64 v61; // r13
  __int64 v62; // rdx
  int v63; // esi
  int v64; // eax
  bool v65; // al
  __int64 *v66; // rax
  int v67; // edi
  __int64 v68; // rsi
  bool v69; // al
  __int64 *v70; // rax
  __int128 v71; // [rsp-50h] [rbp-1D0h]
  __int128 v72; // [rsp-30h] [rbp-1B0h]
  __int128 v73; // [rsp-20h] [rbp-1A0h]
  __int128 v74; // [rsp-20h] [rbp-1A0h]
  __int128 v75; // [rsp-20h] [rbp-1A0h]
  unsigned __int64 v76; // [rsp+8h] [rbp-178h]
  unsigned __int64 v77; // [rsp+10h] [rbp-170h]
  unsigned __int64 v78; // [rsp+10h] [rbp-170h]
  unsigned __int64 v79; // [rsp+18h] [rbp-168h]
  __int128 *v80; // [rsp+18h] [rbp-168h]
  __int64 v81; // [rsp+20h] [rbp-160h]
  __int64 v82; // [rsp+20h] [rbp-160h]
  __int64 v83; // [rsp+20h] [rbp-160h]
  __int64 v84; // [rsp+20h] [rbp-160h]
  unsigned int v85; // [rsp+28h] [rbp-158h]
  int v86; // [rsp+28h] [rbp-158h]
  __int64 v87; // [rsp+28h] [rbp-158h]
  __int64 v88; // [rsp+28h] [rbp-158h]
  int v89; // [rsp+30h] [rbp-150h]
  char v90; // [rsp+30h] [rbp-150h]
  unsigned int v91; // [rsp+30h] [rbp-150h]
  __int64 v92; // [rsp+38h] [rbp-148h]
  __int64 v93; // [rsp+40h] [rbp-140h]
  int v94; // [rsp+40h] [rbp-140h]
  unsigned __int64 v95; // [rsp+48h] [rbp-138h]
  unsigned __int64 v96; // [rsp+50h] [rbp-130h]
  unsigned int v97; // [rsp+5Ch] [rbp-124h]
  unsigned int v98; // [rsp+60h] [rbp-120h]
  unsigned __int64 v99; // [rsp+60h] [rbp-120h]
  int v100; // [rsp+70h] [rbp-110h]
  unsigned __int64 v101; // [rsp+70h] [rbp-110h]
  __int64 v102; // [rsp+70h] [rbp-110h]
  unsigned int v103; // [rsp+70h] [rbp-110h]
  __int64 v104; // [rsp+78h] [rbp-108h]
  __int128 *v105; // [rsp+78h] [rbp-108h]
  __int64 v106; // [rsp+78h] [rbp-108h]
  __int64 v107; // [rsp+78h] [rbp-108h]
  __int64 v108; // [rsp+80h] [rbp-100h]
  __int64 v109; // [rsp+80h] [rbp-100h]
  int v110; // [rsp+80h] [rbp-100h]
  __int64 v111; // [rsp+80h] [rbp-100h]
  int v112; // [rsp+80h] [rbp-100h]
  __int64 v113; // [rsp+80h] [rbp-100h]
  __int64 v114; // [rsp+88h] [rbp-F8h]
  __int128 v115; // [rsp+90h] [rbp-F0h]
  __int64 v116; // [rsp+B0h] [rbp-D0h] BYREF
  int v117; // [rsp+B8h] [rbp-C8h]
  __int64 v118; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v119; // [rsp+C8h] [rbp-B8h]
  __int64 v120; // [rsp+D0h] [rbp-B0h]
  __int64 v121; // [rsp+D8h] [rbp-A8h]
  __int64 v122; // [rsp+E0h] [rbp-A0h]
  __int64 *v123; // [rsp+E8h] [rbp-98h]
  __int64 v124; // [rsp+F0h] [rbp-90h]
  __int64 v125; // [rsp+F8h] [rbp-88h]
  __int64 v126; // [rsp+100h] [rbp-80h]
  int v127; // [rsp+108h] [rbp-78h]
  __int64 v128; // [rsp+110h] [rbp-70h]
  __int64 v129; // [rsp+118h] [rbp-68h]
  __int64 v130; // [rsp+120h] [rbp-60h] BYREF
  __int64 v131; // [rsp+128h] [rbp-58h]
  __int64 *v132; // [rsp+130h] [rbp-50h]
  __int64 v133; // [rsp+138h] [rbp-48h]
  __int64 v134; // [rsp+140h] [rbp-40h] BYREF

  v2 = a2;
  v4 = *(const __m128i **)(a2 + 40);
  v5 = v4[2].m128i_u64[1];
  v6 = v4->m128i_i64[0];
  v7 = v4->m128i_i64[1];
  v8 = v4->m128i_i64[0];
  v9 = v4[3].m128i_i32[0];
  v10 = v5;
  v115 = (__int128)_mm_loadu_si128(v4 + 5);
  v11 = *(_DWORD *)(v5 + 24);
  v12 = v4[3].m128i_u64[0];
  v13 = *(_QWORD *)(v5 + 56);
  if ( v11 == 52 )
  {
    if ( !v13 )
      return 0;
    v33 = *(_QWORD *)(v5 + 56);
    a2 = 1;
    do
    {
      while ( *(_DWORD *)(v33 + 8) != v9 )
      {
        v33 = *(_QWORD *)(v33 + 32);
        if ( !v33 )
          goto LABEL_55;
      }
      if ( !(_DWORD)a2 )
        goto LABEL_4;
      v34 = *(_QWORD *)(v33 + 32);
      if ( !v34 )
        goto LABEL_56;
      if ( *(_DWORD *)(v34 + 8) == v9 )
        goto LABEL_4;
      v33 = *(_QWORD *)(v34 + 32);
      a2 = 0;
    }
    while ( v33 );
LABEL_55:
    if ( (_DWORD)a2 == 1 )
      goto LABEL_4;
LABEL_56:
    v35 = *(_QWORD *)(v2 + 80);
    v36 = *(__int128 **)(v5 + 40);
    v37 = *a1;
    v38 = *(_DWORD *)(v2 + 28);
    v118 = v35;
    if ( v35 )
    {
      v106 = v2;
      v110 = v38;
      sub_B96E90((__int64)&v118, v35, 1);
      v2 = v106;
      v38 = v110;
    }
    LODWORD(v119) = *(_DWORD *)(v2 + 72);
    v39 = sub_340EC60(v37, 305, (unsigned int)&v118, 1, 0, v38, v6, v7, *v36, v115);
    v32 = v118;
    v16 = v39;
    if ( !v118 )
      return v16;
LABEL_46:
    sub_B91220((__int64)&v118, v32);
    return v16;
  }
  if ( v11 != 208 )
  {
    if ( !v13 )
      return 0;
    goto LABEL_4;
  }
  v18 = 1;
  if ( !v13 )
    goto LABEL_38;
  do
  {
    while ( *(_DWORD *)(v13 + 8) != v9 )
    {
      v13 = *(_QWORD *)(v13 + 32);
      if ( !v13 )
        goto LABEL_22;
    }
    if ( !v18 )
      goto LABEL_38;
    v19 = *(_QWORD *)(v13 + 32);
    if ( !v19 )
      goto LABEL_23;
    if ( *(_DWORD *)(v19 + 8) == v9 )
      goto LABEL_38;
    v13 = *(_QWORD *)(v19 + 32);
    v18 = 0;
  }
  while ( v13 );
LABEL_22:
  if ( v18 == 1 )
    goto LABEL_38;
LABEL_23:
  v20 = *(__int128 **)(v5 + 40);
  v21 = *((_DWORD *)v20 + 12);
  v96 = *((_QWORD *)v20 + 1);
  v104 = *(_QWORD *)v20;
  v98 = *((_DWORD *)v20 + 2);
  v95 = *((_QWORD *)v20 + 6);
  v108 = *((_QWORD *)v20 + 5);
  v97 = *(_DWORD *)(*((_QWORD *)v20 + 10) + 96LL);
  v22 = *(_DWORD *)(*(_QWORD *)v20 + 24LL);
  if ( v22 == 11 || (v93 = 0, v22 == 35) )
    v93 = *(_QWORD *)v20;
  v23 = *(_DWORD *)(v108 + 24);
  v100 = v23;
  if ( v23 == 11 || (v24 = 0, v23 == 35) )
    v24 = *((_QWORD *)v20 + 5);
  if ( v22 != 52 )
    goto LABEL_28;
  v60 = *(_QWORD *)(v104 + 56);
  if ( !v60 )
    goto LABEL_28;
  v92 = v7;
  v61 = v12;
  v62 = v24;
  v63 = 1;
  while ( v98 != *(_DWORD *)(v60 + 8) )
  {
LABEL_84:
    v60 = *(_QWORD *)(v60 + 32);
    if ( !v60 )
    {
      v67 = v63;
      v68 = v62;
      v12 = v61;
      v7 = v92;
      if ( v67 != 1 )
        goto LABEL_97;
      goto LABEL_28;
    }
  }
  if ( !v63 )
    goto LABEL_90;
  v60 = *(_QWORD *)(v60 + 32);
  if ( v60 )
  {
    if ( v98 != *(_DWORD *)(v60 + 8) )
    {
      v63 = 0;
      goto LABEL_84;
    }
LABEL_90:
    v12 = v61;
    v7 = v92;
LABEL_28:
    if ( v100 != 52 )
      goto LABEL_39;
    goto LABEL_29;
  }
  v68 = v62;
  v12 = v61;
  v7 = v92;
LABEL_97:
  if ( !v68 )
  {
    if ( v100 != 52 )
      goto LABEL_39;
    v26 = *(_QWORD *)(v108 + 56);
    if ( !v26 )
      goto LABEL_38;
    v25 = 0;
    goto LABEL_30;
  }
  v76 = v12;
  v78 = v5;
  v84 = v2;
  v88 = v8;
  v91 = *((_DWORD *)v20 + 12);
  v80 = *(__int128 **)(v5 + 40);
  v69 = sub_32642F0(v97, v68);
  v21 = v91;
  v8 = v88;
  v2 = v84;
  v5 = v78;
  v12 = v76;
  if ( !v69 )
  {
    v70 = *(__int64 **)(v104 + 40);
    v104 = *v70;
    v96 = *((unsigned int *)v70 + 2) | v96 & 0xFFFFFFFF00000000LL;
    v98 = *((_DWORD *)v70 + 2);
    if ( v100 == 52 )
    {
      v25 = 1;
      v26 = *(_QWORD *)(v108 + 56);
      if ( v26 )
        goto LABEL_30;
    }
LABEL_74:
    v47 = *(_QWORD *)(v10 + 80);
    v102 = *a1;
    v94 = *(_DWORD *)(v2 + 28);
    v48 = *(unsigned __int16 **)(v10 + 48);
    v49 = *v48;
    v50 = *((_QWORD *)v48 + 1);
    v118 = v47;
    if ( v47 )
    {
      v81 = v2;
      v85 = v21;
      v89 = v50;
      sub_B96E90((__int64)&v118, v47, 1);
      v2 = v81;
      v21 = v85;
      LODWORD(v50) = v89;
    }
    v82 = v2;
    LODWORD(v119) = *(_DWORD *)(v10 + 72);
    v86 = v50;
    v51 = v21 | v95 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v52 = sub_33ED040(v102, v97);
    *((_QWORD *)&v74 + 1) = v51;
    *(_QWORD *)&v74 = v108;
    *((_QWORD *)&v72 + 1) = v98 | v96 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v72 = v104;
    v54 = sub_340F900(v102, 208, (unsigned int)&v118, v49, v86, v53, v72, v74, v52);
    v56 = v82;
    v57 = v54;
    v58 = v55;
    v59 = *(_QWORD *)(v82 + 80);
    v116 = v59;
    if ( v59 )
    {
      v114 = v55;
      v113 = v54;
      sub_B96E90((__int64)&v116, v59, 1);
      v56 = v82;
      v57 = v113;
      v58 = v114;
    }
    *((_QWORD *)&v75 + 1) = v58;
    *(_QWORD *)&v75 = v57;
    v117 = *(_DWORD *)(v56 + 72);
    v16 = sub_340EC60(v102, 305, (unsigned int)&v116, 1, 0, v94, v6, v7, v75, v115);
    if ( v116 )
      sub_B91220((__int64)&v116, v116);
    v32 = v118;
    if ( !v118 )
      return v16;
    goto LABEL_46;
  }
  v20 = v80;
  if ( v100 != 52 )
    goto LABEL_39;
LABEL_29:
  v25 = 0;
  v26 = *(_QWORD *)(v108 + 56);
  if ( !v26 )
    goto LABEL_38;
LABEL_30:
  a2 = 1;
  while ( 2 )
  {
    if ( v21 != *(_DWORD *)(v26 + 8) )
    {
LABEL_31:
      v26 = *(_QWORD *)(v26 + 32);
      if ( !v26 )
      {
        a2 = (unsigned int)a2 ^ 1;
        goto LABEL_92;
      }
      continue;
    }
    break;
  }
  if ( !(_DWORD)a2 )
    goto LABEL_73;
  v26 = *(_QWORD *)(v26 + 32);
  if ( v26 )
  {
    if ( *(_DWORD *)(v26 + 8) == v21 )
      goto LABEL_73;
    a2 = 0;
    goto LABEL_31;
  }
  a2 = 1;
LABEL_92:
  if ( v93 )
  {
    if ( (_BYTE)a2 )
    {
      v77 = v12;
      v79 = v5;
      v83 = v2;
      v87 = v8;
      v90 = v25;
      v103 = v21;
      v64 = sub_33CBD20(v97);
      a2 = v93;
      v65 = sub_32642F0(v64, v93);
      v21 = v103;
      v25 = v90;
      v8 = v87;
      v2 = v83;
      v5 = v79;
      v12 = v77;
      if ( !v65 )
      {
        v66 = *(__int64 **)(v108 + 40);
        v21 = *((_DWORD *)v66 + 2);
        v108 = *v66;
        v95 = v21 | v95 & 0xFFFFFFFF00000000LL;
        goto LABEL_74;
      }
    }
  }
LABEL_73:
  if ( v25 )
    goto LABEL_74;
  if ( *(_DWORD *)(v10 + 24) != 208 )
    goto LABEL_68;
LABEL_38:
  v20 = *(__int128 **)(v10 + 40);
LABEL_39:
  a2 = a1[1];
  v27 = 1;
  v28 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v20 + 48LL) + 16LL * *((unsigned int *)v20 + 2));
  if ( (_WORD)v28 != 1 && (!(_WORD)v28 || (v27 = (unsigned __int16)v28, !*(_QWORD *)(a2 + 8 * v28 + 112)))
    || (*(_BYTE *)(a2 + 500LL * v27 + 6720) & 0xFB) != 0 )
  {
LABEL_68:
    v13 = *(_QWORD *)(v10 + 56);
    if ( !v13 )
      return 0;
LABEL_4:
    v14 = 1;
    do
    {
      while ( *(_DWORD *)(v13 + 8) != v9 )
      {
        v13 = *(_QWORD *)(v13 + 32);
        if ( !v13 )
          goto LABEL_11;
      }
      if ( !(_DWORD)v14 )
        return 0;
      v15 = *(_QWORD *)(v13 + 32);
      if ( !v15 )
        goto LABEL_60;
      if ( *(_DWORD *)(v15 + 8) == v9 )
        return 0;
      v13 = *(_QWORD *)(v15 + 32);
      v14 = 0;
    }
    while ( v13 );
LABEL_11:
    if ( (_DWORD)v14 == 1 )
      return 0;
LABEL_60:
    v99 = v12;
    v101 = v5;
    v107 = v2;
    v111 = v8;
    v132 = &v118;
    v124 = sub_33ECD10(1, a2, v12, v14, v8, v5);
    v126 = 0x100000000LL;
    v129 = 0xFFFFFFFFLL;
    v134 = 0;
    v118 = 0;
    v119 = 0;
    v120 = 0;
    v121 = 328;
    v122 = -65536;
    v125 = 0;
    v127 = 0;
    v128 = 0;
    v133 = 0;
    v130 = v6;
    LODWORD(v131) = v7;
    v134 = *(_QWORD *)(v111 + 56);
    if ( v134 )
      *(_QWORD *)(v134 + 24) = &v134;
    v133 = v111 + 56;
    *(_QWORD *)(v111 + 56) = &v130;
    LODWORD(v126) = 1;
    v123 = &v130;
    v40 = sub_32FC610(a1, v101, v99, v111 + 56, v111, v101);
    v42 = v41;
    if ( v40 )
    {
      v43 = v107;
      v44 = *a1;
      v45 = *(_QWORD *)(v107 + 80);
      v46 = *(_DWORD *)(v107 + 28);
      v116 = v45;
      if ( v45 )
      {
        v112 = v46;
        sub_B96E90((__int64)&v116, v45, 1);
        v43 = v107;
        v46 = v112;
      }
      *((_QWORD *)&v73 + 1) = v42;
      *(_QWORD *)&v73 = v40;
      v117 = *(_DWORD *)(v43 + 72);
      v16 = sub_340EC60(v44, 305, (unsigned int)&v116, 1, 0, v46, v130, v131, v73, v115);
      if ( v116 )
        sub_B91220((__int64)&v116, v116);
      sub_33CF710(&v118);
      return v16;
    }
    sub_33CF710(&v118);
    return 0;
  }
  v29 = *(_QWORD *)(v2 + 80);
  v30 = *a1;
  v118 = v29;
  if ( v29 )
  {
    v105 = v20;
    v109 = v2;
    sub_B96E90((__int64)&v118, v29, 1);
    v20 = v105;
    v2 = v109;
  }
  LODWORD(v119) = *(_DWORD *)(v2 + 72);
  *((_QWORD *)&v71 + 1) = v7;
  *(_QWORD *)&v71 = v6;
  v31 = sub_33FC1D0(v30, 306, (unsigned int)&v118, 1, 0, v5, v71, v20[5], *v20, *(__int128 *)((char *)v20 + 40), v115);
  v32 = v118;
  v16 = v31;
  if ( v118 )
    goto LABEL_46;
  return v16;
}
