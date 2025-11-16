// Function: sub_2924A90
// Address: 0x2924a90
//
char __fastcall sub_2924A90(__int64 a1, __int64 a2)
{
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rcx
  char v12; // r12
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rax
  __int64 *v15; // rax
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // r13
  __int64 *v19; // rax
  unsigned __int8 *v20; // r13
  char result; // al
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // r15
  __int64 v28; // r8
  __int64 v29; // r15
  __int64 v30; // r9
  unsigned __int64 v31; // rcx
  unsigned __int64 v32; // r15
  __int64 v33; // rcx
  __int64 v34; // rdx
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  unsigned __int64 v37; // rax
  __int64 v38; // r11
  __int64 v39; // rdx
  __int16 v40; // cx
  __int64 v41; // rsi
  unsigned __int64 v42; // rax
  int v43; // ecx
  __int64 v44; // rax
  __int64 v45; // r9
  __int64 v46; // r15
  __int64 v47; // rdx
  __int64 v48; // rcx
  unsigned int v49; // esi
  __int64 v50; // rdi
  int v51; // eax
  bool v52; // al
  __int64 v53; // r9
  __int64 v54; // rax
  int v55; // r8d
  __int64 **v56; // rax
  unsigned __int64 v57; // rdx
  unsigned __int64 v58; // rdx
  int v59; // eax
  bool v60; // r8
  unsigned __int64 v61; // rax
  unsigned __int16 v62; // cx
  __int64 v63; // r13
  __int64 v64; // rdi
  unsigned int v65; // ebx
  unsigned __int64 v66; // r11
  __int64 v67; // rax
  __int64 v68; // rdx
  unsigned __int64 v69; // rax
  unsigned __int16 v70; // cx
  unsigned __int64 v71; // rax
  unsigned __int64 v72; // rax
  __int64 v73; // rdi
  __int64 v74; // r8
  __int64 v75; // rdx
  unsigned __int64 v76; // rsi
  _QWORD *v77; // rax
  __int64 *v78; // rax
  __int64 v79; // rax
  __int64 v80; // r15
  __int64 v81; // rdx
  __int64 v82; // rdi
  int v83; // eax
  bool v84; // r9
  unsigned __int8 v85; // r9
  unsigned __int64 v86; // rcx
  char v87; // si
  unsigned __int64 v88; // rax
  int v89; // ecx
  __int64 v90; // rax
  __int64 v91; // r13
  __int64 v92; // r14
  __int64 v93; // r14
  int v94; // eax
  __m128i v95; // rax
  unsigned __int8 *v96; // rdi
  __int64 v97; // rsi
  __int64 v98; // rcx
  __int64 v99; // rdx
  __int64 v100; // rax
  unsigned __int64 v101; // rdx
  __int64 v102; // rdx
  int v103; // esi
  unsigned int v104; // [rsp+4h] [rbp-BCh]
  unsigned int v105; // [rsp+8h] [rbp-B8h]
  unsigned __int8 v106; // [rsp+8h] [rbp-B8h]
  __int64 v107; // [rsp+8h] [rbp-B8h]
  __int64 v108; // [rsp+10h] [rbp-B0h]
  __int64 v109; // [rsp+10h] [rbp-B0h]
  unsigned int v110; // [rsp+10h] [rbp-B0h]
  __int64 v111; // [rsp+10h] [rbp-B0h]
  __int64 v112; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v113; // [rsp+18h] [rbp-A8h]
  __int64 v114; // [rsp+18h] [rbp-A8h]
  int v115; // [rsp+18h] [rbp-A8h]
  __int64 v116; // [rsp+18h] [rbp-A8h]
  unsigned int v117; // [rsp+20h] [rbp-A0h]
  __int64 v118; // [rsp+20h] [rbp-A0h]
  __int64 v119; // [rsp+20h] [rbp-A0h]
  __int64 v120; // [rsp+20h] [rbp-A0h]
  unsigned int v121; // [rsp+20h] [rbp-A0h]
  __int64 v122; // [rsp+20h] [rbp-A0h]
  __int64 v123; // [rsp+20h] [rbp-A0h]
  __int64 v124; // [rsp+20h] [rbp-A0h]
  __int64 v125; // [rsp+38h] [rbp-88h] BYREF
  __int64 v126; // [rsp+40h] [rbp-80h] BYREF
  __int64 v127; // [rsp+48h] [rbp-78h]
  __int64 v128; // [rsp+50h] [rbp-70h]
  __int64 v129; // [rsp+58h] [rbp-68h]
  __m128i v130; // [rsp+60h] [rbp-60h] BYREF
  __int64 v131; // [rsp+70h] [rbp-50h]
  __int16 v132; // [rsp+80h] [rbp-40h]

  sub_B91FC0(&v126, a2);
  v7 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( **(_BYTE **)(a2 + 32 * (2 - v7)) != 17 )
  {
    v8 = sub_291C360((__int64 *)a1, a1 + 176, *(_QWORD *)(*(_QWORD *)(a1 + 152) + 8LL));
    v9 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    if ( *(_QWORD *)v9 )
    {
      v10 = *(_QWORD *)(v9 + 8);
      **(_QWORD **)(v9 + 16) = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = *(_QWORD *)(v9 + 16);
    }
    *(_QWORD *)v9 = v8;
    if ( v8 )
    {
      v11 = *(_QWORD *)(v8 + 16);
      *(_QWORD *)(v9 + 8) = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 16) = v9 + 8;
      *(_QWORD *)(v9 + 16) = v8 + 16;
      *(_QWORD *)(v8 + 16) = v9;
    }
    v12 = -1;
    _BitScanReverse64(&v13, 1LL << *(_WORD *)(*(_QWORD *)(a1 + 32) + 2LL));
    v14 = -(__int64)((0x8000000000000000LL >> ((unsigned __int8)v13 ^ 0x3Fu))
                   | (*(_QWORD *)(a1 + 112) - *(_QWORD *)(a1 + 40)))
        & ((0x8000000000000000LL >> ((unsigned __int8)v13 ^ 0x3Fu)) | (*(_QWORD *)(a1 + 112) - *(_QWORD *)(a1 + 40)));
    if ( v14 )
    {
      _BitScanReverse64(&v14, v14);
      v12 = 63 - (v14 ^ 0x3F);
    }
    v15 = (__int64 *)sub_BD5C60(a2);
    *(_QWORD *)(a2 + 72) = sub_A7B980((__int64 *)(a2 + 72), v15, 1, 86);
    v16 = (__int64 *)sub_BD5C60(a2);
    v17 = sub_A77A40(v16, v12);
    v130.m128i_i32[0] = 0;
    v18 = v17;
    v19 = (__int64 *)sub_BD5C60(a2);
    *(_QWORD *)(a2 + 72) = sub_A7B660((__int64 *)(a2 + 72), v19, &v130, 1, v18);
    v20 = *(unsigned __int8 **)(a1 + 152);
    result = sub_F50EE0(v20, 0);
    if ( !result )
      return result;
    v92 = *(_QWORD *)(a1 + 16);
    v130 = (__m128i)4uLL;
    v93 = v92 + 216;
    v131 = (__int64)v20;
    LOBYTE(v22) = v20 != 0;
    if ( v20 + 4096 != 0 && v20 != 0 && v20 != (unsigned __int8 *)-8192LL )
      sub_BD73F0((__int64)&v130);
    sub_D6B260(v93, v130.m128i_i8, v22, v23, v24, v25);
    if ( v131 != -4096 && v131 != 0 && v131 != -8192 )
    {
      sub_BD60C0(&v130);
      return 0;
    }
    return 0;
  }
  v26 = *(_QWORD *)(a1 + 16);
  v130 = (__m128i)4uLL;
  v131 = a2;
  v27 = v26 + 216;
  if ( a2 != -4096 && a2 != -8192 )
    sub_BD73F0((__int64)&v130);
  sub_D6B260(v27, v130.m128i_i8, v7, v4, v5, v6);
  if ( v131 != 0 && v131 != -4096 && v131 != -8192 )
    sub_BD60C0(&v130);
  v28 = *(_QWORD *)(a1 + 32);
  v29 = *(_QWORD *)(v28 + 72);
  v30 = v29;
  if ( (unsigned int)*(unsigned __int8 *)(v29 + 8) - 17 <= 1 )
    v30 = **(_QWORD **)(v29 + 16);
  if ( *(_QWORD *)(a1 + 72) )
    goto LABEL_22;
  if ( *(_QWORD *)(a1 + 64) )
  {
LABEL_38:
    v66 = sub_291C410(
            a1,
            *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
            *(_DWORD *)(a1 + 120) - *(_DWORD *)(a1 + 112));
    if ( *(_QWORD *)(a1 + 64) )
    {
      v67 = *(_QWORD *)(a1 + 40);
      if ( *(_QWORD *)(a1 + 96) != v67 || v67 != *(_QWORD *)(a1 + 104) )
      {
        v68 = *(_QWORD *)(a1 + 32);
        v120 = v66;
        _BitScanReverse64(&v69, 1LL << *(_WORD *)(v68 + 2));
        LOBYTE(v70) = 63 - (v69 ^ 0x3F);
        HIBYTE(v70) = 1;
        v71 = sub_291B4B0((__int64 *)(a1 + 176), *(_QWORD *)(v68 + 72), v68, v70, "oldload");
        v72 = sub_291C8F0(*(_QWORD *)a1, (unsigned int **)(a1 + 176), v71, *(_QWORD *)(a1 + 64));
        v73 = *(_QWORD *)a1;
        v74 = *(_QWORD *)(a1 + 112) - *(_QWORD *)(a1 + 40);
        v132 = 259;
        v130.m128i_i64[0] = (__int64)"insert";
        v66 = sub_291CC20(v73, a1 + 176, v72, v120, v74, &v130);
      }
    }
    v46 = sub_291C8F0(*(_QWORD *)a1, (unsigned int **)(a1 + 176), v66, v29);
    goto LABEL_25;
  }
  v75 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( *(_QWORD *)(a1 + 96) > *(_QWORD *)(a1 + 40) || *(_QWORD *)(a1 + 104) < *(_QWORD *)(a1 + 48) )
    goto LABEL_50;
  v121 = *(_DWORD *)(v75 + 32);
  if ( v121 > 0x40 )
  {
    v116 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    v107 = v30;
    v111 = *(_QWORD *)(a1 + 32);
    v94 = sub_C444A0(v75 + 24);
    v75 = v116;
    if ( v121 - v94 > 0x40 )
      goto LABEL_50;
    v28 = v111;
    v30 = v107;
    v76 = **(_QWORD **)(v116 + 24);
  }
  else
  {
    v76 = *(_QWORD *)(v75 + 24);
  }
  v114 = v30;
  if ( v76 > 0xFFFFFFFF )
  {
LABEL_50:
    v104 = *(_QWORD *)(a1 + 120) - *(_DWORD *)(a1 + 112);
    v80 = sub_AD64C0(*(_QWORD *)(v75 + 8), v104, 0);
    v81 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    v82 = *(_QWORD *)(a2 + 32 * (3 - v81));
    if ( *(_DWORD *)(v82 + 32) <= 0x40u )
    {
      v84 = *(_QWORD *)(v82 + 24) == 0;
    }
    else
    {
      v115 = *(_DWORD *)(v82 + 32);
      v122 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      v83 = sub_C444A0(v82 + 24);
      v81 = v122;
      v84 = v115 == v83;
    }
    v85 = !v84;
    _BitScanReverse64(&v86, 1LL << *(_WORD *)(*(_QWORD *)(a1 + 32) + 2LL));
    v87 = -1;
    v88 = -(__int64)((0x8000000000000000LL >> ((unsigned __int8)v86 ^ 0x3Fu))
                   | (*(_QWORD *)(a1 + 112) - *(_QWORD *)(a1 + 40)))
        & ((0x8000000000000000LL >> ((unsigned __int8)v86 ^ 0x3Fu)) | (*(_QWORD *)(a1 + 112) - *(_QWORD *)(a1 + 40)));
    if ( v88 )
    {
      _BitScanReverse64(&v88, v88);
      v87 = 63 - (v88 ^ 0x3F);
    }
    v89 = 256;
    v106 = v85;
    LOBYTE(v89) = v87;
    v110 = v89;
    v123 = *(_QWORD *)(a2 + 32 * (1 - v81));
    v90 = sub_291C360((__int64 *)a1, a1 + 176, *(_QWORD *)(*(_QWORD *)(a1 + 152) + 8LL));
    v91 = sub_B34240(a1 + 176, v90, v123, v80, v110, v106, 0, 0, 0);
    if ( v126 || v127 || v128 || v129 )
    {
      sub_E00FE0(&v130, &v126, *(_QWORD *)(a1 + 112) - *(_QWORD *)(a1 + 96), v104);
      sub_B9A100(v91, v130.m128i_i64);
    }
    sub_29228E0(
      *(_QWORD *)(a1 + 24),
      *(_BYTE *)(a1 + 137),
      8LL * *(_QWORD *)(a1 + 112),
      8LL * *(_QWORD *)(a1 + 128),
      a2,
      v91,
      *(_QWORD *)(v91 - 32LL * (*(_DWORD *)(v91 + 4) & 0x7FFFFFF)),
      0);
    return 0;
  }
  v77 = (_QWORD *)sub_BD5C60(v28);
  v78 = (__int64 *)sub_BCB2B0(v77);
  v79 = sub_BCDA70(v78, v76);
  if ( !sub_29191E0(*(_QWORD *)a1, v79, v29)
    || (v124 = *(_QWORD *)a1,
        v95.m128i_i64[0] = sub_9208B0(*(_QWORD *)a1, v114),
        v130 = v95,
        v96 = *(unsigned __int8 **)(v124 + 32),
        v97 = *(_QWORD *)(v124 + 40),
        v125 = v95.m128i_i64[0],
        &v96[v97] == sub_29126F0(v96, (__int64)&v96[v97], &v125)) )
  {
    v75 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    goto LABEL_50;
  }
  if ( *(_QWORD *)(a1 + 72) )
  {
LABEL_22:
    v31 = *(_QWORD *)(a1 + 88);
    v32 = (*(_QWORD *)(a1 + 112) - *(_QWORD *)(a1 + 40)) / v31;
    v117 = (*(_QWORD *)(a1 + 120) - *(_QWORD *)(a1 + 40)) / v31 - v32;
    v33 = sub_9208B0(*(_QWORD *)a1, *(_QWORD *)(a1 + 80));
    v35 = v34;
    v130.m128i_i64[0] = v33;
    v36 = v33;
    LODWORD(v33) = *(_DWORD *)(a2 + 4);
    v130.m128i_i64[1] = v35;
    v37 = sub_291C410(a1, *(_QWORD *)(a2 + 32 * (1 - (v33 & 0x7FFFFFF))), v36 >> 3);
    v38 = sub_291C8F0(*(_QWORD *)a1, (unsigned int **)(a1 + 176), v37, *(_QWORD *)(a1 + 80));
    if ( v117 > 1 )
    {
      v130.m128i_i64[0] = (__int64)"vsplat";
      v132 = 259;
      v38 = sub_B37A60((unsigned int **)(a1 + 176), v117, v38, v130.m128i_i64);
    }
    v39 = *(_QWORD *)(a1 + 32);
    v118 = v38;
    v40 = *(_WORD *)(v39 + 2);
    v41 = *(_QWORD *)(v39 + 72);
    v132 = 259;
    _BitScanReverse64(&v42, 1LL << v40);
    v43 = (unsigned __int8)(63 - (v42 ^ 0x3F));
    v130.m128i_i64[0] = (__int64)"oldload";
    BYTE1(v43) = 1;
    v44 = sub_A82CA0((unsigned int **)(a1 + 176), v41, v39, v43, 0, (__int64)&v130);
    v130.m128i_i64[0] = (__int64)"vec";
    v132 = 259;
    v46 = sub_2918170(a1 + 176, v44, v118, (unsigned int)v32, &v130, v45);
    goto LABEL_25;
  }
  if ( *(_QWORD *)(a1 + 64) )
    goto LABEL_38;
  v98 = sub_9208B0(*(_QWORD *)a1, v114);
  v100 = v99;
  v130.m128i_i64[0] = v98;
  v101 = v98;
  LODWORD(v98) = *(_DWORD *)(a2 + 4);
  v130.m128i_i64[1] = v100;
  v102 = sub_291C410(a1, *(_QWORD *)(a2 + 32 * (1 - (v98 & 0x7FFFFFF))), v101 >> 3);
  if ( (unsigned int)*(unsigned __int8 *)(v29 + 8) - 17 <= 1 )
  {
    v103 = *(_DWORD *)(v29 + 32);
    v130.m128i_i64[0] = (__int64)"vsplat";
    v132 = 259;
    v102 = sub_B37A60((unsigned int **)(a1 + 176), v103, v102, v130.m128i_i64);
  }
  v46 = sub_291C8F0(*(_QWORD *)a1, (unsigned int **)(a1 + 176), v102, v29);
LABEL_25:
  v47 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v48 = *(_QWORD *)(a2 + 32 * (3 - v47));
  v49 = *(_DWORD *)(v48 + 32);
  v50 = v48 + 24;
  if ( v49 <= 0x40 )
  {
    v52 = *(_QWORD *)(v48 + 24) == 0;
  }
  else
  {
    v105 = *(_DWORD *)(v48 + 32);
    v108 = *(_QWORD *)(a2 + 32 * (3 - v47));
    v112 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    v119 = v48 + 24;
    v51 = sub_C444A0(v50);
    v49 = v105;
    v48 = v108;
    v47 = v112;
    v50 = v119;
    v52 = v105 == v51;
  }
  v53 = *(_QWORD *)(a1 + 32);
  if ( v52 )
    goto LABEL_58;
  v54 = *(_QWORD *)(v53 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v54 + 8) - 17 <= 1 )
    v54 = **(_QWORD **)(v54 + 16);
  v55 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 - 32 * v47) + 8LL) + 8LL) >> 8;
  if ( v55 == *(_DWORD *)(v54 + 8) >> 8 )
  {
LABEL_58:
    v58 = *(_QWORD *)(a1 + 32);
    if ( v49 > 0x40 )
      goto LABEL_32;
  }
  else
  {
    v56 = (__int64 **)sub_BCE3C0(*(__int64 **)(a1 + 248), v55);
    v57 = *(_QWORD *)(a1 + 32);
    v132 = 257;
    v58 = sub_291AC80((__int64 *)(a1 + 176), 0x32u, v57, v56, (__int64)&v130, 0, v125, 0);
    v53 = *(_QWORD *)(a1 + 32);
    v48 = *(_QWORD *)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    v49 = *(_DWORD *)(v48 + 32);
    v50 = v48 + 24;
    if ( v49 > 0x40 )
    {
LABEL_32:
      v109 = v53;
      v113 = v58;
      v59 = sub_C444A0(v50);
      v53 = v109;
      v58 = v113;
      v60 = v59 == v49;
      goto LABEL_33;
    }
  }
  v60 = *(_QWORD *)(v48 + 24) == 0;
LABEL_33:
  _BitScanReverse64(&v61, 1LL << *(_WORD *)(v53 + 2));
  LOBYTE(v62) = 63 - (v61 ^ 0x3F);
  HIBYTE(v62) = 1;
  v63 = sub_2463EC0((__int64 *)(a1 + 176), v46, v58, v62, !v60);
  v130.m128i_i64[0] = 0x190000000ALL;
  sub_B47C00(v63, a2, v130.m128i_i32, 2);
  if ( v126 || v127 || v128 || v129 )
  {
    sub_E00EB0(&v130, &v126, *(_QWORD *)(a1 + 112) - *(_QWORD *)(a1 + 96), *(_QWORD *)(v46 + 8), *(_QWORD *)a1);
    sub_B9A100(v63, v130.m128i_i64);
  }
  sub_29228E0(
    *(_QWORD *)(a1 + 24),
    *(_BYTE *)(a1 + 137),
    8LL * *(_QWORD *)(a1 + 112),
    8LL * *(_QWORD *)(a1 + 128),
    a2,
    v63,
    *(_QWORD *)(v63 - 32),
    v46);
  v64 = *(_QWORD *)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v65 = *(_DWORD *)(v64 + 32);
  if ( v65 <= 0x40 )
    return *(_QWORD *)(v64 + 24) == 0;
  else
    return v65 == (unsigned int)sub_C444A0(v64 + 24);
}
