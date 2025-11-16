// Function: sub_17CA660
// Address: 0x17ca660
//
__int64 __fastcall sub_17CA660(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  unsigned int v6; // esi
  __int64 *v7; // rdx
  __int64 v8; // r9
  __int64 v9; // r15
  int v11; // edx
  __int64 v12; // r13
  const char *v13; // rax
  __int64 v14; // r15
  __int64 v15; // rax
  _QWORD *v16; // r13
  __int64 *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  _QWORD *v20; // rax
  char v21; // al
  __int64 *v22; // r14
  int v23; // eax
  char v24; // cl
  __int64 v25; // rax
  size_t v26; // rdx
  size_t v27; // r14
  __int64 v28; // rax
  _QWORD *v29; // rax
  int v30; // edx
  __int64 v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 **v34; // rax
  __int64 v35; // rax
  __int128 v36; // rdi
  __int64 v37; // rcx
  char v38; // r13
  __int64 v39; // rcx
  char v40; // r13
  _QWORD *v41; // r12
  char v42; // al
  unsigned int v43; // esi
  __int64 v44; // rcx
  __int64 v45; // r8
  unsigned int v46; // edx
  __int64 *v47; // rax
  __int64 v48; // rdi
  _QWORD *v49; // rdx
  _BYTE *v50; // rsi
  __int64 v51; // rdx
  _BYTE *v52; // rsi
  __int64 *v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rcx
  _QWORD *v56; // rax
  _QWORD *v57; // r10
  __int64 v58; // r9
  char v59; // al
  __int64 v60; // r10
  __int64 **v61; // rax
  int v62; // r10d
  int v63; // r11d
  __int64 *v64; // r10
  int v65; // edi
  int v66; // edi
  int v67; // eax
  int v68; // r8d
  __int64 v69; // rsi
  unsigned int v70; // edx
  __int64 v71; // r9
  int v72; // r11d
  __int64 *v73; // r10
  int v74; // eax
  int v75; // r8d
  __int64 v76; // rsi
  int v77; // r11d
  unsigned int v78; // edx
  __int64 v79; // r9
  char v80; // [rsp+Ch] [rbp-1E4h]
  char v81; // [rsp+10h] [rbp-1E0h]
  __int64 **v82; // [rsp+10h] [rbp-1E0h]
  __int64 *v83; // [rsp+18h] [rbp-1D8h]
  __int64 v84; // [rsp+18h] [rbp-1D8h]
  __int64 *v85; // [rsp+18h] [rbp-1D8h]
  __int64 v86; // [rsp+20h] [rbp-1D0h]
  __int64 v87; // [rsp+28h] [rbp-1C8h]
  int *v88; // [rsp+28h] [rbp-1C8h]
  const void *v89; // [rsp+28h] [rbp-1C8h]
  __int64 v90; // [rsp+30h] [rbp-1C0h]
  __int64 v91; // [rsp+38h] [rbp-1B8h]
  __int64 *v92; // [rsp+38h] [rbp-1B8h]
  __int64 v93; // [rsp+38h] [rbp-1B8h]
  _QWORD *v94; // [rsp+38h] [rbp-1B8h]
  _QWORD *v95; // [rsp+38h] [rbp-1B8h]
  __int64 v96; // [rsp+38h] [rbp-1B8h]
  __int64 v97; // [rsp+38h] [rbp-1B8h]
  __int64 **v98; // [rsp+40h] [rbp-1B0h]
  __int64 v99; // [rsp+48h] [rbp-1A8h]
  __int64 v100; // [rsp+50h] [rbp-1A0h]
  __int64 v101; // [rsp+50h] [rbp-1A0h]
  __int64 v102; // [rsp+58h] [rbp-198h]
  __int64 v103; // [rsp+68h] [rbp-188h] BYREF
  _QWORD v104[2]; // [rsp+70h] [rbp-180h] BYREF
  __m128i v105; // [rsp+80h] [rbp-170h] BYREF
  _QWORD *v106; // [rsp+90h] [rbp-160h]
  const void **v107; // [rsp+A0h] [rbp-150h] BYREF
  __int16 v108; // [rsp+B0h] [rbp-140h]
  const void *v109[2]; // [rsp+C0h] [rbp-130h] BYREF
  _QWORD v110[2]; // [rsp+D0h] [rbp-120h] BYREF
  _QWORD v111[2]; // [rsp+E0h] [rbp-110h] BYREF
  __int64 v112; // [rsp+F0h] [rbp-100h]
  __int64 v113; // [rsp+F8h] [rbp-F8h]
  __int64 v114; // [rsp+100h] [rbp-F0h]
  __int64 v115; // [rsp+108h] [rbp-E8h]
  __int64 *v116; // [rsp+110h] [rbp-E0h]
  _QWORD *v117; // [rsp+120h] [rbp-D0h] BYREF
  size_t v118; // [rsp+128h] [rbp-C8h]
  _QWORD v119[4]; // [rsp+130h] [rbp-C0h] BYREF
  __int64 v120; // [rsp+150h] [rbp-A0h]

  v103 = sub_1649C60(*(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  v4 = *(unsigned int *)(a1 + 136);
  if ( !(_DWORD)v4 )
  {
LABEL_8:
    v106 = 0;
    v105 = 0u;
    goto LABEL_9;
  }
  v5 = *(_QWORD *)(a1 + 120);
  v6 = (v4 - 1) & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
  v7 = (__int64 *)(v5 + 32LL * v6);
  v8 = *v7;
  if ( v103 != *v7 )
  {
    v11 = 1;
    while ( v8 != -8 )
    {
      v62 = v11 + 1;
      v6 = (v4 - 1) & (v11 + v6);
      v7 = (__int64 *)(v5 + 32LL * v6);
      v8 = *v7;
      if ( v103 == *v7 )
        goto LABEL_3;
      v11 = v62;
    }
    goto LABEL_8;
  }
LABEL_3:
  v106 = 0;
  v105 = 0u;
  if ( v7 == (__int64 *)(v5 + 32 * v4) )
  {
LABEL_9:
    v86 = 0;
    v87 = 0;
    goto LABEL_10;
  }
  v9 = v7[2];
  if ( v9 )
    return v9;
  v106 = (_QWORD *)v7[3];
  v105 = _mm_loadu_si128((const __m128i *)(v7 + 1));
  v87 = v105.m128i_u32[1];
  v86 = v105.m128i_u32[0];
LABEL_10:
  v12 = *(_QWORD *)(a1 + 40);
  v100 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL);
  if ( sub_1695900(v100) )
  {
    v111[0] = v12 + 240;
    LOWORD(v112) = 260;
    sub_16E1010((__int64)&v117, (__int64)v111);
    v13 = "__profc_";
    if ( HIDWORD(v120) != 1 )
      v13 = "__profv_";
    v14 = (__int64)v13;
    if ( v117 != v119 )
      j_j___libc_free_0(v117, v119[0] + 1LL);
    sub_17C5450((__int64 *)&v117, a2, v14, 8);
    v99 = sub_1633B90(v12, v117, v118);
    if ( v117 != v119 )
      j_j___libc_free_0(v117, v119[0] + 1LL);
  }
  else
  {
    v99 = 0;
  }
  v15 = *(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  if ( *(_DWORD *)(v15 + 32) <= 0x40u )
    v90 = *(_QWORD *)(v15 + 24);
  else
    v90 = **(_QWORD **)(v15 + 24);
  v16 = **(_QWORD ***)(a1 + 40);
  v17 = (__int64 *)sub_1643360(v16);
  v83 = sub_1645D80(v17, v90);
  v81 = *(_BYTE *)(v103 + 32) & 0xF;
  v91 = sub_15A06D0((__int64 **)v83, v90, v18, v19);
  sub_17C5450((__int64 *)&v117, a2, (__int64)"__profc_", 8);
  v111[0] = &v117;
  LOWORD(v112) = 260;
  v20 = sub_1648A60(88, 1u);
  v9 = (__int64)v20;
  if ( v20 )
    sub_15E51E0((__int64)v20, *(_QWORD *)(a1 + 40), (__int64)v83, 0, v81, v91, (__int64)v111, 0, 0, 0, 0);
  if ( v117 != v119 )
    j_j___libc_free_0(v117, v119[0] + 1LL);
  v21 = *(_BYTE *)(v103 + 32) & 0x30 | *(_BYTE *)(v9 + 32) & 0xCF;
  *(_BYTE *)(v9 + 32) = v21;
  if ( (v21 & 0xFu) - 7 <= 1 || (v21 & 0x30) != 0 && (v21 & 0xF) != 9 )
    *(_BYTE *)(v9 + 33) |= 0x40u;
  sub_1694890((__int64)&v117, 1, *(_DWORD *)(a1 + 100), 1u);
  sub_15E5D20(v9, v117, v118);
  if ( v117 != v119 )
    j_j___libc_free_0(v117, v119[0] + 1LL);
  sub_15E4CC0(v9, 8u);
  *(_QWORD *)(v9 + 48) = v99;
  v82 = (__int64 **)sub_16471D0(v16, 0);
  v84 = sub_1599A20(v82);
  if ( byte_4FA3A60 && !sub_17C5230(*(_QWORD *)(a1 + 40)) && v87 + v86 )
  {
    v53 = (__int64 *)sub_1643360(v16);
    v85 = sub_1645D80(v53, v87 + v86);
    v80 = *(_BYTE *)(v103 + 32) & 0xF;
    v93 = sub_15A06D0((__int64 **)v85, v87 + v86, v54, v55);
    sub_17C5450((__int64 *)&v117, a2, (__int64)"__profvp_", 9);
    LOWORD(v112) = 260;
    v111[0] = &v117;
    v56 = sub_1648A60(88, 1u);
    v57 = v56;
    if ( v56 )
    {
      v58 = v93;
      v94 = v56;
      sub_15E51E0((__int64)v56, *(_QWORD *)(a1 + 40), (__int64)v85, 0, v80, v58, (__int64)v111, 0, 0, 0, 0);
      v57 = v94;
    }
    if ( v117 != v119 )
    {
      v95 = v57;
      j_j___libc_free_0(v117, v119[0] + 1LL);
      v57 = v95;
    }
    v59 = *(_BYTE *)(v103 + 32) & 0x30 | v57[4] & 0xCF;
    *((_BYTE *)v57 + 32) = v59;
    if ( (v59 & 0xFu) - 7 <= 1 || (v59 & 0x30) != 0 && (v59 & 0xF) != 9 )
      *((_BYTE *)v57 + 33) |= 0x40u;
    v96 = (__int64)v57;
    sub_1694890((__int64)&v117, 3, *(_DWORD *)(a1 + 100), 1u);
    sub_15E5D20(v96, v117, v118);
    v60 = v96;
    if ( v117 != v119 )
    {
      j_j___libc_free_0(v117, v119[0] + 1LL);
      v60 = v96;
    }
    v97 = v60;
    sub_15E4CC0(v60, 8u);
    *(_QWORD *)(v97 + 48) = v99;
    v61 = (__int64 **)sub_16471D0(v16, 0);
    v84 = sub_15A4510((__int64 ***)v97, v61, 0);
  }
  v22 = (__int64 *)sub_1643340(v16);
  v92 = sub_1645D80(v22, 2);
  v111[0] = sub_1643360(v16);
  v111[1] = sub_1643360(v16);
  v112 = sub_1647230(v16, 0);
  v113 = sub_16471D0(v16, 0);
  v114 = sub_16471D0(v16, 0);
  v115 = sub_1643350(v16);
  v116 = v92;
  v98 = (__int64 **)sub_1645600(v16, v111, 7, 0);
  v23 = *(_BYTE *)(v100 + 32) & 0xF;
  v24 = *(_BYTE *)(v100 + 32) & 0xF;
  if ( (unsigned int)(v23 - 2) <= 1 )
    goto LABEL_32;
  if ( v23 != 7 )
  {
    if ( v24 == 1 )
      goto LABEL_81;
    if ( v23 != 8 )
      goto LABEL_35;
LABEL_32:
    if ( v24 != 1 )
    {
LABEL_33:
      if ( (unsigned int)(v23 - 7) > 1 )
      {
LABEL_34:
        if ( !(unsigned __int8)sub_15E3650(v100, 0) && (*(_BYTE *)(v100 + 32) & 0xFu) - 2 > 1 )
          goto LABEL_59;
LABEL_35:
        v101 = sub_15A4510((__int64 ***)v100, v82, 0);
        goto LABEL_36;
      }
      goto LABEL_60;
    }
LABEL_81:
    if ( (unsigned __int8)sub_1560180(v100 + 112, 3) )
      goto LABEL_59;
    v23 = *(_BYTE *)(v100 + 32) & 0xF;
    goto LABEL_33;
  }
LABEL_60:
  if ( !*(_QWORD *)(v100 + 48) )
    goto LABEL_34;
LABEL_59:
  v101 = sub_1599A20(v82);
LABEL_36:
  v104[0] = sub_159C470((__int64)v22, v86, 0);
  v104[1] = sub_159C470((__int64)v22, v87, 0);
  v25 = sub_1649C60(*(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  v88 = (int *)sub_1694C30(v25);
  v27 = v26;
  sub_16C1840(&v117);
  sub_16C1A90((int *)&v117, v88, v27);
  sub_16C1AA0(&v117, v109);
  v89 = v109[0];
  v28 = sub_1643360(v16);
  v29 = (_QWORD *)sub_159C470(v28, (__int64)v89, 0);
  v30 = *(_DWORD *)(a2 + 20);
  v117 = v29;
  v31 = *(_QWORD *)(a2 + 24 * (1LL - (v30 & 0xFFFFFFF)));
  if ( *(_DWORD *)(v31 + 32) <= 0x40u )
    v32 = *(_QWORD *)(v31 + 24);
  else
    v32 = **(_QWORD **)(v31 + 24);
  v33 = sub_1643360(v16);
  v118 = sub_159C470(v33, v32, 0);
  v34 = (__int64 **)sub_1647230(v16, 0);
  v119[0] = sub_15A4510((__int64 ***)v9, v34, 0);
  v119[1] = v101;
  v119[2] = v84;
  v35 = sub_1643350(v16);
  *(_QWORD *)&v36 = v92;
  *((_QWORD *)&v36 + 1) = v104;
  v119[3] = sub_159C470(v35, v90, 0);
  v120 = sub_159DFD0(v36, 2, v37);
  v38 = *(_BYTE *)(v103 + 32);
  v102 = sub_159F090(v98, (__int64 *)&v117, 7, v39);
  v40 = v38 & 0xF;
  sub_17C5450((__int64 *)v109, a2, (__int64)"__profd_", 8);
  v108 = 260;
  v107 = v109;
  v41 = sub_1648A60(88, 1u);
  if ( v41 )
    sub_15E51E0((__int64)v41, *(_QWORD *)(a1 + 40), (__int64)v98, 0, v40, v102, (__int64)&v107, 0, 0, 0, 0);
  if ( v109[0] != v110 )
    j_j___libc_free_0(v109[0], v110[0] + 1LL);
  v42 = *(_BYTE *)(v103 + 32) & 0x30 | v41[4] & 0xCF;
  *((_BYTE *)v41 + 32) = v42;
  if ( (v42 & 0xFu) - 7 <= 1 || (v42 & 0x30) != 0 && (v42 & 0xF) != 9 )
    *((_BYTE *)v41 + 33) |= 0x40u;
  sub_1694890((__int64)v109, 0, *(_DWORD *)(a1 + 100), 1u);
  sub_15E5D20((__int64)v41, v109[0], (size_t)v109[1]);
  if ( v109[0] != v110 )
    j_j___libc_free_0(v109[0], v110[0] + 1LL);
  sub_15E4CC0((__int64)v41, 8u);
  v105.m128i_i64[1] = v9;
  v106 = v41;
  v41[6] = v99;
  v43 = *(_DWORD *)(a1 + 136);
  if ( !v43 )
  {
    ++*(_QWORD *)(a1 + 112);
    goto LABEL_97;
  }
  v44 = v103;
  v45 = *(_QWORD *)(a1 + 120);
  v46 = (v43 - 1) & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
  v47 = (__int64 *)(v45 + 32LL * v46);
  v48 = *v47;
  if ( v103 != *v47 )
  {
    v63 = 1;
    v64 = 0;
    while ( v48 != -8 )
    {
      if ( !v64 && v48 == -16 )
        v64 = v47;
      v46 = (v43 - 1) & (v63 + v46);
      v47 = (__int64 *)(v45 + 32LL * v46);
      v48 = *v47;
      if ( v103 == *v47 )
        goto LABEL_48;
      ++v63;
    }
    v65 = *(_DWORD *)(a1 + 128);
    if ( v64 )
      v47 = v64;
    ++*(_QWORD *)(a1 + 112);
    v66 = v65 + 1;
    if ( 4 * v66 < 3 * v43 )
    {
      if ( v43 - *(_DWORD *)(a1 + 132) - v66 > v43 >> 3 )
      {
LABEL_93:
        *(_DWORD *)(a1 + 128) = v66;
        if ( *v47 != -8 )
          --*(_DWORD *)(a1 + 132);
        *v47 = v44;
        v47[2] = 0;
        v47[3] = 0;
        v47[1] = 0;
        goto LABEL_48;
      }
      sub_17CA490(a1 + 112, v43);
      v74 = *(_DWORD *)(a1 + 136);
      if ( v74 )
      {
        v44 = v103;
        v75 = v74 - 1;
        v76 = *(_QWORD *)(a1 + 120);
        v77 = 1;
        v73 = 0;
        v66 = *(_DWORD *)(a1 + 128) + 1;
        v78 = (v74 - 1) & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
        v47 = (__int64 *)(v76 + 32LL * v78);
        v79 = *v47;
        if ( *v47 == v103 )
          goto LABEL_93;
        while ( v79 != -8 )
        {
          if ( v79 == -16 && !v73 )
            v73 = v47;
          v78 = v75 & (v77 + v78);
          v47 = (__int64 *)(v76 + 32LL * v78);
          v79 = *v47;
          if ( v103 == *v47 )
            goto LABEL_93;
          ++v77;
        }
        goto LABEL_101;
      }
      goto LABEL_125;
    }
LABEL_97:
    sub_17CA490(a1 + 112, 2 * v43);
    v67 = *(_DWORD *)(a1 + 136);
    if ( v67 )
    {
      v44 = v103;
      v68 = v67 - 1;
      v69 = *(_QWORD *)(a1 + 120);
      v66 = *(_DWORD *)(a1 + 128) + 1;
      v70 = (v67 - 1) & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
      v47 = (__int64 *)(v69 + 32LL * v70);
      v71 = *v47;
      if ( *v47 == v103 )
        goto LABEL_93;
      v72 = 1;
      v73 = 0;
      while ( v71 != -8 )
      {
        if ( v71 == -16 && !v73 )
          v73 = v47;
        v70 = v68 & (v72 + v70);
        v47 = (__int64 *)(v69 + 32LL * v70);
        v71 = *v47;
        if ( v103 == *v47 )
          goto LABEL_93;
        ++v72;
      }
LABEL_101:
      if ( v73 )
        v47 = v73;
      goto LABEL_93;
    }
LABEL_125:
    ++*(_DWORD *)(a1 + 128);
    BUG();
  }
LABEL_48:
  v49 = v106;
  *(__m128i *)(v47 + 1) = _mm_loadu_si128(&v105);
  v47[3] = (__int64)v49;
  v50 = *(_BYTE **)(a1 + 152);
  v109[0] = v41;
  if ( v50 == *(_BYTE **)(a1 + 160) )
  {
    sub_167C6C0(a1 + 144, v50, v109);
  }
  else
  {
    if ( v50 )
    {
      *(_QWORD *)v50 = v41;
      v50 = *(_BYTE **)(a1 + 152);
    }
    *(_QWORD *)(a1 + 152) = v50 + 8;
  }
  v51 = v103;
  *(_WORD *)(v103 + 32) = *(_WORD *)(v103 + 32) & 0xBFC0 | 0x4008;
  v52 = *(_BYTE **)(a1 + 176);
  if ( v52 == *(_BYTE **)(a1 + 184) )
  {
    sub_17C7180(a1 + 168, v52, &v103);
  }
  else
  {
    if ( v52 )
    {
      *(_QWORD *)v52 = v51;
      v52 = *(_BYTE **)(a1 + 176);
    }
    *(_QWORD *)(a1 + 176) = v52 + 8;
  }
  return v9;
}
