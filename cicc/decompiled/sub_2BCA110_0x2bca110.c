// Function: sub_2BCA110
// Address: 0x2bca110
//
__int64 __fastcall sub_2BCA110(
        __int64 a1,
        unsigned __int8 **a2,
        unsigned __int64 a3,
        __int64 a4,
        __m128i a5,
        __int64 a6,
        unsigned int a7,
        int *a8)
{
  __int64 *v9; // r13
  int v11; // eax
  unsigned __int8 **v12; // r11
  unsigned __int8 **v13; // r15
  unsigned __int8 **v14; // rbx
  unsigned int v15; // esi
  __int64 v16; // rax
  __int64 *v17; // r11
  __int64 v18; // r8
  int v19; // r10d
  _QWORD *v20; // r9
  unsigned int v21; // edi
  _QWORD *v22; // rcx
  __int64 v23; // rdx
  unsigned __int64 v24; // r12
  __int64 v25; // rdx
  __int64 v26; // rsi
  int v27; // eax
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  unsigned __int64 *v30; // rdi
  unsigned __int64 v31; // rsi
  __int64 v32; // rax
  __int64 *v33; // r11
  __int64 v34; // rdx
  unsigned __int64 *v35; // rax
  __int64 v36; // rdx
  unsigned __int64 *v37; // rdi
  __int64 v38; // rcx
  __int64 v39; // rdx
  unsigned __int64 *v40; // rdx
  unsigned int v41; // r12d
  _QWORD *v43; // rdi
  unsigned int v44; // r13d
  __int64 v45; // rcx
  unsigned __int64 v46; // rax
  int v47; // edi
  unsigned int v48; // eax
  __int64 *v49; // rbx
  __int64 *v50; // rdx
  int v51; // eax
  __int64 v52; // rsi
  __int64 v53; // rdi
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // rdx
  __int64 v61; // rcx
  unsigned __int64 v62; // r8
  __int64 v63; // r9
  __int64 *v64; // rax
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  __int64 v68; // rdx
  signed __int64 v69; // r15
  __int64 v70; // rbx
  unsigned int v71; // eax
  _QWORD *v72; // rax
  _QWORD *i; // rdx
  __int64 v74; // rdx
  __int64 v75; // rcx
  unsigned __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // r12
  __int64 v79; // rax
  int v80; // r10d
  __int64 *v81; // r15
  __int64 v82; // rax
  unsigned __int64 *v83; // rbx
  __int64 v84; // rax
  __int64 *v85; // rbx
  _BYTE *v86; // r12
  __int64 v87; // r15
  __int64 *v88; // r12
  __int64 v89; // r15
  __int64 v90; // r15
  signed __int64 v91; // rax
  _BYTE *v92; // r12
  unsigned int v93; // r10d
  __int64 v94; // r12
  __int64 v95; // r12
  __int64 v96; // [rsp+8h] [rbp-2E8h]
  __int64 *v97; // [rsp+8h] [rbp-2E8h]
  __int64 v98; // [rsp+10h] [rbp-2E0h]
  __int64 *v99; // [rsp+18h] [rbp-2D8h]
  unsigned __int8 *v100; // [rsp+18h] [rbp-2D8h]
  __int64 *v101; // [rsp+18h] [rbp-2D8h]
  __int64 *v102; // [rsp+18h] [rbp-2D8h]
  unsigned __int8 **v103; // [rsp+20h] [rbp-2D0h]
  __int64 *v104; // [rsp+20h] [rbp-2D0h]
  __int64 *v105; // [rsp+20h] [rbp-2D0h]
  __int64 *v107; // [rsp+30h] [rbp-2C0h]
  __int64 *v109; // [rsp+38h] [rbp-2B8h]
  __int64 v110; // [rsp+40h] [rbp-2B0h] BYREF
  __int64 v111; // [rsp+48h] [rbp-2A8h]
  __int64 v112; // [rsp+50h] [rbp-2A0h]
  __int64 v113; // [rsp+58h] [rbp-298h]
  unsigned __int64 *v114; // [rsp+60h] [rbp-290h] BYREF
  __int64 v115; // [rsp+68h] [rbp-288h]
  unsigned __int64 v116[4]; // [rsp+70h] [rbp-280h] BYREF
  unsigned __int64 v117[6]; // [rsp+90h] [rbp-260h] BYREF
  __int64 v118; // [rsp+C0h] [rbp-230h] BYREF
  _QWORD *v119; // [rsp+C8h] [rbp-228h]
  __int64 v120; // [rsp+D0h] [rbp-220h]
  unsigned int v121; // [rsp+D8h] [rbp-218h]
  unsigned __int64 v122[6]; // [rsp+E0h] [rbp-210h] BYREF
  void *v123; // [rsp+110h] [rbp-1E0h] BYREF
  __int64 v124; // [rsp+118h] [rbp-1D8h]
  __int64 v125; // [rsp+120h] [rbp-1D0h] BYREF
  unsigned int v126; // [rsp+128h] [rbp-1C8h]
  char v127; // [rsp+140h] [rbp-1B0h] BYREF
  char v128[400]; // [rsp+160h] [rbp-190h] BYREF

  v9 = (__int64 *)a2;
  *a8 = 0;
  v11 = sub_2B49BC0(a4, *a2);
  if ( (!v11
     || (v11 & (v11 - 1)) != 0
     || !sub_2B1F720(*(_QWORD *)(a1 + 8), *(_QWORD *)(*((_QWORD *)*a2 - 8) + 8LL), a3)
     || (unsigned int)a3 <= 1
     || a7 > (unsigned int)a3)
    && (!(_BYTE)qword_500F628 || (unsigned int)a3 < a7 && (_DWORD)a3 + 1 != a7) )
  {
    return 256;
  }
  v110 = 0;
  v114 = v116;
  v111 = 0;
  v112 = 0;
  v12 = &a2[a3];
  v96 = 8 * a3;
  v113 = 0;
  v115 = 0;
  if ( v12 != a2 )
  {
    v13 = &a2[a3];
    v14 = a2;
    v15 = 0;
    v16 = 0;
    v17 = v9;
    while ( 1 )
    {
      v24 = *((_QWORD *)*v14 - 8);
      if ( !v15 )
        break;
      v18 = v15 - 1;
      v19 = 1;
      v20 = 0;
      v21 = v18 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v22 = (_QWORD *)(v16 + 8LL * v21);
      v23 = *v22;
      if ( v24 == *v22 )
      {
LABEL_9:
        if ( v13 == ++v14 )
          goto LABEL_20;
        goto LABEL_10;
      }
      while ( v23 != -4096 )
      {
        if ( v23 != -8192 || v20 )
          v22 = v20;
        v21 = v18 & (v19 + v21);
        v23 = *(_QWORD *)(v16 + 8LL * v21);
        if ( v24 == v23 )
          goto LABEL_9;
        ++v19;
        v20 = v22;
        v22 = (_QWORD *)(v16 + 8LL * v21);
      }
      if ( !v20 )
        v20 = v22;
      ++v110;
      v27 = v112 + 1;
      if ( 4 * ((int)v112 + 1) >= 3 * v15 )
        goto LABEL_13;
      if ( v15 - (v27 + HIDWORD(v112)) <= v15 >> 3 )
      {
        v101 = v17;
        sub_CE2A30((__int64)&v110, v15);
        if ( !(_DWORD)v113 )
        {
LABEL_176:
          LODWORD(v112) = v112 + 1;
          BUG();
        }
        v43 = 0;
        v17 = v101;
        v44 = (v113 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v18 = 1;
        v20 = (_QWORD *)(v111 + 8LL * v44);
        v45 = *v20;
        v27 = v112 + 1;
        if ( v24 != *v20 )
        {
          while ( v45 != -4096 )
          {
            if ( v45 == -8192 && !v43 )
              v43 = v20;
            v93 = v18 + 1;
            v18 = ((_DWORD)v113 - 1) & (v44 + (unsigned int)v18);
            v20 = (_QWORD *)(v111 + 8LL * (unsigned int)v18);
            v44 = v18;
            v45 = *v20;
            if ( v24 == *v20 )
              goto LABEL_15;
            v18 = v93;
          }
          if ( v43 )
            v20 = v43;
        }
      }
LABEL_15:
      LODWORD(v112) = v27;
      if ( *v20 != -4096 )
        --HIDWORD(v112);
      *v20 = v24;
      v28 = (unsigned int)v115;
      v29 = (unsigned int)v115 + 1LL;
      if ( v29 > HIDWORD(v115) )
      {
        v102 = v17;
        sub_C8D5F0((__int64)&v114, v116, v29, 8u, v18, (__int64)v20);
        v28 = (unsigned int)v115;
        v17 = v102;
      }
      ++v14;
      v114[v28] = v24;
      LODWORD(v115) = v115 + 1;
      if ( v13 == v14 )
      {
LABEL_20:
        v30 = v114;
        v31 = (unsigned int)v115;
        v9 = v17;
        v12 = v13;
        goto LABEL_21;
      }
LABEL_10:
      v16 = v111;
      v15 = v113;
    }
    ++v110;
LABEL_13:
    v99 = v17;
    sub_CE2A30((__int64)&v110, 2 * v15);
    if ( !(_DWORD)v113 )
      goto LABEL_176;
    v17 = v99;
    LODWORD(v25) = (v113 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
    v20 = (_QWORD *)(v111 + 8LL * (unsigned int)v25);
    v26 = *v20;
    v27 = v112 + 1;
    if ( v24 != *v20 )
    {
      v80 = 1;
      v18 = 0;
      while ( v26 != -4096 )
      {
        if ( v26 == -8192 && !v18 )
          v18 = (__int64)v20;
        v25 = ((_DWORD)v113 - 1) & (unsigned int)(v25 + v80);
        v20 = (_QWORD *)(v111 + 8 * v25);
        v26 = *v20;
        if ( v24 == *v20 )
          goto LABEL_15;
        ++v80;
      }
      if ( v18 )
        v20 = (_QWORD *)v18;
    }
    goto LABEL_15;
  }
  v30 = v116;
  v31 = 0;
LABEL_21:
  v103 = v12;
  v32 = sub_2B5F980((__int64 *)v30, v31, *(__int64 **)(a1 + 16));
  v33 = (__int64 *)v103;
  v98 = v34;
  v100 = (unsigned __int8 *)v32;
  v35 = v114;
  v36 = 8LL * (unsigned int)v115;
  v37 = &v114[(unsigned __int64)v36 / 8];
  v38 = v36 >> 3;
  v39 = v36 >> 5;
  if ( !v39 )
  {
LABEL_78:
    if ( v38 != 2 )
    {
      if ( v38 != 3 )
      {
        if ( v38 != 1 )
          goto LABEL_29;
        goto LABEL_81;
      }
      if ( *(_BYTE *)*v35 <= 0x1Cu )
        goto LABEL_28;
      ++v35;
    }
    if ( *(_BYTE *)*v35 <= 0x1Cu )
      goto LABEL_28;
    ++v35;
LABEL_81:
    if ( *(_BYTE *)*v35 > 0x1Cu )
      goto LABEL_29;
    goto LABEL_28;
  }
  v40 = &v114[4 * v39];
  while ( *(_BYTE *)*v35 > 0x1Cu )
  {
    if ( *(_BYTE *)v35[1] <= 0x1Cu )
    {
      ++v35;
      break;
    }
    if ( *(_BYTE *)v35[2] <= 0x1Cu )
    {
      v35 += 2;
      break;
    }
    if ( *(_BYTE *)v35[3] <= 0x1Cu )
    {
      v35 += 3;
      break;
    }
    v35 += 4;
    if ( v40 == v35 )
    {
      v38 = v37 - v35;
      goto LABEL_78;
    }
  }
LABEL_28:
  if ( v37 != v35 )
    goto LABEL_30;
LABEL_29:
  if ( (unsigned int)v115 <= 1 )
    goto LABEL_30;
  if ( v96 <= 0 )
  {
    v118 = 0;
    goto LABEL_118;
  }
  if ( v96 >> 3 != 1 )
  {
    _BitScanReverse64(&v46, (v96 >> 3) - 1);
    v118 = 0;
    v47 = 1LL << (64 - ((unsigned __int8)v46 ^ 0x3Fu));
    if ( v47 )
      goto LABEL_59;
LABEL_118:
    v121 = 0;
    goto LABEL_60;
  }
  v118 = 0;
  v47 = 1;
LABEL_59:
  v48 = sub_2B149A0(v47);
  v121 = v48;
  if ( !v48 )
  {
LABEL_60:
    v119 = 0;
    v120 = 0;
    goto LABEL_61;
  }
  v105 = v33;
  v72 = (_QWORD *)sub_C7D670(8LL * v48, 8);
  v120 = 0;
  v119 = v72;
  v33 = v105;
  for ( i = &v72[v121]; i != v72; ++v72 )
  {
    if ( v72 )
      *v72 = -4096;
  }
LABEL_61:
  if ( v33 != v9 )
  {
    v104 = v9;
    v49 = v33;
    do
    {
      v50 = v9++;
      sub_2B78C40((__int64)&v123, (__int64)&v118, v50);
    }
    while ( v49 != v9 );
    v9 = v104;
  }
  if ( sub_2B1F720(*(_QWORD *)(a1 + 8), *(_QWORD *)(*v114 + 8), v115) )
  {
    if ( (unsigned int)v115 <= a3 >> 1 )
      goto LABEL_72;
    goto LABEL_67;
  }
  if ( (_BYTE)qword_500F628 && ((unsigned int)v115 & ((_DWORD)v115 + 1)) == 0 )
  {
    if ( (unsigned int)v115 <= a3 >> 1 )
      goto LABEL_72;
LABEL_67:
    if ( !v100 || !v98 )
      goto LABEL_69;
    goto LABEL_72;
  }
  if ( v100 && v98 )
  {
    if ( *v100 == 61 )
      goto LABEL_72;
    if ( !sub_B469C0(v100) )
    {
LABEL_105:
      v51 = 1;
      goto LABEL_70;
    }
    v81 = (__int64 *)v114;
    v82 = (unsigned int)v115;
    v83 = &v114[v82];
    v84 = (v82 * 8) >> 5;
    v97 = (__int64 *)v83;
    if ( v84 )
    {
      v85 = (__int64 *)v114;
      v107 = (__int64 *)&v114[4 * v84];
      do
      {
        v86 = (_BYTE *)*v85;
        if ( *(_BYTE *)*v85 != 90
          && (a3 < (unsigned int)sub_BD3960(*v85) || sub_2B0FAD0(*((_QWORD *)v86 + 2), 0, (__int64)&v118)) )
        {
          v81 = v85;
          goto LABEL_135;
        }
        v87 = v85[1];
        v88 = v85 + 1;
        if ( *(_BYTE *)v87 != 90
          && (a3 < (unsigned int)sub_BD3960(v85[1]) || sub_2B0FAD0(*(_QWORD *)(v87 + 16), 0, (__int64)&v118))
          || (v89 = v85[2], v88 = v85 + 2, *(_BYTE *)v89 != 90)
          && (a3 < (unsigned int)sub_BD3960(v85[2]) || sub_2B0FAD0(*(_QWORD *)(v89 + 16), 0, (__int64)&v118))
          || (v90 = v85[3], v88 = v85 + 3, *(_BYTE *)v90 != 90)
          && (a3 < (unsigned int)sub_BD3960(v85[3]) || sub_2B0FAD0(*(_QWORD *)(v90 + 16), 0, (__int64)&v118)) )
        {
          v81 = v88;
          goto LABEL_135;
        }
        v85 += 4;
      }
      while ( v107 != v85 );
      v81 = v85;
    }
    v91 = (char *)v97 - (char *)v81;
    if ( (char *)v97 - (char *)v81 != 16 )
    {
      if ( v91 != 24 )
      {
        if ( v91 != 8 )
          goto LABEL_72;
LABEL_153:
        v92 = (_BYTE *)*v81;
        if ( *(_BYTE *)*v81 == 90
          || a3 >= (unsigned int)sub_BD3960(*v81) && !sub_2B0FAD0(*((_QWORD *)v92 + 2), 0, (__int64)&v118) )
        {
          goto LABEL_72;
        }
LABEL_135:
        if ( v97 == v81 )
          goto LABEL_72;
        goto LABEL_105;
      }
      v94 = *v81;
      if ( *(_BYTE *)*v81 != 90
        && (a3 < (unsigned int)sub_BD3960(*v81) || sub_2B0FAD0(*(_QWORD *)(v94 + 16), 0, (__int64)&v118)) )
      {
        goto LABEL_135;
      }
      ++v81;
    }
    v95 = *v81;
    if ( *(_BYTE *)*v81 != 90
      && (a3 < (unsigned int)sub_BD3960(*v81) || sub_2B0FAD0(*(_QWORD *)(v95 + 16), 0, (__int64)&v118)) )
    {
      goto LABEL_135;
    }
    ++v81;
    goto LABEL_153;
  }
  if ( (unsigned int)v115 > a3 >> 1 )
  {
LABEL_69:
    v51 = 2;
LABEL_70:
    v52 = v121;
    v41 = 256;
    v53 = (__int64)v119;
    *a8 = v51;
    sub_C7D6A0(v53, 8 * v52, 8);
    goto LABEL_32;
  }
LABEL_72:
  sub_C7D6A0((__int64)v119, 8LL * v121, 8);
LABEL_30:
  if ( (unsigned __int8)sub_2B2DA70(a4, v9, a3) )
  {
LABEL_31:
    v41 = 257;
    goto LABEL_32;
  }
  sub_2BAAD20(a4, v9, a3);
  if ( !(unsigned __int8)sub_2B2DB00(a4, 0, v54, v55) )
  {
    if ( (unsigned __int8)sub_2B2A740((__int64 **)a4) )
    {
      sub_2BBDBE0((__int64 **)a4, a5, 0, v60, v61, v62, v63);
      sub_2BBFB60(a4, 0, a5, v74, v75, v76, v77);
    }
    sub_2BB0460(a4, 0, v60, v61, v62, v63);
    v64 = &v125;
    v123 = 0;
    v124 = 1;
    do
      *v64++ = -4096;
    while ( v64 != (__int64 *)&v127 );
    sub_2B4F3D0(a4, (__int64)&v123);
    if ( (v124 & 1) == 0 )
      sub_C7D6A0(v125, 8LL * v126, 8);
    sub_2BB3590(a4);
    *a8 = *(_DWORD *)(a4 + 3556);
    if ( v98 && v100 && *v100 == 61 )
      *a8 = 2;
    v69 = sub_2B94A80(a4, 0, 0, v65, v66, v67);
    v70 = v68;
    if ( (_DWORD)v68 )
      v71 = (unsigned int)v68 >> 31;
    else
      LOBYTE(v71) = -(int)qword_5010428 > v69;
    if ( !(_BYTE)v71 )
    {
      v41 = 256;
      goto LABEL_32;
    }
    v109 = *(__int64 **)(a4 + 3352);
    sub_B174A0((__int64)&v123, (__int64)"slp-vectorizer", (__int64)"StoresVectorized", 16, *v9);
    sub_B18290((__int64)&v123, "Stores SLP vectorized with cost ", 0x20u);
    sub_B16D50((__int64)v116, "Cost", 4, v69, v70);
    v78 = sub_23FD640((__int64)&v123, (__int64)v116);
    sub_B18290(v78, " and with tree size ", 0x14u);
    sub_B169E0(&v118, "TreeSize", 8, *(_DWORD *)(a4 + 8));
    v79 = sub_23FD640(v78, (__int64)&v118);
    sub_1049740(v109, v79);
    sub_2240A30(v122);
    sub_2240A30((unsigned __int64 *)&v118);
    sub_2240A30(v117);
    sub_2240A30(v116);
    v123 = &unk_49D9D40;
    sub_23FD590((__int64)v128);
    sub_2BCA0A0(a4);
    goto LABEL_31;
  }
  if ( (unsigned __int8)sub_B19060(a4 + 768, *v9, v56, v57)
    || (unsigned __int8)sub_B19060(a4 + 928, *(_QWORD *)(*v9 - 64), v58, v59) )
  {
    v41 = 0;
  }
  else
  {
    v41 = 256;
    *a8 = *(_DWORD *)(a4 + 3556);
  }
LABEL_32:
  if ( v114 != v116 )
    _libc_free((unsigned __int64)v114);
  sub_C7D6A0(v111, 8LL * (unsigned int)v113, 8);
  return v41;
}
