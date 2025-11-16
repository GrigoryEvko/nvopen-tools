// Function: sub_396A6C0
// Address: 0x396a6c0
//
__int64 __fastcall sub_396A6C0(
        __int64 a1,
        double a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v10; // r13
  __int64 v11; // r14
  __int64 v12; // r15
  _QWORD *v13; // rax
  __int64 v14; // r12
  _QWORD *v15; // rax
  unsigned __int64 v16; // r13
  unsigned __int64 v17; // rdi
  __int64 v18; // rax
  _QWORD *v19; // r12
  _QWORD *v20; // r14
  unsigned __int64 v21; // r15
  unsigned __int64 v22; // rdi
  __int64 v23; // rsi
  int v24; // r14d
  unsigned __int64 v25; // r15
  __int64 *v26; // r13
  __int64 v27; // r8
  unsigned int v28; // ecx
  unsigned __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rdi
  unsigned int v36; // ecx
  __int64 *v37; // rdx
  __int64 v38; // r9
  int v39; // edi
  unsigned __int16 v40; // ax
  int v41; // r13d
  __int64 v42; // r12
  unsigned __int64 v43; // r14
  __int64 v44; // r8
  unsigned int v45; // ecx
  unsigned __int64 v46; // rax
  __int64 v47; // rdi
  __int64 v48; // rdx
  unsigned int v49; // esi
  int v50; // edi
  __int64 v51; // r12
  unsigned int v52; // ecx
  _QWORD *v53; // rax
  __int64 v54; // rcx
  _QWORD *v55; // rdx
  __int64 v56; // rdi
  _QWORD *v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  _QWORD *v60; // rax
  __int64 *v61; // rsi
  __int64 *v62; // r12
  __int64 *v63; // r13
  __int64 *v64; // rax
  __int64 v65; // rax
  __int64 v66; // r14
  size_t v67; // rdx
  __int64 v68; // r12
  char *v69; // r13
  _QWORD *v70; // r13
  _QWORD *v71; // r12
  unsigned __int64 v72; // rdi
  unsigned __int64 v73; // rdi
  _QWORD *v74; // r13
  _QWORD *v75; // r12
  unsigned __int64 v76; // rdi
  unsigned __int64 v77; // rdi
  unsigned __int64 v78; // rdi
  int v80; // edx
  int v81; // r11d
  unsigned __int64 v82; // r10
  int v83; // edi
  double v84; // xmm4_8
  double v85; // xmm5_8
  unsigned __int8 v86; // al
  __int64 v87; // rdx
  _QWORD *v88; // rax
  _QWORD *v89; // rsi
  void *v90; // rax
  __int64 v91; // r12
  char *v92; // rax
  size_t v93; // rdx
  void *v94; // rdi
  unsigned __int64 v95; // rax
  unsigned int v96; // r12d
  void *v97; // rax
  __int64 v98; // rax
  void *v99; // rax
  __int64 v100; // r12
  int v101; // r11d
  unsigned __int64 v102; // r10
  int v103; // edi
  int v104; // r10d
  void *v105; // rax
  unsigned __int64 v106; // r13
  void *v107; // rax
  unsigned __int64 v108; // r13
  __int64 v109; // rax
  __int64 v110; // rax
  __int64 v111; // rax
  __int64 v112; // [rsp+10h] [rbp-90h]
  size_t v113; // [rsp+10h] [rbp-90h]
  unsigned __int8 v114; // [rsp+18h] [rbp-88h]
  _QWORD *v115; // [rsp+28h] [rbp-78h] BYREF
  unsigned __int64 v116; // [rsp+30h] [rbp-70h] BYREF
  __int64 v117; // [rsp+38h] [rbp-68h]
  __int64 *v118; // [rsp+50h] [rbp-50h] BYREF
  __int64 *v119; // [rsp+58h] [rbp-48h]
  __int64 *v120; // [rsp+60h] [rbp-40h]

  sub_3961940(a1);
  v114 = byte_5055300;
  if ( byte_5055300 )
    v114 = sub_3966880((__int64 *)a1);
  v10 = *(_QWORD *)(a1 + 24);
  v11 = *(_QWORD *)(a1 + 16);
  v12 = *(_QWORD *)a1;
  v13 = (_QWORD *)sub_22077B0(0x118u);
  v14 = (__int64)v13;
  if ( v13 )
  {
    *v13 = v12;
    v15 = v13 + 27;
    *(v15 - 26) = v11;
    *(v15 - 25) = v10;
    *(v15 - 24) = 0;
    *(v15 - 23) = 0;
    *(v15 - 22) = 0;
    *((_BYTE *)v15 - 168) = 0;
    *(v15 - 20) = 0;
    *(v15 - 19) = 0;
    *(v15 - 18) = 0;
    *((_DWORD *)v15 - 34) = 0;
    *(v15 - 16) = 0;
    *(v15 - 15) = 0;
    *(v15 - 14) = 0;
    *(v15 - 13) = 0;
    *(v15 - 12) = 0;
    *(v15 - 11) = 0;
    *((_DWORD *)v15 - 20) = 0;
    *(v15 - 9) = 0;
    *(v15 - 8) = 0;
    *(v15 - 7) = 0;
    *((_DWORD *)v15 - 12) = 0;
    *(_QWORD *)(v14 + 176) = 0;
    *(_QWORD *)(v14 + 184) = v15;
    *(_QWORD *)(v14 + 192) = v15;
    *(_QWORD *)(v14 + 200) = 8;
    *(_DWORD *)(v14 + 208) = 0;
  }
  v16 = *(_QWORD *)(a1 + 56);
  *(_QWORD *)(a1 + 56) = v14;
  if ( v16 )
  {
    v17 = *(_QWORD *)(v16 + 192);
    if ( v17 != *(_QWORD *)(v16 + 184) )
      _libc_free(v17);
    j___libc_free_0(*(_QWORD *)(v16 + 152));
    v18 = *(unsigned int *)(v16 + 136);
    if ( (_DWORD)v18 )
    {
      v19 = *(_QWORD **)(v16 + 120);
      v20 = &v19[2 * v18];
      do
      {
        if ( *v19 != -8 && *v19 != -16 )
        {
          v21 = v19[1];
          if ( v21 )
          {
            _libc_free(*(_QWORD *)(v21 + 48));
            _libc_free(*(_QWORD *)(v21 + 24));
            j_j___libc_free_0(v21);
          }
        }
        v19 += 2;
      }
      while ( v20 != v19 );
    }
    j___libc_free_0(*(_QWORD *)(v16 + 120));
    v22 = *(_QWORD *)(v16 + 88);
    if ( v22 )
      j_j___libc_free_0(v22);
    j___libc_free_0(*(_QWORD *)(v16 + 64));
    j_j___libc_free_0(v16);
    v14 = *(_QWORD *)(a1 + 56);
  }
  v23 = *(_QWORD *)v14;
  v24 = 0;
  sub_39659E0(&v118, *(_QWORD *)v14);
  v25 = (unsigned __int64)v118;
  v26 = v119;
  v112 = v14 + 144;
  if ( v119 != v118 )
  {
    do
    {
      v31 = *(_QWORD *)(v14 + 16);
      v32 = 0;
      v33 = *(unsigned int *)(v31 + 48);
      if ( !(_DWORD)v33 )
        goto LABEL_26;
      v34 = *(v26 - 1);
      v35 = *(_QWORD *)(v31 + 32);
      v36 = (v33 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v37 = (__int64 *)(v35 + 16LL * v36);
      v38 = *v37;
      if ( v34 == *v37 )
      {
LABEL_24:
        if ( v37 != (__int64 *)(v35 + 16 * v33) )
        {
          v32 = v37[1];
          goto LABEL_26;
        }
      }
      else
      {
        v80 = 1;
        while ( v38 != -8 )
        {
          v104 = v80 + 1;
          v36 = (v33 - 1) & (v80 + v36);
          v37 = (__int64 *)(v35 + 16LL * v36);
          v38 = *v37;
          if ( v34 == *v37 )
            goto LABEL_24;
          v80 = v104;
        }
      }
      v32 = 0;
LABEL_26:
      v115 = (_QWORD *)v32;
      v23 = *(unsigned int *)(v14 + 168);
      ++v24;
      if ( !(_DWORD)v23 )
      {
        ++*(_QWORD *)(v14 + 144);
        goto LABEL_28;
      }
      v27 = *(_QWORD *)(v14 + 152);
      v28 = (v23 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
      v29 = v27 + 16LL * v28;
      v30 = *(_QWORD *)v29;
      if ( v32 != *(_QWORD *)v29 )
      {
        v81 = 1;
        v82 = 0;
        while ( v30 != -8 )
        {
          if ( !v82 && v30 == -16 )
            v82 = v29;
          v28 = (v23 - 1) & (v81 + v28);
          v29 = v27 + 16LL * v28;
          v30 = *(_QWORD *)v29;
          if ( v32 == *(_QWORD *)v29 )
            goto LABEL_21;
          ++v81;
        }
        v83 = *(_DWORD *)(v14 + 160);
        if ( v82 )
          v29 = v82;
        ++*(_QWORD *)(v14 + 144);
        v39 = v83 + 1;
        if ( 4 * v39 >= (unsigned int)(3 * v23) )
        {
LABEL_28:
          LODWORD(v23) = 2 * v23;
        }
        else if ( (int)v23 - *(_DWORD *)(v14 + 164) - v39 > (unsigned int)v23 >> 3 )
        {
          goto LABEL_82;
        }
        sub_19F5120(v112, v23);
        sub_19E6B80(v112, (__int64 *)&v115, &v116);
        v29 = v116;
        v32 = (__int64)v115;
        v23 = *(unsigned int *)(v14 + 160);
        v39 = v23 + 1;
LABEL_82:
        *(_DWORD *)(v14 + 160) = v39;
        if ( *(_QWORD *)v29 != -8 )
          --*(_DWORD *)(v14 + 164);
        *(_QWORD *)v29 = v32;
        *(_DWORD *)(v29 + 8) = 0;
      }
LABEL_21:
      --v26;
      *(_DWORD *)(v29 + 8) = v24;
    }
    while ( (__int64 *)v25 != v26 );
  }
  sub_3952FD0(v14);
  sub_3967F20(v14, v23);
  if ( v118 )
    j_j___libc_free_0((unsigned __int64)v118);
  sub_3957A80(*(_QWORD *)(a1 + 56), *(_DWORD *)(a1 + 48), dword_5054D80, 1);
  if ( byte_5054E60 )
  {
    v90 = sub_16E8CB0();
    v91 = sub_1263B40((__int64)v90, "Function: ");
    v92 = (char *)sub_1649960(*(_QWORD *)a1);
    v94 = *(void **)(v91 + 24);
    if ( *(_QWORD *)(v91 + 16) - (_QWORD)v94 < v93 )
    {
      v91 = sub_16E7EE0(v91, v92, v93);
    }
    else if ( v93 )
    {
      v113 = v93;
      memcpy(v94, v92, v93);
      *(_QWORD *)(v91 + 24) += v113;
    }
    sub_1263B40(v91, " ");
    v95 = sub_1BF9660(*(_QWORD *)a1);
    v96 = v95;
    if ( (_DWORD)v95 )
    {
      v106 = v95;
      v107 = sub_16E8CB0();
      v108 = HIDWORD(v106);
      v109 = sub_1263B40((__int64)v107, "Launch bounds (");
      sub_16E7A90(v109, v96);
      v110 = (__int64)sub_16E8CB0();
      if ( (_DWORD)v108 )
      {
        v111 = sub_1263B40(v110, ", ");
        v110 = sub_16E7A90(v111, (unsigned int)v108);
      }
      sub_1263B40(v110, ") ");
    }
    v97 = sub_16E8CB0();
    v98 = sub_1263B40((__int64)v97, "Register Target: ");
    v118 = *(__int64 **)(*(_QWORD *)(a1 + 56) + 40LL);
    sub_1DF8250((int *)&v118, v98);
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 56) + 48LL) )
    {
      v105 = sub_16E8CB0();
      sub_1263B40((__int64)v105, " (This fun has per block reg target)");
    }
    v99 = sub_16E8CB0();
    v100 = sub_1263B40((__int64)v99, " Register Pressure: ");
    v118 = *(__int64 **)(*(_QWORD *)(a1 + 56) + 24LL);
    sub_1DF8250((int *)&v118, v100);
    sub_1263B40(v100, "\n");
  }
  v40 = sub_3962650(*(_QWORD *)(a1 + 56), 1.0);
  if ( !(_BYTE)v40 && !HIBYTE(v40) )
    return v114;
  v41 = 0;
  sub_39659E0(&v116, *(_QWORD *)a1);
  v42 = v117;
  v43 = v116;
  if ( v116 != v117 )
  {
    while ( 1 )
    {
      v48 = *(_QWORD *)(v42 - 8);
      v49 = *(_DWORD *)(a1 + 288);
      ++v41;
      v115 = (_QWORD *)v48;
      if ( !v49 )
        break;
      v44 = *(_QWORD *)(a1 + 272);
      v45 = (v49 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
      v46 = v44 + 16LL * v45;
      v47 = *(_QWORD *)v46;
      if ( v48 != *(_QWORD *)v46 )
      {
        v101 = 1;
        v102 = 0;
        while ( v47 != -8 )
        {
          if ( !v102 && v47 == -16 )
            v102 = v46;
          v45 = (v49 - 1) & (v101 + v45);
          v46 = v44 + 16LL * v45;
          v47 = *(_QWORD *)v46;
          if ( v48 == *(_QWORD *)v46 )
            goto LABEL_38;
          ++v101;
        }
        v103 = *(_DWORD *)(a1 + 280);
        if ( v102 )
          v46 = v102;
        ++*(_QWORD *)(a1 + 264);
        v50 = v103 + 1;
        if ( 4 * v50 < 3 * v49 )
        {
          if ( v49 - *(_DWORD *)(a1 + 284) - v50 > v49 >> 3 )
            goto LABEL_105;
          goto LABEL_42;
        }
LABEL_41:
        v49 *= 2;
LABEL_42:
        sub_13FEAC0(a1 + 264, v49);
        sub_13FDDE0(a1 + 264, (__int64 *)&v115, &v118);
        v46 = (unsigned __int64)v118;
        v48 = (__int64)v115;
        v50 = *(_DWORD *)(a1 + 280) + 1;
LABEL_105:
        *(_DWORD *)(a1 + 280) = v50;
        if ( *(_QWORD *)v46 != -8 )
          --*(_DWORD *)(a1 + 284);
        *(_QWORD *)v46 = v48;
        *(_DWORD *)(v46 + 8) = 0;
      }
LABEL_38:
      v42 -= 8;
      *(_DWORD *)(v46 + 8) = v41;
      if ( v43 == v42 )
        goto LABEL_85;
    }
    ++*(_QWORD *)(a1 + 264);
    goto LABEL_41;
  }
LABEL_85:
  sub_3969C60(a1);
  v86 = sub_39692C0(a1, COERCE_DOUBLE(1065353216), a3, a4, a5, v84, v85, a8, a9);
  v87 = *(unsigned int *)(a1 + 188);
  v114 = v86;
  if ( (_DWORD)v87 != *(_DWORD *)(a1 + 192) )
  {
    v88 = *(_QWORD **)(a1 + 176);
    if ( v88 != *(_QWORD **)(a1 + 168) )
      v87 = *(unsigned int *)(a1 + 184);
    v89 = &v88[v87];
    if ( v88 == v89 )
      goto LABEL_91;
    while ( 1 )
    {
      v54 = *v88;
      v55 = v88;
      if ( *v88 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v89 == ++v88 )
        goto LABEL_91;
    }
    if ( v89 == v88 )
    {
LABEL_91:
      v51 = 0;
    }
    else
    {
      v51 = 0;
      do
      {
        v52 = *(_DWORD *)(v54 + 32);
        if ( v52 <= 0x40 )
          v51 = (unsigned int)(1 << v52) | (unsigned __int64)v51;
        v53 = v55 + 1;
        if ( v55 + 1 == v89 )
          break;
        while ( 1 )
        {
          v54 = *v53;
          v55 = v53;
          if ( *v53 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v89 == ++v53 )
            goto LABEL_50;
        }
      }
      while ( v89 != v53 );
    }
LABEL_50:
    v56 = *(_QWORD *)a1;
    v118 = 0;
    v119 = 0;
    v120 = 0;
    v57 = (_QWORD *)sub_15E0530(v56);
    v58 = sub_1643360(v57);
    v59 = sub_159C470(v58, v51, 0);
    v60 = sub_1624210(v59);
    v61 = v119;
    v115 = v60;
    if ( v119 == v120 )
    {
      sub_1273E00((__int64)&v118, v119, &v115);
      v62 = v119;
    }
    else
    {
      if ( v119 )
      {
        *v119 = (__int64)v60;
        v61 = v119;
      }
      v62 = v61 + 1;
      v119 = v61 + 1;
    }
    v63 = v118;
    v64 = (__int64 *)sub_15E0530(*(_QWORD *)a1);
    v65 = sub_1627350(v64, v63, (__int64 *)(v62 - v63), 0, 1);
    v66 = *(_QWORD *)a1;
    v67 = 0;
    v68 = v65;
    v69 = off_4CD4950[0];
    if ( off_4CD4950[0] )
      v67 = strlen(off_4CD4950[0]);
    sub_1627100(v66, v69, v67, v68);
    if ( v118 )
      j_j___libc_free_0((unsigned __int64)v118);
  }
  v70 = *(_QWORD **)(a1 + 112);
  while ( (_QWORD *)(a1 + 112) != v70 )
  {
    v71 = v70;
    v70 = (_QWORD *)*v70;
    v72 = v71[20];
    if ( v72 != v71[19] )
      _libc_free(v72);
    v73 = v71[8];
    if ( (_QWORD *)v73 != v71 + 10 )
      _libc_free(v73);
    j_j___libc_free_0((unsigned __int64)v71);
  }
  *(_QWORD *)(a1 + 120) = v70;
  *(_QWORD *)(a1 + 112) = v70;
  v74 = *(_QWORD **)(a1 + 136);
  *(_QWORD *)(a1 + 128) = 0;
  while ( (_QWORD *)(a1 + 136) != v74 )
  {
    v75 = v74;
    v74 = (_QWORD *)*v74;
    v76 = v75[20];
    if ( (_QWORD *)v76 != v75 + 22 )
      _libc_free(v76);
    v77 = v75[9];
    if ( v77 != v75[8] )
      _libc_free(v77);
    j_j___libc_free_0((unsigned __int64)v75);
  }
  v78 = v116;
  *(_QWORD *)(a1 + 144) = v74;
  *(_QWORD *)(a1 + 136) = v74;
  *(_QWORD *)(a1 + 152) = 0;
  if ( v78 )
    j_j___libc_free_0(v78);
  return v114;
}
