// Function: sub_1AAA850
// Address: 0x1aaa850
//
void __fastcall sub_1AAA850(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        _BYTE *a4,
        _BYTE *a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15,
        __int64 a16,
        char a17)
{
  __int64 v19; // r15
  _BYTE *v20; // rdx
  __int64 v21; // r14
  _QWORD *v22; // rbx
  int v23; // r8d
  int v24; // r9d
  __int64 v25; // rax
  _QWORD *v26; // rax
  __int64 v27; // r15
  __int64 **v28; // r14
  __int64 *v29; // rsi
  __int64 *v30; // r13
  __int64 v31; // rdi
  unsigned __int64 v32; // rax
  __int64 v33; // r13
  __int64 v34; // r14
  __int64 v35; // r15
  __int64 v36; // rdi
  __int64 v37; // rax
  int v38; // r8d
  int v39; // r9d
  __int64 v40; // r15
  __int64 *v41; // r12
  _QWORD *v42; // r14
  bool v43; // zf
  double v44; // xmm4_8
  double v45; // xmm5_8
  __int64 v46; // rsi
  __int64 v47; // rdx
  __int64 v48; // r10
  _QWORD *v49; // r13
  __int64 v50; // rax
  unsigned __int64 v51; // rcx
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  double v57; // xmm4_8
  double v58; // xmm5_8
  __int64 v59; // rsi
  __int64 v60; // r10
  __int64 v61; // r13
  __int64 v62; // r11
  int v63; // eax
  __int64 v64; // rax
  int v65; // edx
  __int64 v66; // rdx
  _QWORD *v67; // rax
  __int64 v68; // rcx
  unsigned __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // rdx
  __int64 v72; // rax
  __int64 v73; // rcx
  __int64 v74; // rdx
  int v75; // eax
  __int64 v76; // rax
  int v77; // edx
  __int64 v78; // rdx
  __int64 *v79; // rax
  __int64 v80; // rcx
  unsigned __int64 v81; // rdx
  __int64 v82; // rdx
  __int64 v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // rsi
  unsigned __int8 *v86; // rsi
  __int64 v87; // rdx
  __int64 v88; // r14
  _QWORD *v89; // rax
  int v90; // r8d
  int v91; // r9d
  __int64 v92; // rax
  _QWORD *v93; // rax
  __int64 v94; // rsi
  __int64 *v95; // r13
  __int64 v96; // r10
  __int64 *v97; // r13
  __int64 *v98; // r14
  __int64 v99; // rdi
  unsigned __int64 v100; // rax
  __int64 v101; // rsi
  unsigned __int8 *v102; // rsi
  __int64 v106; // [rsp+28h] [rbp-E8h]
  __int64 v107; // [rsp+30h] [rbp-E0h]
  __int64 v108; // [rsp+30h] [rbp-E0h]
  __int64 v109; // [rsp+30h] [rbp-E0h]
  __int64 *v111; // [rsp+38h] [rbp-D8h]
  __int64 v112; // [rsp+38h] [rbp-D8h]
  __int64 v113; // [rsp+38h] [rbp-D8h]
  __int64 v114; // [rsp+38h] [rbp-D8h]
  __int64 v115; // [rsp+38h] [rbp-D8h]
  __int64 v116; // [rsp+38h] [rbp-D8h]
  __int64 v118; // [rsp+48h] [rbp-C8h]
  __int64 v119; // [rsp+48h] [rbp-C8h]
  char v120; // [rsp+5Fh] [rbp-B1h] BYREF
  _QWORD v121[2]; // [rsp+60h] [rbp-B0h] BYREF
  const char *v122; // [rsp+70h] [rbp-A0h] BYREF
  _BYTE *v123; // [rsp+78h] [rbp-98h]
  __int16 v124; // [rsp+80h] [rbp-90h]
  __int64 *v125; // [rsp+90h] [rbp-80h] BYREF
  __int64 i; // [rsp+98h] [rbp-78h]
  _WORD v127[56]; // [rsp+A0h] [rbp-70h] BYREF

  v19 = *(_QWORD *)(a1 + 56);
  v127[0] = 773;
  v122 = sub_1649960(a1);
  v123 = v20;
  i = (__int64)a4;
  v125 = (__int64 *)&v122;
  v21 = sub_157E9C0(a1);
  v22 = (_QWORD *)sub_22077B0(64);
  if ( v22 )
    sub_157FB60(v22, v21, (__int64)&v125, v19, a1);
  v25 = *(unsigned int *)(a6 + 8);
  if ( (unsigned int)v25 >= *(_DWORD *)(a6 + 12) )
  {
    sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v23, v24);
    v25 = *(unsigned int *)(a6 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a6 + 8 * v25) = v22;
  ++*(_DWORD *)(a6 + 8);
  v26 = sub_1648A60(56, 1u);
  v27 = (__int64)v26;
  if ( v26 )
    sub_15F8590((__int64)v26, a1, (__int64)v22);
  v28 = (__int64 **)(v27 + 48);
  v29 = *(__int64 **)(sub_157ED20(a1) + 48);
  v125 = v29;
  if ( v29 )
  {
    sub_1623A60((__int64)&v125, (__int64)v29, 2);
    if ( v28 == &v125 )
    {
      if ( v125 )
        sub_161E7C0((__int64)&v125, (__int64)v125);
      goto LABEL_11;
    }
    v85 = *(_QWORD *)(v27 + 48);
    if ( !v85 )
    {
LABEL_67:
      v86 = (unsigned __int8 *)v125;
      *(_QWORD *)(v27 + 48) = v125;
      if ( v86 )
        sub_1623210((__int64)&v125, v86, v27 + 48);
      goto LABEL_11;
    }
LABEL_66:
    sub_161E7C0(v27 + 48, v85);
    goto LABEL_67;
  }
  if ( v28 != &v125 )
  {
    v85 = *(_QWORD *)(v27 + 48);
    if ( v85 )
      goto LABEL_66;
  }
LABEL_11:
  if ( (_DWORD)a3 )
  {
    v30 = a2;
    do
    {
      v31 = *v30++;
      v32 = sub_157EBA0(v31);
      sub_1648780(v32, a1, (__int64)v22);
    }
    while ( &a2[(unsigned int)(a3 - 1) + 1] != v30 );
  }
  v120 = 0;
  sub_1AA9AF0(a1, (__int64)v22, a2, a3, a15, a16, a17, (__int64 *)&v120);
  sub_1AA5740(a1, (__int64)v22, a2, a3, v27, v120);
  v33 = *(_QWORD *)(a1 + 8);
  v125 = (__int64 *)v127;
  for ( i = 0x800000000LL; v33; v33 = *(_QWORD *)(v33 + 8) )
  {
    if ( (unsigned __int8)(*((_BYTE *)sub_1648700(v33) + 16) - 25) <= 9u )
      break;
  }
  v34 = 0;
LABEL_17:
  if ( !v33 )
  {
LABEL_24:
    v40 = 0;
    if ( (_DWORD)v34 )
      goto LABEL_74;
    goto LABEL_25;
  }
  while ( 1 )
  {
    v35 = *(_QWORD *)(v33 + 8);
    if ( v35 )
    {
      do
      {
        if ( (unsigned __int8)(*((_BYTE *)sub_1648700(v35) + 16) - 25) <= 9u )
          break;
        v35 = *(_QWORD *)(v35 + 8);
      }
      while ( v35 );
      v36 = v33;
      v33 = v35;
      v37 = sub_1648700(v36)[5];
      if ( (_QWORD *)v37 == v22 )
        goto LABEL_17;
      goto LABEL_21;
    }
    v37 = sub_1648700(v33)[5];
    if ( v22 == (_QWORD *)v37 )
      break;
LABEL_21:
    if ( HIDWORD(i) <= (unsigned int)v34 )
    {
      v106 = v37;
      sub_16CD150((__int64)&v125, v127, 0, 8, v38, v39);
      v34 = (unsigned int)i;
      v37 = v106;
    }
    v33 = v35;
    v125[v34] = v37;
    v34 = (unsigned int)(i + 1);
    LODWORD(i) = i + 1;
    if ( !v35 )
      goto LABEL_24;
  }
  v40 = 0;
  if ( !(_DWORD)v34 )
    goto LABEL_25;
LABEL_74:
  v109 = *(_QWORD *)(a1 + 56);
  v121[0] = sub_1649960(a1);
  v122 = (const char *)v121;
  v121[1] = v87;
  v124 = 773;
  v123 = a5;
  v88 = sub_157E9C0(a1);
  v89 = (_QWORD *)sub_22077B0(64);
  v40 = (__int64)v89;
  if ( v89 )
    sub_157FB60(v89, v88, (__int64)&v122, v109, a1);
  v92 = *(unsigned int *)(a6 + 8);
  if ( (unsigned int)v92 >= *(_DWORD *)(a6 + 12) )
  {
    sub_16CD150(a6, (const void *)(a6 + 16), 0, 8, v90, v91);
    v92 = *(unsigned int *)(a6 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a6 + 8 * v92) = v40;
  ++*(_DWORD *)(a6 + 8);
  v93 = sub_1648A60(56, 1u);
  v114 = (__int64)v93;
  if ( v93 )
    sub_15F8590((__int64)v93, a1, v40);
  v94 = *(_QWORD *)(sub_157ED20(a1) + 48);
  v95 = (__int64 *)(v114 + 48);
  v122 = (const char *)v94;
  if ( !v94 )
  {
    if ( v95 == (__int64 *)&v122 )
      goto LABEL_84;
    v101 = *(_QWORD *)(v114 + 48);
    if ( !v101 )
      goto LABEL_84;
LABEL_93:
    sub_161E7C0((__int64)v95, v101);
    goto LABEL_94;
  }
  sub_1623A60((__int64)&v122, v94, 2);
  if ( v95 == (__int64 *)&v122 )
  {
    if ( v122 )
      sub_161E7C0((__int64)&v122, (__int64)v122);
    goto LABEL_84;
  }
  v101 = *(_QWORD *)(v114 + 48);
  if ( v101 )
    goto LABEL_93;
LABEL_94:
  v102 = (unsigned __int8 *)v122;
  *(_QWORD *)(v114 + 48) = v122;
  if ( v102 )
    sub_1623210((__int64)&v122, v102, (__int64)v95);
LABEL_84:
  v96 = (unsigned int)i;
  v97 = &v125[(unsigned int)i];
  if ( v125 != v97 )
  {
    v98 = v125;
    do
    {
      v99 = *v98++;
      v100 = sub_157EBA0(v99);
      sub_1648780(v100, a1, v40);
    }
    while ( v97 != v98 );
    v97 = v125;
    v96 = (unsigned int)i;
  }
  v120 = 0;
  sub_1AA9AF0(a1, v40, v97, v96, a15, a16, a17, (__int64 *)&v120);
  sub_1AA5740(a1, v40, v125, (unsigned int)i, v114, v120);
LABEL_25:
  v41 = (__int64 *)sub_157F7B0(a1);
  v42 = (_QWORD *)sub_15F4880((__int64)v41);
  v43 = *a4 == 0;
  v122 = "lpad";
  if ( v43 )
  {
    v124 = 259;
  }
  else
  {
    v123 = a4;
    v124 = 771;
  }
  sub_164B780((__int64)v42, (__int64 *)&v122);
  v111 = (__int64 *)sub_157EE30((__int64)v22);
  sub_157E9D0((__int64)(v22 + 5), (__int64)v42);
  v46 = *v111;
  v47 = v42[3] & 7LL;
  v42[4] = v111;
  v46 &= 0xFFFFFFFFFFFFFFF8LL;
  v42[3] = v46 | v47;
  *(_QWORD *)(v46 + 8) = v42 + 3;
  *v111 = *v111 & 7 | (unsigned __int64)(v42 + 3);
  if ( v40 )
  {
    v48 = sub_15F4880((__int64)v41);
    v43 = *a5 == 0;
    v122 = "lpad";
    if ( v43 )
    {
      v124 = 259;
    }
    else
    {
      v123 = a5;
      v124 = 771;
    }
    v112 = v48;
    sub_164B780(v48, (__int64 *)&v122);
    v49 = (_QWORD *)sub_157EE30(v40);
    sub_157E9D0(v40 + 40, v112);
    v50 = *(_QWORD *)(v112 + 24);
    v51 = *v49 & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v112 + 32) = v49;
    *(_QWORD *)(v112 + 24) = v51 | v50 & 7;
    *(_QWORD *)(v51 + 8) = v112 + 24;
    *v49 = *v49 & 7LL | (v112 + 24);
    if ( v41[1] )
    {
      v122 = "lpad.phi";
      v124 = 259;
      v107 = *v41;
      v52 = sub_1648B60(64);
      v59 = v107;
      v60 = v112;
      v61 = v52;
      if ( v52 )
      {
        v108 = v112;
        v113 = v52;
        sub_15F1EA0(v52, v59, 53, 0, 0, (__int64)v41);
        *(_DWORD *)(v61 + 56) = 2;
        sub_164B780(v61, (__int64 *)&v122);
        v59 = *(unsigned int *)(v61 + 56);
        sub_1648880(v61, v59, 1);
        v60 = v108;
        v62 = v113;
      }
      else
      {
        v62 = 0;
      }
      v63 = *(_DWORD *)(v61 + 20) & 0xFFFFFFF;
      if ( v63 == *(_DWORD *)(v61 + 56) )
      {
        v116 = v62;
        v119 = v60;
        sub_15F55D0(v61, v59, v53, v54, v55, v56);
        v62 = v116;
        v60 = v119;
        v63 = *(_DWORD *)(v61 + 20) & 0xFFFFFFF;
      }
      v64 = (v63 + 1) & 0xFFFFFFF;
      v65 = v64 | *(_DWORD *)(v61 + 20) & 0xF0000000;
      *(_DWORD *)(v61 + 20) = v65;
      if ( (v65 & 0x40000000) != 0 )
        v66 = *(_QWORD *)(v61 - 8);
      else
        v66 = v62 - 24 * v64;
      v67 = (_QWORD *)(v66 + 24LL * (unsigned int)(v64 - 1));
      if ( *v67 )
      {
        v68 = v67[1];
        v69 = v67[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v69 = v68;
        if ( v68 )
        {
          v59 = *(_QWORD *)(v68 + 16) & 3LL;
          *(_QWORD *)(v68 + 16) = v59 | v69;
        }
      }
      *v67 = v42;
      v70 = v42[1];
      v67[1] = v70;
      if ( v70 )
      {
        v59 = (unsigned __int64)(v67 + 1) | *(_QWORD *)(v70 + 16) & 3LL;
        *(_QWORD *)(v70 + 16) = v59;
      }
      v67[2] = (unsigned __int64)(v42 + 1) | v67[2] & 3LL;
      v42[1] = v67;
      v71 = *(_DWORD *)(v61 + 20) & 0xFFFFFFF;
      v72 = (unsigned int)(v71 - 1);
      if ( (*(_BYTE *)(v61 + 23) & 0x40) != 0 )
        v73 = *(_QWORD *)(v61 - 8);
      else
        v73 = v62 - 24 * v71;
      v74 = 3LL * *(unsigned int *)(v61 + 56);
      *(_QWORD *)(v73 + 8 * v72 + 24LL * *(unsigned int *)(v61 + 56) + 8) = v22;
      v75 = *(_DWORD *)(v61 + 20) & 0xFFFFFFF;
      if ( v75 == *(_DWORD *)(v61 + 56) )
      {
        v115 = v62;
        v118 = v60;
        sub_15F55D0(v61, v59, v74, v73, v55, v56);
        v62 = v115;
        v60 = v118;
        v75 = *(_DWORD *)(v61 + 20) & 0xFFFFFFF;
      }
      v76 = (v75 + 1) & 0xFFFFFFF;
      v77 = v76 | *(_DWORD *)(v61 + 20) & 0xF0000000;
      *(_DWORD *)(v61 + 20) = v77;
      if ( (v77 & 0x40000000) != 0 )
        v78 = *(_QWORD *)(v61 - 8);
      else
        v78 = v62 - 24 * v76;
      v79 = (__int64 *)(v78 + 24LL * (unsigned int)(v76 - 1));
      if ( *v79 )
      {
        v80 = v79[1];
        v81 = v79[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v81 = v80;
        if ( v80 )
          *(_QWORD *)(v80 + 16) = *(_QWORD *)(v80 + 16) & 3LL | v81;
      }
      *v79 = v60;
      v82 = *(_QWORD *)(v60 + 8);
      v79[1] = v82;
      if ( v82 )
        *(_QWORD *)(v82 + 16) = (unsigned __int64)(v79 + 1) | *(_QWORD *)(v82 + 16) & 3LL;
      v79[2] = (v60 + 8) | v79[2] & 3;
      *(_QWORD *)(v60 + 8) = v79;
      v83 = *(_DWORD *)(v61 + 20) & 0xFFFFFFF;
      if ( (*(_BYTE *)(v61 + 23) & 0x40) != 0 )
        v84 = *(_QWORD *)(v61 - 8);
      else
        v84 = v62 - 24 * v83;
      *(_QWORD *)(v84 + 8LL * (unsigned int)(v83 - 1) + 24LL * *(unsigned int *)(v61 + 56) + 8) = v40;
      sub_164D160((__int64)v41, v61, a7, a8, a9, a10, v57, v58, a13, a14);
    }
  }
  else
  {
    sub_164D160((__int64)v41, (__int64)v42, a7, a8, a9, a10, v44, v45, a13, a14);
  }
  sub_15F20C0(v41);
  if ( v125 != (__int64 *)v127 )
    _libc_free((unsigned __int64)v125);
}
