// Function: sub_310AE90
// Address: 0x310ae90
//
_QWORD *__fastcall sub_310AE90(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rbp
  __int64 v4; // r12
  _QWORD *v6; // rbx
  __int64 *v7; // r8
  __int64 v8; // rax
  __int64 *v9; // r15
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  _QWORD *v16; // rdi
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // rax
  unsigned int v23; // ecx
  __int64 *v24; // rdx
  __int64 v25; // r9
  __int64 v26; // rsi
  __int64 v27; // rsi
  __int64 *v28; // r8
  __int64 v29; // rax
  __int64 *v30; // r15
  __int64 v31; // r14
  __int64 v32; // rax
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rdx
  _QWORD *v36; // rax
  __int64 v37; // rsi
  __int64 *v38; // r8
  __int64 v39; // rax
  __int64 *v40; // r15
  __int64 v41; // r14
  __int64 v42; // rax
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rdx
  __int64 *v46; // rax
  __int64 v47; // r13
  __int64 v48; // rax
  __int64 *v49; // r8
  __int64 v50; // rax
  __int64 *v51; // r15
  __int64 v52; // r14
  __int64 v53; // rax
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // rdx
  _QWORD *v57; // rax
  __int64 *v58; // r8
  __int64 v59; // rax
  __int64 *v60; // r15
  __int64 v61; // r14
  __int64 v62; // rax
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rdx
  __int64 v66; // rcx
  unsigned __int64 v67; // rax
  __int64 *v68; // r8
  __int64 v69; // rax
  __int64 *v70; // r15
  __int64 v71; // r14
  __int64 v72; // rax
  __int64 v73; // r8
  __int64 v74; // r9
  __int64 v75; // rdx
  __int64 v76; // rcx
  unsigned __int64 v77; // rax
  __int64 *v78; // r8
  __int64 v79; // rax
  __int64 *v80; // r15
  __int64 v81; // r14
  __int64 v82; // rax
  __int64 v83; // r8
  __int64 v84; // r9
  __int64 v85; // rdx
  __int64 v86; // rcx
  _QWORD *v87; // rax
  __int64 *v88; // r8
  __int64 v89; // rax
  __int64 *v90; // r15
  __int64 v91; // r14
  __int64 v92; // rax
  __int64 v93; // r8
  __int64 v94; // r9
  __int64 v95; // rdx
  __int64 v96; // rcx
  unsigned __int64 v97; // rax
  _QWORD *v98; // rax
  int v99; // edx
  int v100; // r10d
  __int64 v101; // [rsp-78h] [rbp-78h]
  __int64 v102; // [rsp-78h] [rbp-78h]
  __int64 v103; // [rsp-78h] [rbp-78h]
  __int64 v104; // [rsp-78h] [rbp-78h]
  __int64 v105; // [rsp-78h] [rbp-78h]
  __int64 v106; // [rsp-78h] [rbp-78h]
  __int64 v107; // [rsp-78h] [rbp-78h]
  __int64 v108; // [rsp-78h] [rbp-78h]
  __int64 *v109; // [rsp-68h] [rbp-68h]
  __int64 *v110; // [rsp-68h] [rbp-68h]
  __int64 *v111; // [rsp-68h] [rbp-68h]
  __int64 *v112; // [rsp-68h] [rbp-68h]
  __int64 *v113; // [rsp-68h] [rbp-68h]
  __int64 *v114; // [rsp-68h] [rbp-68h]
  __int64 *v115; // [rsp-68h] [rbp-68h]
  __int64 *v116; // [rsp-68h] [rbp-68h]
  char v117; // [rsp-59h] [rbp-59h]
  char v118; // [rsp-59h] [rbp-59h]
  char v119; // [rsp-59h] [rbp-59h]
  char v120; // [rsp-59h] [rbp-59h]
  char v121; // [rsp-59h] [rbp-59h]
  char v122; // [rsp-59h] [rbp-59h]
  char v123; // [rsp-59h] [rbp-59h]
  char v124; // [rsp-59h] [rbp-59h]
  _QWORD *v125; // [rsp-58h] [rbp-58h] BYREF
  __int64 v126; // [rsp-50h] [rbp-50h]
  _QWORD v127[9]; // [rsp-48h] [rbp-48h] BYREF

  v127[8] = v3;
  v127[4] = v4;
  v127[3] = v2;
  v6 = (_QWORD *)a2;
  switch ( *(_WORD *)(a2 + 24) )
  {
    case 0:
    case 1:
    case 0x10:
      return v6;
    case 2:
      v26 = sub_310AAC0((__int64)a1, *(_QWORD *)(a2 + 32));
      if ( v26 == v6[4] )
        return v6;
      return sub_DC5200(*a1, v26, v6[5], 0);
    case 3:
      v37 = sub_310AAC0((__int64)a1, *(_QWORD *)(a2 + 32));
      if ( v37 == v6[4] )
        return v6;
      return sub_DC2B70(*a1, v37, v6[5], 0);
    case 4:
      v27 = sub_310AAC0((__int64)a1, *(_QWORD *)(a2 + 32));
      if ( v27 == v6[4] )
        return v6;
      return sub_DC5000(*a1, v27, v6[5], 0);
    case 5:
      v28 = *(__int64 **)(a2 + 32);
      v126 = 0x200000000LL;
      v29 = *(_QWORD *)(a2 + 40);
      v125 = v127;
      v110 = &v28[v29];
      if ( v28 == v110 )
        return v6;
      v118 = 0;
      v30 = v28;
      do
      {
        v31 = *v30;
        v32 = sub_310AAC0((__int64)a1, *v30);
        v35 = (unsigned int)v126;
        if ( (unsigned __int64)(unsigned int)v126 + 1 > HIDWORD(v126) )
        {
          v102 = v32;
          sub_C8D5F0((__int64)&v125, v127, (unsigned int)v126 + 1LL, 8u, v33, v34);
          v35 = (unsigned int)v126;
          v32 = v102;
        }
        v125[v35] = v32;
        v16 = v125;
        LODWORD(v126) = v126 + 1;
        ++v30;
        v118 |= v125[(unsigned int)v126 - 1] != v31;
      }
      while ( v110 != v30 );
      if ( v118 )
      {
        v36 = sub_DC7EB0((__int64 *)*a1, (__int64)&v125, 0, 0);
        v16 = v125;
        v6 = v36;
      }
      goto LABEL_9;
    case 6:
      v38 = *(__int64 **)(a2 + 32);
      v126 = 0x200000000LL;
      v39 = *(_QWORD *)(a2 + 40);
      v125 = v127;
      v111 = &v38[v39];
      if ( v38 == v111 )
        return v6;
      v119 = 0;
      v40 = v38;
      do
      {
        v41 = *v40;
        v42 = sub_310AAC0((__int64)a1, *v40);
        v45 = (unsigned int)v126;
        if ( (unsigned __int64)(unsigned int)v126 + 1 > HIDWORD(v126) )
        {
          v104 = v42;
          sub_C8D5F0((__int64)&v125, v127, (unsigned int)v126 + 1LL, 8u, v43, v44);
          v45 = (unsigned int)v126;
          v42 = v104;
        }
        v125[v45] = v42;
        v16 = v125;
        LODWORD(v126) = v126 + 1;
        ++v40;
        v119 |= v125[(unsigned int)v126 - 1] != v41;
      }
      while ( v111 != v40 );
      if ( v119 )
      {
        v46 = sub_DC8BD0((__int64 *)*a1, (__int64)&v125, 0, 0);
        v16 = v125;
        v6 = v46;
      }
      goto LABEL_9;
    case 7:
      v47 = sub_310AAC0((__int64)a1, *(_QWORD *)(a2 + 32));
      v48 = sub_310AAC0((__int64)a1, *(_QWORD *)(a2 + 40));
      if ( v47 == *(_QWORD *)(a2 + 32) && v48 == *(_QWORD *)(a2 + 40) )
        return v6;
      return (_QWORD *)sub_DCB270(*a1, v47, v48);
    case 8:
      v49 = *(__int64 **)(a2 + 32);
      v126 = 0x200000000LL;
      v50 = *(_QWORD *)(a2 + 40);
      v125 = v127;
      v112 = &v49[v50];
      if ( v49 == v112 )
        return v6;
      v120 = 0;
      v51 = v49;
      do
      {
        v52 = *v51;
        v53 = sub_310AAC0((__int64)a1, *v51);
        v56 = (unsigned int)v126;
        if ( (unsigned __int64)(unsigned int)v126 + 1 > HIDWORD(v126) )
        {
          v103 = v53;
          sub_C8D5F0((__int64)&v125, v127, (unsigned int)v126 + 1LL, 8u, v54, v55);
          v56 = (unsigned int)v126;
          v53 = v103;
        }
        v125[v56] = v53;
        v16 = v125;
        LODWORD(v126) = v126 + 1;
        ++v51;
        v120 |= v125[(unsigned int)v126 - 1] != v52;
      }
      while ( v112 != v51 );
      if ( v120 )
      {
        v57 = sub_DBFF60(*a1, (unsigned int *)&v125, *(_QWORD *)(a2 + 48), *(_WORD *)(a2 + 28) & 7);
        v16 = v125;
        v6 = v57;
      }
      goto LABEL_9;
    case 9:
      v58 = *(__int64 **)(a2 + 32);
      v126 = 0x200000000LL;
      v59 = *(_QWORD *)(a2 + 40);
      v125 = v127;
      v113 = &v58[v59];
      if ( v58 == v113 )
        return v6;
      v121 = 0;
      v60 = v58;
      do
      {
        v61 = *v60;
        v62 = sub_310AAC0((__int64)a1, *v60);
        v65 = (unsigned int)v126;
        if ( (unsigned __int64)(unsigned int)v126 + 1 > HIDWORD(v126) )
        {
          v101 = v62;
          sub_C8D5F0((__int64)&v125, v127, (unsigned int)v126 + 1LL, 8u, v63, v64);
          v65 = (unsigned int)v126;
          v62 = v101;
        }
        v66 = (__int64)v125;
        v125[v65] = v62;
        v16 = v125;
        LODWORD(v126) = v126 + 1;
        ++v60;
        v121 |= v125[(unsigned int)v126 - 1] != v61;
      }
      while ( v113 != v60 );
      if ( v121 )
      {
        v67 = sub_DCE040((__int64 *)*a1, (__int64)&v125, v65, v66, v63);
        v16 = v125;
        v6 = (_QWORD *)v67;
      }
      goto LABEL_9;
    case 0xA:
      v68 = *(__int64 **)(a2 + 32);
      v126 = 0x200000000LL;
      v69 = *(_QWORD *)(a2 + 40);
      v125 = v127;
      v114 = &v68[v69];
      if ( v68 == v114 )
        return v6;
      v122 = 0;
      v70 = v68;
      do
      {
        v71 = *v70;
        v72 = sub_310AAC0((__int64)a1, *v70);
        v75 = (unsigned int)v126;
        if ( (unsigned __int64)(unsigned int)v126 + 1 > HIDWORD(v126) )
        {
          v108 = v72;
          sub_C8D5F0((__int64)&v125, v127, (unsigned int)v126 + 1LL, 8u, v73, v74);
          v75 = (unsigned int)v126;
          v72 = v108;
        }
        v76 = (__int64)v125;
        v125[v75] = v72;
        v16 = v125;
        LODWORD(v126) = v126 + 1;
        ++v70;
        v122 |= v125[(unsigned int)v126 - 1] != v71;
      }
      while ( v114 != v70 );
      if ( v122 )
      {
        v77 = sub_DCDF90((__int64 *)*a1, (__int64)&v125, v75, v76, v73);
        v16 = v125;
        v6 = (_QWORD *)v77;
      }
      goto LABEL_9;
    case 0xB:
      v78 = *(__int64 **)(a2 + 32);
      v126 = 0x200000000LL;
      v79 = *(_QWORD *)(a2 + 40);
      v125 = v127;
      v115 = &v78[v79];
      if ( v78 == v115 )
        return v6;
      v123 = 0;
      v80 = v78;
      do
      {
        v81 = *v80;
        v82 = sub_310AAC0((__int64)a1, *v80);
        v85 = (unsigned int)v126;
        if ( (unsigned __int64)(unsigned int)v126 + 1 > HIDWORD(v126) )
        {
          v107 = v82;
          sub_C8D5F0((__int64)&v125, v127, (unsigned int)v126 + 1LL, 8u, v83, v84);
          v85 = (unsigned int)v126;
          v82 = v107;
        }
        v86 = (__int64)v125;
        v125[v85] = v82;
        v16 = v125;
        LODWORD(v126) = v126 + 1;
        ++v80;
        v123 |= v125[(unsigned int)v126 - 1] != v81;
      }
      while ( v115 != v80 );
      if ( v123 )
      {
        v87 = sub_DCEE50((__int64 *)*a1, (__int64)&v125, 0, v86, v83);
        v16 = v125;
        v6 = v87;
      }
      goto LABEL_9;
    case 0xC:
      v88 = *(__int64 **)(a2 + 32);
      v126 = 0x200000000LL;
      v89 = *(_QWORD *)(a2 + 40);
      v125 = v127;
      v116 = &v88[v89];
      if ( v88 == v116 )
        return v6;
      v124 = 0;
      v90 = v88;
      do
      {
        v91 = *v90;
        v92 = sub_310AAC0((__int64)a1, *v90);
        v95 = (unsigned int)v126;
        if ( (unsigned __int64)(unsigned int)v126 + 1 > HIDWORD(v126) )
        {
          v106 = v92;
          sub_C8D5F0((__int64)&v125, v127, (unsigned int)v126 + 1LL, 8u, v93, v94);
          v95 = (unsigned int)v126;
          v92 = v106;
        }
        v96 = (__int64)v125;
        v125[v95] = v92;
        v16 = v125;
        LODWORD(v126) = v126 + 1;
        ++v90;
        v124 |= v125[(unsigned int)v126 - 1] != v91;
      }
      while ( v116 != v90 );
      if ( v124 )
      {
        v97 = sub_DCE150((__int64 *)*a1, (__int64)&v125, v95, v96, v93);
        v16 = v125;
        v6 = (_QWORD *)v97;
      }
      goto LABEL_9;
    case 0xD:
      v7 = *(__int64 **)(a2 + 32);
      v126 = 0x200000000LL;
      v8 = *(_QWORD *)(a2 + 40);
      v125 = v127;
      v109 = &v7[v8];
      if ( v7 == v109 )
        return v6;
      v117 = 0;
      v9 = v7;
      do
      {
        v10 = *v9;
        v11 = sub_310AAC0((__int64)a1, *v9);
        v14 = (unsigned int)v126;
        if ( (unsigned __int64)(unsigned int)v126 + 1 > HIDWORD(v126) )
        {
          v105 = v11;
          sub_C8D5F0((__int64)&v125, v127, (unsigned int)v126 + 1LL, 8u, v12, v13);
          v14 = (unsigned int)v126;
          v11 = v105;
        }
        v15 = (__int64)v125;
        v125[v14] = v11;
        v16 = v125;
        LODWORD(v126) = v126 + 1;
        ++v9;
        v117 |= v125[(unsigned int)v126 - 1] != v10;
      }
      while ( v109 != v9 );
      if ( v117 )
      {
        v98 = sub_DCEE50((__int64 *)*a1, (__int64)&v125, 1, v15, v12);
        v16 = v125;
        v6 = v98;
      }
LABEL_9:
      if ( v16 != v127 )
        _libc_free((unsigned __int64)v16);
      return v6;
    case 0xE:
      v18 = sub_310AAC0((__int64)a1, *(_QWORD *)(a2 + 32));
      if ( v18 == v6[4] )
        return v6;
      return sub_DD3A70(*a1, v18, v6[5]);
    case 0xF:
      v19 = a1[11];
      v20 = *(_QWORD *)(a2 - 8);
      v21 = *(_QWORD *)(v19 + 8);
      v22 = *(unsigned int *)(v19 + 24);
      if ( !(_DWORD)v22 )
        return v6;
      v23 = (v22 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v24 = (__int64 *)(v21 + 16LL * v23);
      v25 = *v24;
      if ( *v24 == v20 )
        goto LABEL_16;
      v99 = 1;
      break;
    default:
      BUG();
  }
  while ( 1 )
  {
    if ( v25 == -4096 )
      return v6;
    v100 = v99 + 1;
    v23 = (v22 - 1) & (v99 + v23);
    v24 = (__int64 *)(v21 + 16LL * v23);
    v25 = *v24;
    if ( v20 == *v24 )
      break;
    v99 = v100;
  }
LABEL_16:
  if ( v24 == (__int64 *)(v21 + 16 * v22) )
    return v6;
  return (_QWORD *)v24[1];
}
