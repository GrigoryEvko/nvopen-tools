// Function: sub_DD2D80
// Address: 0xdd2d80
//
__int64 __fastcall sub_DD2D80(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r12
  char v5; // di
  int v6; // edi
  _QWORD *v7; // rsi
  int v8; // r9d
  __int64 v9; // rax
  unsigned int v10; // ecx
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rsi
  _QWORD *v16; // r9
  _QWORD *v17; // r15
  _QWORD **v18; // r14
  _QWORD **v19; // rsi
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rdx
  _QWORD *v24; // rdi
  __int64 *v25; // rax
  _QWORD *v26; // r9
  _QWORD *v27; // r15
  _QWORD **v28; // r14
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rdx
  _QWORD *v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rsi
  __int64 v36; // rsi
  _QWORD *v37; // r9
  _QWORD *v38; // r15
  _QWORD **v39; // r14
  __int64 v40; // rax
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rdx
  __int64 v44; // rcx
  char v45; // dl
  _QWORD *v46; // rax
  _QWORD *v47; // r9
  _QWORD *v48; // r15
  _QWORD **v49; // r14
  __int64 v50; // rax
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // rdx
  __int64 v54; // rcx
  unsigned __int64 v55; // rax
  _QWORD *v56; // r9
  _QWORD *v57; // r15
  _QWORD **v58; // r14
  __int64 v59; // rax
  __int64 v60; // r9
  __int64 v61; // rdx
  _QWORD *v62; // r9
  _QWORD *v63; // r15
  _QWORD **v64; // r14
  __int64 v65; // rax
  __int64 v66; // r8
  __int64 v67; // r9
  __int64 v68; // rdx
  __int64 v69; // rcx
  unsigned __int64 v70; // rax
  _QWORD *v71; // r9
  _QWORD *v72; // r15
  _QWORD **v73; // r14
  __int64 v74; // rax
  __int64 v75; // r8
  __int64 v76; // r9
  __int64 v77; // rdx
  __int64 v78; // rcx
  unsigned __int64 v79; // rax
  _QWORD *v80; // r9
  _QWORD *v81; // r15
  _QWORD **v82; // r14
  __int64 v83; // rax
  __int64 v84; // r8
  __int64 v85; // r9
  __int64 v86; // rdx
  _QWORD *v87; // rax
  __int64 v88; // r14
  __int64 v89; // rax
  int v90; // eax
  int v91; // r10d
  __int64 v92; // [rsp+0h] [rbp-90h]
  __int64 v93; // [rsp+0h] [rbp-90h]
  __int64 v94; // [rsp+0h] [rbp-90h]
  __int64 v95; // [rsp+0h] [rbp-90h]
  __int64 v96; // [rsp+0h] [rbp-90h]
  __int64 v97; // [rsp+0h] [rbp-90h]
  __int64 v98; // [rsp+0h] [rbp-90h]
  __int64 v99; // [rsp+0h] [rbp-90h]
  _QWORD *v100; // [rsp+8h] [rbp-88h]
  _QWORD *v101; // [rsp+8h] [rbp-88h]
  _QWORD *v102; // [rsp+8h] [rbp-88h]
  _QWORD *v103; // [rsp+8h] [rbp-88h]
  _QWORD *v104; // [rsp+8h] [rbp-88h]
  _QWORD *v105; // [rsp+8h] [rbp-88h]
  _QWORD *v106; // [rsp+8h] [rbp-88h]
  _QWORD *v107; // [rsp+8h] [rbp-88h]
  char v108; // [rsp+17h] [rbp-79h]
  char v109; // [rsp+17h] [rbp-79h]
  char v110; // [rsp+17h] [rbp-79h]
  char v111; // [rsp+17h] [rbp-79h]
  char v112; // [rsp+17h] [rbp-79h]
  char v113; // [rsp+17h] [rbp-79h]
  char v114; // [rsp+17h] [rbp-79h]
  char v115; // [rsp+17h] [rbp-79h]
  __int64 v116; // [rsp+20h] [rbp-70h] BYREF
  __int64 v117; // [rsp+28h] [rbp-68h] BYREF
  _QWORD *v118; // [rsp+30h] [rbp-60h] BYREF
  __int64 v119; // [rsp+38h] [rbp-58h]
  _QWORD v120[10]; // [rsp+40h] [rbp-50h] BYREF

  v2 = a2;
  if ( *(_BYTE *)(sub_D95540(a2) + 8) != 14 )
    return v2;
  v5 = *((_BYTE *)a1 + 16);
  v116 = a2;
  v6 = v5 & 1;
  if ( v6 )
  {
    v7 = a1 + 3;
    v8 = 3;
  }
  else
  {
    v9 = *((unsigned int *)a1 + 8);
    v7 = (_QWORD *)a1[3];
    if ( !(_DWORD)v9 )
      goto LABEL_14;
    v8 = v9 - 1;
  }
  v10 = v8 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v11 = &v7[2 * v10];
  v12 = *v11;
  if ( v2 != *v11 )
  {
    v90 = 1;
    while ( v12 != -4096 )
    {
      v91 = v90 + 1;
      v10 = v8 & (v90 + v10);
      v11 = &v7[2 * v10];
      v12 = *v11;
      if ( v2 == *v11 )
        goto LABEL_8;
      v90 = v91;
    }
    if ( (_BYTE)v6 )
    {
      v14 = 8;
      goto LABEL_15;
    }
    v9 = *((unsigned int *)a1 + 8);
LABEL_14:
    v14 = 2 * v9;
LABEL_15:
    v11 = &v7[v14];
  }
LABEL_8:
  v13 = 8;
  if ( !(_BYTE)v6 )
    v13 = 2LL * *((unsigned int *)a1 + 8);
  if ( v11 == &v7[v13] )
  {
    switch ( *(_WORD *)(v2 + 24) )
    {
      case 0:
      case 1:
      case 0x10:
        goto LABEL_17;
      case 2:
        v15 = sub_DD2D80(a1, *(_QWORD *)(v2 + 32));
        if ( v15 != *(_QWORD *)(v2 + 32) )
          v2 = (__int64)sub_DC5200(*a1, v15, *(_QWORD *)(v2 + 40), 0);
        goto LABEL_17;
      case 3:
        v35 = sub_DD2D80(a1, *(_QWORD *)(v2 + 32));
        if ( v35 != *(_QWORD *)(v2 + 32) )
          v2 = (__int64)sub_DC2B70(*a1, v35, *(_QWORD *)(v2 + 40), 0);
        goto LABEL_17;
      case 4:
        v34 = sub_DD2D80(a1, *(_QWORD *)(v2 + 32));
        if ( v34 != *(_QWORD *)(v2 + 32) )
          v2 = (__int64)sub_DC5000(*a1, v34, *(_QWORD *)(v2 + 40), 0);
        goto LABEL_17;
      case 5:
        v26 = *(_QWORD **)(v2 + 32);
        v118 = v120;
        v119 = 0x200000000LL;
        v101 = &v26[*(_QWORD *)(v2 + 40)];
        if ( v26 == v101 )
          goto LABEL_17;
        v109 = 0;
        v27 = v26;
        do
        {
          v28 = (_QWORD **)*v27;
          v19 = (_QWORD **)*v27;
          v29 = sub_DD2D80(a1, *v27);
          v32 = (unsigned int)v119;
          if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
          {
            v19 = (_QWORD **)v120;
            v95 = v29;
            sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 8u, v30, v31);
            v32 = (unsigned int)v119;
            v29 = v95;
          }
          v118[v32] = v29;
          v24 = v118;
          LODWORD(v119) = v119 + 1;
          ++v27;
          v109 |= v118[(unsigned int)v119 - 1] != (_QWORD)v28;
        }
        while ( v101 != v27 );
        if ( v109 )
        {
          v19 = &v118;
          v33 = sub_DC7EB0((__int64 *)*a1, (__int64)&v118, *(_WORD *)(v2 + 28) & 7, 0);
          v24 = v118;
          v2 = (__int64)v33;
        }
        goto LABEL_83;
      case 6:
        v16 = *(_QWORD **)(v2 + 32);
        v118 = v120;
        v119 = 0x200000000LL;
        v100 = &v16[*(_QWORD *)(v2 + 40)];
        if ( v16 == v100 )
          goto LABEL_17;
        v108 = 0;
        v17 = v16;
        do
        {
          v18 = (_QWORD **)*v17;
          v19 = (_QWORD **)*v17;
          v20 = sub_DD2D80(a1, *v17);
          v23 = (unsigned int)v119;
          if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
          {
            v19 = (_QWORD **)v120;
            v96 = v20;
            sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 8u, v21, v22);
            v23 = (unsigned int)v119;
            v20 = v96;
          }
          v118[v23] = v20;
          v24 = v118;
          LODWORD(v119) = v119 + 1;
          ++v17;
          v108 |= v118[(unsigned int)v119 - 1] != (_QWORD)v18;
        }
        while ( v100 != v17 );
        if ( v108 )
        {
          v19 = &v118;
          v25 = sub_DC8BD0((__int64 *)*a1, (__int64)&v118, *(_WORD *)(v2 + 28) & 7, 0);
          v24 = v118;
          v2 = (__int64)v25;
        }
        goto LABEL_83;
      case 7:
        v88 = sub_DD2D80(a1, *(_QWORD *)(v2 + 32));
        v89 = sub_DD2D80(a1, *(_QWORD *)(v2 + 40));
        if ( v88 != *(_QWORD *)(v2 + 32) || v89 != *(_QWORD *)(v2 + 40) )
          v2 = sub_DCB270(*a1, v88, v89);
        goto LABEL_17;
      case 8:
        v80 = *(_QWORD **)(v2 + 32);
        v118 = v120;
        v119 = 0x200000000LL;
        v107 = &v80[*(_QWORD *)(v2 + 40)];
        if ( v80 == v107 )
          goto LABEL_17;
        v115 = 0;
        v81 = v80;
        do
        {
          v82 = (_QWORD **)*v81;
          v19 = (_QWORD **)*v81;
          v83 = sub_DD2D80(a1, *v81);
          v86 = (unsigned int)v119;
          if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
          {
            v19 = (_QWORD **)v120;
            v99 = v83;
            sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 8u, v84, v85);
            v86 = (unsigned int)v119;
            v83 = v99;
          }
          v118[v86] = v83;
          v24 = v118;
          LODWORD(v119) = v119 + 1;
          ++v81;
          v115 |= v118[(unsigned int)v119 - 1] != (_QWORD)v82;
        }
        while ( v107 != v81 );
        if ( v115 )
        {
          v19 = &v118;
          v87 = sub_DBFF60(*a1, (unsigned int *)&v118, *(_QWORD *)(v2 + 48), *(_WORD *)(v2 + 28) & 7);
          v24 = v118;
          v2 = (__int64)v87;
        }
        goto LABEL_83;
      case 9:
        v71 = *(_QWORD **)(v2 + 32);
        v118 = v120;
        v119 = 0x200000000LL;
        v106 = &v71[*(_QWORD *)(v2 + 40)];
        if ( v71 == v106 )
          goto LABEL_17;
        v114 = 0;
        v72 = v71;
        do
        {
          v73 = (_QWORD **)*v72;
          v19 = (_QWORD **)*v72;
          v74 = sub_DD2D80(a1, *v72);
          v77 = (unsigned int)v119;
          if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
          {
            v19 = (_QWORD **)v120;
            v94 = v74;
            sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 8u, v75, v76);
            v77 = (unsigned int)v119;
            v74 = v94;
          }
          v78 = (__int64)v118;
          v118[v77] = v74;
          v24 = v118;
          LODWORD(v119) = v119 + 1;
          ++v72;
          v114 |= v118[(unsigned int)v119 - 1] != (_QWORD)v73;
        }
        while ( v106 != v72 );
        if ( v114 )
        {
          v19 = &v118;
          v79 = sub_DCE040((__int64 *)*a1, (__int64)&v118, v77, v78, v75);
          v24 = v118;
          v2 = v79;
        }
        goto LABEL_83;
      case 0xA:
        v62 = *(_QWORD **)(v2 + 32);
        v118 = v120;
        v119 = 0x200000000LL;
        v105 = &v62[*(_QWORD *)(v2 + 40)];
        if ( v62 == v105 )
          goto LABEL_17;
        v113 = 0;
        v63 = v62;
        do
        {
          v64 = (_QWORD **)*v63;
          v19 = (_QWORD **)*v63;
          v65 = sub_DD2D80(a1, *v63);
          v68 = (unsigned int)v119;
          if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
          {
            v19 = (_QWORD **)v120;
            v97 = v65;
            sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 8u, v66, v67);
            v68 = (unsigned int)v119;
            v65 = v97;
          }
          v69 = (__int64)v118;
          v118[v68] = v65;
          v24 = v118;
          LODWORD(v119) = v119 + 1;
          ++v63;
          v113 |= v118[(unsigned int)v119 - 1] != (_QWORD)v64;
        }
        while ( v105 != v63 );
        if ( v113 )
        {
          v19 = &v118;
          v70 = sub_DCDF90((__int64 *)*a1, (__int64)&v118, v68, v69, v66);
          v24 = v118;
          v2 = v70;
        }
        goto LABEL_83;
      case 0xB:
        v56 = *(_QWORD **)(v2 + 32);
        v118 = v120;
        v119 = 0x200000000LL;
        v104 = &v56[*(_QWORD *)(v2 + 40)];
        if ( v56 == v104 )
          goto LABEL_17;
        v112 = 0;
        v57 = v56;
        do
        {
          v58 = (_QWORD **)*v57;
          v19 = (_QWORD **)*v57;
          v59 = sub_DD2D80(a1, *v57);
          v61 = (unsigned int)v119;
          if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
          {
            v19 = (_QWORD **)v120;
            v93 = v59;
            sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 8u, v41, v60);
            v61 = (unsigned int)v119;
            v59 = v93;
          }
          v44 = (__int64)v118;
          v118[v61] = v59;
          v24 = v118;
          LODWORD(v119) = v119 + 1;
          ++v57;
          v112 |= v118[(unsigned int)v119 - 1] != (_QWORD)v58;
        }
        while ( v104 != v57 );
        v45 = 0;
        if ( v112 )
          goto LABEL_47;
        goto LABEL_83;
      case 0xC:
        v47 = *(_QWORD **)(v2 + 32);
        v118 = v120;
        v119 = 0x200000000LL;
        v103 = &v47[*(_QWORD *)(v2 + 40)];
        if ( v47 == v103 )
          goto LABEL_17;
        v111 = 0;
        v48 = v47;
        do
        {
          v49 = (_QWORD **)*v48;
          v19 = (_QWORD **)*v48;
          v50 = sub_DD2D80(a1, *v48);
          v53 = (unsigned int)v119;
          if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
          {
            v19 = (_QWORD **)v120;
            v98 = v50;
            sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 8u, v51, v52);
            v53 = (unsigned int)v119;
            v50 = v98;
          }
          v54 = (__int64)v118;
          v118[v53] = v50;
          v24 = v118;
          LODWORD(v119) = v119 + 1;
          ++v48;
          v111 |= v118[(unsigned int)v119 - 1] != (_QWORD)v49;
        }
        while ( v103 != v48 );
        if ( v111 )
        {
          v19 = &v118;
          v55 = sub_DCE150((__int64 *)*a1, (__int64)&v118, v53, v54, v51);
          v24 = v118;
          v2 = v55;
        }
        goto LABEL_83;
      case 0xD:
        v37 = *(_QWORD **)(v2 + 32);
        v118 = v120;
        v119 = 0x200000000LL;
        v102 = &v37[*(_QWORD *)(v2 + 40)];
        if ( v37 == v102 )
          goto LABEL_17;
        v110 = 0;
        v38 = v37;
        do
        {
          v39 = (_QWORD **)*v38;
          v19 = (_QWORD **)*v38;
          v40 = sub_DD2D80(a1, *v38);
          v43 = (unsigned int)v119;
          if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
          {
            v19 = (_QWORD **)v120;
            v92 = v40;
            sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 8u, v41, v42);
            v43 = (unsigned int)v119;
            v40 = v92;
          }
          v44 = (__int64)v118;
          v118[v43] = v40;
          v24 = v118;
          LODWORD(v119) = v119 + 1;
          ++v38;
          v110 |= v118[(unsigned int)v119 - 1] != (_QWORD)v39;
        }
        while ( v102 != v38 );
        if ( !v110 )
          goto LABEL_83;
        v45 = 1;
LABEL_47:
        v19 = &v118;
        v46 = sub_DCEE50((__int64 *)*a1, (__int64)&v118, v45, v44, v41);
        v24 = v118;
        v2 = (__int64)v46;
LABEL_83:
        if ( v24 != v120 )
          _libc_free(v24, v19);
LABEL_17:
        v117 = v2;
        sub_DB11F0((__int64)&v118, (__int64)(a1 + 1), &v116, &v117);
        v11 = (__int64 *)v120[0];
        break;
      case 0xE:
        v36 = sub_DD2D80(a1, *(_QWORD *)(v2 + 32));
        if ( v36 != *(_QWORD *)(v2 + 32) )
          v2 = sub_DD3A70(*a1, v36, *(_QWORD *)(v2 + 40));
        goto LABEL_17;
      case 0xF:
        v2 = sub_DD3750(*a1, v2, 1);
        goto LABEL_17;
      default:
        BUG();
    }
  }
  return v11[1];
}
