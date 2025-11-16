// Function: sub_DE1010
// Address: 0xde1010
//
__int64 __fastcall sub_DE1010(_QWORD **a1, __int64 a2)
{
  __int64 v2; // r13
  char v4; // di
  int v5; // edi
  _QWORD **v6; // rsi
  int v7; // r9d
  unsigned int v8; // ecx
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // rsi
  _QWORD *v17; // r9
  _QWORD *v18; // r15
  _QWORD **v19; // r14
  _QWORD **v20; // rsi
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rcx
  _QWORD *v26; // rdi
  char v27; // dl
  _QWORD *v28; // rax
  _QWORD *v29; // r9
  _QWORD *v30; // r15
  _QWORD **v31; // r14
  __int64 v32; // rax
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rdx
  __int64 v36; // rcx
  unsigned __int64 v37; // rax
  _QWORD *v38; // r9
  _QWORD *v39; // r15
  _QWORD **v40; // r14
  __int64 v41; // rax
  __int64 v42; // r9
  __int64 v43; // rdx
  _QWORD *v44; // r9
  _QWORD *v45; // r15
  _QWORD **v46; // r14
  __int64 v47; // rax
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rdx
  __int64 v51; // rcx
  unsigned __int64 v52; // rax
  _QWORD *v53; // r9
  _QWORD *v54; // r15
  _QWORD **v55; // r14
  __int64 v56; // rax
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // rdx
  __int64 v60; // rcx
  unsigned __int64 v61; // rax
  _QWORD *v62; // r9
  _QWORD *v63; // r15
  _QWORD **v64; // r14
  __int64 v65; // rax
  __int64 v66; // r8
  __int64 v67; // r9
  __int64 v68; // rdx
  _QWORD *v69; // rax
  _QWORD *v70; // r9
  _QWORD *v71; // r15
  _QWORD **v72; // r14
  __int64 v73; // rax
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 v76; // rdx
  _QWORD *v77; // rax
  __int64 v78; // rsi
  __int64 v79; // r14
  __int64 v80; // rax
  _QWORD *v81; // r9
  _QWORD *v82; // r15
  _QWORD **v83; // r14
  __int64 v84; // rax
  __int64 v85; // r8
  __int64 v86; // r9
  __int64 v87; // rdx
  __int64 *v88; // rax
  __int64 v89; // rsi
  __int64 v90; // rsi
  int v91; // r10d
  __int64 v92; // [rsp+8h] [rbp-98h]
  __int64 v93; // [rsp+8h] [rbp-98h]
  __int64 v94; // [rsp+8h] [rbp-98h]
  __int64 v95; // [rsp+8h] [rbp-98h]
  __int64 v96; // [rsp+8h] [rbp-98h]
  __int64 v97; // [rsp+8h] [rbp-98h]
  __int64 v98; // [rsp+8h] [rbp-98h]
  __int64 v99; // [rsp+8h] [rbp-98h]
  _QWORD *v100; // [rsp+10h] [rbp-90h]
  _QWORD *v101; // [rsp+10h] [rbp-90h]
  _QWORD *v102; // [rsp+10h] [rbp-90h]
  _QWORD *v103; // [rsp+10h] [rbp-90h]
  _QWORD *v104; // [rsp+10h] [rbp-90h]
  _QWORD *v105; // [rsp+10h] [rbp-90h]
  _QWORD *v106; // [rsp+10h] [rbp-90h]
  _QWORD *v107; // [rsp+10h] [rbp-90h]
  char v108; // [rsp+1Fh] [rbp-81h]
  char v109; // [rsp+1Fh] [rbp-81h]
  char v110; // [rsp+1Fh] [rbp-81h]
  char v111; // [rsp+1Fh] [rbp-81h]
  char v112; // [rsp+1Fh] [rbp-81h]
  char v113; // [rsp+1Fh] [rbp-81h]
  char v114; // [rsp+1Fh] [rbp-81h]
  char v115; // [rsp+1Fh] [rbp-81h]
  __int64 v116; // [rsp+28h] [rbp-78h] BYREF
  __int64 v117; // [rsp+38h] [rbp-68h] BYREF
  _QWORD *v118; // [rsp+40h] [rbp-60h] BYREF
  __int64 v119; // [rsp+48h] [rbp-58h]
  _QWORD v120[10]; // [rsp+50h] [rbp-50h] BYREF

  v2 = a2;
  v4 = *((_BYTE *)a1 + 16);
  v116 = a2;
  v5 = v4 & 1;
  if ( v5 )
  {
    v6 = a1 + 3;
    v7 = 3;
  }
  else
  {
    v13 = *((unsigned int *)a1 + 8);
    v6 = (_QWORD **)a1[3];
    if ( !(_DWORD)v13 )
      goto LABEL_12;
    v7 = v13 - 1;
  }
  v8 = v7 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v9 = (__int64 *)&v6[2 * v8];
  v10 = *v9;
  if ( v2 == *v9 )
    goto LABEL_4;
  v15 = 1;
  while ( v10 != -4096 )
  {
    v91 = v15 + 1;
    v8 = v7 & (v15 + v8);
    v9 = (__int64 *)&v6[2 * v8];
    v10 = *v9;
    if ( v2 == *v9 )
      goto LABEL_4;
    v15 = v91;
  }
  if ( (_BYTE)v5 )
  {
    v14 = 8;
    goto LABEL_13;
  }
  v13 = *((unsigned int *)a1 + 8);
LABEL_12:
  v14 = 2 * v13;
LABEL_13:
  v9 = (__int64 *)&v6[v14];
LABEL_4:
  v11 = 8;
  if ( !(_BYTE)v5 )
    v11 = 2LL * *((unsigned int *)a1 + 8);
  if ( v9 == (__int64 *)&v6[v11] )
  {
    switch ( *(_WORD *)(v2 + 24) )
    {
      case 0:
        v2 = (__int64)sub_DA26C0(*a1, *(_QWORD *)(v2 + 32) + 24LL);
        goto LABEL_19;
      case 1:
        goto LABEL_19;
      case 2:
        v90 = sub_DE1010(a1, *(_QWORD *)(v2 + 32));
        if ( v90 != *(_QWORD *)(v2 + 32) )
          v2 = (__int64)sub_DC5200((__int64)*a1, v90, *(_QWORD *)(v2 + 40), 0);
        goto LABEL_19;
      case 3:
        v89 = sub_DE1010(a1, *(_QWORD *)(v2 + 32));
        if ( v89 != *(_QWORD *)(v2 + 32) )
          v2 = (__int64)sub_DC2B70((__int64)*a1, v89, *(_QWORD *)(v2 + 40), 0);
        goto LABEL_19;
      case 4:
        v78 = sub_DE1010(a1, *(_QWORD *)(v2 + 32));
        if ( v78 != *(_QWORD *)(v2 + 32) )
          v2 = (__int64)sub_DC5000((__int64)*a1, v78, *(_QWORD *)(v2 + 40), 0);
        goto LABEL_19;
      case 5:
        v118 = v120;
        v119 = 0x200000000LL;
        v70 = *(_QWORD **)(v2 + 32);
        v106 = &v70[*(_QWORD *)(v2 + 40)];
        if ( v70 == v106 )
          goto LABEL_19;
        v114 = 0;
        v71 = *(_QWORD **)(v2 + 32);
        do
        {
          v72 = (_QWORD **)*v71;
          v20 = (_QWORD **)*v71;
          v73 = sub_DE1010(a1, *v71);
          v76 = (unsigned int)v119;
          if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
          {
            v20 = (_QWORD **)v120;
            v97 = v73;
            sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 8u, v74, v75);
            v76 = (unsigned int)v119;
            v73 = v97;
          }
          v118[v76] = v73;
          v26 = v118;
          LODWORD(v119) = v119 + 1;
          ++v71;
          v114 |= v118[(unsigned int)v119 - 1] != (_QWORD)v72;
        }
        while ( v106 != v71 );
        if ( v114 )
        {
          v20 = &v118;
          v77 = sub_DC7EB0(*a1, (__int64)&v118, 0, 0);
          v26 = v118;
          v2 = (__int64)v77;
        }
        goto LABEL_60;
      case 6:
        v118 = v120;
        v119 = 0x200000000LL;
        v81 = *(_QWORD **)(v2 + 32);
        v107 = &v81[*(_QWORD *)(v2 + 40)];
        if ( v81 == v107 )
          goto LABEL_19;
        v115 = 0;
        v82 = *(_QWORD **)(v2 + 32);
        do
        {
          v83 = (_QWORD **)*v82;
          v20 = (_QWORD **)*v82;
          v84 = sub_DE1010(a1, *v82);
          v87 = (unsigned int)v119;
          if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
          {
            v20 = (_QWORD **)v120;
            v93 = v84;
            sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 8u, v85, v86);
            v87 = (unsigned int)v119;
            v84 = v93;
          }
          v118[v87] = v84;
          v26 = v118;
          LODWORD(v119) = v119 + 1;
          ++v82;
          v115 |= v118[(unsigned int)v119 - 1] != (_QWORD)v83;
        }
        while ( v107 != v82 );
        if ( v115 )
        {
          v20 = &v118;
          v88 = sub_DC8BD0(*a1, (__int64)&v118, 0, 0);
          v26 = v118;
          v2 = (__int64)v88;
        }
        goto LABEL_60;
      case 7:
        v79 = sub_DE1010(a1, *(_QWORD *)(v2 + 32));
        v80 = sub_DE1010(a1, *(_QWORD *)(v2 + 40));
        if ( v79 != *(_QWORD *)(v2 + 32) || v80 != *(_QWORD *)(v2 + 40) )
          v2 = sub_DCB270((__int64)*a1, v79, v80);
        goto LABEL_19;
      case 8:
        v118 = v120;
        v119 = 0x200000000LL;
        v62 = *(_QWORD **)(v2 + 32);
        v105 = &v62[*(_QWORD *)(v2 + 40)];
        if ( v62 == v105 )
          goto LABEL_19;
        v113 = 0;
        v63 = *(_QWORD **)(v2 + 32);
        do
        {
          v64 = (_QWORD **)*v63;
          v20 = (_QWORD **)*v63;
          v65 = sub_DE1010(a1, *v63);
          v68 = (unsigned int)v119;
          if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
          {
            v20 = (_QWORD **)v120;
            v92 = v65;
            sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 8u, v66, v67);
            v68 = (unsigned int)v119;
            v65 = v92;
          }
          v118[v68] = v65;
          v26 = v118;
          LODWORD(v119) = v119 + 1;
          ++v63;
          v113 |= v118[(unsigned int)v119 - 1] != (_QWORD)v64;
        }
        while ( v105 != v63 );
        if ( v113 )
        {
          v20 = &v118;
          v69 = sub_DBFF60((__int64)*a1, (unsigned int *)&v118, *(_QWORD *)(v2 + 48), *(_WORD *)(v2 + 28) & 7);
          v26 = v118;
          v2 = (__int64)v69;
        }
        goto LABEL_60;
      case 9:
        v118 = v120;
        v119 = 0x200000000LL;
        v53 = *(_QWORD **)(v2 + 32);
        v104 = &v53[*(_QWORD *)(v2 + 40)];
        if ( v53 == v104 )
          goto LABEL_19;
        v112 = 0;
        v54 = *(_QWORD **)(v2 + 32);
        do
        {
          v55 = (_QWORD **)*v54;
          v20 = (_QWORD **)*v54;
          v56 = sub_DE1010(a1, *v54);
          v59 = (unsigned int)v119;
          if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
          {
            v20 = (_QWORD **)v120;
            v96 = v56;
            sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 8u, v57, v58);
            v59 = (unsigned int)v119;
            v56 = v96;
          }
          v60 = (__int64)v118;
          v118[v59] = v56;
          v26 = v118;
          LODWORD(v119) = v119 + 1;
          ++v54;
          v112 |= v118[(unsigned int)v119 - 1] != (_QWORD)v55;
        }
        while ( v104 != v54 );
        if ( v112 )
        {
          v20 = &v118;
          v61 = sub_DCE040(*a1, (__int64)&v118, v59, v60, v57);
          v26 = v118;
          v2 = v61;
        }
        goto LABEL_60;
      case 0xA:
        v118 = v120;
        v119 = 0x200000000LL;
        v44 = *(_QWORD **)(v2 + 32);
        v103 = &v44[*(_QWORD *)(v2 + 40)];
        if ( v44 == v103 )
          goto LABEL_19;
        v111 = 0;
        v45 = *(_QWORD **)(v2 + 32);
        do
        {
          v46 = (_QWORD **)*v45;
          v20 = (_QWORD **)*v45;
          v47 = sub_DE1010(a1, *v45);
          v50 = (unsigned int)v119;
          if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
          {
            v20 = (_QWORD **)v120;
            v99 = v47;
            sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 8u, v48, v49);
            v50 = (unsigned int)v119;
            v47 = v99;
          }
          v51 = (__int64)v118;
          v118[v50] = v47;
          v26 = v118;
          LODWORD(v119) = v119 + 1;
          ++v45;
          v111 |= v118[(unsigned int)v119 - 1] != (_QWORD)v46;
        }
        while ( v103 != v45 );
        if ( v111 )
        {
          v20 = &v118;
          v52 = sub_DCDF90(*a1, (__int64)&v118, v50, v51, v48);
          v26 = v118;
          v2 = v52;
        }
        goto LABEL_60;
      case 0xB:
        v118 = v120;
        v119 = 0x200000000LL;
        v38 = *(_QWORD **)(v2 + 32);
        v102 = &v38[*(_QWORD *)(v2 + 40)];
        if ( v38 == v102 )
          goto LABEL_19;
        v110 = 0;
        v39 = *(_QWORD **)(v2 + 32);
        do
        {
          v40 = (_QWORD **)*v39;
          v20 = (_QWORD **)*v39;
          v41 = sub_DE1010(a1, *v39);
          v43 = (unsigned int)v119;
          if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
          {
            v20 = (_QWORD **)v120;
            v98 = v41;
            sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 8u, v22, v42);
            v43 = (unsigned int)v119;
            v41 = v98;
          }
          v25 = (__int64)v118;
          v118[v43] = v41;
          v26 = v118;
          LODWORD(v119) = v119 + 1;
          ++v39;
          v110 |= v118[(unsigned int)v119 - 1] != (_QWORD)v40;
        }
        while ( v102 != v39 );
        v27 = 0;
        if ( v110 )
          goto LABEL_31;
        goto LABEL_60;
      case 0xC:
        v118 = v120;
        v119 = 0x200000000LL;
        v29 = *(_QWORD **)(v2 + 32);
        v101 = &v29[*(_QWORD *)(v2 + 40)];
        if ( v29 == v101 )
          goto LABEL_19;
        v109 = 0;
        v30 = *(_QWORD **)(v2 + 32);
        do
        {
          v31 = (_QWORD **)*v30;
          v20 = (_QWORD **)*v30;
          v32 = sub_DE1010(a1, *v30);
          v35 = (unsigned int)v119;
          if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
          {
            v20 = (_QWORD **)v120;
            v95 = v32;
            sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 8u, v33, v34);
            v35 = (unsigned int)v119;
            v32 = v95;
          }
          v36 = (__int64)v118;
          v118[v35] = v32;
          v26 = v118;
          LODWORD(v119) = v119 + 1;
          ++v30;
          v109 |= v118[(unsigned int)v119 - 1] != (_QWORD)v31;
        }
        while ( v101 != v30 );
        if ( v109 )
        {
          v20 = &v118;
          v37 = sub_DCE150(*a1, (__int64)&v118, v35, v36, v33);
          v26 = v118;
          v2 = v37;
        }
        goto LABEL_60;
      case 0xD:
        v118 = v120;
        v119 = 0x200000000LL;
        v17 = *(_QWORD **)(v2 + 32);
        v100 = &v17[*(_QWORD *)(v2 + 40)];
        if ( v17 == v100 )
          goto LABEL_19;
        v108 = 0;
        v18 = *(_QWORD **)(v2 + 32);
        do
        {
          v19 = (_QWORD **)*v18;
          v20 = (_QWORD **)*v18;
          v21 = sub_DE1010(a1, *v18);
          v24 = (unsigned int)v119;
          if ( (unsigned __int64)(unsigned int)v119 + 1 > HIDWORD(v119) )
          {
            v20 = (_QWORD **)v120;
            v94 = v21;
            sub_C8D5F0((__int64)&v118, v120, (unsigned int)v119 + 1LL, 8u, v22, v23);
            v24 = (unsigned int)v119;
            v21 = v94;
          }
          v25 = (__int64)v118;
          v118[v24] = v21;
          v26 = v118;
          LODWORD(v119) = v119 + 1;
          ++v18;
          v108 |= v118[(unsigned int)v119 - 1] != (_QWORD)v19;
        }
        while ( v100 != v18 );
        if ( !v108 )
          goto LABEL_60;
        v27 = 1;
LABEL_31:
        v20 = &v118;
        v28 = sub_DCEE50(*a1, (__int64)&v118, v27, v25, v22);
        v26 = v118;
        v2 = (__int64)v28;
LABEL_60:
        if ( v26 != v120 )
          _libc_free(v26, v20);
LABEL_19:
        v117 = v2;
        sub_DB11F0((__int64)&v118, (__int64)(a1 + 1), &v116, &v117);
        v9 = (__int64 *)v120[0];
        break;
      case 0xE:
        v16 = sub_DE1010(a1, *(_QWORD *)(v2 + 32));
        if ( v16 != *(_QWORD *)(v2 + 32) )
          v2 = (__int64)sub_DD3A70((__int64)*a1, v16, *(_QWORD *)(v2 + 40));
        goto LABEL_19;
      case 0xF:
        v2 = (__int64)sub_DA3860(*a1, *(_QWORD *)(v2 - 8));
        goto LABEL_19;
      case 0x10:
        v2 = sub_D970F0((__int64)*a1);
        goto LABEL_19;
      default:
        BUG();
    }
  }
  return v9[1];
}
