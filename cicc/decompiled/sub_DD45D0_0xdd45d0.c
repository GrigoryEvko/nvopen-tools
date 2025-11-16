// Function: sub_DD45D0
// Address: 0xdd45d0
//
__int64 __fastcall sub_DD45D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  char v4; // di
  int v5; // edi
  __int64 v6; // rsi
  int v7; // r9d
  unsigned int v8; // ecx
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rsi
  _QWORD *v19; // r9
  _QWORD *v20; // r15
  _QWORD **v21; // r14
  _QWORD **v22; // rsi
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rdx
  __int64 v27; // rcx
  _QWORD *v28; // rdi
  char v29; // dl
  _QWORD *v30; // rax
  _QWORD *v31; // r9
  _QWORD *v32; // r15
  _QWORD **v33; // r14
  __int64 v34; // rax
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rdx
  __int64 v38; // rcx
  unsigned __int64 v39; // rax
  _QWORD *v40; // r9
  _QWORD *v41; // r15
  _QWORD **v42; // r14
  __int64 v43; // rax
  __int64 v44; // r9
  __int64 v45; // rdx
  _QWORD *v46; // r9
  _QWORD *v47; // r15
  _QWORD **v48; // r14
  __int64 v49; // rax
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // rdx
  __int64 v53; // rcx
  unsigned __int64 v54; // rax
  _QWORD *v55; // r9
  _QWORD *v56; // r15
  _QWORD **v57; // r14
  __int64 v58; // rax
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rdx
  __int64 v62; // rcx
  unsigned __int64 v63; // rax
  _QWORD *v64; // r9
  _QWORD *v65; // r15
  _QWORD **v66; // r14
  __int64 v67; // rax
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // rdx
  _QWORD *v71; // rax
  _QWORD *v72; // r9
  _QWORD *v73; // r15
  _QWORD **v74; // r14
  __int64 v75; // rax
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // rdx
  __int64 *v79; // rax
  __int64 v80; // rsi
  __int64 v81; // rsi
  __int64 v82; // rsi
  int v83; // r10d
  __int64 v84; // [rsp+8h] [rbp-98h]
  __int64 v85; // [rsp+8h] [rbp-98h]
  __int64 v86; // [rsp+8h] [rbp-98h]
  __int64 v87; // [rsp+8h] [rbp-98h]
  __int64 v88; // [rsp+8h] [rbp-98h]
  __int64 v89; // [rsp+8h] [rbp-98h]
  __int64 v90; // [rsp+8h] [rbp-98h]
  _QWORD *v91; // [rsp+10h] [rbp-90h]
  _QWORD *v92; // [rsp+10h] [rbp-90h]
  _QWORD *v93; // [rsp+10h] [rbp-90h]
  _QWORD *v94; // [rsp+10h] [rbp-90h]
  _QWORD *v95; // [rsp+10h] [rbp-90h]
  _QWORD *v96; // [rsp+10h] [rbp-90h]
  _QWORD *v97; // [rsp+10h] [rbp-90h]
  char v98; // [rsp+1Fh] [rbp-81h]
  char v99; // [rsp+1Fh] [rbp-81h]
  char v100; // [rsp+1Fh] [rbp-81h]
  char v101; // [rsp+1Fh] [rbp-81h]
  char v102; // [rsp+1Fh] [rbp-81h]
  char v103; // [rsp+1Fh] [rbp-81h]
  char v104; // [rsp+1Fh] [rbp-81h]
  __int64 v105; // [rsp+28h] [rbp-78h] BYREF
  __int64 v106; // [rsp+38h] [rbp-68h] BYREF
  _QWORD *v107; // [rsp+40h] [rbp-60h] BYREF
  __int64 v108; // [rsp+48h] [rbp-58h]
  _QWORD v109[10]; // [rsp+50h] [rbp-50h] BYREF

  v2 = a2;
  v4 = *(_BYTE *)(a1 + 16);
  v105 = a2;
  v5 = v4 & 1;
  if ( v5 )
  {
    v6 = a1 + 24;
    v7 = 3;
  }
  else
  {
    v13 = *(unsigned int *)(a1 + 32);
    v6 = *(_QWORD *)(a1 + 24);
    if ( !(_DWORD)v13 )
      goto LABEL_12;
    v7 = v13 - 1;
  }
  v8 = v7 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v9 = (__int64 *)(v6 + 16LL * v8);
  v10 = *v9;
  if ( v2 == *v9 )
    goto LABEL_4;
  v15 = 1;
  while ( v10 != -4096 )
  {
    v83 = v15 + 1;
    v8 = v7 & (v15 + v8);
    v9 = (__int64 *)(v6 + 16LL * v8);
    v10 = *v9;
    if ( v2 == *v9 )
      goto LABEL_4;
    v15 = v83;
  }
  if ( (_BYTE)v5 )
  {
    v14 = 64;
    goto LABEL_13;
  }
  v13 = *(unsigned int *)(a1 + 32);
LABEL_12:
  v14 = 16 * v13;
LABEL_13:
  v9 = (__int64 *)(v6 + v14);
LABEL_4:
  v11 = 64;
  if ( !(_BYTE)v5 )
    v11 = 16LL * *(unsigned int *)(a1 + 32);
  if ( v9 == (__int64 *)(v6 + v11) )
  {
    switch ( *(_WORD *)(v2 + 24) )
    {
      case 0:
      case 1:
      case 0x10:
        goto LABEL_20;
      case 2:
        v81 = sub_DD45D0(a1, *(_QWORD *)(v2 + 32));
        if ( v81 != *(_QWORD *)(v2 + 32) )
          v2 = (__int64)sub_DC5200(*(_QWORD *)a1, v81, *(_QWORD *)(v2 + 40), 0);
        goto LABEL_20;
      case 3:
        v80 = sub_DD45D0(a1, *(_QWORD *)(v2 + 32));
        if ( v80 != *(_QWORD *)(v2 + 32) )
          v2 = (__int64)sub_DC2B70(*(_QWORD *)a1, v80, *(_QWORD *)(v2 + 40), 0);
        goto LABEL_20;
      case 4:
        v82 = sub_DD45D0(a1, *(_QWORD *)(v2 + 32));
        if ( v82 != *(_QWORD *)(v2 + 32) )
          v2 = (__int64)sub_DC5000(*(_QWORD *)a1, v82, *(_QWORD *)(v2 + 40), 0);
        goto LABEL_20;
      case 5:
        v107 = v109;
        v108 = 0x200000000LL;
        v64 = *(_QWORD **)(v2 + 32);
        v96 = &v64[*(_QWORD *)(v2 + 40)];
        if ( v64 == v96 )
          goto LABEL_20;
        v103 = 0;
        v65 = *(_QWORD **)(v2 + 32);
        do
        {
          v66 = (_QWORD **)*v65;
          v22 = (_QWORD **)*v65;
          v67 = sub_DD45D0(a1, *v65);
          v70 = (unsigned int)v108;
          if ( (unsigned __int64)(unsigned int)v108 + 1 > HIDWORD(v108) )
          {
            v22 = (_QWORD **)v109;
            v89 = v67;
            sub_C8D5F0((__int64)&v107, v109, (unsigned int)v108 + 1LL, 8u, v68, v69);
            v70 = (unsigned int)v108;
            v67 = v89;
          }
          v107[v70] = v67;
          v28 = v107;
          LODWORD(v108) = v108 + 1;
          ++v65;
          v103 |= v107[(unsigned int)v108 - 1] != (_QWORD)v66;
        }
        while ( v96 != v65 );
        if ( v103 )
        {
          v22 = &v107;
          v71 = sub_DC7EB0(*(__int64 **)a1, (__int64)&v107, 0, 0);
          v28 = v107;
          v2 = (__int64)v71;
        }
        goto LABEL_77;
      case 6:
        v107 = v109;
        v108 = 0x200000000LL;
        v72 = *(_QWORD **)(v2 + 32);
        v97 = &v72[*(_QWORD *)(v2 + 40)];
        if ( v72 == v97 )
          goto LABEL_20;
        v104 = 0;
        v73 = *(_QWORD **)(v2 + 32);
        do
        {
          v74 = (_QWORD **)*v73;
          v22 = (_QWORD **)*v73;
          v75 = sub_DD45D0(a1, *v73);
          v78 = (unsigned int)v108;
          if ( (unsigned __int64)(unsigned int)v108 + 1 > HIDWORD(v108) )
          {
            v22 = (_QWORD **)v109;
            v84 = v75;
            sub_C8D5F0((__int64)&v107, v109, (unsigned int)v108 + 1LL, 8u, v76, v77);
            v78 = (unsigned int)v108;
            v75 = v84;
          }
          v107[v78] = v75;
          v28 = v107;
          LODWORD(v108) = v108 + 1;
          ++v73;
          v104 |= v107[(unsigned int)v108 - 1] != (_QWORD)v74;
        }
        while ( v97 != v73 );
        if ( v104 )
        {
          v22 = &v107;
          v79 = sub_DC8BD0(*(__int64 **)a1, (__int64)&v107, 0, 0);
          v28 = v107;
          v2 = (__int64)v79;
        }
        goto LABEL_77;
      case 7:
        v16 = sub_DD45D0(a1, *(_QWORD *)(v2 + 32));
        v17 = sub_DD45D0(a1, *(_QWORD *)(v2 + 40));
        if ( v16 != *(_QWORD *)(v2 + 32) || v17 != *(_QWORD *)(v2 + 40) )
          v2 = sub_DCB270(*(_QWORD *)a1, v16, v17);
        goto LABEL_20;
      case 8:
        if ( *(_QWORD *)(v2 + 48) == *(_QWORD *)(a1 + 88) )
          v2 = **(_QWORD **)(v2 + 32);
        else
          *(_BYTE *)(a1 + 97) = 1;
        goto LABEL_20;
      case 9:
        v107 = v109;
        v108 = 0x200000000LL;
        v55 = *(_QWORD **)(v2 + 32);
        v95 = &v55[*(_QWORD *)(v2 + 40)];
        if ( v55 == v95 )
          goto LABEL_20;
        v102 = 0;
        v56 = *(_QWORD **)(v2 + 32);
        do
        {
          v57 = (_QWORD **)*v56;
          v22 = (_QWORD **)*v56;
          v58 = sub_DD45D0(a1, *v56);
          v61 = (unsigned int)v108;
          if ( (unsigned __int64)(unsigned int)v108 + 1 > HIDWORD(v108) )
          {
            v22 = (_QWORD **)v109;
            v87 = v58;
            sub_C8D5F0((__int64)&v107, v109, (unsigned int)v108 + 1LL, 8u, v59, v60);
            v61 = (unsigned int)v108;
            v58 = v87;
          }
          v62 = (__int64)v107;
          v107[v61] = v58;
          v28 = v107;
          LODWORD(v108) = v108 + 1;
          ++v56;
          v102 |= v107[(unsigned int)v108 - 1] != (_QWORD)v57;
        }
        while ( v95 != v56 );
        if ( v102 )
        {
          v22 = &v107;
          v63 = sub_DCE040(*(__int64 **)a1, (__int64)&v107, v61, v62, v59);
          v28 = v107;
          v2 = v63;
        }
        goto LABEL_77;
      case 0xA:
        v107 = v109;
        v108 = 0x200000000LL;
        v46 = *(_QWORD **)(v2 + 32);
        v94 = &v46[*(_QWORD *)(v2 + 40)];
        if ( v46 == v94 )
          goto LABEL_20;
        v101 = 0;
        v47 = *(_QWORD **)(v2 + 32);
        do
        {
          v48 = (_QWORD **)*v47;
          v22 = (_QWORD **)*v47;
          v49 = sub_DD45D0(a1, *v47);
          v52 = (unsigned int)v108;
          if ( (unsigned __int64)(unsigned int)v108 + 1 > HIDWORD(v108) )
          {
            v22 = (_QWORD **)v109;
            v88 = v49;
            sub_C8D5F0((__int64)&v107, v109, (unsigned int)v108 + 1LL, 8u, v50, v51);
            v52 = (unsigned int)v108;
            v49 = v88;
          }
          v53 = (__int64)v107;
          v107[v52] = v49;
          v28 = v107;
          LODWORD(v108) = v108 + 1;
          ++v47;
          v101 |= v107[(unsigned int)v108 - 1] != (_QWORD)v48;
        }
        while ( v94 != v47 );
        if ( v101 )
        {
          v22 = &v107;
          v54 = sub_DCDF90(*(__int64 **)a1, (__int64)&v107, v52, v53, v50);
          v28 = v107;
          v2 = v54;
        }
        goto LABEL_77;
      case 0xB:
        v107 = v109;
        v108 = 0x200000000LL;
        v40 = *(_QWORD **)(v2 + 32);
        v93 = &v40[*(_QWORD *)(v2 + 40)];
        if ( v40 == v93 )
          goto LABEL_20;
        v100 = 0;
        v41 = *(_QWORD **)(v2 + 32);
        do
        {
          v42 = (_QWORD **)*v41;
          v22 = (_QWORD **)*v41;
          v43 = sub_DD45D0(a1, *v41);
          v45 = (unsigned int)v108;
          if ( (unsigned __int64)(unsigned int)v108 + 1 > HIDWORD(v108) )
          {
            v22 = (_QWORD **)v109;
            v86 = v43;
            sub_C8D5F0((__int64)&v107, v109, (unsigned int)v108 + 1LL, 8u, v24, v44);
            v45 = (unsigned int)v108;
            v43 = v86;
          }
          v27 = (__int64)v107;
          v107[v45] = v43;
          v28 = v107;
          LODWORD(v108) = v108 + 1;
          ++v41;
          v100 |= v107[(unsigned int)v108 - 1] != (_QWORD)v42;
        }
        while ( v93 != v41 );
        v29 = 0;
        if ( v100 )
          goto LABEL_32;
        goto LABEL_77;
      case 0xC:
        v107 = v109;
        v108 = 0x200000000LL;
        v31 = *(_QWORD **)(v2 + 32);
        v92 = &v31[*(_QWORD *)(v2 + 40)];
        if ( v31 == v92 )
          goto LABEL_20;
        v99 = 0;
        v32 = *(_QWORD **)(v2 + 32);
        do
        {
          v33 = (_QWORD **)*v32;
          v22 = (_QWORD **)*v32;
          v34 = sub_DD45D0(a1, *v32);
          v37 = (unsigned int)v108;
          if ( (unsigned __int64)(unsigned int)v108 + 1 > HIDWORD(v108) )
          {
            v22 = (_QWORD **)v109;
            v85 = v34;
            sub_C8D5F0((__int64)&v107, v109, (unsigned int)v108 + 1LL, 8u, v35, v36);
            v37 = (unsigned int)v108;
            v34 = v85;
          }
          v38 = (__int64)v107;
          v107[v37] = v34;
          v28 = v107;
          LODWORD(v108) = v108 + 1;
          ++v32;
          v99 |= v107[(unsigned int)v108 - 1] != (_QWORD)v33;
        }
        while ( v92 != v32 );
        if ( v99 )
        {
          v22 = &v107;
          v39 = sub_DCE150(*(__int64 **)a1, (__int64)&v107, v37, v38, v35);
          v28 = v107;
          v2 = v39;
        }
        goto LABEL_77;
      case 0xD:
        v107 = v109;
        v108 = 0x200000000LL;
        v19 = *(_QWORD **)(v2 + 32);
        v91 = &v19[*(_QWORD *)(v2 + 40)];
        if ( v19 == v91 )
          goto LABEL_20;
        v98 = 0;
        v20 = *(_QWORD **)(v2 + 32);
        do
        {
          v21 = (_QWORD **)*v20;
          v22 = (_QWORD **)*v20;
          v23 = sub_DD45D0(a1, *v20);
          v26 = (unsigned int)v108;
          if ( (unsigned __int64)(unsigned int)v108 + 1 > HIDWORD(v108) )
          {
            v22 = (_QWORD **)v109;
            v90 = v23;
            sub_C8D5F0((__int64)&v107, v109, (unsigned int)v108 + 1LL, 8u, v24, v25);
            v26 = (unsigned int)v108;
            v23 = v90;
          }
          v27 = (__int64)v107;
          v107[v26] = v23;
          v28 = v107;
          LODWORD(v108) = v108 + 1;
          ++v20;
          v98 |= v107[(unsigned int)v108 - 1] != (_QWORD)v21;
        }
        while ( v91 != v20 );
        if ( !v98 )
          goto LABEL_77;
        v29 = 1;
LABEL_32:
        v22 = &v107;
        v30 = sub_DCEE50(*(__int64 **)a1, (__int64)&v107, v29, v27, v24);
        v28 = v107;
        v2 = (__int64)v30;
LABEL_77:
        if ( v28 != v109 )
          _libc_free(v28, v22);
LABEL_20:
        v106 = v2;
        sub_DB11F0((__int64)&v107, a1 + 8, &v105, &v106);
        v9 = (__int64 *)v109[0];
        break;
      case 0xE:
        v18 = sub_DD45D0(a1, *(_QWORD *)(v2 + 32));
        if ( v18 != *(_QWORD *)(v2 + 32) )
          v2 = (__int64)sub_DD3A70(*(_QWORD *)a1, v18, *(_QWORD *)(v2 + 40));
        goto LABEL_20;
      case 0xF:
        if ( !sub_DADE90(*(_QWORD *)a1, v2, *(_QWORD *)(a1 + 88)) )
          *(_BYTE *)(a1 + 96) = 1;
        goto LABEL_20;
      default:
        BUG();
    }
  }
  return v9[1];
}
