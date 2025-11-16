// Function: sub_DD3C70
// Address: 0xdd3c70
//
__int64 __fastcall sub_DD3C70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  char v7; // di
  int v8; // edi
  __int64 v9; // rsi
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  int v17; // eax
  __int64 *v18; // r12
  __int64 v19; // rax
  __int64 v20; // rsi
  _QWORD *v21; // r8
  _QWORD *v22; // r15
  _QWORD **v23; // r14
  _QWORD **v24; // rsi
  __int64 v25; // rax
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rdx
  __int64 v29; // rcx
  _QWORD *v30; // rdi
  char v31; // dl
  _QWORD *v32; // rax
  _QWORD *v33; // r8
  _QWORD *v34; // r15
  _QWORD **v35; // r14
  __int64 v36; // rax
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rdx
  __int64 v40; // rcx
  unsigned __int64 v41; // rax
  _QWORD *v42; // r8
  _QWORD *v43; // r15
  _QWORD **v44; // r14
  __int64 v45; // rax
  __int64 v46; // r9
  __int64 v47; // rdx
  _QWORD *v48; // r8
  _QWORD *v49; // r15
  _QWORD **v50; // r14
  __int64 v51; // rax
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // rdx
  __int64 v55; // rcx
  unsigned __int64 v56; // rax
  _QWORD *v57; // r8
  _QWORD *v58; // r15
  _QWORD **v59; // r14
  __int64 v60; // rax
  __int64 v61; // r8
  __int64 v62; // r9
  __int64 v63; // rdx
  __int64 v64; // rcx
  unsigned __int64 v65; // rax
  _QWORD *v66; // r8
  _QWORD *v67; // r15
  _QWORD **v68; // r14
  __int64 v69; // rax
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // rdx
  _QWORD *v73; // rax
  __int64 v74; // r14
  __int64 v75; // rax
  _QWORD *v76; // r8
  _QWORD *v77; // r15
  _QWORD **v78; // r14
  __int64 v79; // rax
  __int64 v80; // r8
  __int64 v81; // r9
  __int64 v82; // rdx
  __int64 *v83; // rax
  __int64 v84; // rsi
  __int64 v85; // rsi
  __int64 v86; // rsi
  int v87; // r10d
  __int64 v88; // [rsp+8h] [rbp-98h]
  __int64 v89; // [rsp+8h] [rbp-98h]
  __int64 v90; // [rsp+8h] [rbp-98h]
  __int64 v91; // [rsp+8h] [rbp-98h]
  __int64 v92; // [rsp+8h] [rbp-98h]
  __int64 v93; // [rsp+8h] [rbp-98h]
  __int64 v94; // [rsp+8h] [rbp-98h]
  _QWORD *v95; // [rsp+10h] [rbp-90h]
  _QWORD *v96; // [rsp+10h] [rbp-90h]
  _QWORD *v97; // [rsp+10h] [rbp-90h]
  _QWORD *v98; // [rsp+10h] [rbp-90h]
  _QWORD *v99; // [rsp+10h] [rbp-90h]
  _QWORD *v100; // [rsp+10h] [rbp-90h]
  _QWORD *v101; // [rsp+10h] [rbp-90h]
  char v102; // [rsp+1Fh] [rbp-81h]
  char v103; // [rsp+1Fh] [rbp-81h]
  char v104; // [rsp+1Fh] [rbp-81h]
  char v105; // [rsp+1Fh] [rbp-81h]
  char v106; // [rsp+1Fh] [rbp-81h]
  char v107; // [rsp+1Fh] [rbp-81h]
  char v108; // [rsp+1Fh] [rbp-81h]
  __int64 v109; // [rsp+28h] [rbp-78h] BYREF
  __int64 v110; // [rsp+38h] [rbp-68h] BYREF
  _QWORD *v111; // [rsp+40h] [rbp-60h] BYREF
  __int64 v112; // [rsp+48h] [rbp-58h]
  _QWORD v113[10]; // [rsp+50h] [rbp-50h] BYREF

  v5 = a2;
  v7 = *(_BYTE *)(a1 + 16);
  v109 = a2;
  v8 = v7 & 1;
  if ( v8 )
  {
    v9 = a1 + 24;
    a5 = 3;
  }
  else
  {
    v15 = *(unsigned int *)(a1 + 32);
    v9 = *(_QWORD *)(a1 + 24);
    if ( !(_DWORD)v15 )
      goto LABEL_12;
    a5 = (unsigned int)(v15 - 1);
  }
  a4 = (unsigned int)a5 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v10 = (__int64 *)(v9 + 16 * a4);
  v11 = *v10;
  if ( v5 == *v10 )
    goto LABEL_4;
  v17 = 1;
  while ( v11 != -4096 )
  {
    v87 = v17 + 1;
    a4 = (unsigned int)a5 & (v17 + (_DWORD)a4);
    v10 = (__int64 *)(v9 + 16LL * (unsigned int)a4);
    v11 = *v10;
    if ( v5 == *v10 )
      goto LABEL_4;
    v17 = v87;
  }
  if ( (_BYTE)v8 )
  {
    v16 = 64;
    goto LABEL_13;
  }
  v15 = *(unsigned int *)(a1 + 32);
LABEL_12:
  v16 = 16 * v15;
LABEL_13:
  v10 = (__int64 *)(v9 + v16);
LABEL_4:
  v12 = 64;
  if ( !(_BYTE)v8 )
    v12 = 16LL * *(unsigned int *)(a1 + 32);
  v13 = v9 + v12;
  if ( v10 == (__int64 *)v13 )
  {
    switch ( *(_WORD *)(v5 + 24) )
    {
      case 0:
      case 1:
      case 0x10:
        goto LABEL_20;
      case 2:
        v85 = sub_DD3C70(a1, *(_QWORD *)(v5 + 32));
        if ( v85 != *(_QWORD *)(v5 + 32) )
          v5 = (__int64)sub_DC5200(*(_QWORD *)a1, v85, *(_QWORD *)(v5 + 40), 0);
        goto LABEL_20;
      case 3:
        v84 = sub_DD3C70(a1, *(_QWORD *)(v5 + 32));
        if ( v84 != *(_QWORD *)(v5 + 32) )
          v5 = (__int64)sub_DC2B70(*(_QWORD *)a1, v84, *(_QWORD *)(v5 + 40), 0);
        goto LABEL_20;
      case 4:
        v86 = sub_DD3C70(a1, *(_QWORD *)(v5 + 32));
        if ( v86 != *(_QWORD *)(v5 + 32) )
          v5 = (__int64)sub_DC5000(*(_QWORD *)a1, v86, *(_QWORD *)(v5 + 40), 0);
        goto LABEL_20;
      case 5:
        v111 = v113;
        v112 = 0x200000000LL;
        v66 = *(_QWORD **)(v5 + 32);
        v100 = &v66[*(_QWORD *)(v5 + 40)];
        if ( v66 == v100 )
          goto LABEL_20;
        v107 = 0;
        v67 = *(_QWORD **)(v5 + 32);
        do
        {
          v68 = (_QWORD **)*v67;
          v24 = (_QWORD **)*v67;
          v69 = sub_DD3C70(a1, *v67);
          v72 = (unsigned int)v112;
          if ( (unsigned __int64)(unsigned int)v112 + 1 > HIDWORD(v112) )
          {
            v24 = (_QWORD **)v113;
            v93 = v69;
            sub_C8D5F0((__int64)&v111, v113, (unsigned int)v112 + 1LL, 8u, v70, v71);
            v72 = (unsigned int)v112;
            v69 = v93;
          }
          v111[v72] = v69;
          v30 = v111;
          LODWORD(v112) = v112 + 1;
          ++v67;
          v107 |= v111[(unsigned int)v112 - 1] != (_QWORD)v68;
        }
        while ( v100 != v67 );
        if ( v107 )
        {
          v24 = &v111;
          v73 = sub_DC7EB0(*(__int64 **)a1, (__int64)&v111, 0, 0);
          v30 = v111;
          v5 = (__int64)v73;
        }
        goto LABEL_79;
      case 6:
        v111 = v113;
        v112 = 0x200000000LL;
        v76 = *(_QWORD **)(v5 + 32);
        v101 = &v76[*(_QWORD *)(v5 + 40)];
        if ( v76 == v101 )
          goto LABEL_20;
        v108 = 0;
        v77 = *(_QWORD **)(v5 + 32);
        do
        {
          v78 = (_QWORD **)*v77;
          v24 = (_QWORD **)*v77;
          v79 = sub_DD3C70(a1, *v77);
          v82 = (unsigned int)v112;
          if ( (unsigned __int64)(unsigned int)v112 + 1 > HIDWORD(v112) )
          {
            v24 = (_QWORD **)v113;
            v88 = v79;
            sub_C8D5F0((__int64)&v111, v113, (unsigned int)v112 + 1LL, 8u, v80, v81);
            v82 = (unsigned int)v112;
            v79 = v88;
          }
          v111[v82] = v79;
          v30 = v111;
          LODWORD(v112) = v112 + 1;
          ++v77;
          v108 |= v111[(unsigned int)v112 - 1] != (_QWORD)v78;
        }
        while ( v101 != v77 );
        if ( v108 )
        {
          v24 = &v111;
          v83 = sub_DC8BD0(*(__int64 **)a1, (__int64)&v111, 0, 0);
          v30 = v111;
          v5 = (__int64)v83;
        }
        goto LABEL_79;
      case 7:
        v74 = sub_DD3C70(a1, *(_QWORD *)(v5 + 32));
        v75 = sub_DD3C70(a1, *(_QWORD *)(v5 + 40));
        if ( v74 != *(_QWORD *)(v5 + 32) || v75 != *(_QWORD *)(v5 + 40) )
          v5 = sub_DCB270(*(_QWORD *)a1, v74, v75);
        goto LABEL_20;
      case 8:
        if ( *(_QWORD *)(v5 + 48) == *(_QWORD *)(a1 + 88) && *(_QWORD *)(v5 + 40) == 2 )
        {
          v18 = *(__int64 **)a1;
          v19 = sub_D33D80((_QWORD *)v5, *(_QWORD *)a1, v13, a4, a5);
          v5 = (__int64)sub_DCC810(v18, v5, v19, 0, 0);
        }
        else
        {
          *(_BYTE *)(a1 + 96) = 0;
        }
        goto LABEL_20;
      case 9:
        v111 = v113;
        v112 = 0x200000000LL;
        v57 = *(_QWORD **)(v5 + 32);
        v99 = &v57[*(_QWORD *)(v5 + 40)];
        if ( v57 == v99 )
          goto LABEL_20;
        v106 = 0;
        v58 = *(_QWORD **)(v5 + 32);
        do
        {
          v59 = (_QWORD **)*v58;
          v24 = (_QWORD **)*v58;
          v60 = sub_DD3C70(a1, *v58);
          v63 = (unsigned int)v112;
          if ( (unsigned __int64)(unsigned int)v112 + 1 > HIDWORD(v112) )
          {
            v24 = (_QWORD **)v113;
            v91 = v60;
            sub_C8D5F0((__int64)&v111, v113, (unsigned int)v112 + 1LL, 8u, v61, v62);
            v63 = (unsigned int)v112;
            v60 = v91;
          }
          v64 = (__int64)v111;
          v111[v63] = v60;
          v30 = v111;
          LODWORD(v112) = v112 + 1;
          ++v58;
          v106 |= v111[(unsigned int)v112 - 1] != (_QWORD)v59;
        }
        while ( v99 != v58 );
        if ( v106 )
        {
          v24 = &v111;
          v65 = sub_DCE040(*(__int64 **)a1, (__int64)&v111, v63, v64, v61);
          v30 = v111;
          v5 = v65;
        }
        goto LABEL_79;
      case 0xA:
        v111 = v113;
        v112 = 0x200000000LL;
        v48 = *(_QWORD **)(v5 + 32);
        v98 = &v48[*(_QWORD *)(v5 + 40)];
        if ( v48 == v98 )
          goto LABEL_20;
        v105 = 0;
        v49 = *(_QWORD **)(v5 + 32);
        do
        {
          v50 = (_QWORD **)*v49;
          v24 = (_QWORD **)*v49;
          v51 = sub_DD3C70(a1, *v49);
          v54 = (unsigned int)v112;
          if ( (unsigned __int64)(unsigned int)v112 + 1 > HIDWORD(v112) )
          {
            v24 = (_QWORD **)v113;
            v92 = v51;
            sub_C8D5F0((__int64)&v111, v113, (unsigned int)v112 + 1LL, 8u, v52, v53);
            v54 = (unsigned int)v112;
            v51 = v92;
          }
          v55 = (__int64)v111;
          v111[v54] = v51;
          v30 = v111;
          LODWORD(v112) = v112 + 1;
          ++v49;
          v105 |= v111[(unsigned int)v112 - 1] != (_QWORD)v50;
        }
        while ( v98 != v49 );
        if ( v105 )
        {
          v24 = &v111;
          v56 = sub_DCDF90(*(__int64 **)a1, (__int64)&v111, v54, v55, v52);
          v30 = v111;
          v5 = v56;
        }
        goto LABEL_79;
      case 0xB:
        v111 = v113;
        v112 = 0x200000000LL;
        v42 = *(_QWORD **)(v5 + 32);
        v97 = &v42[*(_QWORD *)(v5 + 40)];
        if ( v42 == v97 )
          goto LABEL_20;
        v104 = 0;
        v43 = *(_QWORD **)(v5 + 32);
        do
        {
          v44 = (_QWORD **)*v43;
          v24 = (_QWORD **)*v43;
          v45 = sub_DD3C70(a1, *v43);
          v47 = (unsigned int)v112;
          if ( (unsigned __int64)(unsigned int)v112 + 1 > HIDWORD(v112) )
          {
            v24 = (_QWORD **)v113;
            v90 = v45;
            sub_C8D5F0((__int64)&v111, v113, (unsigned int)v112 + 1LL, 8u, v26, v46);
            v47 = (unsigned int)v112;
            v45 = v90;
          }
          v29 = (__int64)v111;
          v111[v47] = v45;
          v30 = v111;
          LODWORD(v112) = v112 + 1;
          ++v43;
          v104 |= v111[(unsigned int)v112 - 1] != (_QWORD)v44;
        }
        while ( v97 != v43 );
        v31 = 0;
        if ( v104 )
          goto LABEL_32;
        goto LABEL_79;
      case 0xC:
        v111 = v113;
        v112 = 0x200000000LL;
        v33 = *(_QWORD **)(v5 + 32);
        v96 = &v33[*(_QWORD *)(v5 + 40)];
        if ( v33 == v96 )
          goto LABEL_20;
        v103 = 0;
        v34 = *(_QWORD **)(v5 + 32);
        do
        {
          v35 = (_QWORD **)*v34;
          v24 = (_QWORD **)*v34;
          v36 = sub_DD3C70(a1, *v34);
          v39 = (unsigned int)v112;
          if ( (unsigned __int64)(unsigned int)v112 + 1 > HIDWORD(v112) )
          {
            v24 = (_QWORD **)v113;
            v89 = v36;
            sub_C8D5F0((__int64)&v111, v113, (unsigned int)v112 + 1LL, 8u, v37, v38);
            v39 = (unsigned int)v112;
            v36 = v89;
          }
          v40 = (__int64)v111;
          v111[v39] = v36;
          v30 = v111;
          LODWORD(v112) = v112 + 1;
          ++v34;
          v103 |= v111[(unsigned int)v112 - 1] != (_QWORD)v35;
        }
        while ( v96 != v34 );
        if ( v103 )
        {
          v24 = &v111;
          v41 = sub_DCE150(*(__int64 **)a1, (__int64)&v111, v39, v40, v37);
          v30 = v111;
          v5 = v41;
        }
        goto LABEL_79;
      case 0xD:
        v111 = v113;
        v112 = 0x200000000LL;
        v21 = *(_QWORD **)(v5 + 32);
        v95 = &v21[*(_QWORD *)(v5 + 40)];
        if ( v21 == v95 )
          goto LABEL_20;
        v102 = 0;
        v22 = *(_QWORD **)(v5 + 32);
        do
        {
          v23 = (_QWORD **)*v22;
          v24 = (_QWORD **)*v22;
          v25 = sub_DD3C70(a1, *v22);
          v28 = (unsigned int)v112;
          if ( (unsigned __int64)(unsigned int)v112 + 1 > HIDWORD(v112) )
          {
            v24 = (_QWORD **)v113;
            v94 = v25;
            sub_C8D5F0((__int64)&v111, v113, (unsigned int)v112 + 1LL, 8u, v26, v27);
            v28 = (unsigned int)v112;
            v25 = v94;
          }
          v29 = (__int64)v111;
          v111[v28] = v25;
          v30 = v111;
          LODWORD(v112) = v112 + 1;
          ++v22;
          v102 |= v111[(unsigned int)v112 - 1] != (_QWORD)v23;
        }
        while ( v95 != v22 );
        if ( !v102 )
          goto LABEL_79;
        v31 = 1;
LABEL_32:
        v24 = &v111;
        v32 = sub_DCEE50(*(__int64 **)a1, (__int64)&v111, v31, v29, v26);
        v30 = v111;
        v5 = (__int64)v32;
LABEL_79:
        if ( v30 != v113 )
          _libc_free(v30, v24);
LABEL_20:
        v110 = v5;
        sub_DB11F0((__int64)&v111, a1 + 8, &v109, &v110);
        v10 = (__int64 *)v113[0];
        break;
      case 0xE:
        v20 = sub_DD3C70(a1, *(_QWORD *)(v5 + 32));
        if ( v20 != *(_QWORD *)(v5 + 32) )
          v5 = (__int64)sub_DD3A70(*(_QWORD *)a1, v20, *(_QWORD *)(v5 + 40));
        goto LABEL_20;
      case 0xF:
        if ( !sub_DADE90(*(_QWORD *)a1, v5, *(_QWORD *)(a1 + 88)) )
          *(_BYTE *)(a1 + 96) = 0;
        goto LABEL_20;
      default:
        BUG();
    }
  }
  return v10[1];
}
