// Function: sub_2F649E0
// Address: 0x2f649e0
//
__int64 __fastcall sub_2F649E0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rcx
  __int64 v5; // r12
  unsigned __int64 v6; // rax
  int v7; // esi
  __int64 v8; // rdi
  __int64 i; // rsi
  __int16 v10; // dx
  __int64 v11; // rsi
  unsigned int v12; // edi
  unsigned int v13; // ecx
  __int64 *v14; // rdx
  __int64 v15; // r9
  int v16; // r8d
  __int64 v17; // r14
  unsigned __int64 v18; // rdx
  unsigned int v19; // eax
  __int64 v20; // r15
  unsigned int v21; // eax
  __int64 v22; // rcx
  __int64 *v23; // r15
  __int64 v24; // rax
  unsigned __int64 v25; // r15
  __int64 v26; // rax
  __int64 *v27; // rcx
  __int64 *v28; // rdx
  __int64 *v29; // rax
  __int64 v30; // r8
  __int64 v31; // r12
  int v32; // r9d
  __int64 v33; // r8
  unsigned int v34; // eax
  unsigned __int64 v35; // rdx
  __int64 v36; // r10
  __int64 v37; // r12
  __int64 *v38; // rdx
  __int64 *v39; // rax
  __int64 v40; // rax
  __int64 *v41; // rdx
  __int64 v42; // r14
  __int64 v43; // rsi
  __int64 *v44; // rcx
  __int64 v45; // r14
  __int64 *v46; // rax
  __int64 v47; // r13
  unsigned __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // r15
  _BYTE *v54; // rdx
  _BYTE *v55; // r15
  _BYTE *v56; // r14
  _BYTE *v57; // r13
  __int64 *v59; // rax
  __int64 *v60; // rax
  unsigned int v61; // eax
  __int64 v62; // rcx
  __int64 *v63; // r13
  __int64 v64; // rax
  __int64 *v65; // rax
  __int64 v66; // r8
  unsigned __int64 v67; // rsi
  __int64 v68; // r9
  unsigned __int64 k; // rax
  __int64 m; // rdi
  __int16 v71; // cx
  __int64 v72; // rdi
  __int64 v73; // r8
  unsigned int v74; // esi
  __int64 *v75; // rcx
  __int64 v76; // r10
  __int64 v77; // r13
  __int64 *v78; // rax
  __int64 *v79; // rax
  __int64 v80; // r14
  __int64 *v81; // rax
  int v82; // ecx
  int v83; // edx
  __int64 *v84; // rax
  unsigned int v85; // r12d
  __int64 v86; // r14
  _BYTE *v87; // rax
  __int64 v88; // r9
  unsigned __int64 v89; // r10
  __int64 *v90; // rax
  __int64 *v91; // rsi
  __int64 v92; // r12
  unsigned __int64 v93; // r11
  _QWORD *v94; // rax
  _QWORD *v95; // rsi
  int v96; // r10d
  int v97; // r11d
  __int64 v98; // [rsp+8h] [rbp-68h]
  unsigned __int64 v99; // [rsp+10h] [rbp-60h]
  unsigned __int64 v100; // [rsp+10h] [rbp-60h]
  __int64 v101; // [rsp+18h] [rbp-58h]
  __int64 v102; // [rsp+18h] [rbp-58h]
  __int64 v103; // [rsp+18h] [rbp-58h]
  __int64 v104; // [rsp+18h] [rbp-58h]
  int v105; // [rsp+18h] [rbp-58h]
  __int64 *v107; // [rsp+28h] [rbp-48h]
  __int64 v108; // [rsp+28h] [rbp-48h]
  __int64 j; // [rsp+28h] [rbp-48h]
  int v110; // [rsp+28h] [rbp-48h]
  __int64 *v111; // [rsp+28h] [rbp-48h]
  __int64 *v112; // [rsp+28h] [rbp-48h]
  _QWORD *v113; // [rsp+28h] [rbp-48h]
  __int64 v114; // [rsp+28h] [rbp-48h]
  int v115; // [rsp+28h] [rbp-48h]
  __int64 v116; // [rsp+28h] [rbp-48h]
  unsigned int v117; // [rsp+30h] [rbp-40h] BYREF
  int v118; // [rsp+34h] [rbp-3Ch] BYREF
  unsigned int v119; // [rsp+38h] [rbp-38h] BYREF
  unsigned int v120[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v3 = a1[3];
  v117 = 0;
  v118 = 0;
  v119 = 0;
  v120[0] = 0;
  if ( !(unsigned __int8)sub_2F61710(v3, a2, &v117, &v118, (int *)&v119, (int *)v120) )
    return 0;
  v4 = a2;
  v5 = a1[5];
  v6 = a2;
  v7 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  v8 = *(_QWORD *)(v5 + 32);
  if ( (*(_DWORD *)(a2 + 44) & 4) != 0 )
  {
    do
      v6 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v6 + 44) & 4) != 0 );
  }
  if ( (v7 & 8) != 0 )
  {
    do
      v4 = *(_QWORD *)(v4 + 8);
    while ( (*(_BYTE *)(v4 + 44) & 8) != 0 );
  }
  for ( i = *(_QWORD *)(v4 + 8); i != v6; v6 = *(_QWORD *)(v6 + 8) )
  {
    v10 = *(_WORD *)(v6 + 68);
    if ( (unsigned __int16)(v10 - 14) > 4u && v10 != 24 )
      break;
  }
  v11 = *(_QWORD *)(v8 + 128);
  v12 = *(_DWORD *)(v8 + 144);
  if ( !v12 )
    goto LABEL_103;
  v13 = (v12 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v14 = (__int64 *)(v11 + 16LL * v13);
  v15 = *v14;
  if ( v6 != *v14 )
  {
    v83 = 1;
    while ( v15 != -4096 )
    {
      v96 = v83 + 1;
      v13 = (v12 - 1) & (v83 + v13);
      v14 = (__int64 *)(v11 + 16LL * v13);
      v15 = *v14;
      if ( *v14 == v6 )
        goto LABEL_12;
      v83 = v96;
    }
LABEL_103:
    v14 = (__int64 *)(v11 + 16LL * v12);
  }
LABEL_12:
  v16 = v117;
  v17 = v14[1];
  v18 = *(unsigned int *)(v5 + 160);
  v19 = v117 & 0x7FFFFFFF;
  v20 = 8LL * (v117 & 0x7FFFFFFF);
  if ( (v117 & 0x7FFFFFFF) >= (unsigned int)v18 || (v27 = *(__int64 **)(*(_QWORD *)(v5 + 152) + 8LL * v19)) == 0 )
  {
    v21 = v19 + 1;
    if ( (unsigned int)v18 < v21 && v21 != v18 )
    {
      if ( v21 >= v18 )
      {
        v88 = *(_QWORD *)(v5 + 168);
        v89 = v21 - v18;
        if ( v21 > (unsigned __int64)*(unsigned int *)(v5 + 164) )
        {
          v99 = v21 - v18;
          v104 = *(_QWORD *)(v5 + 168);
          v115 = v117;
          sub_C8D5F0(v5 + 152, (const void *)(v5 + 168), v21, 8u, v117, v88);
          v89 = v99;
          v88 = v104;
          v18 = *(unsigned int *)(v5 + 160);
          v16 = v115;
        }
        v22 = *(_QWORD *)(v5 + 152);
        v90 = (__int64 *)(v22 + 8 * v18);
        v91 = &v90[v89];
        if ( v90 != v91 )
        {
          do
            *v90++ = v88;
          while ( v91 != v90 );
          LODWORD(v18) = *(_DWORD *)(v5 + 160);
          v22 = *(_QWORD *)(v5 + 152);
        }
        *(_DWORD *)(v5 + 160) = v89 + v18;
LABEL_15:
        v23 = (__int64 *)(v22 + v20);
        v24 = sub_2E10F30(v16);
        *v23 = v24;
        v107 = (__int64 *)v24;
        v25 = v17 & 0xFFFFFFFFFFFFFFF8LL;
        sub_2E11E80((_QWORD *)v5, v24);
        v26 = v119;
        v27 = v107;
        if ( !v119 )
          goto LABEL_64;
        goto LABEL_16;
      }
      *(_DWORD *)(v5 + 160) = v21;
    }
    v22 = *(_QWORD *)(v5 + 152);
    goto LABEL_15;
  }
  v26 = v119;
  v25 = v17 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v119 )
    goto LABEL_64;
LABEL_16:
  v28 = (__int64 *)v27[13];
  if ( v28 )
  {
    v29 = (__int64 *)(*(_QWORD *)(a1[3] + 272LL) + 16 * v26);
    v30 = *v29;
    v31 = v29[1];
    while ( 1 )
    {
      if ( v30 & v28[14] | v31 & v28[15] )
      {
        v102 = v30;
        v111 = v28;
        v59 = (__int64 *)sub_2E09D00(v28, v17);
        v28 = v111;
        v30 = v102;
        if ( v59 != (__int64 *)(*v111 + 24LL * *((unsigned int *)v111 + 2))
          && (*(_DWORD *)((*v59 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v59 >> 1) & 3) <= ((unsigned int)(v17 >> 1)
                                                                                                 & 3
                                                                                                 | *(_DWORD *)(v25 + 24)) )
        {
          return 0;
        }
      }
      v28 = (__int64 *)v28[13];
      if ( !v28 )
        goto LABEL_20;
    }
  }
LABEL_64:
  v112 = v27;
  v60 = (__int64 *)sub_2E09D00(v27, v17);
  if ( v60 != (__int64 *)(*v112 + 24LL * *((unsigned int *)v112 + 2))
    && (*(_DWORD *)((*v60 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v60 >> 1) & 3) <= (*(_DWORD *)(v25 + 24)
                                                                                           | (unsigned int)(v17 >> 1)
                                                                                           & 3) )
  {
    return 0;
  }
LABEL_20:
  v32 = v118;
  v33 = a1[5];
  v34 = v118 & 0x7FFFFFFF;
  v35 = *(unsigned int *)(v33 + 160);
  v36 = 8LL * (v118 & 0x7FFFFFFF);
  if ( (v118 & 0x7FFFFFFFu) >= (unsigned int)v35 || (v37 = *(_QWORD *)(*(_QWORD *)(v33 + 152) + 8LL * v34)) == 0 )
  {
    v61 = v34 + 1;
    if ( (unsigned int)v35 < v61 && v61 != v35 )
    {
      if ( v61 >= v35 )
      {
        v92 = *(_QWORD *)(v33 + 168);
        v93 = v61 - v35;
        if ( v61 > (unsigned __int64)*(unsigned int *)(v33 + 164) )
        {
          v98 = 8LL * (v118 & 0x7FFFFFFF);
          v100 = v61 - v35;
          v105 = v118;
          v116 = a1[5];
          sub_C8D5F0(v33 + 152, (const void *)(v33 + 168), v61, 8u, v33, (unsigned int)v118);
          v33 = v116;
          v36 = v98;
          v93 = v100;
          v32 = v105;
          v35 = *(unsigned int *)(v116 + 160);
        }
        v62 = *(_QWORD *)(v33 + 152);
        v94 = (_QWORD *)(v62 + 8 * v35);
        v95 = &v94[v93];
        if ( v94 != v95 )
        {
          do
            *v94++ = v92;
          while ( v95 != v94 );
          LODWORD(v35) = *(_DWORD *)(v33 + 160);
          v62 = *(_QWORD *)(v33 + 152);
        }
        *(_DWORD *)(v33 + 160) = v93 + v35;
        goto LABEL_69;
      }
      *(_DWORD *)(v33 + 160) = v61;
    }
    v62 = *(_QWORD *)(v33 + 152);
LABEL_69:
    v113 = (_QWORD *)v33;
    v63 = (__int64 *)(v62 + v36);
    v64 = sub_2E10F30(v32);
    *v63 = v64;
    v37 = v64;
    sub_2E11E80(v113, v64);
  }
  v101 = v25 | 4;
  v38 = (__int64 *)sub_2E09D00((__int64 *)v37, v25 | 4);
  if ( v38 == (__int64 *)(*(_QWORD *)v37 + 24LL * *(unsigned int *)(v37 + 8))
    || (*(_DWORD *)((*v38 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v38 >> 1) & 3) > (*(_DWORD *)(v25 + 24) | 2u) )
  {
    BUG();
  }
  v108 = v38[1];
  v39 = (__int64 *)sub_2E09D00((__int64 *)v37, v108);
  if ( v39 == (__int64 *)(*(_QWORD *)v37 + 24LL * *(unsigned int *)(v37 + 8))
    || (*(_DWORD *)((*v39 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v39 >> 1) & 3) > (*(_DWORD *)((v108 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                          | (unsigned int)(v108 >> 1)
                                                                                          & 3)
    || (v40 = v39[2]) == 0 )
  {
    v84 = (__int64 *)sub_2E09D00((__int64 *)v37, v17);
    if ( v84 != (__int64 *)(*(_QWORD *)v37 + 24LL * *(unsigned int *)(v37 + 8))
      && (*(_DWORD *)((*v84 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v84 >> 1) & 3) <= (*(_DWORD *)(v25 + 24)
                                                                                             | (unsigned int)(v17 >> 1)
                                                                                             & 3) )
    {
LABEL_28:
      v41 = (__int64 *)sub_2E09D00((__int64 *)v37, v17);
      if ( v41 == (__int64 *)(*(_QWORD *)v37 + 24LL * *(unsigned int *)(v37 + 8))
        || (*(_DWORD *)((*v41 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v41 >> 1) & 3) > (*(_DWORD *)(v25 + 24)
                                                                                              | (unsigned int)(v17 >> 1)
                                                                                              & 3)
        || (v42 = v41[2]) == 0 )
      {
        sub_2E14FC0(a1[5], v37, v101);
        v51 = a1[2];
        v52 = (unsigned int)v118;
        if ( v118 >= 0 )
        {
LABEL_38:
          v53 = *(_QWORD *)(*(_QWORD *)(v51 + 304) + 8 * v52);
          goto LABEL_39;
        }
      }
      else
      {
        v43 = 0;
        v44 = (__int64 *)sub_2E09D00((__int64 *)v37, v101);
        if ( v44 != (__int64 *)(*(_QWORD *)v37 + 24LL * *(unsigned int *)(v37 + 8))
          && (*(_DWORD *)((*v44 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v44 >> 1) & 3) <= (*(_DWORD *)(v25 + 24)
                                                                                                 | 2u) )
        {
          v43 = v44[2];
        }
        sub_2E0AAF0(v37, v43, v42);
        v45 = *(_QWORD *)(v37 + 104);
        v46 = (__int64 *)(*(_QWORD *)(a1[3] + 272LL) + 16LL * v120[0]);
        v47 = *v46;
        for ( j = v46[1]; v45; v45 = *(_QWORD *)(v45 + 104) )
        {
          if ( v47 & *(_QWORD *)(v45 + 112) | *(_QWORD *)(v45 + 120) & j )
          {
            v65 = (__int64 *)sub_2E09D00((__int64 *)v45, v101);
            v66 = 0;
            if ( v65 != (__int64 *)(*(_QWORD *)v45 + 24LL * *(unsigned int *)(v45 + 8))
              && (*(_DWORD *)((*v65 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v65 >> 1) & 3) <= (*(_DWORD *)(v25 + 24) | 2u) )
            {
              v66 = v65[2];
            }
            sub_2E0A600(v45, v66);
          }
        }
        sub_2E0AF60(v37);
        v51 = a1[2];
        v52 = (unsigned int)v118;
        if ( v118 >= 0 )
          goto LABEL_38;
      }
      v53 = *(_QWORD *)(*(_QWORD *)(v51 + 56) + 16 * (v52 & 0x7FFFFFFF) + 8);
LABEL_39:
      if ( !v53 )
      {
LABEL_45:
        v54 = *(_BYTE **)(a2 + 32);
        v55 = &v54[40 * (*(_DWORD *)(a2 + 40) & 0xFFFFFF)];
        if ( v54 != v55 )
        {
          while ( 1 )
          {
            v56 = v54;
            if ( sub_2DADC00(v54) )
              break;
            v54 = v56 + 40;
            if ( v55 == v56 + 40 )
              goto LABEL_58;
          }
          if ( v56 != v55 )
          {
            v110 = v118;
            do
            {
              if ( v110 == *((_DWORD *)v56 + 2) )
                v56[4] |= 1u;
              if ( v56 + 40 == v55 )
                break;
              v57 = v56 + 40;
              while ( 1 )
              {
                v56 = v57;
                if ( sub_2DADC00(v57) )
                  break;
                v57 += 40;
                if ( v55 == v57 )
                  goto LABEL_58;
              }
            }
            while ( v57 != v55 );
          }
        }
LABEL_58:
        sub_2E168A0((_QWORD *)a1[5], v37, 0, v48, v49, v50);
        return a2;
      }
      while ( (*(_BYTE *)(v53 + 4) & 8) != 0 )
      {
        v53 = *(_QWORD *)(v53 + 32);
        if ( !v53 )
          goto LABEL_45;
      }
LABEL_41:
      if ( (*(_BYTE *)(v53 + 3) & 0x10) != 0 )
        goto LABEL_44;
      v67 = *(_QWORD *)(v53 + 16);
      v68 = *(_QWORD *)(a1[5] + 32LL);
      for ( k = v67; (*(_BYTE *)(k + 44) & 4) != 0; k = *(_QWORD *)k & 0xFFFFFFFFFFFFFFF8LL )
        ;
      for ( ; (*(_BYTE *)(v67 + 44) & 8) != 0; v67 = *(_QWORD *)(v67 + 8) )
        ;
      for ( m = *(_QWORD *)(v67 + 8); m != k; k = *(_QWORD *)(k + 8) )
      {
        v71 = *(_WORD *)(k + 68);
        if ( (unsigned __int16)(v71 - 14) > 4u && v71 != 24 )
          break;
      }
      v72 = *(unsigned int *)(v68 + 144);
      v73 = *(_QWORD *)(v68 + 128);
      if ( (_DWORD)v72 )
      {
        v74 = (v72 - 1) & (((unsigned int)k >> 9) ^ ((unsigned int)k >> 4));
        v75 = (__int64 *)(v73 + 16LL * v74);
        v76 = *v75;
        if ( k == *v75 )
          goto LABEL_87;
        v82 = 1;
        while ( v76 != -4096 )
        {
          v97 = v82 + 1;
          v74 = (v72 - 1) & (v82 + v74);
          v75 = (__int64 *)(v73 + 16LL * v74);
          v76 = *v75;
          if ( *v75 == k )
            goto LABEL_87;
          v82 = v97;
        }
      }
      v75 = (__int64 *)(v73 + 16 * v72);
LABEL_87:
      v77 = v75[1];
      v48 = a1[3];
      v78 = (__int64 *)(*(_QWORD *)(v48 + 272) + 16LL * ((*(_DWORD *)v53 >> 8) & 0xFFF));
      v49 = v78[1];
      v50 = *v78;
      if ( (*v78 & v49) == 0xFFFFFFFFFFFFFFFFLL || (v80 = *(_QWORD *)(v37 + 104)) == 0 )
      {
        v79 = (__int64 *)sub_2E09D00((__int64 *)v37, v77);
        v48 = *(_QWORD *)v37 + 24LL * *(unsigned int *)(v37 + 8);
        if ( v79 != (__int64 *)v48 )
        {
          v48 = v77 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_DWORD *)((*v79 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v79 >> 1) & 3) <= (*(_DWORD *)((v77 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v77 >> 1) & 3) )
            goto LABEL_44;
        }
LABEL_90:
        *(_BYTE *)(v53 + 4) |= 1u;
        goto LABEL_44;
      }
      while ( 1 )
      {
        if ( v49 & *(_QWORD *)(v80 + 120) | v50 & *(_QWORD *)(v80 + 112) )
        {
          v103 = v50;
          v114 = v49;
          v81 = (__int64 *)sub_2E09D00((__int64 *)v80, v77);
          v49 = v114;
          v50 = v103;
          if ( v81 != (__int64 *)(*(_QWORD *)v80 + 24LL * *(unsigned int *)(v80 + 8))
            && (*(_DWORD *)((*v81 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v81 >> 1) & 3) <= (*(_DWORD *)((v77 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v77 >> 1) & 3) )
          {
            break;
          }
        }
        v80 = *(_QWORD *)(v80 + 104);
        if ( !v80 )
          goto LABEL_90;
      }
LABEL_44:
      while ( 1 )
      {
        v53 = *(_QWORD *)(v53 + 32);
        if ( !v53 )
          goto LABEL_45;
        if ( (*(_BYTE *)(v53 + 4) & 8) == 0 )
          goto LABEL_41;
      }
    }
  }
  else if ( (*(_BYTE *)(v40 + 8) & 6) != 0 )
  {
    goto LABEL_28;
  }
  if ( (*(_DWORD *)(a2 + 40) & 0xFFFFFF) != 0 )
  {
    v85 = (*(_DWORD *)(a2 + 40) & 0xFFFFFF) - 1;
    v86 = 40LL * v85;
    while ( 1 )
    {
      v87 = (_BYTE *)(v86 + *(_QWORD *)(a2 + 32));
      if ( *v87 || (v87[3] & 0x10) == 0 )
      {
        v86 -= 40;
        sub_2E8A650(a2, v85);
        if ( !v85 )
          break;
      }
      else
      {
        v86 -= 40;
        if ( !v85 )
          break;
      }
      --v85;
    }
  }
  sub_2E88D70(a2, (unsigned __int16 *)(*(_QWORD *)(a1[4] + 8LL) - 400LL));
  return a2;
}
