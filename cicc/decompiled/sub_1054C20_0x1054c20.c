// Function: sub_1054C20
// Address: 0x1054c20
//
unsigned __int64 __fastcall sub_1054C20(__int64 a1, unsigned __int64 a2)
{
  char v4; // cl
  __int64 v5; // rdi
  int v6; // r8d
  unsigned int v7; // esi
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  unsigned __int64 result; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  int v14; // r8d
  unsigned int v15; // esi
  unsigned __int64 *v16; // rdx
  unsigned __int64 v17; // r9
  __int64 v18; // rax
  unsigned int v19; // esi
  int v20; // eax
  _QWORD *v21; // r15
  char v22; // r12
  _BYTE *v23; // r13
  _BYTE *v24; // rsi
  __int64 v25; // rax
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rdx
  __int64 v29; // rcx
  _BYTE *v30; // rdi
  _QWORD *v31; // r15
  char v32; // r12
  _BYTE *v33; // r13
  __int64 v34; // rax
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rdx
  __int64 v38; // rcx
  _QWORD *v39; // r15
  char v40; // r12
  _BYTE *v41; // r13
  __int64 v42; // rax
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rdx
  __int64 v46; // rcx
  _QWORD *v47; // r15
  char v48; // r12
  _BYTE *v49; // r13
  __int64 v50; // rax
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // rsi
  __int64 v56; // rsi
  __int64 v57; // r12
  __int64 v58; // rax
  _QWORD *v59; // r15
  char v60; // r12
  _BYTE *v61; // r13
  __int64 v62; // rax
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rdx
  __int64 v66; // rsi
  _QWORD *v67; // r15
  char v68; // r12
  _BYTE *v69; // r13
  __int64 v70; // rax
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // rdx
  __int64 v74; // rcx
  _QWORD *v75; // r13
  _QWORD *i; // r12
  __int64 v77; // r8
  __int64 v78; // r9
  __int64 v79; // r15
  __int64 v80; // rax
  unsigned __int64 v81; // rdx
  int v82; // r13d
  __int64 v83; // r12
  _QWORD *v84; // rax
  _QWORD *v85; // r15
  char v86; // r12
  _BYTE *v87; // r13
  __int64 v88; // rax
  __int64 v89; // r8
  __int64 v90; // r9
  __int64 v91; // rdx
  __int64 v92; // rsi
  unsigned int v93; // edx
  unsigned __int64 *v94; // r11
  int v95; // edi
  unsigned int v96; // ecx
  int v97; // r12d
  int v98; // r11d
  __int64 v99; // rdi
  int v100; // ecx
  unsigned int v101; // edx
  unsigned __int64 v102; // rsi
  __int64 v103; // rdi
  int v104; // ecx
  unsigned int v105; // edx
  unsigned __int64 v106; // rsi
  int v107; // r9d
  unsigned __int64 *v108; // r8
  int v109; // ecx
  int v110; // ecx
  __int64 v111; // r15
  __int64 *v112; // rdi
  __int64 v113; // r13
  __int64 v114; // rdx
  __int64 v115; // rax
  _QWORD *v116; // rax
  int v117; // r9d
  _QWORD *v118; // [rsp+8h] [rbp-C8h]
  __int64 v119; // [rsp+10h] [rbp-C0h]
  __int64 v120; // [rsp+10h] [rbp-C0h]
  __int64 v121; // [rsp+10h] [rbp-C0h]
  __int64 v122; // [rsp+10h] [rbp-C0h]
  __int64 v123; // [rsp+10h] [rbp-C0h]
  __int64 v124; // [rsp+10h] [rbp-C0h]
  __int64 v125; // [rsp+10h] [rbp-C0h]
  __int64 v126; // [rsp+18h] [rbp-B8h]
  _QWORD *v127; // [rsp+20h] [rbp-B0h]
  _QWORD *v128; // [rsp+20h] [rbp-B0h]
  _QWORD *v129; // [rsp+20h] [rbp-B0h]
  _QWORD *v130; // [rsp+20h] [rbp-B0h]
  _QWORD *v131; // [rsp+20h] [rbp-B0h]
  _QWORD *v132; // [rsp+20h] [rbp-B0h]
  _QWORD *v133; // [rsp+20h] [rbp-B0h]
  unsigned __int64 v134; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v135; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v136; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v137; // [rsp+28h] [rbp-A8h]
  _QWORD v138[2]; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD v139[2]; // [rsp+40h] [rbp-90h] BYREF
  _BYTE *v140; // [rsp+50h] [rbp-80h] BYREF
  __int64 v141; // [rsp+58h] [rbp-78h]
  _BYTE v142[112]; // [rsp+60h] [rbp-70h] BYREF

  v4 = *(_BYTE *)(a1 + 16) & 1;
  if ( v4 )
  {
    v5 = a1 + 24;
    v6 = 3;
  }
  else
  {
    v12 = *(unsigned int *)(a1 + 32);
    v5 = *(_QWORD *)(a1 + 24);
    if ( !(_DWORD)v12 )
      goto LABEL_17;
    v6 = v12 - 1;
  }
  v7 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v5 + 16LL * (v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
  v9 = *v8;
  if ( *v8 == a2 )
    goto LABEL_4;
  v20 = 1;
  while ( v9 != -4096 )
  {
    v98 = v20 + 1;
    v7 = v6 & (v20 + v7);
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( *v8 == a2 )
      goto LABEL_4;
    v20 = v98;
  }
  if ( v4 )
  {
    v18 = 64;
    goto LABEL_18;
  }
  v12 = *(unsigned int *)(a1 + 32);
LABEL_17:
  v18 = 16 * v12;
LABEL_18:
  v8 = (__int64 *)(v5 + v18);
LABEL_4:
  v10 = 64;
  if ( !v4 )
    v10 = 16LL * *(unsigned int *)(a1 + 32);
  if ( v8 != (__int64 *)(v5 + v10) )
    return v8[1];
  switch ( *(_WORD *)(a2 + 24) )
  {
    case 0:
    case 1:
    case 0xF:
    case 0x10:
      goto LABEL_11;
    case 2:
      v56 = sub_1054C20(a1, *(_QWORD *)(a2 + 32));
      if ( v56 == *(_QWORD *)(a2 + 32) )
        goto LABEL_106;
      result = (unsigned __int64)sub_DC5200(*(_QWORD *)a1, v56, *(_QWORD *)(a2 + 40), 0);
      v4 = *(_BYTE *)(a1 + 16) & 1;
      goto LABEL_12;
    case 3:
      v55 = sub_1054C20(a1, *(_QWORD *)(a2 + 32));
      if ( v55 == *(_QWORD *)(a2 + 32) )
        goto LABEL_106;
      result = (unsigned __int64)sub_DC2B70(*(_QWORD *)a1, v55, *(_QWORD *)(a2 + 40), 0);
      v4 = *(_BYTE *)(a1 + 16) & 1;
      goto LABEL_12;
    case 4:
      v92 = sub_1054C20(a1, *(_QWORD *)(a2 + 32));
      if ( v92 == *(_QWORD *)(a2 + 32) )
        goto LABEL_106;
      result = (unsigned __int64)sub_DC5000(*(_QWORD *)a1, v92, *(_QWORD *)(a2 + 40), 0);
      v4 = *(_BYTE *)(a1 + 16) & 1;
      goto LABEL_12;
    case 5:
      v85 = *(_QWORD **)(a2 + 32);
      v140 = v142;
      v141 = 0x200000000LL;
      v133 = &v85[*(_QWORD *)(a2 + 40)];
      if ( v85 == v133 )
        goto LABEL_11;
      v86 = 0;
      do
      {
        v87 = (_BYTE *)*v85;
        v24 = (_BYTE *)*v85;
        v88 = sub_1054C20(a1, *v85);
        v91 = (unsigned int)v141;
        if ( (unsigned __int64)(unsigned int)v141 + 1 > HIDWORD(v141) )
        {
          v24 = v142;
          v119 = v88;
          sub_C8D5F0((__int64)&v140, v142, (unsigned int)v141 + 1LL, 8u, v89, v90);
          v91 = (unsigned int)v141;
          v88 = v119;
        }
        *(_QWORD *)&v140[8 * v91] = v88;
        v30 = v140;
        LODWORD(v141) = v141 + 1;
        ++v85;
        v86 |= *(_QWORD *)&v140[8 * (unsigned int)v141 - 8] != (_QWORD)v87;
      }
      while ( v133 != v85 );
      result = a2;
      if ( v86 )
      {
        v24 = &v140;
        result = (unsigned __int64)sub_DC7EB0(*(__int64 **)a1, (__int64)&v140, 0, 0);
        v30 = v140;
      }
      goto LABEL_94;
    case 6:
      v59 = *(_QWORD **)(a2 + 32);
      v140 = v142;
      v141 = 0x200000000LL;
      v131 = &v59[*(_QWORD *)(a2 + 40)];
      if ( v59 == v131 )
        goto LABEL_11;
      v60 = 0;
      do
      {
        v61 = (_BYTE *)*v59;
        v24 = (_BYTE *)*v59;
        v62 = sub_1054C20(a1, *v59);
        v65 = (unsigned int)v141;
        if ( (unsigned __int64)(unsigned int)v141 + 1 > HIDWORD(v141) )
        {
          v24 = v142;
          v124 = v62;
          sub_C8D5F0((__int64)&v140, v142, (unsigned int)v141 + 1LL, 8u, v63, v64);
          v65 = (unsigned int)v141;
          v62 = v124;
        }
        *(_QWORD *)&v140[8 * v65] = v62;
        v30 = v140;
        LODWORD(v141) = v141 + 1;
        ++v59;
        v60 |= *(_QWORD *)&v140[8 * (unsigned int)v141 - 8] != (_QWORD)v61;
      }
      while ( v131 != v59 );
      result = a2;
      if ( v60 )
      {
        v24 = &v140;
        result = (unsigned __int64)sub_DC8BD0(*(__int64 **)a1, (__int64)&v140, 0, 0);
        v30 = v140;
      }
      goto LABEL_94;
    case 7:
      v57 = sub_1054C20(a1, *(_QWORD *)(a2 + 32));
      v58 = sub_1054C20(a1, *(_QWORD *)(a2 + 40));
      if ( v57 == *(_QWORD *)(a2 + 32) && v58 == *(_QWORD *)(a2 + 40) )
        goto LABEL_106;
      result = sub_DCB270(*(_QWORD *)a1, v57, v58);
      goto LABEL_59;
    case 8:
      v75 = *(_QWORD **)(a2 + 32);
      v140 = v142;
      v141 = 0x800000000LL;
      for ( i = &v75[*(_QWORD *)(a2 + 40)]; i != v75; LODWORD(v141) = v141 + 1 )
      {
        v79 = sub_1054C20(a1, *v75);
        v80 = (unsigned int)v141;
        v81 = (unsigned int)v141 + 1LL;
        if ( v81 > HIDWORD(v141) )
        {
          sub_C8D5F0((__int64)&v140, v142, v81, 8u, v77, v78);
          v80 = (unsigned int)v141;
        }
        ++v75;
        *(_QWORD *)&v140[8 * v80] = v79;
      }
      if ( (*(unsigned __int8 (__fastcall **)(_QWORD, unsigned __int64))(a1 + 96))(*(_QWORD *)(a1 + 104), a2) )
      {
        if ( *(_DWORD *)(a1 + 88) == 1 )
        {
          if ( (int)v141 - 1 > 0 )
          {
            v111 = 8;
            v126 = 8LL * (unsigned int)(v141 - 2) + 16;
            do
            {
              v112 = *(__int64 **)a1;
              v113 = v111 - 8;
              v114 = *(_QWORD *)&v140[v111];
              v115 = *(_QWORD *)&v140[v111 - 8];
              v138[0] = v139;
              v139[0] = v115;
              v139[1] = v114;
              v138[1] = 0x200000002LL;
              v116 = sub_DC7EB0(v112, (__int64)v138, 0, 0);
              if ( (_QWORD *)v138[0] != v139 )
              {
                v118 = v116;
                _libc_free(v138[0], v138);
                v116 = v118;
              }
              v111 += 8;
              *(_QWORD *)&v140[v113] = v116;
            }
            while ( v126 != v111 );
          }
        }
        else
        {
          v82 = v141 - 2;
          if ( (int)v141 - 2 >= 0 )
          {
            v83 = 8LL * v82;
            do
            {
              --v82;
              v84 = sub_DCC810(*(__int64 **)a1, *(_QWORD *)&v140[v83], *(_QWORD *)&v140[v83 + 8], 0, 0);
              *(_QWORD *)&v140[v83] = v84;
              v83 -= 8;
            }
            while ( v82 != -1 );
          }
        }
      }
      result = (unsigned __int64)sub_DBFF60(*(_QWORD *)a1, (unsigned int *)&v140, *(_QWORD *)(a2 + 48), 0);
      if ( v140 != v142 )
      {
        v134 = result;
        _libc_free(v140, &v140);
        result = v134;
      }
      goto LABEL_59;
    case 9:
      v67 = *(_QWORD **)(a2 + 32);
      v140 = v142;
      v141 = 0x200000000LL;
      v132 = &v67[*(_QWORD *)(a2 + 40)];
      if ( v67 == v132 )
        goto LABEL_11;
      v68 = 0;
      do
      {
        v69 = (_BYTE *)*v67;
        v24 = (_BYTE *)*v67;
        v70 = sub_1054C20(a1, *v67);
        v73 = (unsigned int)v141;
        if ( (unsigned __int64)(unsigned int)v141 + 1 > HIDWORD(v141) )
        {
          v24 = v142;
          v121 = v70;
          sub_C8D5F0((__int64)&v140, v142, (unsigned int)v141 + 1LL, 8u, v71, v72);
          v73 = (unsigned int)v141;
          v70 = v121;
        }
        v74 = (__int64)v140;
        *(_QWORD *)&v140[8 * v73] = v70;
        v30 = v140;
        LODWORD(v141) = v141 + 1;
        ++v67;
        v68 |= *(_QWORD *)&v140[8 * (unsigned int)v141 - 8] != (_QWORD)v69;
      }
      while ( v132 != v67 );
      result = a2;
      if ( v68 )
      {
        v24 = &v140;
        result = sub_DCE040(*(__int64 **)a1, (__int64)&v140, v73, v74, v71);
        v30 = v140;
      }
      goto LABEL_94;
    case 0xA:
      v47 = *(_QWORD **)(a2 + 32);
      v140 = v142;
      v141 = 0x200000000LL;
      v130 = &v47[*(_QWORD *)(a2 + 40)];
      if ( v47 == v130 )
        goto LABEL_11;
      v48 = 0;
      do
      {
        v49 = (_BYTE *)*v47;
        v24 = (_BYTE *)*v47;
        v50 = sub_1054C20(a1, *v47);
        v53 = (unsigned int)v141;
        if ( (unsigned __int64)(unsigned int)v141 + 1 > HIDWORD(v141) )
        {
          v24 = v142;
          v123 = v50;
          sub_C8D5F0((__int64)&v140, v142, (unsigned int)v141 + 1LL, 8u, v51, v52);
          v53 = (unsigned int)v141;
          v50 = v123;
        }
        v54 = (__int64)v140;
        *(_QWORD *)&v140[8 * v53] = v50;
        v30 = v140;
        LODWORD(v141) = v141 + 1;
        ++v47;
        v48 |= *(_QWORD *)&v140[8 * (unsigned int)v141 - 8] != (_QWORD)v49;
      }
      while ( v130 != v47 );
      result = a2;
      if ( v48 )
      {
        v24 = &v140;
        result = sub_DCDF90(*(__int64 **)a1, (__int64)&v140, v53, v54, v51);
        v30 = v140;
      }
      goto LABEL_94;
    case 0xB:
      v39 = *(_QWORD **)(a2 + 32);
      v140 = v142;
      v141 = 0x200000000LL;
      v129 = &v39[*(_QWORD *)(a2 + 40)];
      if ( v39 == v129 )
        goto LABEL_11;
      v40 = 0;
      do
      {
        v41 = (_BYTE *)*v39;
        v24 = (_BYTE *)*v39;
        v42 = sub_1054C20(a1, *v39);
        v45 = (unsigned int)v141;
        if ( (unsigned __int64)(unsigned int)v141 + 1 > HIDWORD(v141) )
        {
          v24 = v142;
          v120 = v42;
          sub_C8D5F0((__int64)&v140, v142, (unsigned int)v141 + 1LL, 8u, v43, v44);
          v45 = (unsigned int)v141;
          v42 = v120;
        }
        v46 = (__int64)v140;
        *(_QWORD *)&v140[8 * v45] = v42;
        v30 = v140;
        LODWORD(v141) = v141 + 1;
        ++v39;
        v40 |= *(_QWORD *)&v140[8 * (unsigned int)v141 - 8] != (_QWORD)v41;
      }
      while ( v129 != v39 );
      result = a2;
      if ( v40 )
      {
        v24 = &v140;
        result = (unsigned __int64)sub_DCEE50(*(__int64 **)a1, (__int64)&v140, 0, v46, v43);
        v30 = v140;
      }
      goto LABEL_94;
    case 0xC:
      v31 = *(_QWORD **)(a2 + 32);
      v140 = v142;
      v141 = 0x200000000LL;
      v128 = &v31[*(_QWORD *)(a2 + 40)];
      if ( v31 == v128 )
        goto LABEL_11;
      v32 = 0;
      do
      {
        v33 = (_BYTE *)*v31;
        v24 = (_BYTE *)*v31;
        v34 = sub_1054C20(a1, *v31);
        v37 = (unsigned int)v141;
        if ( (unsigned __int64)(unsigned int)v141 + 1 > HIDWORD(v141) )
        {
          v24 = v142;
          v125 = v34;
          sub_C8D5F0((__int64)&v140, v142, (unsigned int)v141 + 1LL, 8u, v35, v36);
          v37 = (unsigned int)v141;
          v34 = v125;
        }
        v38 = (__int64)v140;
        *(_QWORD *)&v140[8 * v37] = v34;
        v30 = v140;
        LODWORD(v141) = v141 + 1;
        ++v31;
        v32 |= *(_QWORD *)&v140[8 * (unsigned int)v141 - 8] != (_QWORD)v33;
      }
      while ( v128 != v31 );
      result = a2;
      if ( v32 )
      {
        v24 = &v140;
        result = sub_DCE150(*(__int64 **)a1, (__int64)&v140, v37, v38, v35);
        v30 = v140;
      }
      goto LABEL_94;
    case 0xD:
      v21 = *(_QWORD **)(a2 + 32);
      v140 = v142;
      v141 = 0x200000000LL;
      v127 = &v21[*(_QWORD *)(a2 + 40)];
      if ( v21 == v127 )
      {
LABEL_11:
        result = a2;
      }
      else
      {
        v22 = 0;
        do
        {
          v23 = (_BYTE *)*v21;
          v24 = (_BYTE *)*v21;
          v25 = sub_1054C20(a1, *v21);
          v28 = (unsigned int)v141;
          if ( (unsigned __int64)(unsigned int)v141 + 1 > HIDWORD(v141) )
          {
            v24 = v142;
            v122 = v25;
            sub_C8D5F0((__int64)&v140, v142, (unsigned int)v141 + 1LL, 8u, v26, v27);
            v28 = (unsigned int)v141;
            v25 = v122;
          }
          v29 = (__int64)v140;
          *(_QWORD *)&v140[8 * v28] = v25;
          v30 = v140;
          LODWORD(v141) = v141 + 1;
          ++v21;
          v22 |= *(_QWORD *)&v140[8 * (unsigned int)v141 - 8] != (_QWORD)v23;
        }
        while ( v127 != v21 );
        result = a2;
        if ( v22 )
        {
          v24 = &v140;
          result = (unsigned __int64)sub_DCEE50(*(__int64 **)a1, (__int64)&v140, 1, v29, v26);
          v30 = v140;
        }
LABEL_94:
        if ( v30 == v142 )
        {
LABEL_59:
          v4 = *(_BYTE *)(a1 + 16) & 1;
        }
        else
        {
          v135 = result;
          _libc_free(v30, v24);
          result = v135;
          v4 = *(_BYTE *)(a1 + 16) & 1;
        }
      }
LABEL_12:
      if ( v4 )
      {
        v13 = a1 + 24;
        v14 = 3;
      }
      else
      {
        v19 = *(_DWORD *)(a1 + 32);
        v13 = *(_QWORD *)(a1 + 24);
        if ( !v19 )
        {
          v93 = *(_DWORD *)(a1 + 16);
          ++*(_QWORD *)(a1 + 8);
          v94 = 0;
          v95 = (v93 >> 1) + 1;
LABEL_99:
          v96 = 3 * v19;
          goto LABEL_100;
        }
        v14 = v19 - 1;
      }
      v15 = v14 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v16 = (unsigned __int64 *)(v13 + 16LL * v15);
      v17 = *v16;
      if ( *v16 == a2 )
        return v16[1];
      v97 = 1;
      v94 = 0;
      while ( v17 != -4096 )
      {
        if ( !v94 && v17 == -8192 )
          v94 = v16;
        v15 = v14 & (v97 + v15);
        v16 = (unsigned __int64 *)(v13 + 16LL * v15);
        v17 = *v16;
        if ( *v16 == a2 )
          return v16[1];
        ++v97;
      }
      if ( !v94 )
        v94 = v16;
      v93 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)(a1 + 8);
      v95 = (v93 >> 1) + 1;
      if ( !v4 )
      {
        v19 = *(_DWORD *)(a1 + 32);
        goto LABEL_99;
      }
      v96 = 12;
      v19 = 4;
LABEL_100:
      if ( 4 * v95 >= v96 )
      {
        v136 = result;
        sub_DB0DD0(a1 + 8, 2 * v19);
        result = v136;
        if ( (*(_BYTE *)(a1 + 16) & 1) != 0 )
        {
          v99 = a1 + 24;
          v100 = 3;
        }
        else
        {
          v109 = *(_DWORD *)(a1 + 32);
          v99 = *(_QWORD *)(a1 + 24);
          if ( !v109 )
            goto LABEL_155;
          v100 = v109 - 1;
        }
        v101 = v100 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v94 = (unsigned __int64 *)(v99 + 16LL * v101);
        v102 = *v94;
        if ( *v94 != a2 )
        {
          v117 = 1;
          v108 = 0;
          while ( v102 != -4096 )
          {
            if ( v102 == -8192 && !v108 )
              v108 = v94;
            v101 = v100 & (v117 + v101);
            v94 = (unsigned __int64 *)(v99 + 16LL * v101);
            v102 = *v94;
            if ( *v94 == a2 )
              goto LABEL_118;
            ++v117;
          }
          goto LABEL_124;
        }
LABEL_118:
        v93 = *(_DWORD *)(a1 + 16);
        goto LABEL_102;
      }
      if ( v19 - *(_DWORD *)(a1 + 20) - v95 <= v19 >> 3 )
      {
        v137 = result;
        sub_DB0DD0(a1 + 8, v19);
        result = v137;
        if ( (*(_BYTE *)(a1 + 16) & 1) != 0 )
        {
          v103 = a1 + 24;
          v104 = 3;
          goto LABEL_121;
        }
        v110 = *(_DWORD *)(a1 + 32);
        v103 = *(_QWORD *)(a1 + 24);
        if ( v110 )
        {
          v104 = v110 - 1;
LABEL_121:
          v105 = v104 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v94 = (unsigned __int64 *)(v103 + 16LL * v105);
          v106 = *v94;
          if ( *v94 != a2 )
          {
            v107 = 1;
            v108 = 0;
            while ( v106 != -4096 )
            {
              if ( !v108 && v106 == -8192 )
                v108 = v94;
              v105 = v104 & (v107 + v105);
              v94 = (unsigned __int64 *)(v103 + 16LL * v105);
              v106 = *v94;
              if ( *v94 == a2 )
                goto LABEL_118;
              ++v107;
            }
LABEL_124:
            if ( v108 )
              v94 = v108;
            goto LABEL_118;
          }
          goto LABEL_118;
        }
LABEL_155:
        *(_DWORD *)(a1 + 16) = (2 * (*(_DWORD *)(a1 + 16) >> 1) + 2) | *(_DWORD *)(a1 + 16) & 1;
        BUG();
      }
LABEL_102:
      *(_DWORD *)(a1 + 16) = (2 * (v93 >> 1) + 2) | v93 & 1;
      if ( *v94 != -4096 )
        --*(_DWORD *)(a1 + 20);
      *v94 = a2;
      v94[1] = result;
      return result;
    case 0xE:
      v66 = sub_1054C20(a1, *(_QWORD *)(a2 + 32));
      if ( v66 == *(_QWORD *)(a2 + 32) )
      {
LABEL_106:
        result = a2;
        v4 = *(_BYTE *)(a1 + 16) & 1;
      }
      else
      {
        result = (unsigned __int64)sub_DD3A70(*(_QWORD *)a1, v66, *(_QWORD *)(a2 + 40));
        v4 = *(_BYTE *)(a1 + 16) & 1;
      }
      goto LABEL_12;
    default:
      BUG();
  }
}
