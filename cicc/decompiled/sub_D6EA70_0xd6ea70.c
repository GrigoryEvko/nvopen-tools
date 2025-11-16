// Function: sub_D6EA70
// Address: 0xd6ea70
//
__int64 __fastcall sub_D6EA70(__int64 *a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5, char a6)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  unsigned __int64 *v8; // r15
  unsigned __int64 *v9; // r13
  unsigned __int64 *v10; // rdi
  __int64 (__fastcall *v11)(__int64); // rax
  __int64 *v12; // rax
  unsigned __int64 v13; // rdi
  __int64 v14; // r8
  unsigned __int64 *v15; // r15
  unsigned __int64 *v16; // r13
  unsigned __int64 *v17; // rdi
  __int64 (__fastcall *v18)(__int64); // rax
  unsigned __int64 v19; // rdi
  __int64 (__fastcall **v20)(__int64 *); // rax
  __int64 *v21; // rbx
  int v22; // ecx
  unsigned int v23; // edx
  unsigned __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // r14
  bool v27; // zf
  _QWORD *v28; // rax
  unsigned int v29; // edx
  unsigned int v30; // esi
  __int64 v31; // rdx
  int v32; // eax
  __int64 v33; // r12
  char v34; // bl
  __int64 v35; // rax
  char v36; // di
  __int64 (__fastcall **v37)(__int64 *); // r8
  unsigned int v38; // ecx
  __int64 (__fastcall **v39)(__int64 *); // rdx
  __int64 (__fastcall *v40)(__int64 *); // r10
  __int64 v41; // r8
  __int64 v42; // r9
  unsigned __int64 v43; // rbx
  char v44; // di
  _QWORD *v45; // rax
  __int64 v46; // rcx
  __int64 *v47; // rdx
  __int64 v48; // rdx
  __int64 v49; // rbx
  __int64 v50; // rax
  _BYTE *v51; // rcx
  __int64 v52; // r14
  int v53; // eax
  int v54; // edi
  unsigned int v55; // edx
  unsigned __int64 v56; // rax
  __int64 v57; // r8
  __int64 v58; // r13
  _QWORD *v59; // rax
  _QWORD *v60; // rdx
  int v61; // esi
  _QWORD *v62; // rdx
  int v63; // eax
  __int64 *v64; // r12
  __int64 *v65; // r13
  unsigned int v66; // eax
  _QWORD *v67; // rcx
  __int64 v68; // rdi
  _QWORD *v69; // r10
  int v70; // edx
  int v71; // r11d
  int v72; // eax
  __int64 v73; // rbx
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // r9
  int v78; // eax
  int v79; // edx
  int v80; // r9d
  int v81; // ecx
  int v82; // r8d
  __int64 *v83; // [rsp+28h] [rbp-208h]
  __int64 *v85; // [rsp+40h] [rbp-1F0h]
  unsigned __int64 v88; // [rsp+58h] [rbp-1D8h]
  unsigned __int64 v89; // [rsp+58h] [rbp-1D8h]
  __int64 *v90; // [rsp+60h] [rbp-1D0h]
  _BYTE *v91; // [rsp+60h] [rbp-1D0h]
  __int64 v92; // [rsp+68h] [rbp-1C8h]
  unsigned __int64 v93; // [rsp+68h] [rbp-1C8h]
  __int64 v94; // [rsp+68h] [rbp-1C8h]
  __int64 v95; // [rsp+68h] [rbp-1C8h]
  __int64 v96; // [rsp+68h] [rbp-1C8h]
  __int64 *v97; // [rsp+70h] [rbp-1C0h]
  __int64 *v98; // [rsp+80h] [rbp-1B0h] BYREF
  __int64 v99; // [rsp+88h] [rbp-1A8h] BYREF
  unsigned __int64 v100; // [rsp+90h] [rbp-1A0h] BYREF
  __int64 v101; // [rsp+98h] [rbp-198h] BYREF
  __int64 v102; // [rsp+A0h] [rbp-190h]
  __int64 v103; // [rsp+A8h] [rbp-188h]
  __int64 v104; // [rsp+B0h] [rbp-180h]
  unsigned __int64 v105; // [rsp+C0h] [rbp-170h] BYREF
  __int64 v106; // [rsp+C8h] [rbp-168h] BYREF
  __int64 v107; // [rsp+D0h] [rbp-160h]
  __int64 v108; // [rsp+D8h] [rbp-158h]
  _QWORD v109[4]; // [rsp+E0h] [rbp-150h] BYREF
  unsigned __int64 v110; // [rsp+100h] [rbp-130h] BYREF
  __int64 v111; // [rsp+108h] [rbp-128h]
  __int64 (__fastcall *v112)(__int64 *); // [rsp+110h] [rbp-120h] BYREF
  __int64 v113; // [rsp+118h] [rbp-118h]
  char v114; // [rsp+120h] [rbp-110h] BYREF
  __int64 v115; // [rsp+150h] [rbp-E0h] BYREF
  __int64 v116; // [rsp+158h] [rbp-D8h]
  __int64 v117; // [rsp+160h] [rbp-D0h]
  __int64 v118; // [rsp+168h] [rbp-C8h]
  __int64 *v119; // [rsp+170h] [rbp-C0h] BYREF
  __int64 v120; // [rsp+178h] [rbp-B8h]
  _BYTE v121[176]; // [rsp+180h] [rbp-B0h] BYREF

  v119 = (__int64 *)v121;
  v120 = 0x1000000000LL;
  v6 = *(_QWORD *)(a2 + 48);
  v7 = *(_QWORD *)(a2 + 40);
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v118 = 0;
  v100 = a3;
  v101 = v6;
  v88 = a3 + 8 * a4;
  v92 = v7;
  v102 = v88;
  v103 = v7;
  if ( v7 == v6 )
    goto LABEL_18;
  do
  {
    do
    {
      v8 = &v110;
      v9 = &v110;
      v10 = &v100;
      v113 = 0;
      v112 = sub_D67840;
      v11 = sub_D67820;
      if ( ((unsigned __int8)sub_D67820 & 1) != 0 )
LABEL_3:
        v11 = *(__int64 (__fastcall **)(__int64))((char *)v11 + *v10 - 1);
      v12 = (__int64 *)v11((__int64)v10);
      if ( !v12 )
      {
        while ( 1 )
        {
          v8 += 2;
          if ( &v114 == (char *)v8 )
            break;
          v13 = v9[3];
          v11 = (__int64 (__fastcall *)(__int64))v9[2];
          v9 = v8;
          v10 = (unsigned __int64 *)((char *)&v100 + v13);
          if ( ((unsigned __int8)v11 & 1) != 0 )
            goto LABEL_3;
          v12 = (__int64 *)v11((__int64)v10);
          if ( v12 )
            goto LABEL_8;
        }
LABEL_149:
        BUG();
      }
LABEL_8:
      v99 = *v12;
      if ( (_DWORD)v117 )
      {
        v15 = &v105;
        v7 = (__int64)&v99;
        if ( (unsigned __int8)sub_D6B660((__int64)&v115, &v99, &v105) )
          goto LABEL_11;
        v61 = v118;
        v62 = (_QWORD *)v105;
        ++v115;
        v63 = v117 + 1;
        v110 = v105;
        if ( 4 * ((int)v117 + 1) >= (unsigned int)(3 * v118) )
        {
          v61 = 2 * v118;
        }
        else if ( (int)v118 - HIDWORD(v117) - v63 > (unsigned int)v118 >> 3 )
        {
LABEL_81:
          LODWORD(v117) = v63;
          if ( *v62 != -4096 )
            --HIDWORD(v117);
          *v62 = v99;
          v7 = v99;
          sub_B1A4E0((__int64)&v119, v99);
          goto LABEL_11;
        }
        sub_CF28B0((__int64)&v115, v61);
        sub_D6B660((__int64)&v115, &v99, &v110);
        v62 = (_QWORD *)v110;
        v63 = v117 + 1;
        goto LABEL_81;
      }
      v7 = (__int64)&v119[(unsigned int)v120];
      if ( (_QWORD *)v7 != sub_D67AA0(v119, v7, &v99) )
      {
LABEL_10:
        v15 = &v105;
        goto LABEL_11;
      }
      v7 = v14;
      v15 = &v105;
      sub_B1A4E0((__int64)&v119, v14);
      if ( (unsigned int)v120 > 0x10 )
      {
        v64 = v119;
        v65 = &v119[(unsigned int)v120];
        while ( 1 )
        {
          v7 = (unsigned int)v118;
          if ( !(_DWORD)v118 )
            break;
          v66 = (v118 - 1) & (((unsigned int)*v64 >> 9) ^ ((unsigned int)*v64 >> 4));
          v67 = (_QWORD *)(v116 + 8LL * v66);
          v68 = *v67;
          if ( *v67 != *v64 )
          {
            v71 = 1;
            v69 = 0;
            while ( v68 != -4096 )
            {
              if ( v68 == -8192 && !v69 )
                v69 = v67;
              v66 = (v118 - 1) & (v71 + v66);
              v67 = (_QWORD *)(v116 + 8LL * v66);
              v68 = *v67;
              if ( *v64 == *v67 )
                goto LABEL_87;
              ++v71;
            }
            if ( !v69 )
              v69 = v67;
            ++v115;
            v70 = v117 + 1;
            v110 = (unsigned __int64)v69;
            if ( 4 * ((int)v117 + 1) < (unsigned int)(3 * v118) )
            {
              if ( (int)v118 - HIDWORD(v117) - v70 > (unsigned int)v118 >> 3 )
                goto LABEL_98;
              goto LABEL_91;
            }
LABEL_90:
            LODWORD(v7) = 2 * v118;
LABEL_91:
            sub_CF28B0((__int64)&v115, v7);
            v7 = (__int64)v64;
            sub_D6B660((__int64)&v115, v64, &v110);
            v69 = (_QWORD *)v110;
            v70 = v117 + 1;
LABEL_98:
            LODWORD(v117) = v70;
            if ( *v69 != -4096 )
              --HIDWORD(v117);
            *v69 = *v64;
          }
LABEL_87:
          if ( v65 == ++v64 )
            goto LABEL_10;
        }
        ++v115;
        v110 = 0;
        goto LABEL_90;
      }
LABEL_11:
      v16 = &v105;
      v108 = 0;
      v17 = &v100;
      v107 = (__int64)sub_D677F0;
      v18 = sub_D677C0;
      if ( ((unsigned __int8)sub_D677C0 & 1) != 0 )
LABEL_12:
        v18 = *(__int64 (__fastcall **)(__int64))((char *)v18 + *v17 - 1);
      while ( !(unsigned __int8)v18((__int64)v17) )
      {
        v15 += 2;
        if ( v109 == v15 )
          goto LABEL_149;
        v19 = v16[3];
        v18 = (__int64 (__fastcall *)(__int64))v16[2];
        v16 = v15;
        v17 = (unsigned __int64 *)((char *)&v100 + v19);
        if ( ((unsigned __int8)v18 & 1) != 0 )
          goto LABEL_12;
      }
    }
    while ( v92 != v101 );
LABEL_18:
    ;
  }
  while ( v88 != v100 || v92 != v103 || v88 != v102 );
  v20 = &v112;
  v110 = 0;
  v98 = &v115;
  v111 = 1;
  do
  {
    *v20 = (__int64 (__fastcall *)(__int64 *))-4096LL;
    v20 += 2;
  }
  while ( v20 != (__int64 (__fastcall **)(__int64 *))&v115 );
  v21 = v119;
  v90 = &v119[(unsigned int)v120];
  if ( v90 == v119 )
  {
    v36 = v111 & 1;
    goto LABEL_115;
  }
  while ( 2 )
  {
    while ( 2 )
    {
      v32 = *(_DWORD *)(a5 + 24);
      v33 = *v21;
      if ( v32 )
      {
        v22 = v32 - 1;
        v7 = *(_QWORD *)(a5 + 8);
        v106 = 2;
        v107 = 0;
        v108 = -4096;
        v23 = (v32 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
        v109[0] = 0;
        v24 = v7 + ((unsigned __int64)v23 << 6);
        v25 = *(_QWORD *)(v24 + 24);
        if ( v33 != v25 )
        {
          v78 = 1;
          while ( v25 != -4096 )
          {
            v82 = v78 + 1;
            v23 = v22 & (v78 + v23);
            v24 = v7 + ((unsigned __int64)v23 << 6);
            v25 = *(_QWORD *)(v24 + 24);
            if ( v33 == v25 )
              goto LABEL_26;
            v78 = v82;
          }
          v105 = (unsigned __int64)&unk_49DB368;
          sub_D68D70(&v106);
          break;
        }
LABEL_26:
        v93 = v24;
        v105 = (unsigned __int64)&unk_49DB368;
        sub_D68D70(&v106);
        if ( v93 == *(_QWORD *)(a5 + 8) + ((unsigned __int64)*(unsigned int *)(a5 + 24) << 6) )
          break;
        v105 = 6;
        v106 = 0;
        v107 = *(_QWORD *)(v93 + 56);
        v26 = v107;
        if ( v107 != -4096 && v107 != 0 && v107 != -8192 )
        {
          v7 = *(_QWORD *)(v93 + 40) & 0xFFFFFFFFFFFFFFF8LL;
          sub_BD6050(&v105, v7);
          v26 = v107;
        }
        if ( !v26 )
          goto LABEL_43;
        sub_D68D70(&v105);
        v94 = *a1;
        v99 = sub_D68B40(*a1, v33);
        if ( !v99 )
          goto LABEL_40;
        v95 = sub_10420D0(v94, v26);
        v27 = (unsigned __int8)sub_D69D00((__int64)&v110, &v99, &v100) == 0;
        v28 = (_QWORD *)v100;
        if ( !v27 )
        {
LABEL_39:
          v28[1] = v95;
LABEL_40:
          v7 = v33;
          ++v21;
          sub_D69A00(
            a1,
            v33,
            v26,
            a5,
            (__int64)&v110,
            0,
            (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_D679F0,
            (__int64)&v98);
          if ( v90 == v21 )
            goto LABEL_44;
          continue;
        }
        ++v110;
        v105 = v100;
        v29 = ((unsigned int)v111 >> 1) + 1;
        if ( (v111 & 1) != 0 )
        {
          v30 = 4;
          if ( 4 * v29 < 0xC )
          {
LABEL_35:
            if ( v30 - (v29 + HIDWORD(v111)) > v30 >> 3 )
            {
LABEL_36:
              LODWORD(v111) = v111 & 1 | (2 * v29);
              if ( *v28 != -4096 )
                --HIDWORD(v111);
              v31 = v99;
              v28[1] = 0;
              *v28 = v31;
              goto LABEL_39;
            }
LABEL_126:
            sub_D6B9C0((__int64)&v110, v30);
            sub_D69D00((__int64)&v110, &v99, &v105);
            v28 = (_QWORD *)v105;
            v29 = ((unsigned int)v111 >> 1) + 1;
            goto LABEL_36;
          }
        }
        else
        {
          v30 = v113;
          if ( 4 * v29 < 3 * (int)v113 )
            goto LABEL_35;
        }
        v30 *= 2;
        goto LABEL_126;
      }
      break;
    }
    v105 = 6;
    v106 = 0;
    v107 = 0;
LABEL_43:
    ++v21;
    sub_D68D70(&v105);
    if ( v90 != v21 )
      continue;
    break;
  }
LABEL_44:
  v34 = v111;
  v83 = &v119[(unsigned int)v120];
  if ( v83 != v119 )
  {
    v97 = v119;
    while ( 1 )
    {
      v7 = *v97;
      v35 = sub_D68B40(*a1, *v97);
      v36 = v34 & 1;
      if ( v35 )
      {
        if ( v36 )
        {
          v37 = &v112;
          v7 = 3;
        }
        else
        {
          v7 = (unsigned int)v113;
          v37 = (__int64 (__fastcall **)(__int64 *))v112;
          if ( !(_DWORD)v113 )
            goto LABEL_114;
          v7 = (unsigned int)(v113 - 1);
        }
        v38 = v7 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
        v39 = &v37[2 * v38];
        v40 = *v39;
        if ( *v39 == (__int64 (__fastcall *)(__int64 *))v35 )
        {
LABEL_50:
          v85 = (__int64 *)v39[1];
          if ( v85 )
          {
            v99 = v35;
            v105 = *(_QWORD *)(v85[8] + 16);
            sub_D4B000((__int64 *)&v105);
            v43 = v105;
            v105 = 0;
            v107 = 4;
            v106 = (__int64)v109;
            LODWORD(v108) = 0;
            BYTE4(v108) = 1;
            if ( v43 )
            {
              v44 = 1;
              v7 = *(_QWORD *)(*(_QWORD *)(v43 + 24) + 40LL);
LABEL_53:
              v45 = (_QWORD *)v106;
              v46 = HIDWORD(v107);
              v47 = (__int64 *)(v106 + 8LL * HIDWORD(v107));
              if ( (__int64 *)v106 == v47 )
              {
LABEL_120:
                if ( HIDWORD(v107) < (unsigned int)v107 )
                {
                  v46 = (unsigned int)++HIDWORD(v107);
                  *v47 = v7;
                  v44 = BYTE4(v108);
                  ++v105;
                  goto LABEL_57;
                }
                goto LABEL_60;
              }
              while ( v7 != *v45 )
              {
                if ( v47 == ++v45 )
                  goto LABEL_120;
              }
LABEL_57:
              while ( 1 )
              {
                v43 = *(_QWORD *)(v43 + 8);
                if ( !v43 )
                  break;
                while ( 1 )
                {
                  v47 = *(__int64 **)(v43 + 24);
                  if ( (unsigned __int8)(*(_BYTE *)v47 - 30) > 0xAu )
                    break;
                  v7 = v47[5];
                  if ( v44 )
                    goto LABEL_53;
LABEL_60:
                  sub_C8CC70((__int64)&v105, v7, (__int64)v47, v46, v41, v42);
                  v43 = *(_QWORD *)(v43 + 8);
                  v44 = BYTE4(v108);
                  if ( !v43 )
                    goto LABEL_61;
                }
              }
            }
LABEL_61:
            v48 = v99;
            if ( (*(_DWORD *)(v99 + 4) & 0x7FFFFFF) != 0 )
            {
              v49 = 0;
              v96 = 8LL * (*(_DWORD *)(v99 + 4) & 0x7FFFFFF);
              while ( 1 )
              {
                v50 = *(_QWORD *)(v48 - 8);
                v51 = *(_BYTE **)(v50 + 4 * v49);
                v52 = *(_QWORD *)(32LL * *(unsigned int *)(v48 + 76) + v50 + v49);
                v53 = *(_DWORD *)(a5 + 24);
                v91 = v51;
                if ( !v53 )
                  goto LABEL_104;
                v54 = v53 - 1;
                v7 = *(_QWORD *)(a5 + 8);
                v101 = 2;
                v102 = 0;
                v103 = -4096;
                v55 = (v53 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
                v104 = 0;
                v56 = v7 + ((unsigned __int64)v55 << 6);
                v57 = *(_QWORD *)(v56 + 24);
                if ( v57 != v52 )
                  break;
LABEL_65:
                v89 = v56;
                v100 = (unsigned __int64)&unk_49DB368;
                sub_D68D70(&v101);
                if ( v89 == *(_QWORD *)(a5 + 8) + ((unsigned __int64)*(unsigned int *)(a5 + 24) << 6) )
                  goto LABEL_104;
                v100 = 6;
                v101 = 0;
                v102 = *(_QWORD *)(v89 + 56);
                v58 = v102;
                LOBYTE(v7) = v102 != 0;
                if ( v102 != -4096 && v102 != 0 && v102 != -8192 )
                {
                  v7 = *(_QWORD *)(v89 + 40) & 0xFFFFFFFFFFFFFFF8LL;
                  sub_BD6050(&v100, v7);
                  v58 = v102;
                }
                if ( v58 )
                {
                  sub_D68D70(&v100);
                  if ( BYTE4(v108) )
                    goto LABEL_71;
                  goto LABEL_107;
                }
LABEL_105:
                v58 = v52;
                sub_D68D70(&v100);
                if ( a6 )
                  goto LABEL_76;
                if ( BYTE4(v108) )
                {
LABEL_71:
                  v59 = (_QWORD *)v106;
                  v60 = (_QWORD *)(v106 + 8LL * HIDWORD(v107));
                  if ( (_QWORD *)v106 != v60 )
                  {
                    while ( v58 != *v59 )
                    {
                      if ( v60 == ++v59 )
                        goto LABEL_76;
                    }
LABEL_75:
                    v7 = (__int64)sub_D69810(
                                    v91,
                                    a5,
                                    (__int64)&v110,
                                    *a1,
                                    (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_D679F0,
                                    (__int64)&v98);
                    sub_D689D0((__int64)v85, v7, v58);
                  }
LABEL_76:
                  v49 += 8;
                  if ( v96 == v49 )
                    goto LABEL_109;
                  goto LABEL_77;
                }
LABEL_107:
                v7 = v58;
                if ( sub_C8CA60((__int64)&v105, v58) )
                  goto LABEL_75;
                v49 += 8;
                if ( v96 == v49 )
                  goto LABEL_109;
LABEL_77:
                v48 = v99;
              }
              v72 = 1;
              while ( v57 != -4096 )
              {
                v81 = v72 + 1;
                v55 = v54 & (v72 + v55);
                v56 = v7 + ((unsigned __int64)v55 << 6);
                v57 = *(_QWORD *)(v56 + 24);
                if ( v52 == v57 )
                  goto LABEL_65;
                v72 = v81;
              }
              v100 = (unsigned __int64)&unk_49DB368;
              sub_D68D70(&v101);
LABEL_104:
              v100 = 6;
              v101 = 0;
              v102 = 0;
              goto LABEL_105;
            }
LABEL_109:
            v73 = sub_D67DF0(v85);
            if ( v73 )
            {
              v7 = (__int64)v85;
              *sub_D6BCB0((__int64)&v110, &v99) = v73;
              sub_D6E4B0(a1, (__int64)v85, 0, v74, v75, v76);
            }
            if ( !BYTE4(v108) )
              _libc_free(v106, v7);
            v34 = v111;
            v36 = v111 & 1;
          }
        }
        else
        {
          v79 = 1;
          while ( v40 != (__int64 (__fastcall *)(__int64 *))-4096LL )
          {
            v80 = v79 + 1;
            v38 = v7 & (v79 + v38);
            v39 = &v37[2 * v38];
            v40 = *v39;
            if ( (__int64 (__fastcall *)(__int64 *))v35 == *v39 )
              goto LABEL_50;
            v79 = v80;
          }
        }
      }
LABEL_114:
      if ( v83 == ++v97 )
        goto LABEL_115;
    }
  }
  v36 = v111 & 1;
LABEL_115:
  if ( !v36 )
  {
    v7 = 16LL * (unsigned int)v113;
    sub_C7D6A0((__int64)v112, v7, 8);
  }
  if ( v119 != (__int64 *)v121 )
    _libc_free(v119, v7);
  return sub_C7D6A0(v116, 8LL * (unsigned int)v118, 8);
}
