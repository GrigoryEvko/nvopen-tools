// Function: sub_29E7B10
// Address: 0x29e7b10
//
void __fastcall sub_29E7B10(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4, __int64 a5, __int64 a6)
{
  const char *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r15
  __int64 v9; // rbx
  __int64 v10; // rdi
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // r12
  __int64 v19; // r13
  unsigned __int64 v20; // rax
  __int64 *v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rsi
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 *v26; // rax
  __int64 *v27; // rax
  __int64 *v28; // r12
  __int64 v29; // r13
  __int64 *v30; // rbx
  __int64 v31; // r14
  __int64 v32; // r15
  __int64 v33; // rbx
  __int64 v34; // r15
  __int64 v35; // r13
  int v36; // eax
  unsigned int v37; // edx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // r12
  int v42; // eax
  __int64 v43; // rbx
  __int64 v44; // r13
  int v45; // eax
  int v46; // eax
  unsigned int v47; // edx
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rdx
  __int64 v51; // rbx
  __int64 v52; // r12
  const char *v53; // r14
  unsigned __int16 v54; // r13
  _QWORD *v55; // rdi
  __int64 v56; // rbx
  __int64 v57; // r14
  __int64 v58; // r15
  int v59; // eax
  unsigned int v60; // edx
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rdx
  __int64 v64; // r13
  int v65; // eax
  _QWORD *v66; // r12
  __int64 *v67; // r13
  const char *v68; // rax
  __int64 *v69; // rsi
  int v70; // r13d
  __int64 v71; // rdx
  __int64 v72; // r15
  __int64 v73; // rbx
  int v74; // edx
  unsigned int v75; // esi
  __int64 v76; // rdx
  __int64 v77; // rsi
  __int64 v78; // rsi
  __int64 v79; // rdx
  __int64 v80; // rax
  __int64 v81; // r14
  _QWORD *v82; // r15
  __int64 v83; // rcx
  int v84; // edx
  __int64 v85; // r12
  __int64 v86; // rax
  __int64 v87; // rbx
  _QWORD *v88; // r13
  __int64 v89; // rbx
  __int64 v90; // r13
  __int64 v91; // r12
  int v92; // eax
  int v93; // eax
  unsigned int v94; // edx
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // rdx
  __int64 v98; // r12
  __int64 v99; // r14
  __int64 v100; // rdx
  __int64 v101; // rsi
  __int64 *v102; // rax
  __int64 v104; // [rsp+10h] [rbp-1D0h]
  __int64 v106; // [rsp+30h] [rbp-1B0h]
  int v107; // [rsp+3Ch] [rbp-1A4h]
  __int64 v108; // [rsp+40h] [rbp-1A0h]
  __int64 v109; // [rsp+48h] [rbp-198h]
  __int64 v110; // [rsp+48h] [rbp-198h]
  unsigned int v111; // [rsp+58h] [rbp-188h]
  unsigned __int64 v112; // [rsp+60h] [rbp-180h]
  __int64 *v113; // [rsp+60h] [rbp-180h]
  __int64 v114; // [rsp+68h] [rbp-178h]
  __int64 v115; // [rsp+68h] [rbp-178h]
  const char *v116; // [rsp+70h] [rbp-170h] BYREF
  __int64 v117; // [rsp+78h] [rbp-168h]
  char *v118; // [rsp+80h] [rbp-160h]
  __int16 v119; // [rsp+90h] [rbp-150h]
  _QWORD *v120; // [rsp+A0h] [rbp-140h]
  __int64 v121; // [rsp+A8h] [rbp-138h]
  __int64 v122; // [rsp+B0h] [rbp-130h]
  __int64 v123; // [rsp+B8h] [rbp-128h]
  _BYTE *v124; // [rsp+C0h] [rbp-120h] BYREF
  __int64 v125; // [rsp+C8h] [rbp-118h]
  _BYTE v126[64]; // [rsp+D0h] [rbp-110h] BYREF
  __int64 v127; // [rsp+110h] [rbp-D0h] BYREF
  __int64 *v128; // [rsp+118h] [rbp-C8h]
  __int64 v129; // [rsp+120h] [rbp-C0h]
  int v130; // [rsp+128h] [rbp-B8h]
  char v131; // [rsp+12Ch] [rbp-B4h]
  char v132; // [rsp+130h] [rbp-B0h] BYREF

  v8 = *(_QWORD *)(a2 + 72);
  v9 = *(_QWORD *)(a1 + 40);
  v10 = *(_QWORD *)(a1 - 64);
  v124 = v126;
  v11 = *(_QWORD *)(v10 + 56);
  v104 = v10;
  v120 = (_QWORD *)v10;
  v121 = 0;
  v122 = 0;
  v123 = 0;
  v125 = 0x800000000LL;
  while ( 1 )
  {
    if ( !v11 )
      BUG();
    if ( *(_BYTE *)(v11 - 24) != 84 )
      break;
    v12 = *(_QWORD *)(v11 - 32);
    v13 = 0x1FFFFFFFE0LL;
    if ( (*(_DWORD *)(v11 - 20) & 0x7FFFFFF) != 0 )
    {
      v14 = 0;
      do
      {
        if ( v9 == *(_QWORD *)(v12 + 32LL * *(unsigned int *)(v11 + 48) + 8 * v14) )
        {
          v13 = 32 * v14;
          goto LABEL_9;
        }
        ++v14;
      }
      while ( (*(_DWORD *)(v11 - 20) & 0x7FFFFFF) != (_DWORD)v14 );
      v13 = 0x1FFFFFFFE0LL;
    }
LABEL_9:
    v15 = *(_QWORD *)(v12 + v13);
    v16 = (unsigned int)v125;
    v17 = (unsigned int)v125 + 1LL;
    if ( v17 > HIDWORD(v125) )
    {
      v115 = v15;
      sub_C8D5F0((__int64)&v124, v126, v17, 8u, v15, a6);
      v16 = (unsigned int)v125;
      v15 = v115;
    }
    *(_QWORD *)&v124[8 * v16] = v15;
    LODWORD(v125) = v125 + 1;
    v11 = *(_QWORD *)(v11 + 8);
  }
  v127 = 0;
  v122 = v11 - 24;
  v106 = v8 + 72;
  v18 = a2 + 24;
  v19 = v8 + 72;
  v128 = (__int64 *)&v132;
  v129 = 16;
  v130 = 0;
  v131 = 1;
  v114 = a2 + 24;
  if ( a2 + 24 != v8 + 72 )
  {
    while ( 1 )
    {
      v20 = *(_QWORD *)(v18 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v20 == v18 + 24 )
        goto LABEL_141;
      if ( !v20 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v20 - 24) - 30 > 0xA )
LABEL_141:
        BUG();
      if ( *(_BYTE *)(v20 - 24) == 34 )
      {
        v23 = sub_B4B0F0(v20 - 24);
        if ( !v131 )
          goto LABEL_26;
        v26 = v128;
        v22 = HIDWORD(v129);
        v21 = &v128[HIDWORD(v129)];
        if ( v128 == v21 )
        {
LABEL_48:
          if ( HIDWORD(v129) >= (unsigned int)v129 )
          {
LABEL_26:
            sub_C8CC70((__int64)&v127, v23, (__int64)v21, v22, v24, v25);
            v18 = *(_QWORD *)(v18 + 8);
            if ( v19 != v18 )
              goto LABEL_24;
LABEL_27:
            v27 = v128;
            if ( v131 )
              v28 = &v128[HIDWORD(v129)];
            else
              v28 = &v128[(unsigned int)v129];
            if ( v128 != v28 )
            {
              while ( 1 )
              {
                v29 = *v27;
                v30 = v27;
                if ( (unsigned __int64)*v27 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v28 == ++v27 )
                  goto LABEL_32;
              }
              if ( v28 != v27 )
              {
                v113 = v28;
                v98 = v122;
                do
                {
                  v99 = 0;
                  v111 = *(_DWORD *)(v98 + 4) & 0x7FFFFFF;
                  sub_B490C0(v29, v111);
                  if ( v111 )
                  {
                    do
                    {
                      if ( (*(_BYTE *)(v98 + 7) & 0x40) != 0 )
                        v100 = *(_QWORD *)(v98 - 8);
                      else
                        v100 = v98 - 32LL * (*(_DWORD *)(v98 + 4) & 0x7FFFFFF);
                      v101 = *(_QWORD *)(v100 + v99);
                      v99 += 32;
                      sub_B49100(v29, v101);
                    }
                    while ( v99 != 32LL * v111 );
                  }
                  if ( (*(_BYTE *)(v98 + 2) & 1) != 0 )
                    *(_WORD *)(v29 + 2) |= 1u;
                  v102 = v30 + 1;
                  if ( v30 + 1 == v113 )
                    break;
                  v29 = *v102;
                  for ( ++v30; (unsigned __int64)*v102 >= 0xFFFFFFFFFFFFFFFELL; v30 = v102 )
                  {
                    if ( v113 == ++v102 )
                      goto LABEL_32;
                    v29 = *v102;
                  }
                }
                while ( v113 != v30 );
              }
            }
LABEL_32:
            if ( *a3 )
            {
              if ( v114 )
              {
                v31 = sub_29E7990(v114 - 24, (__int64)v120, 0);
                if ( v31 )
                {
                  v32 = (unsigned int)v125;
                  v33 = v120[7];
                  if ( (_DWORD)v125 )
                    goto LABEL_36;
                }
LABEL_67:
                v112 = *(_QWORD *)(v114 + 24) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v112 == v114 + 24 )
                  goto LABEL_143;
                if ( !v112 )
                  BUG();
                if ( (unsigned int)*(unsigned __int8 *)(v112 - 24) - 30 > 0xA )
LABEL_143:
                  BUG();
                if ( *(_BYTE *)(v112 - 24) == 35 )
                {
                  v51 = v121;
                  if ( !v121 )
                  {
                    v66 = v120;
                    v67 = *(__int64 **)(v122 + 32);
                    v68 = sub_BD5D20((__int64)v120);
                    v119 = 773;
                    v69 = v67;
                    v70 = 0;
                    v116 = v68;
                    v117 = v71;
                    v118 = ".body";
                    v121 = sub_AA8550(v66, v69, 0, (__int64)&v116, 0);
                    v72 = *(_QWORD *)(v121 + 56);
                    v73 = v120[7];
                    v107 = v125;
                    if ( (_DWORD)v125 )
                    {
                      v108 = *(_QWORD *)(v121 + 56);
                      do
                      {
                        if ( !v73 )
                        {
                          v6 = sub_BD5D20(0);
                          v119 = 773;
                          v116 = v6;
                          v117 = v7;
                          v118 = ".lpad-body";
                          BUG();
                        }
                        v116 = sub_BD5D20(v73 - 24);
                        v119 = 773;
                        v117 = v79;
                        v118 = ".lpad-body";
                        v109 = *(_QWORD *)(v73 - 16);
                        v80 = sub_BD2DA0(80);
                        v81 = v80;
                        if ( v80 )
                        {
                          v82 = (_QWORD *)v80;
                          sub_B44260(v80, v109, 55, 0x8000000u, 0, 0);
                          *(_DWORD *)(v81 + 72) = 2;
                          sub_BD6B50((unsigned __int8 *)v81, &v116);
                          sub_BD2A10(v81, *(_DWORD *)(v81 + 72), 1);
                        }
                        else
                        {
                          v82 = 0;
                        }
                        sub_B44220(v82, v108, 1);
                        sub_BD84D0(v73 - 24, v81);
                        v83 = (__int64)v120;
                        v84 = *(_DWORD *)(v81 + 4) & 0x7FFFFFF;
                        if ( v84 == *(_DWORD *)(v81 + 72) )
                        {
                          v110 = (__int64)v120;
                          sub_B48D90(v81);
                          v83 = v110;
                          v84 = *(_DWORD *)(v81 + 4) & 0x7FFFFFF;
                        }
                        v74 = (v84 + 1) & 0x7FFFFFF;
                        v75 = v74 | *(_DWORD *)(v81 + 4) & 0xF8000000;
                        v76 = *(_QWORD *)(v81 - 8) + 32LL * (unsigned int)(v74 - 1);
                        *(_DWORD *)(v81 + 4) = v75;
                        if ( *(_QWORD *)v76 )
                        {
                          v77 = *(_QWORD *)(v76 + 8);
                          **(_QWORD **)(v76 + 16) = v77;
                          if ( v77 )
                            *(_QWORD *)(v77 + 16) = *(_QWORD *)(v76 + 16);
                        }
                        *(_QWORD *)v76 = v73 - 24;
                        v78 = *(_QWORD *)(v73 - 8);
                        *(_QWORD *)(v76 + 8) = v78;
                        if ( v78 )
                          *(_QWORD *)(v78 + 16) = v76 + 8;
                        *(_QWORD *)(v76 + 16) = v73 - 8;
                        ++v70;
                        *(_QWORD *)(v73 - 8) = v76;
                        *(_QWORD *)(32LL * *(unsigned int *)(v81 + 72)
                                  + 8LL * ((*(_DWORD *)(v81 + 4) & 0x7FFFFFFu) - 1)
                                  + *(_QWORD *)(v81 - 8)) = v83;
                        v73 = *(_QWORD *)(v73 + 8);
                      }
                      while ( v107 != v70 );
                      v72 = v108;
                    }
                    v116 = "eh.lpad-body";
                    v119 = 259;
                    v85 = *(_QWORD *)(v122 + 8);
                    v86 = sub_BD2DA0(80);
                    v87 = v86;
                    if ( v86 )
                    {
                      v88 = (_QWORD *)v86;
                      sub_B44260(v86, v85, 55, 0x8000000u, 0, 0);
                      *(_DWORD *)(v87 + 72) = 2;
                      sub_BD6B50((unsigned __int8 *)v87, &v116);
                      sub_BD2A10(v87, *(_DWORD *)(v87 + 72), 1);
                    }
                    else
                    {
                      v88 = 0;
                    }
                    v123 = v87;
                    sub_B44220(v88, v72, 1);
                    sub_BD84D0(v122, v123);
                    v89 = v123;
                    v90 = (__int64)v120;
                    v91 = v122;
                    v92 = *(_DWORD *)(v123 + 4) & 0x7FFFFFF;
                    if ( v92 == *(_DWORD *)(v123 + 72) )
                    {
                      sub_B48D90(v123);
                      v92 = *(_DWORD *)(v89 + 4) & 0x7FFFFFF;
                    }
                    v93 = (v92 + 1) & 0x7FFFFFF;
                    v94 = v93 | *(_DWORD *)(v89 + 4) & 0xF8000000;
                    v95 = *(_QWORD *)(v89 - 8) + 32LL * (unsigned int)(v93 - 1);
                    *(_DWORD *)(v89 + 4) = v94;
                    if ( *(_QWORD *)v95 )
                    {
                      v96 = *(_QWORD *)(v95 + 8);
                      **(_QWORD **)(v95 + 16) = v96;
                      if ( v96 )
                        *(_QWORD *)(v96 + 16) = *(_QWORD *)(v95 + 16);
                    }
                    *(_QWORD *)v95 = v91;
                    if ( v91 )
                    {
                      v97 = *(_QWORD *)(v91 + 16);
                      *(_QWORD *)(v95 + 8) = v97;
                      if ( v97 )
                        *(_QWORD *)(v97 + 16) = v95 + 8;
                      *(_QWORD *)(v95 + 16) = v91 + 16;
                      *(_QWORD *)(v91 + 16) = v95;
                    }
                    *(_QWORD *)(*(_QWORD *)(v89 - 8)
                              + 32LL * *(unsigned int *)(v89 + 72)
                              + 8LL * ((*(_DWORD *)(v89 + 4) & 0x7FFFFFFu) - 1)) = v90;
                    v51 = v121;
                  }
                  v52 = *(_QWORD *)(v112 + 16);
                  sub_B43C20((__int64)&v116, v52);
                  v53 = v116;
                  v54 = v117;
                  v55 = sub_BD2C40(72, 1u);
                  if ( v55 )
                    sub_B4C8F0((__int64)v55, v51, 1u, (__int64)v53, v54);
                  v56 = *(_QWORD *)(v51 + 56);
                  v57 = 0;
                  v58 = 8LL * (unsigned int)v125;
                  if ( (_DWORD)v125 )
                  {
                    do
                    {
                      if ( !v56 )
                        BUG();
                      v64 = *(_QWORD *)&v124[v57];
                      v65 = *(_DWORD *)(v56 - 20) & 0x7FFFFFF;
                      if ( v65 == *(_DWORD *)(v56 + 48) )
                      {
                        sub_B48D90(v56 - 24);
                        v65 = *(_DWORD *)(v56 - 20) & 0x7FFFFFF;
                      }
                      v59 = (v65 + 1) & 0x7FFFFFF;
                      v60 = v59 | *(_DWORD *)(v56 - 20) & 0xF8000000;
                      v61 = *(_QWORD *)(v56 - 32) + 32LL * (unsigned int)(v59 - 1);
                      *(_DWORD *)(v56 - 20) = v60;
                      if ( *(_QWORD *)v61 )
                      {
                        v62 = *(_QWORD *)(v61 + 8);
                        **(_QWORD **)(v61 + 16) = v62;
                        if ( v62 )
                          *(_QWORD *)(v62 + 16) = *(_QWORD *)(v61 + 16);
                      }
                      *(_QWORD *)v61 = v64;
                      if ( v64 )
                      {
                        v63 = *(_QWORD *)(v64 + 16);
                        *(_QWORD *)(v61 + 8) = v63;
                        if ( v63 )
                          *(_QWORD *)(v63 + 16) = v61 + 8;
                        *(_QWORD *)(v61 + 16) = v64 + 16;
                        *(_QWORD *)(v64 + 16) = v61;
                      }
                      v57 += 8;
                      *(_QWORD *)(*(_QWORD *)(v56 - 32)
                                + 32LL * *(unsigned int *)(v56 + 48)
                                + 8LL * ((*(_DWORD *)(v56 - 20) & 0x7FFFFFFu) - 1)) = v52;
                      v56 = *(_QWORD *)(v56 + 8);
                    }
                    while ( v58 != v57 );
                  }
                  v43 = v123;
                  v44 = *(_QWORD *)(v112 - 56);
                  v45 = *(_DWORD *)(v123 + 4) & 0x7FFFFFF;
                  if ( v45 == *(_DWORD *)(v123 + 72) )
                  {
                    sub_B48D90(v123);
                    v45 = *(_DWORD *)(v43 + 4) & 0x7FFFFFF;
                  }
                  v46 = (v45 + 1) & 0x7FFFFFF;
                  v47 = v46 | *(_DWORD *)(v43 + 4) & 0xF8000000;
                  v48 = *(_QWORD *)(v43 - 8) + 32LL * (unsigned int)(v46 - 1);
                  *(_DWORD *)(v43 + 4) = v47;
                  if ( *(_QWORD *)v48 )
                  {
                    v49 = *(_QWORD *)(v48 + 8);
                    **(_QWORD **)(v48 + 16) = v49;
                    if ( v49 )
                      *(_QWORD *)(v49 + 16) = *(_QWORD *)(v48 + 16);
                  }
                  *(_QWORD *)v48 = v44;
                  if ( v44 )
                  {
                    v50 = *(_QWORD *)(v44 + 16);
                    *(_QWORD *)(v48 + 8) = v50;
                    if ( v50 )
                      *(_QWORD *)(v50 + 16) = v48 + 8;
                    *(_QWORD *)(v48 + 16) = v44 + 16;
                    *(_QWORD *)(v44 + 16) = v48;
                  }
                  *(_QWORD *)(*(_QWORD *)(v43 - 8)
                            + 32LL * *(unsigned int *)(v43 + 72)
                            + 8LL * ((*(_DWORD *)(v43 + 4) & 0x7FFFFFFu) - 1)) = v52;
                  sub_B43D60((_QWORD *)(v112 - 24));
                }
                v114 = *(_QWORD *)(v114 + 8);
                if ( v106 == v114 )
                  break;
                goto LABEL_32;
              }
              v31 = sub_29E7990(0, (__int64)v120, 0);
              if ( !v31 || (v32 = (unsigned int)v125, v33 = v120[7], !(_DWORD)v125) )
LABEL_139:
                BUG();
LABEL_36:
              v34 = 8 * v32;
              v35 = 0;
              do
              {
                if ( !v33 )
                  BUG();
                v41 = *(_QWORD *)&v124[v35];
                v42 = *(_DWORD *)(v33 - 20) & 0x7FFFFFF;
                if ( v42 == *(_DWORD *)(v33 + 48) )
                {
                  sub_B48D90(v33 - 24);
                  v42 = *(_DWORD *)(v33 - 20) & 0x7FFFFFF;
                }
                v36 = (v42 + 1) & 0x7FFFFFF;
                v37 = v36 | *(_DWORD *)(v33 - 20) & 0xF8000000;
                v38 = *(_QWORD *)(v33 - 32) + 32LL * (unsigned int)(v36 - 1);
                *(_DWORD *)(v33 - 20) = v37;
                if ( *(_QWORD *)v38 )
                {
                  v39 = *(_QWORD *)(v38 + 8);
                  **(_QWORD **)(v38 + 16) = v39;
                  if ( v39 )
                    *(_QWORD *)(v39 + 16) = *(_QWORD *)(v38 + 16);
                }
                *(_QWORD *)v38 = v41;
                if ( v41 )
                {
                  v40 = *(_QWORD *)(v41 + 16);
                  *(_QWORD *)(v38 + 8) = v40;
                  if ( v40 )
                    *(_QWORD *)(v40 + 16) = v38 + 8;
                  *(_QWORD *)(v38 + 16) = v41 + 16;
                  *(_QWORD *)(v41 + 16) = v38;
                }
                v35 += 8;
                *(_QWORD *)(*(_QWORD *)(v33 - 32)
                          + 32LL * *(unsigned int *)(v33 + 48)
                          + 8LL * ((*(_DWORD *)(v33 - 20) & 0x7FFFFFFu) - 1)) = v31;
                v33 = *(_QWORD *)(v33 + 8);
              }
              while ( v35 != v34 );
            }
            if ( !v114 )
              goto LABEL_139;
            goto LABEL_67;
          }
          ++HIDWORD(v129);
          *v21 = v23;
          ++v127;
        }
        else
        {
          while ( v23 != *v26 )
          {
            if ( v21 == ++v26 )
              goto LABEL_48;
          }
        }
      }
      v18 = *(_QWORD *)(v18 + 8);
      if ( v19 == v18 )
        goto LABEL_27;
LABEL_24:
      if ( !v18 )
        BUG();
    }
  }
  sub_AA5980(v104, *(_QWORD *)(a1 + 40), 0);
  if ( !v131 )
    _libc_free((unsigned __int64)v128);
  if ( v124 != v126 )
    _libc_free((unsigned __int64)v124);
}
