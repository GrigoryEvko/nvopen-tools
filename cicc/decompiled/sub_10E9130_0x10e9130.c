// Function: sub_10E9130
// Address: 0x10e9130
//
__int64 __fastcall sub_10E9130(_QWORD *a1, unsigned __int8 *a2)
{
  __int64 v3; // r12
  unsigned __int8 *v4; // rax
  __int64 v5; // r14
  unsigned int v6; // r15d
  unsigned int v8; // eax
  __int64 v9; // rcx
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  char v14; // bl
  int v15; // edx
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rbx
  int v21; // ebx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rbx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  unsigned int v28; // ebx
  __int64 v29; // rax
  int v30; // ebx
  __int64 v31; // rax
  __int64 v32; // rax
  char v33; // r13
  char v34; // r12
  int v35; // r12d
  __int64 v36; // r13
  __int64 v37; // rax
  __int64 v38; // rax
  const char **v39; // rsi
  unsigned __int64 v40; // r9
  __int64 v41; // rbx
  int v42; // r14d
  __int64 v43; // r12
  __int64 v44; // r8
  __int64 v45; // rsi
  __int64 v46; // rdx
  __int64 *v47; // rdi
  __int64 v48; // rax
  unsigned __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // r9
  __int64 v53; // rdx
  unsigned __int64 v54; // r8
  __int64 v55; // rbx
  __int64 v56; // rax
  __int64 v57; // r8
  __int64 v58; // rdx
  __int64 v59; // rax
  unsigned __int64 v60; // rdx
  int v61; // ebx
  __int64 *v62; // r12
  __int64 v63; // rax
  unsigned __int64 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // rdx
  unsigned __int64 v67; // r8
  __int64 v68; // rax
  __int64 v69; // rax
  int v70; // eax
  __int64 *v71; // rdi
  __int64 v72; // rax
  unsigned __int64 v73; // rbx
  unsigned __int64 v74; // rax
  unsigned __int64 v75; // rax
  bool v76; // zf
  unsigned int **v77; // rdi
  __int64 v78; // r15
  __int64 v79; // rax
  __int64 v80; // rsi
  __int64 *v81; // r14
  __int64 v82; // rax
  __int64 v83; // rsi
  const char *v84; // r13
  const char *v85; // r12
  __int64 v86; // rdi
  __int64 v87; // r8
  __int64 v88; // rsi
  unsigned __int8 *v89; // rsi
  unsigned __int64 v90; // [rsp-10h] [rbp-280h]
  __int64 v91; // [rsp+8h] [rbp-268h]
  __int64 v92; // [rsp+8h] [rbp-268h]
  __int64 v93; // [rsp+10h] [rbp-260h]
  __int64 v94; // [rsp+18h] [rbp-258h]
  __int64 *v95; // [rsp+20h] [rbp-250h]
  int v96; // [rsp+44h] [rbp-22Ch]
  unsigned int v97; // [rsp+48h] [rbp-228h]
  __int64 v98; // [rsp+58h] [rbp-218h]
  __int64 v99; // [rsp+60h] [rbp-210h]
  unsigned int v100; // [rsp+78h] [rbp-1F8h]
  __int64 v101; // [rsp+78h] [rbp-1F8h]
  unsigned __int8 *v102; // [rsp+80h] [rbp-1F0h]
  __int64 *v103; // [rsp+80h] [rbp-1F0h]
  unsigned int v104; // [rsp+88h] [rbp-1E8h]
  _QWORD *v105; // [rsp+88h] [rbp-1E8h]
  int v106; // [rsp+90h] [rbp-1E0h]
  __int64 v107; // [rsp+A0h] [rbp-1D0h]
  __int64 v108; // [rsp+A0h] [rbp-1D0h]
  __int64 *v109; // [rsp+A0h] [rbp-1D0h]
  __int64 v110; // [rsp+A0h] [rbp-1D0h]
  _QWORD *v111; // [rsp+A0h] [rbp-1D0h]
  unsigned __int64 v112; // [rsp+A0h] [rbp-1D0h]
  __int64 v113; // [rsp+A0h] [rbp-1D0h]
  __int64 v114; // [rsp+A0h] [rbp-1D0h]
  __int64 v115; // [rsp+A8h] [rbp-1C8h]
  __int64 v116; // [rsp+A8h] [rbp-1C8h]
  unsigned __int64 v117; // [rsp+A8h] [rbp-1C8h]
  char *v118; // [rsp+A8h] [rbp-1C8h]
  __int64 v119; // [rsp+B0h] [rbp-1C0h] BYREF
  __int64 v120; // [rsp+B8h] [rbp-1B8h] BYREF
  __int64 v121[4]; // [rsp+C0h] [rbp-1B0h] BYREF
  __int16 v122; // [rsp+E0h] [rbp-190h]
  const char *v123; // [rsp+F0h] [rbp-180h] BYREF
  __int64 v124; // [rsp+F8h] [rbp-178h]
  _BYTE v125[16]; // [rsp+100h] [rbp-170h] BYREF
  _QWORD *v126; // [rsp+110h] [rbp-160h]
  _BYTE *v127; // [rsp+140h] [rbp-130h] BYREF
  __int64 v128; // [rsp+148h] [rbp-128h]
  _BYTE v129[64]; // [rsp+150h] [rbp-120h] BYREF
  _BYTE *v130; // [rsp+190h] [rbp-E0h] BYREF
  __int64 v131; // [rsp+198h] [rbp-D8h]
  _BYTE v132[16]; // [rsp+1A0h] [rbp-D0h] BYREF
  _QWORD *v133; // [rsp+1B0h] [rbp-C0h]
  __int64 v134; // [rsp+1E0h] [rbp-90h] BYREF
  _BYTE *v135; // [rsp+1E8h] [rbp-88h]
  _BYTE v136[120]; // [rsp+1F8h] [rbp-78h] BYREF

  v3 = (__int64)a2;
  v4 = sub_BD3990(*((unsigned __int8 **)a2 - 4), (__int64)a2);
  if ( *v4 )
    return 0;
  v5 = (__int64)v4;
  if ( sub_B2FC80((__int64)v4) )
    return 0;
  if ( (unsigned __int8)sub_B2D620(v5, "thunk", 5u) )
    return 0;
  if ( (unsigned __int8)sub_B2D610(v5, 20) )
    return 0;
  LOBYTE(v8) = sub_B49200((__int64)a2);
  v6 = v8;
  if ( (_BYTE)v8 )
    return 0;
  v9 = *((_QWORD *)a2 + 1);
  v119 = *((_QWORD *)a2 + 9);
  v98 = v9;
  v115 = *(_QWORD *)(v5 + 24);
  v10 = **(_QWORD **)(v115 + 16);
  v99 = v10;
  if ( v9 == v10 )
    goto LABEL_26;
  if ( *(_BYTE *)(v10 + 8) == 15 )
    return 0;
  if ( !(unsigned __int8)sub_B50C50(v10, v9, a1[11]) )
  {
    if ( *((_QWORD *)a2 + 2) )
      return 0;
    goto LABEL_26;
  }
  v11 = *((_QWORD *)a2 + 2);
  if ( !v119 )
    goto LABEL_17;
  if ( !v11 )
  {
LABEL_26:
    v15 = *a2;
    goto LABEL_27;
  }
  v12 = sub_A74610(&v119);
  sub_A74940((__int64)&v134, *(_QWORD *)v115, v12);
  v13 = sub_A74610(&v119);
  sub_A751C0((__int64)&v130, v10, v13, 3);
  v14 = sub_A74BD0((__int64)&v134, (__int64)&v130);
  sub_10DF690(v133, (__int64)&v130);
  if ( !v14 )
  {
    if ( v135 != v136 )
      _libc_free(v135, &v130);
    v11 = *((_QWORD *)a2 + 2);
LABEL_17:
    v15 = *a2;
    if ( v11 && (_BYTE)v15 == 34 )
    {
      v16 = *((_QWORD *)a2 - 12);
      if ( v16 )
      {
        while ( 1 )
        {
          v17 = *(_QWORD *)(v11 + 24);
          if ( *(_BYTE *)v17 == 84 && *(_QWORD *)(v17 + 40) == v16 )
            return 0;
          v11 = *(_QWORD *)(v11 + 8);
          if ( !v11 )
            goto LABEL_52;
        }
      }
      goto LABEL_52;
    }
LABEL_27:
    if ( v15 == 40 )
    {
      v107 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
LABEL_29:
      if ( (a2[7] & 0x80u) != 0 )
      {
        v18 = sub_BD2BC0((__int64)a2);
        v20 = v18 + v19;
        if ( (a2[7] & 0x80u) == 0 )
        {
          if ( (unsigned int)(v20 >> 4) )
            goto LABEL_146;
        }
        else if ( (unsigned int)((v20 - sub_BD2BC0((__int64)a2)) >> 4) )
        {
          if ( (a2[7] & 0x80u) != 0 )
          {
            v21 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
            if ( (a2[7] & 0x80u) == 0 )
              BUG();
            v22 = sub_BD2BC0((__int64)a2);
            v24 = 32LL * (unsigned int)(*(_DWORD *)(v22 + v23 - 4) - v21);
LABEL_35:
            v104 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
            v106 = *(_DWORD *)(v115 + 12);
            v130 = *(_BYTE **)(v5 + 120);
            if ( (unsigned __int8)sub_A74390((__int64 *)&v130, 83, 0) )
              return v6;
            v134 = *(_QWORD *)(v5 + 120);
            if ( (unsigned __int8)sub_A74390(&v134, 84, 0) )
              return v6;
            v27 = 32LL * v104 - 32 - v107 - v24;
            v28 = v106 - 1;
            v29 = v27 >> 5;
            v97 = v29;
            if ( v106 - 1 > (unsigned int)v29 )
              v28 = v29;
            v96 = v29;
            v100 = v28;
            v102 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
            if ( v28 )
            {
              v30 = 0;
              v108 = 8;
              v105 = a1;
              while ( 1 )
              {
                v35 = v30++;
                v36 = *(_QWORD *)(*(_QWORD *)(v115 + 16) + v108);
                if ( !(unsigned __int8)sub_B50C50(*(_QWORD *)(*(_QWORD *)&v102[4 * v108 - 32] + 8LL), v36, v105[11]) )
                  return 0;
                v31 = sub_A744E0(&v119, v35);
                sub_A74940((__int64)&v134, *(_QWORD *)v115, v31);
                v32 = sub_A744E0(&v119, v35);
                sub_A751C0((__int64)&v130, v36, v32, 2);
                v33 = sub_A74BD0((__int64)&v134, (__int64)&v130);
                sub_10DF690(v133, (__int64)&v130);
                if ( v135 != v136 )
                  _libc_free(v135, &v130);
                if ( v33 )
                  return 0;
                if ( (unsigned __int8)sub_B49B80((__int64)a2, v35, 83) )
                  return 0;
                if ( (unsigned __int8)sub_A74710(&v119, v30, 84) )
                  return 0;
                if ( (unsigned __int8)sub_A74710(&v119, v30, 74) )
                  return 0;
                v34 = sub_A74710(&v119, v30, 81);
                v134 = *(_QWORD *)(v5 + 120);
                if ( (unsigned __int8)sub_A74710(&v134, v30, 81) != v34 )
                  return 0;
                v108 += 8;
                if ( v30 == v100 )
                {
                  a1 = v105;
                  v3 = (__int64)a2;
                  break;
                }
              }
            }
            if ( *(_DWORD *)(v115 + 12) - 1 >= v97
              || !(*(_DWORD *)(v115 + 8) >> 8)
              || !v119
              || !(unsigned __int8)sub_A74390(&v119, 85, &v134)
              || (int)v134 - 1 < (unsigned int)(*(_DWORD *)(v115 + 12) - 1) )
            {
              v127 = v129;
              v128 = 0x800000000LL;
              v130 = v132;
              v131 = 0x800000000LL;
              if ( v97 > 8uLL )
                sub_C8D5F0((__int64)&v127, v129, v97, 8u, v25, v26);
              if ( v97 > (unsigned __int64)HIDWORD(v131) )
                sub_C8D5F0((__int64)&v130, v132, v97, 8u, v25, v26);
              v37 = sub_A74610(&v119);
              sub_A74940((__int64)&v134, *(_QWORD *)v115, v37);
              v38 = sub_A74610(&v119);
              sub_A751C0((__int64)&v123, v10, v38, 3);
              v39 = &v123;
              sub_A74A10((__int64)&v134, (__int64)&v123);
              sub_10DF690(v126, (__int64)&v123);
              v103 = (__int64 *)sub_BD5C60(v3);
              v95 = (__int64 *)(v3 - 32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF));
              if ( v100 )
              {
                v94 = v5;
                v41 = 0;
                v93 = v3;
                v109 = (__int64 *)(v3 - 32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF));
                do
                {
                  v42 = v41++;
                  v43 = *(_QWORD *)(*(_QWORD *)(v115 + 16) + 8 * v41);
                  v44 = *v109;
                  if ( *(_QWORD *)(*v109 + 8) != v43 )
                  {
                    v45 = *v109;
                    v46 = *(_QWORD *)(*(_QWORD *)(v115 + 16) + 8 * v41);
                    v47 = (__int64 *)a1[4];
                    LOWORD(v126) = 257;
                    v44 = sub_10E0940(v47, v45, v46, (__int64)&v123);
                  }
                  v48 = (unsigned int)v128;
                  v49 = (unsigned int)v128 + 1LL;
                  if ( v49 > HIDWORD(v128) )
                  {
                    v92 = v44;
                    sub_C8D5F0((__int64)&v127, v129, v49, 8u, v44, v40);
                    v48 = (unsigned int)v128;
                    v44 = v92;
                  }
                  *(_QWORD *)&v127[8 * v48] = v44;
                  LODWORD(v128) = v128 + 1;
                  v50 = sub_A744E0(&v119, v42);
                  sub_A751C0((__int64)&v123, v43, v50, 1);
                  v39 = (const char **)v103;
                  v121[0] = sub_A744E0(&v119, v42);
                  v51 = sub_A7A3C0(v121, v103, (__int64)&v123);
                  v53 = (unsigned int)v131;
                  v54 = (unsigned int)v131 + 1LL;
                  if ( v54 > HIDWORD(v131) )
                  {
                    v39 = (const char **)v132;
                    v91 = v51;
                    sub_C8D5F0((__int64)&v130, v132, (unsigned int)v131 + 1LL, 8u, v54, v52);
                    v53 = (unsigned int)v131;
                    v51 = v91;
                  }
                  *(_QWORD *)&v130[8 * v53] = v51;
                  LODWORD(v131) = v131 + 1;
                  sub_10DF690(v126, (__int64)v39);
                  v109 += 4;
                }
                while ( v100 != v41 );
                v5 = v94;
                v3 = v93;
                v95 += 4 * v100;
              }
              if ( *(_DWORD *)(v115 + 12) - 1 != v100 )
              {
                LODWORD(v55) = v100;
                do
                {
                  v55 = (unsigned int)(v55 + 1);
                  v56 = sub_AD6530(*(_QWORD *)(*(_QWORD *)(v115 + 16) + 8 * v55), (__int64)v39);
                  v58 = (unsigned int)v128;
                  v40 = (unsigned int)v128 + 1LL;
                  if ( v40 > HIDWORD(v128) )
                  {
                    v39 = (const char **)v129;
                    v101 = v56;
                    sub_C8D5F0((__int64)&v127, v129, (unsigned int)v128 + 1LL, 8u, v57, v40);
                    v58 = (unsigned int)v128;
                    v56 = v101;
                  }
                  *(_QWORD *)&v127[8 * v58] = v56;
                  v59 = (unsigned int)v131;
                  LODWORD(v128) = v128 + 1;
                  v60 = (unsigned int)v131 + 1LL;
                  if ( v60 > HIDWORD(v131) )
                  {
                    v39 = (const char **)v132;
                    sub_C8D5F0((__int64)&v130, v132, v60, 8u, v57, v40);
                    v59 = (unsigned int)v131;
                  }
                  *(_QWORD *)&v130[8 * v59] = 0;
                  LODWORD(v131) = v131 + 1;
                }
                while ( *(_DWORD *)(v115 + 12) - 1 != (_DWORD)v55 );
                v100 = v55;
              }
              if ( v97 > v100 && *(_DWORD *)(v115 + 8) >> 8 )
              {
                v116 = v3;
                v61 = v100;
                v62 = v95;
                do
                {
                  v87 = *v62;
                  v68 = *(_QWORD *)(*v62 + 8);
                  if ( *(_BYTE *)(v68 + 8) == 12 && *(_DWORD *)(v68 + 8) <= 0x1FFFu )
                  {
                    v69 = sub_BCB2D0(*(_QWORD **)v68);
                    v87 = *v62;
                    if ( *(_QWORD *)(*v62 + 8) != v69 )
                    {
                      v110 = v69;
                      v70 = sub_B50D10(*v62, 0, v69, 0);
                      v71 = (__int64 *)a1[4];
                      LOWORD(v126) = 257;
                      v72 = sub_10E0690(v71, v70, *v62, v110, (__int64)&v123, 0, v121[0], 0);
                      v40 = v90;
                      v87 = v72;
                    }
                  }
                  v63 = (unsigned int)v128;
                  v64 = (unsigned int)v128 + 1LL;
                  if ( v64 > HIDWORD(v128) )
                  {
                    v114 = v87;
                    sub_C8D5F0((__int64)&v127, v129, v64, 8u, v87, v40);
                    v63 = (unsigned int)v128;
                    v87 = v114;
                  }
                  *(_QWORD *)&v127[8 * v63] = v87;
                  LODWORD(v128) = v128 + 1;
                  v65 = sub_A744E0(&v119, v61);
                  v66 = (unsigned int)v131;
                  v67 = (unsigned int)v131 + 1LL;
                  if ( v67 > HIDWORD(v131) )
                  {
                    v113 = v65;
                    sub_C8D5F0((__int64)&v130, v132, (unsigned int)v131 + 1LL, 8u, v67, v40);
                    v66 = (unsigned int)v131;
                    v65 = v113;
                  }
                  ++v61;
                  v62 += 4;
                  *(_QWORD *)&v130[8 * v66] = v65;
                  LODWORD(v131) = v131 + 1;
                }
                while ( v61 != v96 );
                v3 = v116;
              }
              v117 = sub_A74680(&v119);
              if ( *(_BYTE *)(v99 + 8) == 7 )
              {
                LOWORD(v126) = 257;
                sub_BD6B50((unsigned __int8 *)v3, &v123);
              }
              v73 = (unsigned int)v131;
              v111 = v130;
              v74 = sub_A7A280(v103, (__int64)&v134);
              v75 = sub_A78180(v103, v117, v74, v111, v73);
              v123 = v125;
              v112 = v75;
              v124 = 0x100000000LL;
              sub_B56970(v3, (__int64)&v123);
              v76 = *(_BYTE *)v3 == 34;
              v77 = (unsigned int **)a1[4];
              v122 = 257;
              if ( v76 )
              {
                v78 = sub_B33310(
                        v77,
                        *(_QWORD *)(v5 + 24),
                        v5,
                        *(_QWORD *)(v3 - 96),
                        *(_QWORD *)(v3 - 64),
                        (__int64)v121,
                        (__int64)v127,
                        (unsigned int)v128,
                        (__int64)v123,
                        (unsigned int)v124);
              }
              else
              {
                v78 = sub_B33530(
                        v77,
                        *(_QWORD *)(v5 + 24),
                        v5,
                        (int)v127,
                        v128,
                        (__int64)v121,
                        (__int64)v123,
                        (unsigned int)v124,
                        0);
                *(_WORD *)(v78 + 2) = *(_WORD *)(v3 + 2) & 3 | *(_WORD *)(v78 + 2) & 0xFFFC;
              }
              v118 = (char *)v78;
              sub_BD6B90((unsigned __int8 *)v78, (unsigned __int8 *)v3);
              *(_WORD *)(v78 + 2) = *(_WORD *)(v3 + 2) & 0xFFC | *(_WORD *)(v78 + 2) & 0xF003;
              *(_QWORD *)(v78 + 72) = v112;
              LODWORD(v121[0]) = 2;
              sub_B47C00(v78, v3, (int *)v121, 1);
              if ( v98 == *(_QWORD *)(v78 + 8) )
                goto LABEL_110;
              if ( !*(_QWORD *)(v3 + 16) )
              {
                if ( (*(_BYTE *)(v3 + 1) & 1) == 0 )
                  goto LABEL_112;
                goto LABEL_130;
              }
              v122 = 257;
              v79 = sub_B52260(v78, v98, (__int64)v121, 0, 0);
              v80 = *(_QWORD *)(v3 + 48);
              v78 = v79;
              v121[0] = v80;
              if ( v80 )
              {
                v81 = (__int64 *)(v79 + 48);
                sub_B96E90((__int64)v121, v80, 1);
                if ( v81 == v121 )
                {
                  if ( v121[0] )
                    sub_B91220((__int64)v121, v121[0]);
                  goto LABEL_109;
                }
                v88 = *(_QWORD *)(v78 + 48);
                if ( !v88 )
                {
LABEL_137:
                  v89 = (unsigned __int8 *)v121[0];
                  *(_QWORD *)(v78 + 48) = v121[0];
                  if ( v89 )
                    sub_B976B0((__int64)v121, v89, (__int64)v81);
                  goto LABEL_109;
                }
              }
              else
              {
                v81 = (__int64 *)(v79 + 48);
                if ( (__int64 *)(v79 + 48) == v121 || (v88 = *(_QWORD *)(v79 + 48)) == 0 )
                {
LABEL_109:
                  sub_B445D0((__int64)v121, v118);
                  sub_B44220((_QWORD *)v78, v121[0], v121[1]);
                  v82 = a1[5];
                  v120 = v78;
                  sub_10E8740(v82 + 2096, &v120);
                  sub_10A5FE0(a1[5], v3);
LABEL_110:
                  if ( *(_QWORD *)(v3 + 16) )
                  {
                    sub_F162A0((__int64)a1, v3, v78);
LABEL_112:
                    v83 = v3;
                    sub_F207A0((__int64)a1, (__int64 *)v3);
                    v84 = v123;
                    v85 = &v123[56 * (unsigned int)v124];
                    if ( v123 != v85 )
                    {
                      do
                      {
                        v86 = *((_QWORD *)v85 - 3);
                        v85 -= 56;
                        if ( v86 )
                        {
                          v83 = *((_QWORD *)v85 + 6) - v86;
                          j_j___libc_free_0(v86, v83);
                        }
                        if ( *(const char **)v85 != v85 + 16 )
                        {
                          v83 = *((_QWORD *)v85 + 2) + 1LL;
                          j_j___libc_free_0(*(_QWORD *)v85, v83);
                        }
                      }
                      while ( v84 != v85 );
                      v84 = v123;
                    }
                    if ( v84 != v125 )
                      _libc_free(v84, v83);
                    if ( v135 != v136 )
                      _libc_free(v135, v83);
                    if ( v130 != v132 )
                      _libc_free(v130, v83);
                    if ( v127 != v129 )
                      _libc_free(v127, v83);
                    return 1;
                  }
                  if ( (*(_BYTE *)(v3 + 1) & 1) == 0 )
                    goto LABEL_112;
                  if ( v98 == *(_QWORD *)(v78 + 8) )
                  {
                    sub_BD7FF0(v3, v78);
                    goto LABEL_112;
                  }
LABEL_130:
                  sub_BD6EE0(v3);
                  goto LABEL_112;
                }
              }
              sub_B91220((__int64)v81, v88);
              goto LABEL_137;
            }
            return 0;
          }
LABEL_146:
          BUG();
        }
      }
      v24 = 0;
      goto LABEL_35;
    }
    v107 = 0;
    if ( v15 == 85 )
      goto LABEL_29;
    if ( v15 != 34 )
      BUG();
LABEL_52:
    v107 = 64;
    goto LABEL_29;
  }
  if ( v135 != v136 )
    _libc_free(v135, &v130);
  return v6;
}
