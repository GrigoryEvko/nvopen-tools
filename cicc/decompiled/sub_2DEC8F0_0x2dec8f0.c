// Function: sub_2DEC8F0
// Address: 0x2dec8f0
//
__int64 __fastcall sub_2DEC8F0(unsigned __int8 *a1, _QWORD *a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 v7; // r13
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rbx
  char v14; // r12
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // r9
  unsigned __int64 *v18; // r13
  unsigned __int64 *v19; // rbx
  unsigned __int64 v20; // rdi
  unsigned __int8 *v21; // rbx
  __int64 v22; // r12
  _DWORD *v23; // rax
  unsigned int v24; // r13d
  unsigned __int64 v25; // rbx
  unsigned __int64 v26; // r13
  unsigned __int64 v27; // rdi
  unsigned int v28; // eax
  unsigned __int64 v29; // r14
  __int64 v30; // rax
  _QWORD *v31; // rax
  _QWORD *v32; // rdx
  _QWORD *v33; // r13
  bool v34; // r8
  __int64 v35; // rax
  _QWORD *v36; // rax
  _QWORD *v37; // rdx
  _QWORD *v38; // r12
  char v39; // r13
  __int64 v40; // rax
  unsigned int v41; // ebx
  _QWORD *v42; // rax
  __int64 v43; // rax
  _QWORD *v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rsi
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rax
  __int64 v52; // r14
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // r12
  unsigned __int64 v58; // rdi
  unsigned __int64 v59; // r14
  unsigned __int64 v60; // r12
  unsigned __int64 v61; // rdi
  __int64 v62; // r14
  unsigned __int64 v63; // r12
  unsigned __int64 v64; // rdi
  _QWORD *v65; // rdx
  __int64 v66; // rcx
  _BYTE *v67; // r8
  __int64 v68; // r9
  __int64 v69; // rax
  __int64 v70; // rcx
  unsigned __int8 *v71; // r12
  __int64 i; // rbx
  unsigned int v73; // r14d
  __int64 v74; // rax
  unsigned __int64 v75; // rdx
  unsigned __int64 v76; // r12
  unsigned __int64 v77; // r13
  unsigned __int64 v78; // rdi
  unsigned __int8 *v79; // rbx
  __int64 v80; // rax
  unsigned __int64 v81; // rdx
  __int64 v82; // r14
  char v83; // r12
  __int64 v84; // rax
  unsigned __int64 v85; // rdx
  unsigned int v86; // r14d
  unsigned int v87; // eax
  unsigned int v88; // eax
  __int64 v89; // r9
  __int64 v90; // r9
  unsigned __int64 *v91; // r13
  unsigned __int64 v92; // rbx
  unsigned __int64 v93; // rdi
  __int64 v94; // rdx
  unsigned __int64 v95; // rax
  unsigned __int64 *v96; // r13
  unsigned __int64 v97; // r12
  unsigned __int64 v98; // rdi
  unsigned __int64 v99; // rdx
  unsigned __int64 v100; // r8
  unsigned __int64 *v101; // rcx
  __int64 v102; // rax
  unsigned __int64 v103; // rsi
  unsigned __int64 v104; // rax
  int v105; // edx
  unsigned __int64 v106; // rdx
  unsigned __int64 v107; // r8
  unsigned __int64 *v108; // rcx
  __int64 v109; // rax
  unsigned __int64 v110; // rsi
  unsigned __int64 v111; // rax
  int v112; // edx
  unsigned __int64 *v113; // r13
  unsigned __int64 v114; // r12
  unsigned __int64 v115; // rdi
  _BYTE *v116; // [rsp+8h] [rbp-248h]
  __int64 v117; // [rsp+10h] [rbp-240h]
  __int64 v118; // [rsp+10h] [rbp-240h]
  __int64 v119; // [rsp+10h] [rbp-240h]
  __int64 v120; // [rsp+10h] [rbp-240h]
  __int64 v121; // [rsp+28h] [rbp-228h]
  char v123; // [rsp+40h] [rbp-210h]
  int v124; // [rsp+40h] [rbp-210h]
  __int64 v125; // [rsp+48h] [rbp-208h] BYREF
  _QWORD *v126; // [rsp+50h] [rbp-200h] BYREF
  __int64 v127; // [rsp+58h] [rbp-1F8h]
  unsigned int v128; // [rsp+60h] [rbp-1F0h] BYREF
  unsigned __int64 v129; // [rsp+68h] [rbp-1E8h]
  unsigned __int64 *v130; // [rsp+70h] [rbp-1E0h] BYREF
  __int64 v131; // [rsp+78h] [rbp-1D8h]
  _BYTE v132[96]; // [rsp+80h] [rbp-1D0h] BYREF
  char *v133; // [rsp+E0h] [rbp-170h] BYREF
  unsigned int v134; // [rsp+E8h] [rbp-168h]
  _QWORD *v135; // [rsp+F0h] [rbp-160h] BYREF
  unsigned __int64 v136; // [rsp+F8h] [rbp-158h]
  _BYTE *v137; // [rsp+100h] [rbp-150h] BYREF
  __int64 v138; // [rsp+108h] [rbp-148h]
  _BYTE v139[96]; // [rsp+110h] [rbp-140h] BYREF
  unsigned __int64 v140; // [rsp+170h] [rbp-E0h] BYREF
  unsigned int v141; // [rsp+178h] [rbp-D8h]
  unsigned __int64 v142[18]; // [rsp+180h] [rbp-D0h] BYREF
  __int64 v143; // [rsp+210h] [rbp-40h]

  v130 = (unsigned __int64 *)v132;
  v125 = (__int64)a1;
  v128 = -1;
  v129 = 0;
  v131 = 0x400000000LL;
  v134 = 1;
  v133 = 0;
  v3 = a1[2] & 1;
  if ( (a1[2] & 1) != 0 )
  {
    return 0;
  }
  else if ( !sub_B46500(a1) )
  {
    v7 = *(_QWORD *)(a2[17] + 24LL);
    v8 = sub_9208B0(a3, v7);
    v10 = v9;
    v11 = v8;
    v12 = v10;
    v142[0] = v11;
    v13 = v11 + 7;
    v142[1] = v12;
    v14 = v12;
    v142[0] = sub_9208B0(a3, v7);
    v142[1] = v15;
    if ( v142[0] == (v13 & 0xFFFFFFFFFFFFFFF8LL) && LOBYTE(v142[1]) == v14 )
    {
      v21 = *(unsigned __int8 **)(v125 - 32);
      v22 = *((_QWORD *)v21 + 1);
      if ( *(_BYTE *)(v22 + 8) == 14 )
      {
        while ( 1 )
        {
          v23 = sub_AE2980(a3, *(_DWORD *)(v22 + 8) >> 8);
          v15 = *v21;
          if ( (unsigned __int8)v15 <= 0x1Cu )
            goto LABEL_38;
          v24 = v23[3];
          if ( (unsigned int)(unsigned __int8)v15 - 67 > 0xC )
            break;
          if ( (_BYTE)v15 != 78 )
          {
            LODWORD(v142[17]) = v23[3];
            v142[2] = (unsigned __int64)&v142[4];
            LODWORD(v142[0]) = 0;
            v142[1] = 0;
            v142[3] = 0x400000000LL;
            if ( v24 > 0x40 )
            {
              sub_C43690((__int64)&v142[16], 0, 0);
              if ( LODWORD(v142[17]) > 0x40 )
              {
                if ( v142[16] )
                  j_j___libc_free_0_0(v142[16]);
              }
            }
            v113 = (unsigned __int64 *)v142[2];
            v114 = v142[2] + 24LL * LODWORD(v142[3]);
            if ( v142[2] != v114 )
            {
              do
              {
                v114 -= 24LL;
                if ( *(_DWORD *)(v114 + 16) > 0x40u )
                {
                  v115 = *(_QWORD *)(v114 + 8);
                  if ( v115 )
                    j_j___libc_free_0_0(v115);
                }
              }
              while ( v113 != (unsigned __int64 *)v114 );
              v113 = (unsigned __int64 *)v142[2];
            }
            if ( v113 != &v142[4] )
              _libc_free((unsigned __int64)v113);
            goto LABEL_41;
          }
          v21 = (unsigned __int8 *)*((_QWORD *)v21 - 4);
          v22 = *((_QWORD *)v21 + 1);
          if ( *(_BYTE *)(v22 + 8) != 14 )
            goto LABEL_25;
        }
        if ( (_BYTE)v15 != 63 )
        {
LABEL_38:
          if ( (unsigned int)*(unsigned __int8 *)(v22 + 8) - 17 <= 1 )
            v22 = **(_QWORD **)(v22 + 16);
          v28 = sub_AE2980(a3, *(_DWORD *)(v22 + 8) >> 8)[3];
          v142[2] = (unsigned __int64)&v142[4];
          LODWORD(v142[0]) = 0;
          v142[1] = 0;
          v142[3] = 0x400000000LL;
          LODWORD(v142[17]) = v28;
          if ( v28 > 0x40 )
          {
            sub_C43690((__int64)&v142[16], 0, 0);
            if ( LODWORD(v142[17]) <= 0x40 )
            {
              v76 = v142[2];
              v77 = v142[2] + 24LL * LODWORD(v142[3]);
            }
            else
            {
              if ( v142[16] )
                j_j___libc_free_0_0(v142[16]);
              v76 = v142[2];
              v77 = v142[2] + 24LL * LODWORD(v142[3]);
            }
            if ( v76 != v77 )
            {
              do
              {
                v77 -= 24LL;
                if ( *(_DWORD *)(v77 + 16) > 0x40u )
                {
                  v78 = *(_QWORD *)(v77 + 8);
                  if ( v78 )
                    j_j___libc_free_0_0(v78);
                }
              }
              while ( v76 != v77 );
              v77 = v142[2];
            }
            if ( (unsigned __int64 *)v77 != &v142[4] )
              _libc_free(v77);
          }
          goto LABEL_41;
        }
        LODWORD(v127) = v23[3];
        if ( v24 > 0x40 )
          sub_C43690((__int64)&v126, 0, 0);
        else
          v126 = 0;
        if ( (unsigned __int8)sub_B4DE60((__int64)v21, a3, (__int64)&v126) )
        {
          LODWORD(v142[0]) = 0;
          v142[2] = (unsigned __int64)&v142[4];
          v142[3] = 0x400000000LL;
          v142[1] = 0;
          LODWORD(v142[17]) = v127;
          if ( (unsigned int)v127 > 0x40 )
          {
            sub_C43780((__int64)&v142[16], (const void **)&v126);
            v94 = LODWORD(v142[0]);
            v95 = v142[1];
          }
          else
          {
            v94 = 0;
            v142[16] = (unsigned __int64)v126;
            v95 = 0;
          }
          v128 = v94;
          v129 = v95;
          sub_2DEB400((__int64)&v130, &v142[2], v94, v66, (__int64)v67, v68);
          if ( v134 > 0x40 && v133 )
            j_j___libc_free_0_0((unsigned __int64)v133);
          v96 = (unsigned __int64 *)v142[2];
          v133 = (char *)v142[16];
          v134 = v142[17];
          v97 = v142[2] + 24LL * LODWORD(v142[3]);
          if ( v142[2] != v97 )
          {
            do
            {
              v97 -= 24LL;
              if ( *(_DWORD *)(v97 + 16) > 0x40u )
              {
                v98 = *(_QWORD *)(v97 + 8);
                if ( v98 )
                  j_j___libc_free_0_0(v98);
              }
            }
            while ( v96 != (unsigned __int64 *)v97 );
            v96 = (unsigned __int64 *)v142[2];
          }
          if ( v96 != &v142[4] )
            _libc_free((unsigned __int64)v96);
          v21 = *(unsigned __int8 **)&v21[-32 * (*((_DWORD *)v21 + 1) & 0x7FFFFFF)];
        }
        else
        {
          v135 = &v137;
          v136 = 0x400000000LL;
          v69 = *((_DWORD *)v21 + 1) & 0x7FFFFFF;
          v124 = v69;
          if ( (unsigned int)v69 <= 1 )
            goto LABEL_148;
          v70 = (unsigned int)(v69 - 1);
          v68 = 1;
          v121 = v70;
          v71 = v21;
          for ( i = 1; ; ++i )
          {
            v73 = i;
            v65 = (_QWORD *)(i - v69);
            v67 = *(_BYTE **)&v71[32 * (i - v69)];
            if ( *v67 != 17 )
              break;
            v74 = (unsigned int)v136;
            v70 = HIDWORD(v136);
            v75 = (unsigned int)v136 + 1LL;
            if ( v75 > HIDWORD(v136) )
            {
              v116 = v67;
              sub_C8D5F0((__int64)&v135, &v137, v75, 8u, (__int64)v67, v68);
              v74 = (unsigned int)v136;
              v67 = v116;
            }
            v65 = v135;
            v73 = i + 1;
            v135[v74] = v67;
            LODWORD(v136) = v136 + 1;
            if ( i == v121 )
              break;
            v69 = *((_DWORD *)v71 + 1) & 0x7FFFFFF;
          }
          v79 = v71;
          if ( v124 == v73 + 1 )
          {
            sub_2DEBFC0(
              *(_QWORD *)&v71[32 * (v73 - (unsigned __int64)(*((_DWORD *)v71 + 1) & 0x7FFFFFF))],
              (__int64)&v128,
              *((_DWORD *)v71 + 1) & 0x7FFFFFF,
              v70,
              (__int64)v67,
              v68);
            v80 = sub_AE54E0(a3, *((_QWORD *)v71 + 9), v135, (unsigned int)v136);
            if ( (unsigned int)v127 > 0x40 )
            {
              *v126 = v80;
              memset(v126 + 1, 0, 8 * (unsigned int)(((unsigned __int64)(unsigned int)v127 + 63) >> 6) - 8);
            }
            else
            {
              v81 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v127;
              if ( !(_DWORD)v127 )
                v81 = 0;
              v126 = (_QWORD *)(v81 & v80);
            }
            v82 = *((_QWORD *)v71 + 10);
            v83 = sub_AE5020(a3, v82);
            v84 = sub_9208B0(a3, v82);
            v142[1] = v85;
            v142[0] = (((unsigned __int64)(v84 + 7) >> 3) + (1LL << v83) - 1) >> v83 << v83;
            v86 = sub_CA1930(v142);
            v87 = v134;
            if ( v24 < v134 )
            {
              if ( v128 != -1 )
              {
                v88 = v128 - v134 + v24;
                if ( v134 - v24 >= v128 )
                  v88 = 0;
                v128 = v88;
              }
              sub_C44740((__int64)v142, &v133, v24);
              if ( v134 > 0x40 && v133 )
                j_j___libc_free_0_0((unsigned __int64)v133);
              v133 = (char *)v142[0];
              v87 = v142[1];
              v134 = v142[1];
              if ( v129 )
              {
                v99 = (unsigned int)v131;
                v100 = v24;
                v101 = v142;
                v142[1] = v24;
                v102 = (__int64)v130;
                LODWORD(v142[0]) = 3;
                v103 = (unsigned int)v131 + 1LL;
                LODWORD(v142[2]) = 32;
                if ( v103 > HIDWORD(v131) )
                {
                  if ( v130 > v142
                    || (v120 = (__int64)v130,
                        v99 = (unsigned __int64)&v130[3 * (unsigned int)v131],
                        (unsigned __int64)v142 >= v99) )
                  {
                    sub_2DEABD0((__int64)&v130, v103, v99, (__int64)v142, v24, v89);
                    v99 = (unsigned int)v131;
                    v102 = (__int64)v130;
                    v101 = v142;
                    v100 = v24;
                  }
                  else
                  {
                    sub_2DEABD0((__int64)&v130, v103, v99, (__int64)v142, v24, v89);
                    v99 = (unsigned int)v131;
                    v100 = v24;
                    v102 = (__int64)v130;
                    v101 = (unsigned __int64 *)((char *)v142 + (_QWORD)v130 - v120);
                  }
                }
                v104 = v102 + 24 * v99;
                if ( v104 )
                {
                  *(_DWORD *)v104 = *(_DWORD *)v101;
                  v105 = *((_DWORD *)v101 + 4);
                  *((_DWORD *)v101 + 4) = 0;
                  *(_DWORD *)(v104 + 16) = v105;
                  *(_QWORD *)(v104 + 8) = v101[1];
                }
                LODWORD(v131) = v131 + 1;
                if ( LODWORD(v142[2]) > 0x40 && v100 )
                  j_j___libc_free_0_0(v100);
                v87 = v134;
              }
            }
            if ( v24 > v87 )
            {
              if ( v128 != -1 )
              {
                if ( v24 + v128 - v87 <= v87 )
                  v87 = v24 + v128 - v87;
                v128 = v87;
              }
              sub_C44830((__int64)v142, &v133, v24);
              if ( v134 > 0x40 && v133 )
                j_j___libc_free_0_0((unsigned __int64)v133);
              v133 = (char *)v142[0];
              v134 = v142[1];
              if ( v129 )
              {
                v106 = (unsigned int)v131;
                v107 = v24;
                v108 = v142;
                v142[1] = v24;
                v109 = (__int64)v130;
                LODWORD(v142[0]) = 2;
                v110 = (unsigned int)v131 + 1LL;
                LODWORD(v142[2]) = 32;
                if ( v110 > HIDWORD(v131) )
                {
                  if ( v130 > v142
                    || (v119 = (__int64)v130,
                        v106 = (unsigned __int64)&v130[3 * (unsigned int)v131],
                        (unsigned __int64)v142 >= v106) )
                  {
                    sub_2DEABD0((__int64)&v130, v110, v106, (__int64)v142, v24, v90);
                    v106 = (unsigned int)v131;
                    v109 = (__int64)v130;
                    v108 = v142;
                    v107 = v24;
                  }
                  else
                  {
                    sub_2DEABD0((__int64)&v130, v110, v106, (__int64)v142, v24, v90);
                    v106 = (unsigned int)v131;
                    v107 = v24;
                    v109 = (__int64)v130;
                    v108 = (unsigned __int64 *)((char *)v142 + (_QWORD)v130 - v119);
                  }
                }
                v111 = v109 + 24 * v106;
                if ( v111 )
                {
                  *(_DWORD *)v111 = *(_DWORD *)v108;
                  v112 = *((_DWORD *)v108 + 4);
                  *((_DWORD *)v108 + 4) = 0;
                  *(_DWORD *)(v111 + 16) = v112;
                  *(_QWORD *)(v111 + 8) = v108[1];
                }
                LODWORD(v131) = v131 + 1;
                if ( LODWORD(v142[2]) > 0x40 && v107 )
                  j_j___libc_free_0_0(v107);
              }
            }
            LODWORD(v142[1]) = v24;
            if ( v24 > 0x40 )
              sub_C43690((__int64)v142, v86, 0);
            else
              v142[0] = v86;
            sub_2DEB8E0((__int64)&v128, (__int64)v142);
            if ( LODWORD(v142[1]) > 0x40 && v142[0] )
              j_j___libc_free_0_0(v142[0]);
            if ( (_DWORD)v127 == v134 )
              sub_C45EE0((__int64)&v133, (__int64 *)&v126);
            else
              v128 = -1;
            v21 = *(unsigned __int8 **)&v79[-32 * (*((_DWORD *)v79 + 1) & 0x7FFFFFF)];
            if ( v135 != &v137 )
              _libc_free((unsigned __int64)v135);
          }
          else
          {
LABEL_148:
            v128 = -1;
            v129 = 0;
            memset(v142, 0, sizeof(v142));
            v142[2] = (unsigned __int64)&v142[4];
            LODWORD(v142[0]) = -1;
            v142[3] = 0x400000000LL;
            LODWORD(v142[17]) = 1;
            sub_2DEB400((__int64)&v130, &v142[2], (__int64)v65, 0, (__int64)v67, v68);
            if ( v134 > 0x40 && v133 )
              j_j___libc_free_0_0((unsigned __int64)v133);
            v91 = (unsigned __int64 *)v142[2];
            v133 = (char *)v142[16];
            v134 = v142[17];
            v92 = v142[2] + 24LL * LODWORD(v142[3]);
            if ( v142[2] != v92 )
            {
              do
              {
                v92 -= 24LL;
                if ( *(_DWORD *)(v92 + 16) > 0x40u )
                {
                  v93 = *(_QWORD *)(v92 + 8);
                  if ( v93 )
                    j_j___libc_free_0_0(v93);
                }
              }
              while ( v91 != (unsigned __int64 *)v92 );
              v91 = (unsigned __int64 *)v142[2];
            }
            if ( v91 != &v142[4] )
              _libc_free((unsigned __int64)v91);
            if ( v135 != &v137 )
              _libc_free((unsigned __int64)v135);
            v21 = 0;
          }
        }
        if ( (unsigned int)v127 > 0x40 && v126 )
          j_j___libc_free_0_0((unsigned __int64)v126);
      }
      else
      {
LABEL_25:
        v128 = -1;
        v129 = 0;
        memset(v142, 0, sizeof(v142));
        v142[2] = (unsigned __int64)&v142[4];
        LODWORD(v142[0]) = -1;
        v142[3] = 0x400000000LL;
        LODWORD(v142[17]) = 1;
        sub_2DEB400((__int64)&v130, &v142[2], v15, 0, v16, v17);
        if ( v134 > 0x40 && v133 )
          j_j___libc_free_0_0((unsigned __int64)v133);
        v25 = v142[2];
        v133 = (char *)v142[16];
        v134 = v142[17];
        v26 = v142[2] + 24LL * LODWORD(v142[3]);
        if ( v142[2] != v26 )
        {
          do
          {
            v26 -= 24LL;
            if ( *(_DWORD *)(v26 + 16) > 0x40u )
            {
              v27 = *(_QWORD *)(v26 + 8);
              if ( v27 )
                j_j___libc_free_0_0(v27);
            }
          }
          while ( v25 != v26 );
          v26 = v142[2];
        }
        if ( (unsigned __int64 *)v26 != &v142[4] )
          _libc_free(v26);
        v21 = 0;
      }
LABEL_41:
      v29 = v125;
      v30 = *(_QWORD *)(v125 + 40);
      a2[2] = v21;
      a2[1] = v30;
      v31 = sub_2DEC850((__int64)(a2 + 3), (unsigned __int64 *)&v125);
      v33 = v32;
      if ( v32 )
      {
        v34 = 1;
        if ( !v31 && a2 + 4 != v32 )
          v34 = v29 < v32[4];
        v123 = v34;
        v35 = sub_22077B0(0x28u);
        *(_QWORD *)(v35 + 32) = v125;
        sub_220F040(v123, v35, v33, a2 + 4);
        ++a2[8];
        v29 = v125;
      }
      v142[0] = v29;
      v36 = sub_2D11AF0((__int64)(a2 + 9), v142);
      v38 = v37;
      if ( v37 )
      {
        v39 = 1;
        if ( !v36 && a2 + 10 != v37 )
          v39 = v29 < v37[4];
        v40 = sub_22077B0(0x28u);
        *(_QWORD *)(v40 + 32) = v142[0];
        sub_220F040(v39, v40, v38, a2 + 10);
        ++a2[14];
      }
      if ( *(_DWORD *)(a2[17] + 32LL) )
      {
        v41 = 0;
        do
        {
          v42 = (_QWORD *)sub_BD5C60(v125);
          v43 = sub_BCB2D0(v42);
          v126 = (_QWORD *)sub_ACD640(v43, 0, 0);
          v44 = (_QWORD *)sub_BD5C60(v125);
          v45 = sub_BCB2D0(v44);
          v46 = sub_ACD640(v45, v41, 0);
          v47 = a2[17];
          v127 = v46;
          v49 = sub_AE54E0(a3, v47, &v126, 2);
          v51 = 0;
          if ( !v41 )
            v51 = v125;
          v52 = v51;
          LODWORD(v135) = v128;
          v136 = v129;
          v137 = v139;
          v138 = 0x400000000LL;
          if ( (_DWORD)v131 )
          {
            v117 = v49;
            sub_2DEB050((__int64)&v137, (__int64 *)&v130, (unsigned int)v131, v48, v49, v50);
            v49 = v117;
          }
          v141 = v134;
          if ( v134 > 0x40 )
          {
            v118 = v49;
            sub_C43780((__int64)&v140, (const void **)&v133);
            v49 = v118;
          }
          else
          {
            v140 = (unsigned __int64)v133;
          }
          sub_C46A40((__int64)&v140, v49);
          LODWORD(v142[0]) = (_DWORD)v135;
          v142[1] = v136;
          v142[2] = (unsigned __int64)&v142[4];
          v142[3] = 0x400000000LL;
          if ( (_DWORD)v138 )
            sub_2DEB050((__int64)&v142[2], (__int64 *)&v137, v53, v54, v55, v56);
          LODWORD(v142[17]) = v141;
          if ( v141 > 0x40 )
            sub_C43780((__int64)&v142[16], (const void **)&v140);
          else
            v142[16] = v140;
          v143 = v52;
          v57 = a2[16] + 152LL * v41;
          *(_DWORD *)v57 = v142[0];
          *(_QWORD *)(v57 + 8) = v142[1];
          sub_2DEB400(v57 + 16, &v142[2], 19LL * v41, v54, v55, v56);
          if ( *(_DWORD *)(v57 + 136) > 0x40u )
          {
            v58 = *(_QWORD *)(v57 + 128);
            if ( v58 )
              j_j___libc_free_0_0(v58);
          }
          *(_QWORD *)(v57 + 128) = v142[16];
          *(_DWORD *)(v57 + 136) = v142[17];
          *(_QWORD *)(v57 + 144) = v143;
          v59 = v142[2];
          v60 = v142[2] + 24LL * LODWORD(v142[3]);
          if ( v142[2] != v60 )
          {
            do
            {
              v60 -= 24LL;
              if ( *(_DWORD *)(v60 + 16) > 0x40u )
              {
                v61 = *(_QWORD *)(v60 + 8);
                if ( v61 )
                  j_j___libc_free_0_0(v61);
              }
            }
            while ( v59 != v60 );
            v60 = v142[2];
          }
          if ( (unsigned __int64 *)v60 != &v142[4] )
            _libc_free(v60);
          if ( v141 > 0x40 && v140 )
            j_j___libc_free_0_0(v140);
          v62 = (__int64)v137;
          v63 = (unsigned __int64)&v137[24 * (unsigned int)v138];
          if ( v137 != (_BYTE *)v63 )
          {
            do
            {
              v63 -= 24LL;
              if ( *(_DWORD *)(v63 + 16) > 0x40u )
              {
                v64 = *(_QWORD *)(v63 + 8);
                if ( v64 )
                  j_j___libc_free_0_0(v64);
              }
            }
            while ( v62 != v63 );
            v63 = (unsigned __int64)v137;
          }
          if ( (_BYTE *)v63 != v139 )
            _libc_free(v63);
          ++v41;
        }
        while ( v41 < *(_DWORD *)(a2[17] + 32LL) );
      }
      v3 = 1;
    }
    else
    {
      v3 = 0;
    }
    if ( v134 <= 0x40 )
    {
      v18 = v130;
      v19 = &v130[3 * (unsigned int)v131];
    }
    else
    {
      if ( v133 )
        j_j___libc_free_0_0((unsigned __int64)v133);
      v18 = v130;
      v19 = &v130[3 * (unsigned int)v131];
    }
    if ( v19 != v18 )
    {
      do
      {
        v19 -= 3;
        if ( *((_DWORD *)v19 + 4) > 0x40u )
        {
          v20 = v19[1];
          if ( v20 )
            j_j___libc_free_0_0(v20);
        }
      }
      while ( v18 != v19 );
      v18 = v130;
    }
    if ( v18 != (unsigned __int64 *)v132 )
      _libc_free((unsigned __int64)v18);
  }
  return v3;
}
