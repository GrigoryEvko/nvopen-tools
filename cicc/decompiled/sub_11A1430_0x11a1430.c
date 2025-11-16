// Function: sub_11A1430
// Address: 0x11a1430
//
__int64 __fastcall sub_11A1430(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 **a4, unsigned int a5, __m128i *a6)
{
  unsigned int v9; // ecx
  unsigned int v10; // r8d
  int v11; // eax
  __m128i *v12; // rcx
  __int64 v13; // r13
  unsigned int v14; // ebx
  __int64 v15; // rax
  __int64 v16; // r14
  bool v17; // cc
  char v18; // r12
  __int64 v19; // r12
  __m128i *v21; // rcx
  unsigned int v22; // ebx
  __int64 v23; // rax
  __int64 v24; // r14
  char v25; // al
  bool v26; // zf
  unsigned int v27; // r13d
  unsigned __int64 v28; // rax
  unsigned int v29; // edx
  unsigned __int64 v30; // rbx
  unsigned int v31; // ebx
  unsigned int v32; // r12d
  unsigned __int64 v33; // rax
  unsigned int v34; // r12d
  unsigned __int8 *v35; // rdx
  unsigned int v36; // edx
  char v37; // al
  __int64 *v38; // r10
  unsigned __int8 *v39; // rcx
  __int64 *v40; // rdi
  __int64 v41; // r8
  int v42; // eax
  __int64 *v43; // rdi
  unsigned int v44; // r13d
  __int64 v45; // rax
  __int64 v46; // r14
  char v47; // al
  __int64 *v48; // r10
  unsigned __int8 *v49; // rcx
  __int64 *v50; // rdi
  __int64 v51; // r8
  int v52; // eax
  __int64 *v53; // rdi
  unsigned int v54; // r13d
  __int64 v55; // rax
  __int64 v56; // r14
  char v57; // al
  unsigned int v58; // edx
  __int64 v59; // r13
  unsigned __int64 v60; // r13
  char v61; // al
  bool v62; // r14
  unsigned int v63; // r13d
  __int64 v64; // r14
  unsigned __int64 v65; // r14
  char v66; // bl
  unsigned __int8 *v67; // rcx
  __int64 *v68; // rdi
  __int64 v69; // r8
  int v70; // eax
  __int64 *v71; // rdi
  unsigned int v72; // r13d
  __int64 v73; // rax
  __int64 v74; // r14
  char v75; // al
  unsigned int v76; // r14d
  __int64 v77; // r13
  unsigned __int64 v78; // r13
  char v79; // bl
  unsigned int v80; // ebx
  __int64 v81; // r13
  unsigned __int64 v82; // r13
  char v83; // r12
  unsigned __int8 *v84; // r10
  int v85; // eax
  unsigned int v86; // ebx
  unsigned __int64 v87; // rcx
  unsigned int v88; // r12d
  unsigned __int64 v89; // rax
  unsigned int v90; // r12d
  unsigned __int8 *v91; // rdx
  unsigned int v92; // ebx
  __int64 *v93; // r10
  _BYTE *v94; // rax
  __int64 v95; // rbx
  unsigned int v96; // r14d
  unsigned __int64 v97; // rdx
  unsigned __int64 *v98; // rax
  char v99; // bl
  unsigned int v100; // r15d
  unsigned __int64 v101; // rax
  unsigned int v102; // eax
  unsigned __int8 v103; // bl
  char v104; // bl
  __int64 *v105; // rdi
  __int64 v106; // r8
  int v107; // eax
  __int64 *v108; // rdi
  __int64 *v109; // rdi
  char v110; // al
  unsigned __int8 *v111; // r10
  unsigned int v112; // [rsp+8h] [rbp-F8h]
  __int64 *v113; // [rsp+8h] [rbp-F8h]
  __int64 v114; // [rsp+18h] [rbp-E8h]
  unsigned int v115; // [rsp+18h] [rbp-E8h]
  char v116; // [rsp+18h] [rbp-E8h]
  unsigned int v117; // [rsp+18h] [rbp-E8h]
  char v118; // [rsp+18h] [rbp-E8h]
  unsigned int v119; // [rsp+18h] [rbp-E8h]
  char v120; // [rsp+18h] [rbp-E8h]
  unsigned int v121; // [rsp+20h] [rbp-E0h]
  char v123; // [rsp+28h] [rbp-D8h]
  char v124; // [rsp+28h] [rbp-D8h]
  char v125; // [rsp+28h] [rbp-D8h]
  unsigned int v126; // [rsp+28h] [rbp-D8h]
  char v127; // [rsp+28h] [rbp-D8h]
  __int64 v128; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v129; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v130; // [rsp+48h] [rbp-B8h] BYREF
  unsigned __int64 v131; // [rsp+50h] [rbp-B0h] BYREF
  unsigned int v132; // [rsp+58h] [rbp-A8h]
  unsigned __int64 v133; // [rsp+60h] [rbp-A0h] BYREF
  unsigned int v134; // [rsp+68h] [rbp-98h]
  unsigned __int64 v135; // [rsp+70h] [rbp-90h] BYREF
  unsigned int v136; // [rsp+78h] [rbp-88h]
  unsigned __int64 v137; // [rsp+80h] [rbp-80h] BYREF
  unsigned int v138; // [rsp+88h] [rbp-78h]
  unsigned __int64 v139; // [rsp+90h] [rbp-70h] BYREF
  unsigned int v140; // [rsp+98h] [rbp-68h]
  __int64 *v141; // [rsp+A0h] [rbp-60h] BYREF
  __int64 *v142; // [rsp+A8h] [rbp-58h] BYREF
  __int64 *v143; // [rsp+B0h] [rbp-50h]
  __int64 *v144; // [rsp+B8h] [rbp-48h] BYREF
  char v145; // [rsp+C0h] [rbp-40h]

  v9 = *(_DWORD *)(a3 + 8);
  v114 = *((_QWORD *)a2 + 1);
  v10 = v9;
  v134 = v9;
  if ( v9 > 0x40 )
  {
    v121 = v9;
    v112 = v9;
    sub_C43690((__int64)&v133, 0, 0);
    v136 = v121;
    sub_C43690((__int64)&v135, 0, 0);
    v138 = v121;
    sub_C43690((__int64)&v137, 0, 0);
    v140 = v121;
    sub_C43690((__int64)&v139, 0, 0);
    v9 = v121;
    v10 = v112;
    switch ( *a2 )
    {
      case '*':
        v31 = *(_DWORD *)(a3 + 8);
        if ( v31 <= 0x40 )
        {
          v28 = *(_QWORD *)a3;
          v29 = v31 - 64;
          if ( *(_QWORD *)a3 )
          {
LABEL_43:
            _BitScanReverse64(&v30, v28);
            v31 = v29 + (v30 ^ 0x3F);
            goto LABEL_44;
          }
          v132 = v121;
          v32 = v121 - v31;
        }
        else
        {
          v85 = sub_C444A0(a3);
          LOBYTE(v9) = v121;
          LOBYTE(v31) = v85;
          v132 = v121;
          v32 = v121 - v85;
        }
        goto LABEL_187;
      case ',':
        v9 = *(_DWORD *)(a3 + 8);
LABEL_199:
        v86 = v9;
        if ( v9 > 0x40 )
        {
          v117 = v10;
          v102 = sub_C444A0(a3);
          v10 = v117;
          v86 = v102;
        }
        else if ( *(_QWORD *)a3 )
        {
          _BitScanReverse64(&v87, *(_QWORD *)a3);
          v86 = v86 - 64 + (v87 ^ 0x3F);
        }
        v132 = v10;
        v88 = v10 - v86;
        if ( v10 > 0x40 )
        {
          v120 = v10;
          sub_C43690((__int64)&v131, 0, 0);
          LOBYTE(v10) = v120;
        }
        else
        {
          v131 = 0;
        }
        if ( v88 )
        {
          if ( v88 > 0x40 )
          {
            sub_C43C90(&v131, 0, v88);
          }
          else
          {
            v89 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v86 - (unsigned __int8)v10 + 64);
            if ( v132 > 0x40 )
              *(_QWORD *)v131 |= v89;
            else
              v131 |= v89;
          }
        }
        v90 = a5 + 1;
        if ( (a2[7] & 0x40) != 0 )
          v91 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        else
          v91 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        sub_9AC0E0(*((_QWORD *)v91 + 4), &v137, v90, a6);
        v92 = v132;
        if ( v132 <= 0x40 )
        {
          if ( (v131 & ~v137) == 0 )
          {
LABEL_212:
            if ( (a2[7] & 0x40) != 0 )
              v93 = (__int64 *)*((_QWORD *)a2 - 1);
            else
              v93 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
            v19 = *v93;
            if ( v92 <= 0x40 )
              goto LABEL_16;
            goto LABEL_58;
          }
        }
        else if ( (unsigned __int8)sub_C446F0((__int64 *)&v131, (__int64 *)&v137) )
        {
          goto LABEL_212;
        }
        v103 = a2[1];
        v118 = (v103 & 4) != 0;
        v104 = (v103 & 2) != 0;
        if ( (a2[7] & 0x40) != 0 )
          v105 = (__int64 *)*((_QWORD *)a2 - 1);
        else
          v105 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        sub_9AC0E0(*v105, &v133, v90, a6);
        sub_C70430((__int64)&v141, 0, v118, v104, (__int64)&v133, (__int64)&v137);
        if ( *((_DWORD *)a4 + 2) <= 0x40u )
          goto LABEL_249;
LABEL_247:
        if ( *a4 )
          j_j___libc_free_0_0(*a4);
LABEL_249:
        v17 = *((_DWORD *)a4 + 6) <= 0x40u;
        *a4 = v141;
        v107 = (int)v142;
        LODWORD(v142) = 0;
        *((_DWORD *)a4 + 2) = v107;
        if ( v17 || (v108 = a4[2]) == 0 )
        {
          a4[2] = v143;
          *((_DWORD *)a4 + 6) = (_DWORD)v144;
        }
        else
        {
          j_j___libc_free_0_0(v108);
          v17 = (unsigned int)v142 <= 0x40;
          a4[2] = v143;
          *((_DWORD *)a4 + 6) = (_DWORD)v144;
          if ( !v17 && v141 )
            j_j___libc_free_0_0(v141);
        }
        sub_99B5E0((__int64)a2, (__int64)a4, a5, a6->m128i_i64, v106);
        if ( v132 <= 0x40 )
          goto LABEL_41;
        goto LABEL_255;
      case '8':
LABEL_29:
        v21 = a6;
        v13 = (__int64)(a4 + 2);
        sub_9AC0E0((__int64)a2, (unsigned __int64 *)a4, a5, v21);
        v22 = *((_DWORD *)a4 + 2);
        v132 = v22;
        if ( v22 <= 0x40 )
        {
          v23 = (__int64)*a4;
LABEL_31:
          v131 = (unsigned __int64)a4[2] | v23;
          v24 = v131;
          goto LABEL_32;
        }
        sub_C43780((__int64)&v131, (const void **)a4);
        v22 = v132;
        if ( v132 <= 0x40 )
        {
          v23 = v131;
          goto LABEL_31;
        }
        sub_C43BD0(&v131, (__int64 *)a4 + 2);
        v22 = v132;
        v24 = v131;
LABEL_32:
        v17 = *(_DWORD *)(a3 + 8) <= 0x40u;
        LODWORD(v142) = v22;
        v141 = (__int64 *)v24;
        v132 = 0;
        if ( v17 )
          v25 = (*(_QWORD *)a3 & ~v24) == 0;
        else
          v25 = sub_C446F0((__int64 *)a3, (__int64 *)&v141);
        if ( v22 > 0x40 )
        {
          if ( v24 )
          {
            v123 = v25;
            j_j___libc_free_0_0(v24);
            v25 = v123;
            if ( v132 > 0x40 )
            {
              if ( v131 )
              {
                j_j___libc_free_0_0(v131);
                v25 = v123;
              }
            }
          }
        }
        if ( v25 )
          goto LABEL_15;
        v26 = *a2 == 56;
        LOBYTE(v143) = 0;
        v141 = &v130;
        v27 = *(_DWORD *)(a3 + 8);
        v142 = &v129;
        v144 = &v128;
        v145 = 0;
        if ( !v26 )
          goto LABEL_41;
        v94 = (_BYTE *)*((_QWORD *)a2 - 8);
        if ( *v94 != 54 )
          goto LABEL_41;
        if ( !*((_QWORD *)v94 - 8) )
          goto LABEL_41;
        v130 = *((_QWORD *)v94 - 8);
        if ( !(unsigned __int8)sub_991580((__int64)&v142, *((_QWORD *)v94 - 4)) )
          goto LABEL_41;
        if ( !(unsigned __int8)sub_991580((__int64)&v144, *((_QWORD *)a2 - 4)) )
          goto LABEL_41;
        v95 = v128;
        if ( v129 != v128 )
          goto LABEL_41;
        v96 = *(_DWORD *)(v128 + 8);
        if ( v96 > 0x40 )
        {
          if ( v96 - (unsigned int)sub_C444A0(v128) > 0x40 )
            goto LABEL_41;
          v97 = **(_QWORD **)v95;
        }
        else
        {
          v97 = *(_QWORD *)v128;
        }
        if ( v27 <= v97 )
          goto LABEL_41;
        v98 = *(unsigned __int64 **)v95;
        if ( *(_DWORD *)(v95 + 8) > 0x40u )
          v98 = (unsigned __int64 *)*v98;
        v132 = v27;
        v99 = (char)v98;
        v100 = v27 - (_DWORD)v98;
        if ( v27 > 0x40 )
          sub_C43690((__int64)&v131, 0, 0);
        else
          v131 = 0;
        if ( !v100 )
          goto LABEL_307;
        if ( v100 > 0x40 )
        {
          sub_C43C90(&v131, 0, v100);
        }
        else
        {
          v101 = 0xFFFFFFFFFFFFFFFFLL >> (v99 - (unsigned __int8)v27 + 64);
          if ( v132 <= 0x40 )
          {
            v131 |= v101;
            if ( sub_10024C0((__int64 *)a3, (__int64 *)&v131) )
            {
LABEL_234:
              v19 = v130;
              goto LABEL_16;
            }
            goto LABEL_41;
          }
          *(_QWORD *)v131 |= v101;
        }
LABEL_307:
        if ( sub_10024C0((__int64 *)a3, (__int64 *)&v131) )
        {
          if ( v132 > 0x40 && v131 )
            j_j___libc_free_0_0(v131);
          goto LABEL_234;
        }
        if ( v132 > 0x40 )
        {
LABEL_255:
          if ( v131 )
            j_j___libc_free_0_0(v131);
        }
LABEL_41:
        v19 = 0;
        goto LABEL_16;
      case '9':
LABEL_136:
        if ( (a2[7] & 0x40) != 0 )
          v67 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        else
          v67 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        sub_9AC0E0(*((_QWORD *)v67 + 4), &v137, a5 + 1, a6);
        if ( (a2[7] & 0x40) != 0 )
          v68 = (__int64 *)*((_QWORD *)a2 - 1);
        else
          v68 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        sub_9AC0E0(*v68, &v133, a5 + 1, a6);
        sub_9ADBC0((__int64 *)&v141, (__int64)a2, (__int64)&v133, (__int64)&v137, a5, a6);
        if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
          j_j___libc_free_0_0(*a4);
        v17 = *((_DWORD *)a4 + 6) <= 0x40u;
        *a4 = v141;
        v70 = (int)v142;
        LODWORD(v142) = 0;
        *((_DWORD *)a4 + 2) = v70;
        v113 = (__int64 *)(a4 + 2);
        if ( v17 || (v71 = a4[2]) == 0 )
        {
          a4[2] = v143;
          *((_DWORD *)a4 + 6) = (_DWORD)v144;
        }
        else
        {
          j_j___libc_free_0_0(v71);
          v17 = (unsigned int)v142 <= 0x40;
          a4[2] = v143;
          *((_DWORD *)a4 + 6) = (_DWORD)v144;
          if ( !v17 && v141 )
            j_j___libc_free_0_0(v141);
        }
        sub_99B5E0((__int64)a2, (__int64)a4, a5, a6->m128i_i64, v69);
        v72 = *((_DWORD *)a4 + 2);
        v132 = v72;
        if ( v72 <= 0x40 )
        {
          v73 = (__int64)*a4;
LABEL_151:
          v131 = (unsigned __int64)a4[2] | v73;
          v74 = v131;
          goto LABEL_152;
        }
        sub_C43780((__int64)&v131, (const void **)a4);
        v72 = v132;
        if ( v132 <= 0x40 )
        {
          v73 = v131;
          goto LABEL_151;
        }
        sub_C43BD0(&v131, v113);
        v74 = v131;
        v72 = v132;
LABEL_152:
        v17 = *(_DWORD *)(a3 + 8) <= 0x40u;
        LODWORD(v142) = v72;
        v141 = (__int64 *)v74;
        v132 = 0;
        if ( v17 )
          v75 = (*(_QWORD *)a3 & ~v74) == 0;
        else
          v75 = sub_C446F0((__int64 *)a3, (__int64 *)&v141);
        if ( v72 > 0x40 )
        {
          if ( v74 )
          {
            v127 = v75;
            j_j___libc_free_0_0(v74);
            v75 = v127;
            if ( v132 > 0x40 )
            {
              if ( v131 )
              {
                j_j___libc_free_0_0(v131);
                v75 = v127;
              }
            }
          }
        }
        if ( v75 )
          goto LABEL_188;
        v76 = v134;
        v132 = v134;
        if ( v134 > 0x40 )
        {
          sub_C43780((__int64)&v131, (const void **)&v133);
          v76 = v132;
          if ( v132 > 0x40 )
          {
            sub_C43BD0(&v131, (__int64 *)&v139);
            v76 = v132;
            v78 = v131;
LABEL_163:
            v17 = *(_DWORD *)(a3 + 8) <= 0x40u;
            LODWORD(v142) = v76;
            v141 = (__int64 *)v78;
            v132 = 0;
            if ( v17 )
              v79 = (*(_QWORD *)a3 & ~v78) == 0;
            else
              v79 = sub_C446F0((__int64 *)a3, (__int64 *)&v141);
            if ( v76 > 0x40 )
            {
              if ( v78 )
              {
                j_j___libc_free_0_0(v78);
                if ( v132 > 0x40 )
                {
                  if ( v131 )
                    j_j___libc_free_0_0(v131);
                }
              }
            }
            if ( !v79 )
            {
              v80 = v138;
              v132 = v138;
              if ( v138 > 0x40 )
              {
                sub_C43780((__int64)&v131, (const void **)&v137);
                v80 = v132;
                if ( v132 > 0x40 )
                {
                  sub_C43BD0(&v131, (__int64 *)&v135);
                  v82 = v131;
                  v80 = v132;
LABEL_174:
                  v17 = *(_DWORD *)(a3 + 8) <= 0x40u;
                  LODWORD(v142) = v80;
                  v141 = (__int64 *)v82;
                  v132 = 0;
                  if ( v17 )
                    v83 = (*(_QWORD *)a3 & ~v82) == 0;
                  else
                    v83 = sub_C446F0((__int64 *)a3, (__int64 *)&v141);
                  if ( v80 > 0x40 )
                  {
                    if ( v82 )
                    {
                      j_j___libc_free_0_0(v82);
                      if ( v132 > 0x40 )
                      {
                        if ( v131 )
                          j_j___libc_free_0_0(v131);
                      }
                    }
                  }
                  if ( !v83 )
                    goto LABEL_41;
                  goto LABEL_182;
                }
                v81 = v131;
              }
              else
              {
                v81 = v137;
              }
              v82 = v135 | v81;
              v131 = v82;
              goto LABEL_174;
            }
LABEL_86:
            if ( (a2[7] & 0x40) != 0 )
              v48 = (__int64 *)*((_QWORD *)a2 - 1);
            else
              v48 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
            v19 = *v48;
            goto LABEL_16;
          }
          v77 = v131;
        }
        else
        {
          v77 = v133;
        }
        v78 = v139 | v77;
        v131 = v78;
        goto LABEL_163;
      case ':':
LABEL_89:
        if ( (a2[7] & 0x40) != 0 )
          v49 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        else
          v49 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        sub_9AC0E0(*((_QWORD *)v49 + 4), &v137, a5 + 1, a6);
        if ( (a2[7] & 0x40) != 0 )
          v50 = (__int64 *)*((_QWORD *)a2 - 1);
        else
          v50 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        sub_9AC0E0(*v50, &v133, a5 + 1, a6);
        sub_9ADBC0((__int64 *)&v141, (__int64)a2, (__int64)&v133, (__int64)&v137, a5, a6);
        if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
          j_j___libc_free_0_0(*a4);
        v17 = *((_DWORD *)a4 + 6) <= 0x40u;
        *a4 = v141;
        v52 = (int)v142;
        LODWORD(v142) = 0;
        *((_DWORD *)a4 + 2) = v52;
        v113 = (__int64 *)(a4 + 2);
        if ( v17 || (v53 = a4[2]) == 0 )
        {
          a4[2] = v143;
          *((_DWORD *)a4 + 6) = (_DWORD)v144;
        }
        else
        {
          j_j___libc_free_0_0(v53);
          v17 = (unsigned int)v142 <= 0x40;
          a4[2] = v143;
          *((_DWORD *)a4 + 6) = (_DWORD)v144;
          if ( !v17 && v141 )
            j_j___libc_free_0_0(v141);
        }
        sub_99B5E0((__int64)a2, (__int64)a4, a5, a6->m128i_i64, v51);
        v54 = *((_DWORD *)a4 + 2);
        v132 = v54;
        if ( v54 <= 0x40 )
        {
          v55 = (__int64)*a4;
LABEL_104:
          v131 = (unsigned __int64)a4[2] | v55;
          v56 = v131;
          goto LABEL_105;
        }
        sub_C43780((__int64)&v131, (const void **)a4);
        v54 = v132;
        if ( v132 <= 0x40 )
        {
          v55 = v131;
          goto LABEL_104;
        }
        sub_C43BD0(&v131, v113);
        v56 = v131;
        v54 = v132;
LABEL_105:
        v17 = *(_DWORD *)(a3 + 8) <= 0x40u;
        LODWORD(v142) = v54;
        v141 = (__int64 *)v56;
        v132 = 0;
        if ( v17 )
          v57 = (*(_QWORD *)a3 & ~v56) == 0;
        else
          v57 = sub_C446F0((__int64 *)a3, (__int64 *)&v141);
        if ( v54 > 0x40 )
        {
          if ( v56 )
          {
            v125 = v57;
            j_j___libc_free_0_0(v56);
            v57 = v125;
            if ( v132 > 0x40 )
            {
              if ( v131 )
              {
                j_j___libc_free_0_0(v131);
                v57 = v125;
              }
            }
          }
        }
        if ( v57 )
          goto LABEL_188;
        v58 = v136;
        v132 = v136;
        if ( v136 > 0x40 )
        {
          sub_C43780((__int64)&v131, (const void **)&v135);
          v58 = v132;
          if ( v132 > 0x40 )
          {
            sub_C43BD0(&v131, (__int64 *)&v137);
            v58 = v132;
            v60 = v131;
LABEL_116:
            v17 = *(_DWORD *)(a3 + 8) <= 0x40u;
            LODWORD(v142) = v58;
            v141 = (__int64 *)v60;
            v132 = 0;
            if ( v17 )
            {
              v62 = (*(_QWORD *)a3 & ~v60) == 0;
            }
            else
            {
              v126 = v58;
              v61 = sub_C446F0((__int64 *)a3, (__int64 *)&v141);
              v58 = v126;
              v62 = v61;
            }
            if ( v58 > 0x40 )
            {
              if ( v60 )
              {
                j_j___libc_free_0_0(v60);
                if ( v132 > 0x40 )
                {
                  if ( v131 )
                    j_j___libc_free_0_0(v131);
                }
              }
            }
            if ( v62 )
              goto LABEL_86;
            v63 = v140;
            v132 = v140;
            if ( v140 > 0x40 )
            {
              sub_C43780((__int64)&v131, (const void **)&v139);
              v63 = v132;
              if ( v132 > 0x40 )
              {
                sub_C43BD0(&v131, (__int64 *)&v133);
                v63 = v132;
                v65 = v131;
LABEL_127:
                v17 = *(_DWORD *)(a3 + 8) <= 0x40u;
                LODWORD(v142) = v63;
                v141 = (__int64 *)v65;
                v132 = 0;
                if ( v17 )
                  v66 = (*(_QWORD *)a3 & ~v65) == 0;
                else
                  v66 = sub_C446F0((__int64 *)a3, (__int64 *)&v141);
                if ( v63 > 0x40 )
                {
                  if ( v65 )
                  {
                    j_j___libc_free_0_0(v65);
                    if ( v132 > 0x40 )
                    {
                      if ( v131 )
                        j_j___libc_free_0_0(v131);
                    }
                  }
                }
                if ( !v66 )
                  goto LABEL_41;
                goto LABEL_182;
              }
              v64 = v131;
            }
            else
            {
              v64 = v139;
            }
            v65 = v133 | v64;
            v131 = v65;
            goto LABEL_127;
          }
          v59 = v131;
        }
        else
        {
          v59 = v135;
        }
        v60 = v137 | v59;
        v131 = v60;
        goto LABEL_116;
      case ';':
LABEL_60:
        if ( (a2[7] & 0x40) != 0 )
          v39 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        else
          v39 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        sub_9AC0E0(*((_QWORD *)v39 + 4), &v137, a5 + 1, a6);
        if ( (a2[7] & 0x40) != 0 )
          v40 = (__int64 *)*((_QWORD *)a2 - 1);
        else
          v40 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        sub_9AC0E0(*v40, &v133, a5 + 1, a6);
        sub_9ADBC0((__int64 *)&v141, (__int64)a2, (__int64)&v133, (__int64)&v137, a5, a6);
        if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
          j_j___libc_free_0_0(*a4);
        v17 = *((_DWORD *)a4 + 6) <= 0x40u;
        *a4 = v141;
        v42 = (int)v142;
        LODWORD(v142) = 0;
        *((_DWORD *)a4 + 2) = v42;
        v113 = (__int64 *)(a4 + 2);
        if ( v17 || (v43 = a4[2]) == 0 )
        {
          a4[2] = v143;
          *((_DWORD *)a4 + 6) = (_DWORD)v144;
        }
        else
        {
          j_j___libc_free_0_0(v43);
          v17 = (unsigned int)v142 <= 0x40;
          a4[2] = v143;
          *((_DWORD *)a4 + 6) = (_DWORD)v144;
          if ( !v17 && v141 )
            j_j___libc_free_0_0(v141);
        }
        sub_99B5E0((__int64)a2, (__int64)a4, a5, a6->m128i_i64, v41);
        v44 = *((_DWORD *)a4 + 2);
        v132 = v44;
        if ( v44 > 0x40 )
        {
          sub_C43780((__int64)&v131, (const void **)a4);
          v44 = v132;
          if ( v132 > 0x40 )
          {
            sub_C43BD0(&v131, v113);
            v44 = v132;
            v46 = v131;
LABEL_76:
            v17 = *(_DWORD *)(a3 + 8) <= 0x40u;
            LODWORD(v142) = v44;
            v141 = (__int64 *)v46;
            v132 = 0;
            if ( v17 )
              v47 = (*(_QWORD *)a3 & ~v46) == 0;
            else
              v47 = sub_C446F0((__int64 *)a3, (__int64 *)&v141);
            if ( v44 > 0x40 )
            {
              if ( v46 )
              {
                v124 = v47;
                j_j___libc_free_0_0(v46);
                v47 = v124;
                if ( v132 > 0x40 )
                {
                  if ( v131 )
                  {
                    j_j___libc_free_0_0(v131);
                    v47 = v124;
                  }
                }
              }
            }
            if ( !v47 )
            {
              if ( *(_DWORD *)(a3 + 8) <= 0x40u )
              {
                if ( (*(_QWORD *)a3 & ~v137) == 0 )
                  goto LABEL_86;
                if ( (*(_QWORD *)a3 & ~v133) != 0 )
                  goto LABEL_41;
              }
              else
              {
                if ( (unsigned __int8)sub_C446F0((__int64 *)a3, (__int64 *)&v137) )
                  goto LABEL_86;
                if ( !(unsigned __int8)sub_C446F0((__int64 *)a3, (__int64 *)&v133) )
                  goto LABEL_41;
              }
LABEL_182:
              if ( (a2[7] & 0x40) != 0 )
                v84 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
              else
                v84 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
              v19 = *((_QWORD *)v84 + 4);
              goto LABEL_16;
            }
LABEL_188:
            v19 = sub_AD6220(v114, (__int64)v113);
            goto LABEL_16;
          }
          v45 = v131;
        }
        else
        {
          v45 = (__int64)*a4;
        }
        v131 = (unsigned __int64)a4[2] | v45;
        v46 = v131;
        goto LABEL_76;
      default:
LABEL_4:
        v12 = a6;
        v13 = (__int64)(a4 + 2);
        sub_9AC0E0((__int64)a2, (unsigned __int64 *)a4, a5, v12);
        v14 = *((_DWORD *)a4 + 2);
        v132 = v14;
        if ( v14 > 0x40 )
        {
          sub_C43780((__int64)&v131, (const void **)a4);
          v14 = v132;
          if ( v132 > 0x40 )
          {
            sub_C43BD0(&v131, (__int64 *)a4 + 2);
            v14 = v132;
            v16 = v131;
LABEL_7:
            v17 = *(_DWORD *)(a3 + 8) <= 0x40u;
            LODWORD(v142) = v14;
            v141 = (__int64 *)v16;
            v132 = 0;
            if ( v17 )
              v18 = (*(_QWORD *)a3 & ~v16) == 0;
            else
              v18 = sub_C446F0((__int64 *)a3, (__int64 *)&v141);
            if ( v14 > 0x40 )
            {
              if ( v16 )
              {
                j_j___libc_free_0_0(v16);
                if ( v132 > 0x40 )
                {
                  if ( v131 )
                    j_j___libc_free_0_0(v131);
                }
              }
            }
            if ( !v18 )
              goto LABEL_41;
LABEL_15:
            v19 = sub_AD6220(v114, v13);
            goto LABEL_16;
          }
          v15 = v131;
        }
        else
        {
          v15 = (__int64)*a4;
        }
        v131 = (unsigned __int64)a4[2] | v15;
        v16 = v131;
        goto LABEL_7;
    }
  }
  v11 = *a2;
  v138 = v9;
  v133 = 0;
  v136 = v9;
  v135 = 0;
  v137 = 0;
  v140 = v9;
  v139 = 0;
  switch ( v11 )
  {
    case '*':
      v28 = *(_QWORD *)a3;
      v29 = v9 - 64;
      if ( *(_QWORD *)a3 )
        goto LABEL_43;
      v31 = v9;
      break;
    case ',':
      goto LABEL_199;
    case '8':
      goto LABEL_29;
    case '9':
      goto LABEL_136;
    case ':':
      goto LABEL_89;
    case ';':
      goto LABEL_60;
    default:
      goto LABEL_4;
  }
LABEL_44:
  v132 = v9;
  v32 = v9 - v31;
  if ( v9 > 0x40 )
  {
LABEL_187:
    v116 = v9;
    sub_C43690((__int64)&v131, 0, 0);
    LOBYTE(v9) = v116;
  }
  else
  {
    v131 = 0;
  }
  if ( v32 )
  {
    if ( v32 > 0x40 )
    {
      sub_C43C90(&v131, 0, v32);
    }
    else
    {
      v33 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v31 - (unsigned __int8)v9 + 64);
      if ( v132 > 0x40 )
        *(_QWORD *)v131 |= v33;
      else
        v131 |= v33;
    }
  }
  v34 = a5 + 1;
  if ( (a2[7] & 0x40) != 0 )
    v35 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
  else
    v35 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  sub_9AC0E0(*((_QWORD *)v35 + 4), &v137, v34, a6);
  v36 = v132;
  if ( v132 <= 0x40 )
  {
    if ( (v131 & ~v137) == 0 )
      goto LABEL_54;
  }
  else
  {
    v115 = v132;
    v37 = sub_C446F0((__int64 *)&v131, (__int64 *)&v137);
    v36 = v115;
    if ( v37 )
    {
LABEL_54:
      if ( (a2[7] & 0x40) != 0 )
        v38 = (__int64 *)*((_QWORD *)a2 - 1);
      else
        v38 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v19 = *v38;
      goto LABEL_57;
    }
  }
  if ( (a2[7] & 0x40) != 0 )
    v109 = (__int64 *)*((_QWORD *)a2 - 1);
  else
    v109 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  sub_9AC0E0(*v109, &v133, v34, a6);
  v36 = v132;
  if ( v132 <= 0x40 )
  {
    if ( (v131 & ~v133) == 0 )
      goto LABEL_267;
LABEL_295:
    sub_C70430((__int64)&v141, 1, (a2[1] & 4) != 0, (a2[1] & 2) != 0, (__int64)&v133, (__int64)&v137);
    if ( *((_DWORD *)a4 + 2) <= 0x40u )
      goto LABEL_249;
    goto LABEL_247;
  }
  v119 = v132;
  v110 = sub_C446F0((__int64 *)&v131, (__int64 *)&v133);
  v36 = v119;
  if ( !v110 )
    goto LABEL_295;
LABEL_267:
  if ( (a2[7] & 0x40) != 0 )
    v111 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
  else
    v111 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  v19 = *((_QWORD *)v111 + 4);
LABEL_57:
  if ( v36 > 0x40 )
  {
LABEL_58:
    if ( v131 )
      j_j___libc_free_0_0(v131);
  }
LABEL_16:
  if ( v140 > 0x40 && v139 )
    j_j___libc_free_0_0(v139);
  if ( v138 > 0x40 && v137 )
    j_j___libc_free_0_0(v137);
  if ( v136 > 0x40 && v135 )
    j_j___libc_free_0_0(v135);
  if ( v134 > 0x40 && v133 )
    j_j___libc_free_0_0(v133);
  return v19;
}
