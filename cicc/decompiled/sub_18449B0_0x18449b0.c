// Function: sub_18449B0
// Address: 0x18449b0
//
__int64 __fastcall sub_18449B0(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // r14
  unsigned int v10; // eax
  __int64 v11; // rsi
  unsigned int v12; // r8d
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rdx
  const void *v18; // r12
  const void *v19; // r15
  __int64 v20; // rbx
  __int64 v21; // rax
  char v22; // r13
  __int64 v23; // r12
  char v24; // r13
  __int64 v25; // rcx
  double v26; // xmm4_8
  double v27; // xmm5_8
  __int64 *v28; // r13
  __int64 v29; // rbx
  __int64 v30; // rdi
  unsigned __int64 v31; // r12
  unsigned __int8 v32; // al
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // r12
  __int64 v35; // rax
  __int64 *v36; // r14
  __int64 *v37; // r15
  __int64 v38; // r8
  _QWORD *j; // rax
  unsigned __int64 v40; // rax
  bool v41; // zf
  _WORD *v42; // rdi
  unsigned __int64 v43; // rdx
  int v44; // esi
  __int64 v45; // rax
  _QWORD *v46; // rax
  __int64 v47; // r15
  unsigned __int64 v48; // r15
  __int64 v49; // rdi
  int v50; // esi
  __int64 *v51; // r9
  int v52; // eax
  __int64 v53; // rsi
  double v54; // xmm4_8
  double v55; // xmm5_8
  unsigned __int64 v56; // r12
  _QWORD *v57; // r14
  __int64 v58; // rdi
  __int64 *v60; // r15
  __int64 *v61; // rbx
  unsigned __int64 *v62; // r12
  unsigned __int64 v63; // rdx
  unsigned __int64 v64; // rcx
  __int64 v65; // r12
  __int64 v66; // rdx
  __int64 v67; // rbx
  __int64 i; // r13
  __int64 v69; // rsi
  unsigned __int64 v70; // r12
  _WORD *v71; // rbx
  __int64 v72; // rdx
  unsigned int v73; // esi
  __int64 v74; // rax
  double v75; // xmm4_8
  double v76; // xmm5_8
  __int64 *v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rsi
  unsigned __int8 *v80; // rsi
  int v81; // ebx
  int v82; // r8d
  __int64 v83; // r9
  __int64 v84; // rax
  unsigned int v85; // eax
  _WORD *v86; // rcx
  unsigned __int64 v87; // r8
  __int64 v88; // r15
  __int64 *v89; // rax
  __int64 v90; // r15
  _WORD *v91; // r8
  unsigned __int64 v92; // rdi
  int v93; // esi
  __int64 v94; // rax
  _QWORD *v95; // rax
  unsigned __int64 v96; // rcx
  __int64 v97; // rax
  __int64 *v98; // r10
  __int64 v99; // r11
  __int64 v100; // rsi
  __int64 v101; // rdx
  int v102; // r8d
  int v103; // eax
  int v104; // edx
  unsigned __int64 v105; // rax
  __int64 *v106; // rax
  _QWORD *v107; // rax
  size_t n; // [rsp+8h] [rbp-178h]
  int v109; // [rsp+14h] [rbp-16Ch]
  _QWORD *dest; // [rsp+20h] [rbp-160h]
  __int64 v112; // [rsp+28h] [rbp-158h]
  __int64 v113; // [rsp+30h] [rbp-150h]
  __int64 v114; // [rsp+38h] [rbp-148h]
  __int64 *v115; // [rsp+40h] [rbp-140h]
  unsigned __int64 v116; // [rsp+40h] [rbp-140h]
  _WORD *v117; // [rsp+40h] [rbp-140h]
  __int64 v118; // [rsp+48h] [rbp-138h]
  _QWORD *v119; // [rsp+48h] [rbp-138h]
  __int64 v120; // [rsp+48h] [rbp-138h]
  __int64 v121; // [rsp+48h] [rbp-138h]
  __int64 v122; // [rsp+48h] [rbp-138h]
  __int64 v123; // [rsp+50h] [rbp-130h]
  __int64 v124; // [rsp+50h] [rbp-130h]
  __int64 v125; // [rsp+50h] [rbp-130h]
  unsigned __int64 v126; // [rsp+50h] [rbp-130h]
  __int64 v127; // [rsp+50h] [rbp-130h]
  unsigned __int64 v128; // [rsp+50h] [rbp-130h]
  __int64 v129; // [rsp+58h] [rbp-128h]
  unsigned __int64 v130; // [rsp+60h] [rbp-120h]
  int v131; // [rsp+90h] [rbp-F0h]
  unsigned int v132; // [rsp+90h] [rbp-F0h]
  __int64 *v133; // [rsp+90h] [rbp-F0h]
  unsigned int v134; // [rsp+90h] [rbp-F0h]
  __int64 v135; // [rsp+98h] [rbp-E8h]
  __int64 v136; // [rsp+A0h] [rbp-E0h]
  __int64 v137; // [rsp+A8h] [rbp-D8h]
  __int64 v138; // [rsp+B0h] [rbp-D0h]
  __int64 v139; // [rsp+B0h] [rbp-D0h]
  __int64 *v140; // [rsp+B0h] [rbp-D0h]
  __int64 v141; // [rsp+B8h] [rbp-C8h]
  __int64 v142; // [rsp+C8h] [rbp-B8h]
  __int64 v143; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v144; // [rsp+D8h] [rbp-A8h] BYREF
  __int64 v145[2]; // [rsp+E0h] [rbp-A0h] BYREF
  __int16 v146; // [rsp+F0h] [rbp-90h]
  _WORD *v147; // [rsp+100h] [rbp-80h] BYREF
  __int64 v148; // [rsp+108h] [rbp-78h]
  _WORD v149[56]; // [rsp+110h] [rbp-70h] BYREF

  v9 = a1;
  v10 = sub_1560180(a1 + 112, 18);
  if ( (_BYTE)v10 )
    return 0;
  v11 = *(_QWORD *)(a1 + 80);
  v12 = v10;
  v142 = a1 + 72;
  if ( v11 == a1 + 72 )
  {
LABEL_13:
    v15 = *(_QWORD *)(a1 + 24);
    v16 = *(_QWORD *)(v15 + 16);
    v17 = 8LL * *(unsigned int *)(v15 + 12);
    v18 = (const void *)(v16 + 8);
    v19 = (const void *)(v16 + v17);
    n = v17 - 8;
    v20 = (v17 - 8) >> 3;
    if ( (unsigned __int64)(v17 - 8) > 0x7FFFFFFFFFFFFFF8LL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    dest = 0;
    if ( v20 )
      dest = (_QWORD *)sub_22077B0(n);
    if ( v19 != v18 )
      memcpy(dest, v18, n);
    v21 = sub_1644EA0(**(__int64 ***)(v15 + 16), dest, v20, 0);
    v22 = *(_BYTE *)(a1 + 32);
    v109 = v20;
    v23 = v21;
    v149[0] = 257;
    v24 = v22 & 0xF;
    v141 = sub_1648B60(120);
    if ( v141 )
      sub_15E2490(v141, v23, v24, (__int64)&v147, 0);
    sub_15E4330(v141, a1);
    *(_QWORD *)(v141 + 48) = *(_QWORD *)(a1 + 48);
    sub_1631B60(*(_QWORD *)(a1 + 40) + 24LL, v141);
    v25 = *(_QWORD *)(a1 + 56);
    *(_QWORD *)(v141 + 64) = a1 + 56;
    v25 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v141 + 56) = v25 | *(_QWORD *)(v141 + 56) & 7LL;
    *(_QWORD *)(v25 + 8) = v141 + 56;
    *(_QWORD *)(a1 + 56) = *(_QWORD *)(a1 + 56) & 7LL | (v141 + 56);
    sub_164B7C0(v141, a1);
    if ( !*(_QWORD *)(a1 + 8) )
    {
      v60 = 0;
      v137 = 0;
LABEL_67:
      if ( v142 != (*(_QWORD *)(v9 + 72) & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v61 = *(__int64 **)(v9 + 80);
        v62 = *(unsigned __int64 **)(v141 + 80);
        if ( (unsigned __int64 *)v142 != v62 )
        {
          if ( v142 != v141 + 72 )
            sub_15809C0(v141 + 72, v142, *(_QWORD *)(v9 + 80), v142);
          if ( (unsigned __int64 *)v142 != v62 && (__int64 *)v142 != v61 )
          {
            v63 = *(_QWORD *)(v9 + 72) & 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)((*v61 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v142;
            *(_QWORD *)(v9 + 72) = *(_QWORD *)(v9 + 72) & 7LL | *v61 & 0xFFFFFFFFFFFFFFF8LL;
            v64 = *v62;
            *(_QWORD *)(v63 + 8) = v62;
            v64 &= 0xFFFFFFFFFFFFFFF8LL;
            *v61 = v64 | *v61 & 7;
            *(_QWORD *)(v64 + 8) = v61;
            *v62 = v63 | *v62 & 7;
          }
        }
      }
      if ( (*(_BYTE *)(v9 + 18) & 1) != 0 )
      {
        sub_15E08E0(v9, v142);
        v65 = *(_QWORD *)(v9 + 88);
        if ( (*(_BYTE *)(v9 + 18) & 1) != 0 )
          sub_15E08E0(v9, v142);
        v66 = *(_QWORD *)(v9 + 88);
      }
      else
      {
        v65 = *(_QWORD *)(v9 + 88);
        v66 = v65;
      }
      v67 = v66 + 40LL * *(_QWORD *)(v9 + 96);
      if ( (*(_BYTE *)(v141 + 18) & 1) != 0 )
        sub_15E08E0(v141, v142);
      for ( i = *(_QWORD *)(v141 + 88); v65 != v67; i += 40 )
      {
        sub_164D160(v65, i, a2, a3, a4, a5, v26, v27, a8, a9);
        v69 = v65;
        v65 += 40;
        sub_164B7C0(i, v69);
      }
      v147 = v149;
      v148 = 0x100000000LL;
      sub_1626D60(v9, (__int64)&v147);
      v70 = (unsigned __int64)v147;
      v71 = &v147[8 * (unsigned int)v148];
      if ( v147 != v71 )
      {
        do
        {
          v72 = *(_QWORD *)(v70 + 8);
          v73 = *(_DWORD *)v70;
          v70 += 16LL;
          sub_16267C0(v141, v73, v72);
        }
        while ( v71 != (_WORD *)v70 );
      }
      v74 = sub_15A4510((__int64 ***)v141, *(__int64 ***)v9, 0);
      sub_164D160(v9, v74, a2, a3, a4, a5, v75, v76, a8, a9);
      sub_159D9E0(v141);
      sub_15E3D00(v9);
      if ( v147 != v149 )
        _libc_free((unsigned __int64)v147);
      if ( v60 )
        j_j___libc_free_0(v60, v137);
      if ( dest )
        j_j___libc_free_0(dest, n);
      return 1;
    }
    v28 = 0;
    v136 = 0;
    v129 = 24LL * (unsigned int)v20;
    v130 = (unsigned int)v20;
    v112 = 8LL * (unsigned int)v20;
    v29 = *(_QWORD *)(a1 + 8);
    while ( 1 )
    {
      v30 = v29;
      v29 = *(_QWORD *)(v29 + 8);
      v31 = (unsigned __int64)sub_1648700(v30);
      v32 = *(_BYTE *)(v31 + 16);
      if ( v32 <= 0x17u )
        break;
      if ( v32 == 78 )
      {
        v33 = v31;
        v34 = v31 & 0xFFFFFFFFFFFFFFF8LL;
        v35 = v33 | 4;
      }
      else
      {
        if ( v32 != 29 )
          break;
        v105 = v31;
        v34 = v31 & 0xFFFFFFFFFFFFFFF8LL;
        v35 = v105 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v143 = v35;
      v137 = v136 - (_QWORD)v28;
      if ( v34 )
      {
        v36 = (__int64 *)(v34 - 24LL * (*(_DWORD *)(v34 + 20) & 0xFFFFFFF));
        v37 = &v36[(unsigned __int64)v129 / 8];
        if ( (v136 - (__int64)v28) >> 3 >= v130 )
        {
          if ( v130 )
          {
            if ( v37 == v36 )
            {
              v131 = 0;
              v139 = 0;
            }
            else
            {
              v106 = v28;
              do
              {
                if ( v106 )
                  *v106 = *v36;
                v36 += 3;
                ++v106;
              }
              while ( v37 != v36 );
              v139 = (__int64)(0x5555555555555558LL * ((unsigned __int64)(v129 - 24) >> 3) + 8) >> 3;
              v131 = v139;
            }
            v40 = v34;
          }
          else
          {
            if ( v129 )
            {
              v77 = v28;
              do
              {
                v78 = *v36;
                ++v77;
                v36 += 3;
                *(v77 - 1) = v78;
              }
              while ( v77 != v28 );
            }
            v139 = 0;
            v40 = v34;
            v131 = 0;
          }
        }
        else
        {
          v38 = 0;
          if ( v130 )
            v38 = sub_22077B0(v112);
          for ( j = (_QWORD *)v38; v37 != v36; ++j )
          {
            if ( j )
              *j = *v36;
            v36 += 3;
          }
          if ( v28 )
          {
            v138 = v38;
            j_j___libc_free_0(v28, v137);
            v38 = v138;
          }
          v28 = (__int64 *)v38;
          v137 = v112;
          v136 = v38 + v112;
          v139 = v112 >> 3;
          v131 = v112 >> 3;
          v40 = v143 & 0xFFFFFFFFFFFFFFF8LL;
        }
        v144 = *(_QWORD *)(v40 + 56);
        if ( v144 )
        {
          v147 = v149;
          v148 = 0x800000000LL;
          if ( v109 )
          {
            v124 = v29;
            v81 = 0;
            do
            {
              v83 = sub_1560230(&v144, v81);
              v84 = (unsigned int)v148;
              if ( (unsigned int)v148 >= HIDWORD(v148) )
              {
                v121 = v83;
                sub_16CD150((__int64)&v147, v149, 0, 8, v82, v83);
                v84 = (unsigned int)v148;
                v83 = v121;
              }
              ++v81;
              *(_QWORD *)&v147[4 * v84] = v83;
              v85 = v148 + 1;
              LODWORD(v148) = v148 + 1;
            }
            while ( v109 != v81 );
            v29 = v124;
            v86 = v147;
            v87 = v85;
          }
          else
          {
            v86 = v149;
            v87 = 0;
          }
          v116 = v87;
          v119 = v86;
          v125 = sub_1560240(&v144);
          v88 = sub_1560250(&v144);
          v89 = (__int64 *)sub_15E0530(a1);
          v144 = sub_155FDB0(v89, v88, v125, v119, v116);
          if ( v147 != v149 )
            _libc_free((unsigned __int64)v147);
        }
        v147 = v149;
        v148 = 0x100000000LL;
        sub_1740980(&v143, (__int64)&v147);
        v41 = *(_BYTE *)(v34 + 16) == 29;
        v146 = 257;
        if ( v41 )
        {
          v123 = *(_QWORD *)(v34 - 24);
          v118 = *(_QWORD *)(v34 - 48);
          v42 = &v147[28 * (unsigned int)v148];
          if ( v147 == v42 )
          {
            v44 = 0;
          }
          else
          {
            v43 = (unsigned __int64)v147;
            v44 = 0;
            do
            {
              v45 = *(_QWORD *)(v43 + 40) - *(_QWORD *)(v43 + 32);
              v43 += 56LL;
              v44 += v45 >> 3;
            }
            while ( v42 != (_WORD *)v43 );
          }
          v113 = *(_QWORD *)(*(_QWORD *)v141 + 24LL);
          v114 = (unsigned int)v148;
          v115 = (__int64 *)v147;
          v132 = v44 + v131 + 3;
          v46 = sub_1648AB0(72, v132, 16 * (int)v148);
          v47 = (__int64)v46;
          if ( v46 )
          {
            sub_15F1EA0((__int64)v46, **(_QWORD **)(v113 + 16), 5, (__int64)&v46[-3 * v132], v132, v34);
            *(_QWORD *)(v47 + 56) = 0;
            sub_15F6500(v47, v113, v141, v118, v123, (__int64)v145, v28, v139, v115, v114);
          }
          v48 = v47 & 0xFFFFFFFFFFFFFFF8LL;
          LOWORD(v49) = *(_WORD *)(v48 + 18);
          v50 = (unsigned __int16)v49;
LABEL_47:
          LOWORD(v50) = v50 & 0x8000;
          v51 = (__int64 *)(v48 + 48);
          v52 = v50 | v49 & 3 | (4 * ((*(unsigned __int16 *)((v143 & 0xFFFFFFFFFFFFFFF8LL) + 18) >> 2) & 0x3FFFDFFF));
          *(_QWORD *)(v48 + 56) = v144;
          *(_WORD *)(v48 + 18) = v52;
          v53 = *(_QWORD *)(v34 + 48);
          v145[0] = v53;
          if ( v53 )
          {
            sub_1623A60((__int64)v145, v53, 2);
            v51 = (__int64 *)(v48 + 48);
            if ( (__int64 *)(v48 + 48) == v145 )
            {
              if ( v145[0] )
                sub_161E7C0((__int64)v145, v145[0]);
              goto LABEL_51;
            }
            v79 = *(_QWORD *)(v48 + 48);
            if ( !v79 )
            {
LABEL_97:
              v80 = (unsigned __int8 *)v145[0];
              *(_QWORD *)(v48 + 48) = v145[0];
              if ( v80 )
                sub_1623210((__int64)v145, v80, (__int64)v51);
              goto LABEL_51;
            }
          }
          else if ( v51 == v145 || (v79 = *(_QWORD *)(v48 + 48)) == 0 )
          {
LABEL_51:
            if ( (unsigned __int8)sub_1625980(v34, v145) )
              sub_15F3B70(v48, v145[0]);
            if ( *(_QWORD *)(v34 + 8) )
              sub_164D160(v34, v48, a2, a3, a4, a5, v54, v55, a8, a9);
            sub_164B7C0(v48, v34);
            sub_15F20C0((_QWORD *)v34);
            v56 = (unsigned __int64)v147;
            v57 = &v147[28 * (unsigned int)v148];
            if ( v147 != (_WORD *)v57 )
            {
              do
              {
                v58 = *(v57 - 3);
                v57 -= 7;
                if ( v58 )
                  j_j___libc_free_0(v58, v57[6] - v58);
                if ( (_QWORD *)*v57 != v57 + 2 )
                  j_j___libc_free_0(*v57, v57[2] + 1LL);
              }
              while ( (_QWORD *)v56 != v57 );
              v57 = v147;
            }
            if ( v57 != (_QWORD *)v149 )
              _libc_free((unsigned __int64)v57);
            goto LABEL_24;
          }
          v140 = v51;
          sub_161E7C0((__int64)v51, v79);
          v51 = v140;
          goto LABEL_97;
        }
        v90 = *(_QWORD *)(*(_QWORD *)v141 + 24LL);
        v91 = &v147[28 * (unsigned int)v148];
        if ( v147 == v91 )
        {
          v122 = (unsigned int)v148;
          v128 = (unsigned __int64)v147;
          v134 = v131 + 1;
          v107 = sub_1648AB0(72, v134, 16 * (int)v148);
          v102 = v134;
          v49 = (__int64)v107;
          if ( v107 )
          {
            v97 = v139;
            v98 = (__int64 *)v128;
            v99 = v122;
            goto LABEL_114;
          }
        }
        else
        {
          v92 = (unsigned __int64)v147;
          v93 = 0;
          do
          {
            v94 = *(_QWORD *)(v92 + 40) - *(_QWORD *)(v92 + 32);
            v92 += 56LL;
            v93 += v94 >> 3;
          }
          while ( v91 != (_WORD *)v92 );
          v117 = &v147[28 * (unsigned int)v148];
          v120 = (unsigned int)v148;
          v126 = (unsigned __int64)v147;
          v95 = sub_1648AB0(72, v131 + 1 + v93, 16 * (int)v148);
          v96 = v126;
          v49 = (__int64)v95;
          if ( v95 )
          {
            v97 = v139;
            v98 = (__int64 *)v126;
            v99 = v120;
            LODWORD(v100) = 0;
            do
            {
              v101 = *(_QWORD *)(v96 + 40) - *(_QWORD *)(v96 + 32);
              v96 += 56LL;
              v100 = (unsigned int)(v101 >> 3) + (unsigned int)v100;
            }
            while ( v117 != (_WORD *)v96 );
            v139 += v100;
            v102 = v131 + 1 + v100;
LABEL_114:
            v127 = v97;
            v133 = v98;
            v135 = v99;
            sub_15F1EA0(v49, **(_QWORD **)(v90 + 16), 54, v49 - 24 * v139 - 24, v102, v34);
            *(_QWORD *)(v49 + 56) = 0;
            sub_15F5B40(v49, v90, v141, v28, v127, (__int64)v145, v133, v135);
          }
        }
        v48 = v49 & 0xFFFFFFFFFFFFFFF8LL;
        v103 = *(_WORD *)((v49 & 0xFFFFFFFFFFFFFFF8LL) + 18) & 0x8000;
        v104 = *(_WORD *)((v49 & 0xFFFFFFFFFFFFFFF8LL) + 18) & 0x7FFC;
        v50 = v103 | v104 | *(_WORD *)(v34 + 18) & 3;
        *(_WORD *)((v49 & 0xFFFFFFFFFFFFFFF8LL) + 18) = v103 | v104 | *(_WORD *)(v34 + 18) & 3;
        LOBYTE(v49) = v50;
        goto LABEL_47;
      }
LABEL_24:
      if ( !v29 )
      {
        v9 = a1;
        v60 = v28;
        goto LABEL_67;
      }
    }
    v137 = v136 - (_QWORD)v28;
    goto LABEL_24;
  }
  while ( 1 )
  {
    if ( !v11 )
      BUG();
    v13 = *(_QWORD *)(v11 + 24);
    if ( v13 != v11 + 16 )
      break;
LABEL_12:
    v11 = *(_QWORD *)(v11 + 8);
    if ( v142 == v11 )
      goto LABEL_13;
  }
  while ( 1 )
  {
    if ( !v13 )
      BUG();
    if ( *(_BYTE *)(v13 - 8) == 78 )
    {
      if ( (*(_WORD *)(v13 - 6) & 3) == 2 )
        return v12;
      v14 = *(_QWORD *)(v13 - 48);
      if ( !*(_BYTE *)(v14 + 16) && (*(_BYTE *)(v14 + 33) & 0x20) != 0 && *(_DWORD *)(v14 + 36) == 214 )
        return v12;
    }
    v13 = *(_QWORD *)(v13 + 8);
    if ( v11 + 16 == v13 )
      goto LABEL_12;
  }
}
