// Function: sub_174DF90
// Address: 0x174df90
//
__int64 __fastcall sub_174DF90(
        __int64 *a1,
        __int64 a2,
        __int64 ***a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // r8
  __int64 v14; // r10
  __int64 v15; // rcx
  char v16; // al
  char v17; // dl
  __int64 *v18; // r14
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // r15
  int v22; // eax
  __int64 v23; // r13
  int v24; // eax
  int v25; // eax
  const void **v26; // rsi
  unsigned int v27; // r14d
  __int64 v28; // r10
  unsigned __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rdx
  int v32; // eax
  bool v33; // al
  __int64 **v34; // rdi
  __int64 ***v35; // r13
  __int64 v36; // r12
  _QWORD *v37; // rax
  double v38; // xmm4_8
  double v39; // xmm5_8
  unsigned int v41; // r15d
  __int64 v42; // rdi
  int v43; // eax
  bool v44; // al
  unsigned int v45; // r15d
  __int64 v46; // rdi
  int v47; // eax
  __int64 v48; // r15
  int v49; // eax
  __int64 v50; // rax
  __int64 v51; // r13
  __int64 v52; // rdx
  __int64 ***v53; // rax
  __int64 **v54; // rsi
  __int64 v55; // r14
  const char ***v56; // r10
  __int64 v57; // rax
  __int64 v58; // r13
  __int64 v59; // r12
  _QWORD *v60; // rax
  double v61; // xmm4_8
  double v62; // xmm5_8
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rax
  const char ***v66; // r10
  const char **v67; // rax
  __int64 v68; // rsi
  __int64 v69; // rax
  __int64 *v70; // rsi
  __int64 v71; // rdi
  __int64 v72; // rdx
  const char **v73; // rsi
  __int64 v74; // rsi
  __int64 v75; // rdx
  unsigned __int8 *v76; // rsi
  int v77; // eax
  __int64 v78; // rax
  __int64 v79; // r13
  __int64 v80; // rdx
  __int64 v81; // rax
  unsigned int v82; // r15d
  __int64 v83; // rax
  char v84; // cl
  unsigned int v85; // r13d
  int v86; // eax
  bool v87; // al
  __int64 v88; // rax
  __int64 v89; // rsi
  __int64 v90; // rax
  bool v91; // zf
  __int64 v92; // rsi
  __int64 v93; // rsi
  unsigned __int8 *v94; // rsi
  unsigned int v95; // r15d
  __int64 v96; // rax
  char v97; // cl
  unsigned int v98; // r13d
  int v99; // eax
  bool v100; // al
  unsigned int v101; // [rsp+0h] [rbp-F0h]
  const char ***v102; // [rsp+0h] [rbp-F0h]
  int v103; // [rsp+0h] [rbp-F0h]
  int v104; // [rsp+0h] [rbp-F0h]
  int v105; // [rsp+8h] [rbp-E8h]
  __int64 v106; // [rsp+8h] [rbp-E8h]
  __int64 v107; // [rsp+8h] [rbp-E8h]
  __int64 v108; // [rsp+8h] [rbp-E8h]
  __int64 v109; // [rsp+8h] [rbp-E8h]
  __int64 v110; // [rsp+8h] [rbp-E8h]
  __int64 v111; // [rsp+8h] [rbp-E8h]
  int v112; // [rsp+14h] [rbp-DCh]
  __int64 v113; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v114; // [rsp+18h] [rbp-D8h]
  __int64 v115; // [rsp+18h] [rbp-D8h]
  __int64 v116; // [rsp+18h] [rbp-D8h]
  __int64 v117; // [rsp+18h] [rbp-D8h]
  const char ***v118; // [rsp+18h] [rbp-D8h]
  __int64 *v119; // [rsp+18h] [rbp-D8h]
  const char ***v120; // [rsp+18h] [rbp-D8h]
  __int64 v121; // [rsp+18h] [rbp-D8h]
  __int64 *v122; // [rsp+18h] [rbp-D8h]
  __int64 v123; // [rsp+18h] [rbp-D8h]
  __int64 v124; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v125; // [rsp+28h] [rbp-C8h] BYREF
  _QWORD v126[2]; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v127[2]; // [rsp+40h] [rbp-B0h] BYREF
  __int16 v128; // [rsp+50h] [rbp-A0h]
  _QWORD *v129; // [rsp+60h] [rbp-90h] BYREF
  const char *v130; // [rsp+68h] [rbp-88h]
  __int16 v131; // [rsp+70h] [rbp-80h]
  const char *v132; // [rsp+80h] [rbp-70h] BYREF
  __int64 v133; // [rsp+88h] [rbp-68h]
  __int16 v134; // [rsp+90h] [rbp-60h]
  const char **v135; // [rsp+A0h] [rbp-50h] BYREF
  char *v136; // [rsp+A8h] [rbp-48h]
  __int64 v137; // [rsp+B0h] [rbp-40h]
  unsigned int v138; // [rsp+B8h] [rbp-38h]

  v11 = a2;
  v14 = *(_QWORD *)(a2 - 24);
  v15 = *(_QWORD *)v14;
  v16 = *(_BYTE *)(*(_QWORD *)v14 + 8LL);
  v17 = v16;
  if ( v16 == 16 )
    v17 = *(_BYTE *)(**(_QWORD **)(v15 + 16) + 8LL);
  if ( v17 != 11 )
    return 0;
  v18 = *(__int64 **)(a2 - 48);
  v19 = *(unsigned __int8 *)(v14 + 16);
  v112 = *(_WORD *)(a2 + 18) & 0x7FFF;
  if ( v112 == 40 )
  {
    if ( (_BYTE)v19 == 13 )
    {
      v45 = *(_DWORD *)(v14 + 32);
      if ( v45 <= 0x40 )
      {
        if ( !*(_QWORD *)(v14 + 24) )
          goto LABEL_53;
        goto LABEL_7;
      }
      v109 = a2;
      v46 = v14 + 24;
      v116 = *(_QWORD *)(a2 - 24);
    }
    else
    {
      if ( (unsigned __int8)v19 > 0x10u || v16 != 16 )
        return 0;
      v109 = a2;
      v116 = *(_QWORD *)(a2 - 24);
      v64 = sub_15A1020((_BYTE *)v14, a2, v19, v15);
      v14 = v116;
      v11 = a2;
      if ( !v64 || *(_BYTE *)(v64 + 16) != 13 )
      {
        v103 = *(_QWORD *)(*(_QWORD *)v116 + 32LL);
        if ( !v103 )
          goto LABEL_53;
        v82 = 0;
        while ( 1 )
        {
          v110 = v11;
          v121 = v14;
          v83 = sub_15A0A60(v14, v82);
          v14 = v121;
          v11 = v110;
          if ( !v83 )
            break;
          v84 = *(_BYTE *)(v83 + 16);
          if ( v84 != 9 )
          {
            if ( v84 != 13 )
              break;
            v85 = *(_DWORD *)(v83 + 32);
            if ( v85 <= 0x40 )
            {
              v87 = *(_QWORD *)(v83 + 24) == 0;
            }
            else
            {
              v86 = sub_16A57B0(v83 + 24);
              v14 = v121;
              v11 = v110;
              v87 = v85 == v86;
            }
            if ( !v87 )
              break;
          }
          if ( v103 == ++v82 )
            goto LABEL_53;
        }
LABEL_6:
        if ( *(_BYTE *)(v14 + 16) != 13 )
          return 0;
        goto LABEL_7;
      }
      v45 = *(_DWORD *)(v64 + 32);
      if ( v45 <= 0x40 )
      {
        if ( !*(_QWORD *)(v64 + 24) )
          goto LABEL_53;
        goto LABEL_6;
      }
      v46 = v64 + 24;
    }
    v47 = sub_16A57B0(v46);
    v14 = v116;
    v11 = v109;
    if ( v45 != v47 )
      goto LABEL_6;
    goto LABEL_53;
  }
  if ( v112 != 38 )
    goto LABEL_6;
  if ( (_BYTE)v19 == 13 )
  {
    v41 = *(_DWORD *)(v14 + 32);
    if ( v41 > 0x40 )
    {
      v108 = a2;
      v42 = v14 + 24;
      v115 = *(_QWORD *)(a2 - 24);
LABEL_46:
      v43 = sub_16A58F0(v42);
      v14 = v115;
      v11 = v108;
      v44 = v41 == v43;
      goto LABEL_47;
    }
    if ( *(_QWORD *)(v14 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v41) )
      goto LABEL_53;
LABEL_7:
    v20 = *(_QWORD *)(v11 + 8);
    if ( v20 )
    {
      v21 = *(_QWORD *)(v20 + 8);
      if ( !v21 )
      {
        v22 = *(unsigned __int16 *)(v11 + 18);
        BYTE1(v22) &= ~0x80u;
        if ( (unsigned int)(v22 - 32) <= 1 )
        {
          v23 = v14 + 24;
          if ( *(_DWORD *)(v14 + 32) <= 0x40u )
          {
            v78 = *(_QWORD *)(v14 + 24);
            if ( !v78 || (v78 & (v78 - 1)) == 0 )
            {
LABEL_13:
              v26 = (const void **)v18;
              v106 = v14;
              sub_14C2530((__int64)&v135, v18, a1[333], 0, a1[330], (__int64)a3, a1[332], 0);
              v27 = (unsigned int)v136;
              v28 = v106;
              LODWORD(v133) = (_DWORD)v136;
              if ( (unsigned int)v136 > 0x40 )
              {
                v26 = (const void **)&v135;
                sub_16A4FD0((__int64)&v132, (const void **)&v135);
                v27 = v133;
                v28 = v106;
                if ( (unsigned int)v133 > 0x40 )
                {
                  sub_16A8F40((__int64 *)&v132);
                  v27 = v133;
                  v28 = v106;
                  LODWORD(v130) = v133;
                  v114 = (unsigned __int64)v132;
                  v129 = v132;
                  if ( (unsigned int)v133 > 0x40 )
                  {
                    v77 = sub_16A5940((__int64)&v129);
                    v28 = v106;
                    if ( v77 != 1 )
                    {
LABEL_32:
                      if ( v114 )
                        j_j___libc_free_0_0(v114);
                      goto LABEL_34;
                    }
                    goto LABEL_18;
                  }
LABEL_16:
                  v30 = v114;
                  if ( !v114 || ((v114 - 1) & v114) != 0 )
                    goto LABEL_34;
LABEL_18:
                  v31 = *(unsigned int *)(v28 + 32);
                  if ( (unsigned int)v31 <= 0x40 )
                  {
                    v33 = *(_QWORD *)(v28 + 24) == 0;
                  }
                  else
                  {
                    v101 = *(_DWORD *)(v28 + 32);
                    v107 = v28;
                    v32 = sub_16A57B0(v23);
                    v31 = v101;
                    v28 = v107;
                    v33 = v101 == v32;
                  }
                  if ( !v33 )
                  {
                    if ( (unsigned int)v31 <= 0x40 )
                    {
                      if ( *(_QWORD *)(v28 + 24) == v114 )
                        goto LABEL_31;
                    }
                    else
                    {
                      v26 = (const void **)&v129;
                      if ( sub_16A5220(v23, (const void **)&v129) )
                        goto LABEL_31;
                    }
                    v34 = *a3;
                    if ( v112 == 33 )
                      v35 = (__int64 ***)sub_15A04A0(v34);
                    else
                      v35 = (__int64 ***)sub_15A06D0(v34, (__int64)v26, v31, v30);
                    v21 = (__int64)a3[1];
                    if ( v21 )
                    {
                      v36 = *a1;
                      do
                      {
                        v37 = sub_1648700(v21);
                        sub_170B990(v36, (__int64)v37);
                        v21 = *(_QWORD *)(v21 + 8);
                      }
                      while ( v21 );
                      if ( v35 == a3 )
                        v35 = (__int64 ***)sub_1599EF0(*v35);
                      v21 = (__int64)a3;
                      sub_164D160((__int64)a3, (__int64)v35, a4, a5, a6, a7, v38, v39, a10, a11);
                    }
                  }
LABEL_31:
                  if ( v27 > 0x40 )
                    goto LABEL_32;
LABEL_34:
                  if ( v138 > 0x40 && v137 )
                    j_j___libc_free_0_0(v137);
                  if ( (unsigned int)v136 > 0x40 && v135 )
                    j_j___libc_free_0_0(v135);
                  return v21;
                }
                v29 = (unsigned __int64)v132;
              }
              else
              {
                v29 = (unsigned __int64)v135;
              }
              LODWORD(v130) = v27;
              v114 = ~v29 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v27);
              v129 = (_QWORD *)v114;
              goto LABEL_16;
            }
          }
          else
          {
            v105 = *(_DWORD *)(v14 + 32);
            v113 = v14;
            v24 = sub_16A57B0(v14 + 24);
            v14 = v113;
            if ( v105 == v24 )
              goto LABEL_13;
            v25 = sub_16A5940(v23);
            v14 = v113;
            if ( v25 == 1 )
              goto LABEL_13;
          }
        }
      }
    }
    return 0;
  }
  if ( v16 != 16 || (unsigned __int8)v19 > 0x10u )
    return 0;
  v108 = a2;
  v115 = *(_QWORD *)(a2 - 24);
  v63 = sub_15A1020((_BYTE *)v14, a2, v19, v15);
  v14 = v115;
  v11 = a2;
  if ( !v63 || *(_BYTE *)(v63 + 16) != 13 )
  {
    v104 = *(_QWORD *)(*(_QWORD *)v115 + 32LL);
    if ( !v104 )
      goto LABEL_53;
    v95 = 0;
    while ( 1 )
    {
      v111 = v11;
      v123 = v14;
      v96 = sub_15A0A60(v14, v95);
      v14 = v123;
      v11 = v111;
      if ( !v96 )
        goto LABEL_6;
      v97 = *(_BYTE *)(v96 + 16);
      if ( v97 != 9 )
      {
        if ( v97 != 13 )
          goto LABEL_6;
        v98 = *(_DWORD *)(v96 + 32);
        if ( v98 <= 0x40 )
        {
          v100 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v98) == *(_QWORD *)(v96 + 24);
        }
        else
        {
          v99 = sub_16A58F0(v96 + 24);
          v14 = v123;
          v11 = v111;
          v100 = v98 == v99;
        }
        if ( !v100 )
          goto LABEL_6;
      }
      if ( v104 == ++v95 )
        goto LABEL_53;
    }
  }
  v41 = *(_DWORD *)(v63 + 32);
  if ( v41 > 0x40 )
  {
    v42 = v63 + 24;
    goto LABEL_46;
  }
  v44 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v41) == *(_QWORD *)(v63 + 24);
LABEL_47:
  if ( !v44 )
    goto LABEL_6;
LABEL_53:
  v48 = *v18;
  v49 = sub_16431D0(*v18);
  v50 = sub_15A0680(v48, (unsigned int)(v49 - 1), 0);
  v51 = a1[1];
  v117 = v50;
  v132 = sub_1649960((__int64)v18);
  v133 = v52;
  v135 = &v132;
  LOWORD(v137) = 773;
  v136 = ".lobit";
  v53 = (__int64 ***)sub_173DE00(v51, (__int64)v18, v117, (__int64 *)&v135, 0, *(double *)a4.m128_u64, a5, a6);
  v54 = *a3;
  v55 = (__int64)v53;
  if ( *a3 != *v53 )
  {
    v56 = (const char ***)a1[1];
    v128 = 257;
    if ( v54 != *v53 )
    {
      v118 = v56;
      if ( *((_BYTE *)v53 + 16) > 0x10u )
      {
        v134 = 257;
        v65 = sub_15FE0A0(v53, (__int64)v54, 1, (__int64)&v132, 0);
        v66 = v118;
        v55 = v65;
        v67 = v118[1];
        if ( v67 )
        {
          v102 = v118;
          v119 = (__int64 *)v118[2];
          sub_157E9D0((__int64)(v67 + 5), v55);
          v66 = v102;
          v68 = *v119;
          v69 = *(_QWORD *)(v55 + 24) & 7LL;
          *(_QWORD *)(v55 + 32) = v119;
          v68 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v55 + 24) = v68 | v69;
          *(_QWORD *)(v68 + 8) = v55 + 24;
          *v119 = *v119 & 7 | (v55 + 24);
        }
        v70 = v127;
        v71 = v55;
        v120 = v66;
        sub_164B780(v55, v127);
        v124 = v55;
        if ( !v120[10] )
          goto LABEL_137;
        ((void (__fastcall *)(const char ***, __int64 *))v120[11])(v120 + 8, &v124);
        v73 = *v120;
        if ( *v120 )
        {
          v135 = *v120;
          sub_1623A60((__int64)&v135, (__int64)v73, 2);
          v74 = *(_QWORD *)(v55 + 48);
          v75 = v55 + 48;
          if ( v74 )
          {
            sub_161E7C0(v55 + 48, v74);
            v75 = v55 + 48;
          }
          v76 = (unsigned __int8 *)v135;
          *(_QWORD *)(v55 + 48) = v135;
          if ( v76 )
            sub_1623210((__int64)&v135, v76, v75);
        }
      }
      else
      {
        v55 = sub_15A4750(v53, v54, 1);
        v57 = sub_14DBA30(v55, (__int64)v118[12], 0);
        if ( v57 )
          v55 = v57;
      }
    }
  }
  if ( v112 != 38 )
    goto LABEL_59;
  v79 = a1[1];
  v126[0] = sub_1649960(v55);
  v129 = v126;
  v126[1] = v80;
  v131 = 773;
  v130 = ".not";
  if ( *(_BYTE *)(v55 + 16) <= 0x10u )
  {
    v55 = sub_15A2B00((__int64 *)v55, *(double *)a4.m128_u64, a5, a6);
    v81 = sub_14DBA30(v55, *(_QWORD *)(v79 + 96), 0);
    if ( v81 )
      v55 = v81;
    goto LABEL_59;
  }
  LOWORD(v137) = 257;
  v55 = sub_15FB630((__int64 *)v55, (__int64)&v135, 0);
  v88 = *(_QWORD *)(v79 + 8);
  if ( v88 )
  {
    v122 = *(__int64 **)(v79 + 16);
    sub_157E9D0(v88 + 40, v55);
    v89 = *v122;
    v90 = *(_QWORD *)(v55 + 24) & 7LL;
    *(_QWORD *)(v55 + 32) = v122;
    v89 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v55 + 24) = v89 | v90;
    *(_QWORD *)(v89 + 8) = v55 + 24;
    *v122 = *v122 & 7 | (v55 + 24);
  }
  v70 = (__int64 *)&v129;
  v71 = v55;
  sub_164B780(v55, (__int64 *)&v129);
  v91 = *(_QWORD *)(v79 + 80) == 0;
  v125 = v55;
  if ( v91 )
LABEL_137:
    sub_4263D6(v71, v70, v72);
  (*(void (__fastcall **)(__int64, __int64 *))(v79 + 88))(v79 + 64, &v125);
  v92 = *(_QWORD *)v79;
  if ( *(_QWORD *)v79 )
  {
    v132 = *(const char **)v79;
    sub_1623A60((__int64)&v132, v92, 2);
    v93 = *(_QWORD *)(v55 + 48);
    if ( v93 )
      sub_161E7C0(v55 + 48, v93);
    v94 = (unsigned __int8 *)v132;
    *(_QWORD *)(v55 + 48) = v132;
    if ( v94 )
      sub_1623210((__int64)&v132, v94, v55 + 48);
  }
LABEL_59:
  v58 = (__int64)a3[1];
  if ( !v58 )
    return 0;
  v59 = *a1;
  do
  {
    v60 = sub_1648700(v58);
    sub_170B990(v59, (__int64)v60);
    v58 = *(_QWORD *)(v58 + 8);
  }
  while ( v58 );
  if ( (__int64 ***)v55 == a3 )
    v55 = sub_1599EF0(*(__int64 ***)v55);
  v21 = (__int64)a3;
  sub_164D160((__int64)a3, v55, a4, a5, a6, a7, v61, v62, a10, a11);
  return v21;
}
