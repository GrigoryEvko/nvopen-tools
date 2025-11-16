// Function: sub_174FAC0
// Address: 0x174fac0
//
__int64 __fastcall sub_174FAC0(
        __int64 *a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v12; // rdi
  __int64 v13; // r14
  __int64 v15; // rbx
  __int64 **v16; // r15
  double v17; // xmm4_8
  double v18; // xmm5_8
  unsigned __int8 v19; // al
  __int64 v20; // rax
  char v21; // al
  __int64 v22; // rdx
  unsigned __int64 v23; // rdi
  __int64 v24; // rax
  char v25; // al
  __int64 v26; // rcx
  int v27; // esi
  int v28; // esi
  unsigned __int8 **v29; // rcx
  unsigned __int8 *v30; // r15
  __int64 v31; // rax
  __int64 v32; // rdi
  __int64 v33; // r12
  __int64 *v34; // rax
  __int64 *v35; // r14
  __int64 v36; // rax
  unsigned int v37; // ebx
  unsigned int v38; // eax
  unsigned int v39; // r15d
  unsigned int v40; // edx
  unsigned int v41; // esi
  char v42; // al
  __int64 v43; // rbx
  __int64 v44; // r13
  _QWORD *v45; // rax
  double v46; // xmm4_8
  double v47; // xmm5_8
  __int64 v48; // rsi
  unsigned int v49; // r15d
  unsigned int v50; // eax
  __int64 **v51; // rcx
  __int64 v52; // rdi
  __int64 *v53; // r12
  unsigned __int64 v54; // rax
  __int64 *v55; // rsi
  __int64 v56; // rdx
  const char *v57; // rdi
  __int64 v58; // r15
  __int64 v59; // r11
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  double v63; // xmm4_8
  double v64; // xmm5_8
  __int64 v65; // r11
  __int64 v66; // rbx
  const char *v67; // rax
  __int64 **v68; // rcx
  __int64 v69; // rdx
  unsigned __int8 *v70; // rbx
  const char *v71; // rax
  __int64 **v72; // rcx
  __int64 v73; // rdx
  unsigned __int8 *v74; // rax
  __int64 v75; // r12
  __int64 v76; // rax
  double v77; // xmm4_8
  double v78; // xmm5_8
  __int64 v79; // r11
  unsigned __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // r13
  __int64 v83; // r14
  __int64 v84; // rdx
  unsigned __int8 *v85; // rax
  __int64 v86; // r12
  __int64 v87; // r13
  _QWORD *v88; // rax
  unsigned __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rdx
  __int64 v92; // rcx
  int v93; // edx
  int v94; // edx
  __int64 **v95; // rcx
  __int64 *v96; // r15
  unsigned __int64 v97; // rdi
  __int64 **v98; // rsi
  __int64 v99; // rax
  __int64 v100; // rcx
  int v101; // eax
  int v102; // eax
  unsigned __int8 **v103; // rcx
  unsigned __int64 v104; // rax
  __int64 v105; // rsi
  int v106; // ecx
  int v107; // ecx
  __int64 **v108; // rsi
  __int64 v109; // rax
  char v110; // [rsp+7h] [rbp-A9h]
  char v111; // [rsp+8h] [rbp-A8h]
  __int64 *v112; // [rsp+8h] [rbp-A8h]
  int v113; // [rsp+10h] [rbp-A0h]
  unsigned int v114; // [rsp+10h] [rbp-A0h]
  __int64 v115; // [rsp+10h] [rbp-A0h]
  __int64 v116; // [rsp+18h] [rbp-98h]
  unsigned int v117; // [rsp+18h] [rbp-98h]
  __int64 **v118; // [rsp+18h] [rbp-98h]
  unsigned int v119; // [rsp+18h] [rbp-98h]
  __int64 v120; // [rsp+18h] [rbp-98h]
  __int64 v121; // [rsp+18h] [rbp-98h]
  __int64 v122; // [rsp+18h] [rbp-98h]
  int v123; // [rsp+2Ch] [rbp-84h] BYREF
  unsigned __int64 v124; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v125; // [rsp+38h] [rbp-78h]
  unsigned __int64 v126; // [rsp+40h] [rbp-70h] BYREF
  __int64 v127; // [rsp+48h] [rbp-68h]
  __int16 v128; // [rsp+50h] [rbp-60h]
  unsigned __int64 v129; // [rsp+60h] [rbp-50h] BYREF
  char *v130; // [rsp+68h] [rbp-48h]
  __int16 v131; // [rsp+70h] [rbp-40h]

  v12 = a2[1];
  if ( v12 && !*(_QWORD *)(v12 + 8) && *((_BYTE *)sub_1648700(v12) + 16) == 60 )
    return 0;
  v13 = sub_174B490(a1, (__int64)a2, a3, a4, a5, a6, a7, a8, a9, a10);
  if ( v13 )
    return v13;
  v15 = *(a2 - 3);
  v16 = (__int64 **)*a2;
  v116 = *(_QWORD *)v15;
  if ( *(_BYTE *)(*a2 + 8) != 16 && !(unsigned __int8)sub_1705440((__int64)a1, v116, (__int64)v16)
    || !(unsigned __int8)sub_174AD80(v15, (__int64)v16, &v123, a1) )
  {
    v19 = *(_BYTE *)(v15 + 16);
    if ( v19 <= 0x17u )
      return v13;
    if ( v19 == 60 )
    {
      v112 = *(__int64 **)(v15 - 24);
      v114 = sub_16431D0(*v112);
      v49 = sub_16431D0(*(_QWORD *)v15);
      v118 = (__int64 **)*a2;
      v50 = sub_16431D0(*a2);
      v51 = v118;
      if ( v114 < v50 )
      {
        v125 = v114;
        if ( v114 > 0x40 )
          sub_16A4EF0((__int64)&v124, 0, 0);
        else
          v124 = 0;
        if ( v49 )
        {
          if ( v49 > 0x40 )
          {
            sub_16A5260(&v124, 0, v49);
          }
          else
          {
            v80 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v49);
            if ( v125 > 0x40 )
              *(_QWORD *)v124 |= v80;
            else
              v124 |= v80;
          }
        }
        v81 = sub_15A1070(*v112, (__int64)&v124);
        v82 = a1[1];
        v83 = v81;
        v126 = (unsigned __int64)sub_1649960(v15);
        v127 = v84;
        v130 = ".mask";
        v131 = 773;
        v129 = (unsigned __int64)&v126;
        v85 = sub_1729500(v82, (unsigned __int8 *)v112, v83, (__int64 *)&v129, *(double *)a3.m128_u64, a4, a5);
        v131 = 257;
        v86 = *a2;
        v87 = (__int64)v85;
        v88 = sub_1648A60(56, 1u);
        v13 = (__int64)v88;
        if ( v88 )
          sub_15FC690((__int64)v88, v87, v86, (__int64)&v129, 0);
        if ( v125 <= 0x40 )
          return v13;
        v57 = (const char *)v124;
        if ( !v124 )
          return v13;
      }
      else
      {
        if ( v114 == v50 )
        {
          LODWORD(v127) = v114;
          if ( v114 > 0x40 )
            sub_16A4EF0((__int64)&v126, 0, 0);
          else
            v126 = 0;
          if ( v49 )
          {
            if ( v49 > 0x40 )
            {
              sub_16A5260(&v126, 0, v49);
            }
            else
            {
              v104 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v49);
              if ( (unsigned int)v127 > 0x40 )
                *(_QWORD *)v126 |= v104;
              else
                v126 |= v104;
            }
          }
          v131 = 257;
          v55 = v112;
          v56 = sub_15A1070(*v112, (__int64)&v126);
        }
        else
        {
          if ( v114 <= v50 )
            return v13;
          v52 = a1[1];
          v131 = 257;
          v119 = v50;
          v53 = (__int64 *)sub_1708970(v52, 36, (__int64)v112, v51, (__int64 *)&v129);
          LODWORD(v127) = v119;
          if ( v119 > 0x40 )
            sub_16A4EF0((__int64)&v126, 0, 0);
          else
            v126 = 0;
          if ( v49 )
          {
            if ( v49 > 0x40 )
            {
              sub_16A5260(&v126, 0, v49);
            }
            else
            {
              v54 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v49);
              if ( (unsigned int)v127 > 0x40 )
                *(_QWORD *)v126 |= v54;
              else
                v126 |= v54;
            }
          }
          v131 = 257;
          v55 = v53;
          v56 = sub_15A1070(*v53, (__int64)&v126);
        }
        v13 = sub_15FB440(26, v55, v56, (__int64)&v129, 0);
        if ( (unsigned int)v127 <= 0x40 )
          return v13;
        v57 = (const char *)v126;
        if ( !v126 )
          return v13;
      }
      j_j___libc_free_0_0(v57);
      return v13;
    }
    if ( v19 == 75 )
      return sub_174F020(a1, v15, (__int64)a2, 1, a3, a4, a5, a6, v17, v18, a9, a10);
    if ( (unsigned int)v19 - 35 > 0x11 )
      return v13;
    if ( v19 == 51 )
    {
      v58 = *(_QWORD *)(v15 - 48);
      if ( *(_BYTE *)(v58 + 16) == 75 )
      {
        v59 = *(_QWORD *)(v15 - 24);
        if ( *(_BYTE *)(v59 + 16) == 75 )
        {
          v60 = *(_QWORD *)(v58 + 8);
          if ( v60 )
          {
            if ( !*(_QWORD *)(v60 + 8) )
            {
              v61 = *(_QWORD *)(v59 + 8);
              if ( v61 )
              {
                if ( !*(_QWORD *)(v61 + 8) )
                {
                  v120 = *(_QWORD *)(v15 - 24);
                  v62 = sub_174F020(a1, *(_QWORD *)(v15 - 48), (__int64)a2, 0, a3, a4, a5, a6, v17, v18, a9, a10);
                  v65 = v120;
                  if ( v62
                    || (v109 = sub_174F020(a1, v120, (__int64)a2, 0, a3, a4, a5, a6, v63, v64, a9, a10), v65 = v120, v109) )
                  {
                    v66 = a1[1];
                    v121 = v65;
                    v67 = sub_1649960(v58);
                    v68 = (__int64 **)*a2;
                    v127 = v69;
                    v129 = (unsigned __int64)&v126;
                    v126 = (unsigned __int64)v67;
                    v131 = 261;
                    v70 = sub_1708970(v66, 37, v58, v68, (__int64 *)&v129);
                    v115 = a1[1];
                    v71 = sub_1649960(v121);
                    v72 = (__int64 **)*a2;
                    v127 = v73;
                    v126 = (unsigned __int64)v71;
                    v129 = (unsigned __int64)&v126;
                    v131 = 261;
                    v74 = sub_1708970(v115, 37, v121, v72, (__int64 *)&v129);
                    v131 = 257;
                    v75 = (__int64)v74;
                    v76 = sub_15FB440(27, (__int64 *)v70, (__int64)v74, (__int64)&v129, 0);
                    v79 = v121;
                    v13 = v76;
                    if ( v70[16] == 61 )
                    {
                      sub_174F020(a1, v58, (__int64)v70, 1, a3, a4, a5, a6, v77, v78, a9, a10);
                      v79 = v121;
                    }
                    if ( *(_BYTE *)(v75 + 16) == 61 )
                      sub_174F020(a1, v79, v75, 1, a3, a4, a5, a6, v77, v78, a9, a10);
                    return v13;
                  }
                }
              }
            }
          }
        }
      }
    }
    v20 = *(_QWORD *)(v15 + 8);
    if ( !v20 || *(_QWORD *)(v20 + 8) )
      return v13;
    v21 = *(_BYTE *)(v15 + 16);
    if ( v21 == 50 )
    {
      v92 = *(_QWORD *)(v15 - 48);
      v93 = *(unsigned __int8 *)(v92 + 16);
      if ( (unsigned __int8)v93 > 0x17u )
      {
        v94 = v93 - 24;
      }
      else
      {
        if ( (_BYTE)v93 != 5 )
          goto LABEL_20;
        v94 = *(unsigned __int16 *)(v92 + 18);
      }
      if ( v94 != 36
        || ((*(_BYTE *)(v92 + 23) & 0x40) == 0
          ? (v95 = (__int64 **)(v92 - 24LL * (*(_DWORD *)(v92 + 20) & 0xFFFFFFF)))
          : (v95 = *(__int64 ***)(v92 - 8)),
            (v96 = *v95) == 0 || (v97 = *(_QWORD *)(v15 - 24), *(_BYTE *)(v97 + 16) > 0x10u)) )
      {
LABEL_20:
        if ( v21 != 5 )
          return v13;
        if ( *(_WORD *)(v15 + 18) != 28 )
          return v13;
        v22 = *(_QWORD *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
        if ( !v22 )
          return v13;
        v23 = *(_QWORD *)(v15 + 24 * (1LL - (*(_DWORD *)(v15 + 20) & 0xFFFFFFF)));
        if ( !v23 )
          return v13;
        goto LABEL_24;
      }
    }
    else
    {
      if ( v21 != 5 )
        goto LABEL_96;
      if ( *(_WORD *)(v15 + 18) != 26 )
        goto LABEL_20;
      v105 = *(_QWORD *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
      v106 = *(unsigned __int8 *)(v105 + 16);
      if ( (unsigned __int8)v106 > 0x17u )
      {
        v107 = v106 - 24;
      }
      else
      {
        if ( (_BYTE)v106 != 5 )
          goto LABEL_20;
        v107 = *(unsigned __int16 *)(v105 + 18);
      }
      if ( v107 != 36 )
        goto LABEL_20;
      v108 = (*(_BYTE *)(v105 + 23) & 0x40) != 0
           ? *(__int64 ***)(v105 - 8)
           : (__int64 **)(v105 - 24LL * (*(_DWORD *)(v105 + 20) & 0xFFFFFFF));
      v96 = *v108;
      if ( !*v108 )
        goto LABEL_20;
      v97 = *(_QWORD *)(v15 + 24 * (1LL - (*(_DWORD *)(v15 + 20) & 0xFFFFFFF)));
      if ( !v97 )
        goto LABEL_20;
    }
    v98 = (__int64 **)*a2;
    if ( *a2 == *v96 )
    {
      v131 = 257;
      v99 = sub_15A3CB0(v97, v98, 0);
      return sub_15FB440(26, v96, v99, (__int64)&v129, 0);
    }
LABEL_96:
    if ( v21 == 52 )
    {
      v22 = *(_QWORD *)(v15 - 48);
      if ( !v22 )
        return v13;
      v23 = *(_QWORD *)(v15 - 24);
      if ( *(_BYTE *)(v23 + 16) > 0x10u )
        return v13;
LABEL_24:
      v24 = *(_QWORD *)(v22 + 8);
      if ( !v24 || *(_QWORD *)(v24 + 8) )
        return v13;
      v25 = *(_BYTE *)(v22 + 16);
      if ( v25 == 50 )
      {
        v100 = *(_QWORD *)(v22 - 48);
        v101 = *(unsigned __int8 *)(v100 + 16);
        if ( (unsigned __int8)v101 > 0x17u )
        {
          v102 = v101 - 24;
        }
        else
        {
          if ( (_BYTE)v101 != 5 )
            return v13;
          v102 = *(unsigned __int16 *)(v100 + 18);
        }
        if ( v102 != 36 )
          return v13;
        v103 = (*(_BYTE *)(v100 + 23) & 0x40) != 0
             ? *(unsigned __int8 ***)(v100 - 8)
             : (unsigned __int8 **)(v100 - 24LL * (*(_DWORD *)(v100 + 20) & 0xFFFFFFF));
        v30 = *v103;
        if ( !*v103 || v23 != *(_QWORD *)(v22 - 24) )
          return v13;
      }
      else
      {
        if ( v25 != 5 || *(_WORD *)(v22 + 18) != 26 )
          return v13;
        v26 = *(_QWORD *)(v22 - 24LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF));
        v27 = *(unsigned __int8 *)(v26 + 16);
        if ( (unsigned __int8)v27 > 0x17u )
        {
          v28 = v27 - 24;
        }
        else
        {
          if ( (_BYTE)v27 != 5 )
            return v13;
          v28 = *(unsigned __int16 *)(v26 + 18);
        }
        if ( v28 != 36 )
          return v13;
        v29 = (*(_BYTE *)(v26 + 23) & 0x40) != 0
            ? *(unsigned __int8 ***)(v26 - 8)
            : (unsigned __int8 **)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
        v30 = *v29;
        if ( !*v29 || v23 != *(_QWORD *)(v22 + 24 * (1LL - (*(_DWORD *)(v22 + 20) & 0xFFFFFFF))) )
          return v13;
      }
      if ( *a2 == *(_QWORD *)v30 )
      {
        v31 = sub_15A3CB0(v23, (__int64 **)*a2, 0);
        v32 = a1[1];
        v33 = v31;
        v128 = 257;
        v131 = 257;
        v34 = (__int64 *)sub_1729500(v32, v30, v31, (__int64 *)&v126, *(double *)a3.m128_u64, a4, a5);
        return sub_15FB440(28, v34, v33, (__int64)&v129, 0);
      }
      return v13;
    }
    goto LABEL_20;
  }
  v35 = (__int64 *)sub_174BF40(a1, v15, v16, 0);
  if ( *(_BYTE *)(v15 + 16) > 0x17u )
  {
    v36 = *(_QWORD *)(v15 + 8);
    if ( v36 )
    {
      if ( !*(_QWORD *)(v36 + 8) )
        sub_1AEBB60(v15, v35, a2, a1[332]);
    }
  }
  v113 = sub_16431D0(v116);
  v111 = v123;
  v117 = v113 - v123;
  v37 = v113 - v123;
  v38 = sub_16431D0((__int64)v16);
  LODWORD(v130) = v38;
  v39 = v38;
  if ( v38 <= 0x40 )
  {
    v41 = v37;
    v129 = 0;
    v40 = v38;
  }
  else
  {
    sub_16A4EF0((__int64)&v129, 0, 0);
    v40 = (unsigned int)v130;
    v41 = (_DWORD)v130 + v117 - v39;
  }
  if ( v40 != v41 )
  {
    if ( v41 > 0x3F || v40 > 0x40 )
      sub_16A5260(&v129, v41, v40);
    else
      v129 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v117 - (unsigned __int8)v39 + 64) << v41;
  }
  v42 = sub_14C1670((__int64)v35, (__int64)&v129, a1[333], 0, a1[330], (__int64)a2, a1[332]);
  if ( (unsigned int)v130 > 0x40 && v129 )
  {
    v110 = v42;
    j_j___libc_free_0_0(v129);
    v42 = v110;
  }
  if ( v42 )
  {
    v43 = a2[1];
    if ( !v43 )
      return 0;
    v44 = *a1;
    do
    {
      v45 = sub_1648700(v43);
      sub_170B990(v44, (__int64)v45);
      v43 = *(_QWORD *)(v43 + 8);
    }
    while ( v43 );
    if ( a2 == v35 )
      v35 = (__int64 *)sub_1599EF0((__int64 **)*a2);
    v48 = (__int64)v35;
    v13 = (__int64)a2;
    sub_164D160((__int64)a2, v48, a3, a4, a5, a6, v46, v47, a9, a10);
  }
  else
  {
    LODWORD(v130) = v39;
    if ( v39 > 0x40 )
      sub_16A4EF0((__int64)&v129, 0, 0);
    else
      v129 = 0;
    if ( v117 )
    {
      if ( v117 > 0x40 )
      {
        sub_16A5260(&v129, 0, v117);
      }
      else
      {
        v89 = 0xFFFFFFFFFFFFFFFFLL >> (v111 + 64 - v113);
        if ( (unsigned int)v130 > 0x40 )
          *(_QWORD *)v129 |= v89;
        else
          v129 |= v89;
      }
    }
    v90 = sub_15A1070(*v35, (__int64)&v129);
    v91 = v90;
    if ( (unsigned int)v130 > 0x40 && v129 )
    {
      v122 = v90;
      j_j___libc_free_0_0(v129);
      v91 = v122;
    }
    v131 = 257;
    return sub_15FB440(26, v35, v91, (__int64)&v129, 0);
  }
  return v13;
}
