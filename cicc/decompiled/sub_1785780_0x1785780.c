// Function: sub_1785780
// Address: 0x1785780
//
unsigned __int8 *__fastcall sub_1785780(
        __m128i *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v11; // r12
  __m128i v12; // xmm1
  __m128i v13; // xmm2
  char v14; // al
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // r13
  __int64 v18; // r14
  _QWORD *v19; // rax
  double v20; // xmm4_8
  double v21; // xmm5_8
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r14
  __int64 v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 *v30; // rax
  __int64 v31; // rsi
  __int64 v32; // r14
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 *v35; // rsi
  __int64 v36; // rdx
  int v37; // edi
  unsigned __int8 *v38; // rbx
  __int64 v39; // rax
  __int64 v40; // rsi
  __int64 v41; // rcx
  unsigned __int8 *v42; // r14
  __int64 v43; // rax
  unsigned __int8 *v44; // rdx
  __int64 *v45; // rsi
  unsigned __int8 v46; // al
  __int64 v47; // rdx
  __int64 *v48; // rcx
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rcx
  char v53; // al
  __int64 v54; // rcx
  __int64 *v55; // rdx
  __int64 *v56; // rax
  __int64 v57; // rax
  bool v58; // al
  bool v59; // zf
  __int16 v60; // dx
  __int64 v61; // rcx
  __int64 v62; // rsi
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // r14
  __int64 v66; // rax
  __int64 v67; // rdi
  __int64 *v68; // rax
  char v69; // al
  __int64 v70; // rdx
  unsigned __int8 *v71; // rax
  __int64 v72; // rdi
  _QWORD *v73; // rax
  __int64 v74; // rax
  unsigned __int8 *v75; // rsi
  __int64 v76; // rax
  unsigned __int8 v77; // dl
  unsigned __int64 v78; // rax
  __int64 *v79; // rax
  __int64 v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rcx
  __int64 v83; // rax
  double v84; // xmm4_8
  double v85; // xmm5_8
  _QWORD *v86; // r14
  double v87; // xmm4_8
  double v88; // xmm5_8
  __int64 v89; // rcx
  __int64 v90; // rdi
  __int64 v91; // rax
  unsigned int v92; // edx
  __int64 v93; // rdi
  int v94; // esi
  unsigned int i; // ecx
  __int64 v96; // rax
  unsigned int v97; // ecx
  int v98; // [rsp+0h] [rbp-D0h]
  bool v99; // [rsp+7h] [rbp-C9h]
  __int64 v100; // [rsp+8h] [rbp-C8h]
  __int64 v101; // [rsp+10h] [rbp-C0h]
  __int64 v102; // [rsp+18h] [rbp-B8h]
  __int64 v103; // [rsp+20h] [rbp-B0h] BYREF
  __int64 *v104; // [rsp+28h] [rbp-A8h] BYREF
  unsigned __int8 *v105; // [rsp+30h] [rbp-A0h] BYREF
  __int64 *v106; // [rsp+38h] [rbp-98h]
  __int64 *v107; // [rsp+40h] [rbp-90h]
  __m128i v108; // [rsp+50h] [rbp-80h] BYREF
  __m128i v109; // [rsp+60h] [rbp-70h]
  __int64 v110; // [rsp+70h] [rbp-60h]
  int v111; // [rsp+78h] [rbp-58h]
  __int64 v112; // [rsp+80h] [rbp-50h]
  __int64 v113; // [rsp+88h] [rbp-48h]

  v11 = (__int64 *)a2;
  v12 = _mm_loadu_si128(a1 + 167);
  v110 = a2;
  v13 = _mm_loadu_si128(a1 + 168);
  v108 = v12;
  v109 = v13;
  v14 = sub_15F24E0(a2);
  v15 = sub_13D6F10(*(_QWORD **)(a2 - 48), *(_QWORD *)(a2 - 24), v14, &v108);
  if ( v15 )
  {
    v16 = *(_QWORD *)(a2 + 8);
    if ( v16 )
    {
      v17 = a1->m128i_i64[0];
      v18 = v15;
      do
      {
        v19 = sub_1648700(v16);
        sub_170B990(v17, (__int64)v19);
        v16 = *(_QWORD *)(v16 + 8);
      }
      while ( v16 );
      if ( a2 == v18 )
        v18 = sub_1599EF0(*(__int64 ***)a2);
      sub_164D160(a2, v18, a3, *(double *)v12.m128i_i64, *(double *)v13.m128i_i64, a6, v20, v21, a9, a10);
      return (unsigned __int8 *)v11;
    }
    return 0;
  }
  v23 = (__int64)sub_1707490(
                   (__int64)a1,
                   (unsigned __int8 *)a2,
                   *(double *)a3.m128_u64,
                   *(double *)v12.m128i_i64,
                   *(double *)v13.m128i_i64);
  if ( v23 )
    return (unsigned __int8 *)v23;
  v26 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v26 + 16) <= 0x10u )
  {
    v27 = *(_QWORD *)(a2 - 48);
    v108.m128i_i64[1] = (__int64)&v105;
    if ( (unsigned __int8)sub_171FB50((__int64)&v108, v27, v24, v25) )
    {
      v109.m128i_i16[0] = 257;
      v39 = sub_15A2BF0(
              (__int64 *)v26,
              257,
              v28,
              v29,
              *(double *)a3.m128_u64,
              *(double *)v12.m128i_i64,
              *(double *)v13.m128i_i64);
      v37 = 19;
      v35 = (__int64 *)v105;
      v36 = v39;
    }
    else
    {
      if ( !(unsigned __int8)sub_15A0D80((_BYTE *)v26, v27, v28, v29)
        && (!sub_15F24D0((__int64)v11) || !sub_15A0C20(v26, v27, v24, v25))
        || (a3 = (__m128)0x3FF0000000000000uLL,
            v30 = (__int64 *)sub_15A10B0(*v11, 1.0),
            v31 = v26,
            v32 = sub_15A2CB0(v30, v26, 1.0, *(double *)v12.m128i_i64, *(double *)v13.m128i_i64),
            !sub_15A0C20(v32, v31, v33, v34)) )
      {
LABEL_22:
        v26 = *(v11 - 3);
        goto LABEL_23;
      }
      v35 = (__int64 *)*(v11 - 6);
      v36 = v32;
      v109.m128i_i16[0] = 257;
      v37 = 16;
    }
    v38 = (unsigned __int8 *)sub_15FB440(v37, v35, v36, (__int64)&v108, 0);
    sub_15F2530(v38, (__int64)v11, 1);
    if ( v38 )
      return v38;
    goto LABEL_22;
  }
LABEL_23:
  v102 = *(v11 - 6);
  if ( *(_BYTE *)(v102 + 16) > 0x10u )
  {
LABEL_67:
    v46 = *(_BYTE *)(v26 + 16);
    goto LABEL_30;
  }
  v40 = v26;
  v108.m128i_i64[1] = (__int64)&v105;
  if ( (unsigned __int8)sub_171FB50((__int64)&v108, v26, v24, v25) )
  {
    v42 = v105;
    v109.m128i_i16[0] = 257;
    v43 = sub_15A2BF0(
            (__int64 *)v102,
            v40,
            257,
            v41,
            *(double *)a3.m128_u64,
            *(double *)v12.m128i_i64,
            *(double *)v13.m128i_i64);
    v44 = v42;
    v45 = (__int64 *)v43;
    goto LABEL_26;
  }
  if ( !sub_15F24A0((__int64)v11) )
  {
LABEL_27:
    v26 = *(v11 - 3);
    v46 = *(_BYTE *)(v26 + 16);
    goto LABEL_28;
  }
  v58 = sub_15F24D0((__int64)v11);
  v26 = *(v11 - 3);
  v59 = !v58;
  v46 = *(_BYTE *)(v26 + 16);
  if ( v59 )
    goto LABEL_28;
  if ( v46 == 40 )
  {
    if ( !*(_QWORD *)(v26 - 48) )
      goto LABEL_28;
    v105 = *(unsigned __int8 **)(v26 - 48);
    v62 = *(_QWORD *)(v26 - 24);
    if ( *(_BYTE *)(v62 + 16) > 0x10u )
      goto LABEL_28;
LABEL_75:
    v65 = sub_15A2CB0((__int64 *)v102, v62, *(double *)a3.m128_u64, *(double *)v12.m128i_i64, *(double *)v13.m128i_i64);
    goto LABEL_63;
  }
  if ( v46 != 5 )
  {
    if ( v46 == 43 )
    {
      if ( *(_QWORD *)(v26 - 48) )
      {
        v105 = *(unsigned __int8 **)(v26 - 48);
        v62 = *(_QWORD *)(v26 - 24);
        if ( *(_BYTE *)(v62 + 16) <= 0x10u )
          goto LABEL_62;
      }
    }
    goto LABEL_28;
  }
  v60 = *(_WORD *)(v26 + 18);
  if ( v60 == 16 )
  {
    v89 = *(_DWORD *)(v26 + 20) & 0xFFFFFFF;
    if ( !*(_QWORD *)(v26 - 24 * v89) )
      goto LABEL_28;
    v105 = *(unsigned __int8 **)(v26 - 24 * v89);
    v62 = *(_QWORD *)(v26 + 24 * (1 - v89));
    if ( !v62 )
      goto LABEL_28;
    goto LABEL_75;
  }
  if ( v60 == 19 )
  {
    v61 = *(_DWORD *)(v26 + 20) & 0xFFFFFFF;
    if ( *(_QWORD *)(v26 - 24 * v61) )
    {
      v105 = *(unsigned __int8 **)(v26 - 24 * v61);
      v62 = *(_QWORD *)(v26 + 24 * (1 - v61));
      if ( v62 )
      {
LABEL_62:
        v65 = sub_15A2C50(
                (__int64 *)v102,
                v62,
                *(double *)a3.m128_u64,
                *(double *)v12.m128i_i64,
                *(double *)v13.m128i_i64);
LABEL_63:
        if ( v65 && sub_15A0C20(v65, v62, v63, v64) )
        {
          v45 = (__int64 *)v65;
          v109.m128i_i16[0] = 257;
          v44 = v105;
LABEL_26:
          v38 = (unsigned __int8 *)sub_15FB440(19, v45, (__int64)v44, (__int64)&v108, 0);
          sub_15F2530(v38, (__int64)v11, 1);
          if ( v38 )
            return v38;
          goto LABEL_27;
        }
        goto LABEL_27;
      }
    }
  }
LABEL_28:
  v102 = *(v11 - 6);
  if ( *(_BYTE *)(v102 + 16) <= 0x10u && v46 == 79 )
  {
    v23 = sub_1707470((__int64)a1, v11, v26, *(double *)a3.m128_u64, *(double *)v12.m128i_i64, *(double *)v13.m128i_i64);
    if ( v23 )
      return (unsigned __int8 *)v23;
    goto LABEL_67;
  }
LABEL_30:
  if ( v46 > 0x10u )
    goto LABEL_34;
  if ( !unk_4FA20C0 || v46 != 14 )
  {
LABEL_33:
    if ( *(_BYTE *)(v102 + 16) == 79 )
    {
      v23 = sub_1707470(
              (__int64)a1,
              v11,
              v102,
              *(double *)a3.m128_u64,
              *(double *)v12.m128i_i64,
              *(double *)v13.m128i_i64);
      if ( v23 )
        return (unsigned __int8 *)v23;
    }
LABEL_34:
    if ( !sub_15F24A0((__int64)v11) || !sub_15F24D0((__int64)v11) )
      goto LABEL_36;
    v66 = *(_QWORD *)(v102 + 8);
    if ( v66 && !*(_QWORD *)(v66 + 8) )
    {
      v69 = *(_BYTE *)(v102 + 16);
      if ( v69 == 43 )
      {
        if ( !*(_QWORD *)(v102 - 48) )
          goto LABEL_78;
        v104 = *(__int64 **)(v102 - 48);
        v71 = *(unsigned __int8 **)(v102 - 24);
        if ( !v71 )
          goto LABEL_78;
      }
      else
      {
        if ( v69 != 5 )
          goto LABEL_78;
        if ( *(_WORD *)(v102 + 18) != 19 )
          goto LABEL_78;
        v70 = *(_DWORD *)(v102 + 20) & 0xFFFFFFF;
        if ( !*(_QWORD *)(v102 - 24 * v70) )
          goto LABEL_78;
        v104 = *(__int64 **)(v102 - 24 * v70);
        v71 = *(unsigned __int8 **)(v102 + 24 * (1 - v70));
        if ( !v71 )
          goto LABEL_78;
      }
      v105 = v71;
      if ( v71[16] > 0x10u || *(_BYTE *)(v26 + 16) > 0x10u )
      {
        v72 = a1->m128i_i64[1];
        v109.m128i_i16[0] = 257;
        v73 = sub_1780A30(
                v72,
                (__int64)v71,
                v26,
                (__int64)v11,
                v108.m128i_i64,
                *(double *)a3.m128_u64,
                *(double *)v12.m128i_i64,
                *(double *)v13.m128i_i64);
        v109.m128i_i16[0] = 257;
        return sub_1780390(19, v104, (__int64)v73, (__int64)v11, (__int64)&v108);
      }
    }
LABEL_78:
    v108.m128i_i64[0] = (__int64)&v104;
    v108.m128i_i64[1] = (__int64)&v105;
    if ( (unsigned __int8)sub_1781DA0(&v108, v26) && (v105[16] > 0x10u || *(_BYTE *)(v102 + 16) > 0x10u) )
    {
      v67 = a1->m128i_i64[1];
      v109.m128i_i16[0] = 257;
      v68 = sub_1780A30(
              v67,
              (__int64)v105,
              v102,
              (__int64)v11,
              v108.m128i_i64,
              *(double *)a3.m128_u64,
              *(double *)v12.m128i_i64,
              *(double *)v13.m128i_i64);
      v109.m128i_i16[0] = 257;
      return sub_1780390(19, v68, (__int64)v104, (__int64)v11, (__int64)&v108);
    }
LABEL_36:
    if ( !sub_15F24A0((__int64)v11)
      || (v49 = *(_QWORD *)(v102 + 8)) == 0
      || *(_QWORD *)(v49 + 8)
      || (v50 = *(_QWORD *)(v26 + 8)) == 0
      || (v48 = &v103, (v100 = *(_QWORD *)(v50 + 8)) != 0) )
    {
LABEL_41:
      v106 = &v103;
      if ( (unsigned __int8)sub_171FB50((__int64)&v105, v102, v47, (__int64)v48) )
      {
        v108.m128i_i64[1] = (__int64)&v104;
        if ( (unsigned __int8)sub_171FB50((__int64)&v108, v26, v51, v52) )
        {
          sub_1593B40(v11 - 6, v103);
          sub_1593B40(v11 - 3, (__int64)v104);
          return (unsigned __int8 *)v11;
        }
      }
      if ( !sub_15F24B0((__int64)v11) || !sub_15F24A0((__int64)v11) )
        return 0;
      v53 = *(_BYTE *)(v26 + 16);
      if ( v53 == 40 )
      {
        v56 = *(__int64 **)(v26 - 48);
        v55 = *(__int64 **)(v26 - 24);
        if ( v56 && v56 == (__int64 *)v102 )
        {
LABEL_51:
          if ( v55 )
          {
            v104 = v55;
LABEL_53:
            v57 = sub_15A10B0(*v11, 1.0);
            sub_1593B40(v11 - 6, v57);
            sub_1593B40(v11 - 3, (__int64)v104);
            return (unsigned __int8 *)v11;
          }
          return 0;
        }
        if ( v55 == 0 || v55 != (__int64 *)v102 )
          return 0;
      }
      else
      {
        if ( v53 != 5 || *(_WORD *)(v26 + 18) != 16 )
          return 0;
        v54 = *(_DWORD *)(v26 + 20) & 0xFFFFFFF;
        v55 = *(__int64 **)(v26 - 24 * v54);
        v56 = *(__int64 **)(v26 + 24 * (1 - v54));
        if ( !v55 || v55 != (__int64 *)v102 )
        {
          if ( v56 == 0 || v56 != (__int64 *)v102 )
            return 0;
          goto LABEL_51;
        }
      }
      if ( v56 )
      {
        v104 = v56;
        goto LABEL_53;
      }
      return 0;
    }
    LODWORD(v105) = 194;
    LODWORD(v106) = 0;
    v107 = &v103;
    if ( (unsigned __int8)sub_173F0F0((__int64)&v105, v102)
      && (v108.m128i_i32[0] = 30, v108.m128i_i32[2] = 0, v109.m128i_i64[0] = v103, sub_173F140((__int64)&v108, v26)) )
    {
      v99 = 0;
    }
    else
    {
      LODWORD(v105) = 30;
      LODWORD(v106) = 0;
      v107 = &v103;
      if ( !(unsigned __int8)sub_173F0F0((__int64)&v105, v102) )
        goto LABEL_41;
      v108.m128i_i32[0] = 194;
      v108.m128i_i32[2] = 0;
      v109.m128i_i64[0] = v103;
      v99 = sub_173F140((__int64)&v108, v26);
      if ( !v99 )
        goto LABEL_41;
    }
    if ( !(unsigned __int8)sub_1AB1780(a1[165].m128i_i64[1], *v11, 393, 394, 398) )
      goto LABEL_41;
    v74 = sub_16498A0((__int64)v11);
    v75 = (unsigned __int8 *)v11[6];
    v108.m128i_i64[0] = 0;
    v109.m128i_i64[1] = v74;
    v76 = v11[5];
    v110 = 0;
    v108.m128i_i64[1] = v76;
    v111 = 0;
    v112 = 0;
    v113 = 0;
    v109.m128i_i64[0] = (__int64)(v11 + 3);
    v105 = v75;
    if ( v75 )
    {
      sub_1623A60((__int64)&v105, (__int64)v75, 2);
      if ( v108.m128i_i64[0] )
        sub_161E7C0((__int64)&v108, v108.m128i_i64[0]);
      v108.m128i_i64[0] = (__int64)v105;
      if ( v105 )
        sub_1623210((__int64)&v105, v105, (__int64)&v108);
    }
    v98 = v111;
    v101 = v110;
    v111 = sub_15F24E0((__int64)v11);
    v77 = *(_BYTE *)(v102 + 16);
    v78 = 0;
    if ( v77 <= 0x17u )
      goto LABEL_110;
    if ( v77 == 78 )
    {
      v78 = v102 | 4;
    }
    else
    {
      v78 = 0;
      if ( v77 != 29 )
        goto LABEL_110;
      v78 = v102 & 0xFFFFFFFFFFFFFFFBLL;
    }
    if ( (v78 & 4) != 0 )
    {
      v79 = (__int64 *)((v78 & 0xFFFFFFFFFFFFFFF8LL) - 24);
LABEL_111:
      v80 = *v79;
      if ( *(_BYTE *)(v80 + 16) )
        BUG();
      v81 = 0;
      v104 = *(__int64 **)(v80 + 112);
      v82 = *(_QWORD *)a1[165].m128i_i64[1];
      if ( (((int)*(unsigned __int8 *)(v82 + 98) >> 2) & 3) != 0 )
      {
        if ( (((int)*(unsigned __int8 *)(v82 + 98) >> 2) & 3) != 3 )
        {
          v92 = *(_DWORD *)(v82 + 136);
          v93 = *(_QWORD *)(v82 + 120);
          if ( v92 )
          {
            v94 = 1;
            for ( i = ((_WORD)v92 - 1) & 0x38CD; ; i = (v92 - 1) & v97 )
            {
              v96 = v93 + 40LL * i;
              if ( *(_DWORD *)v96 == 393 )
                break;
              if ( *(_DWORD *)v96 == -1 )
                goto LABEL_142;
              v97 = v94 + i;
              ++v94;
            }
          }
          else
          {
LABEL_142:
            v96 = v93 + 40LL * v92;
          }
          v83 = sub_1AB1BA0(v103, *(_QWORD *)(v96 + 8), *(_QWORD *)(v96 + 16), &v108, &v104);
          goto LABEL_116;
        }
        v81 = qword_4F9B700[787];
        v100 = qword_4F9B700[786];
      }
      v83 = sub_1AB1BA0(v103, v100, v81, &v108, &v104);
LABEL_116:
      v86 = (_QWORD *)v83;
      if ( v99 )
      {
        a3 = (__m128)0x3FF0000000000000uLL;
        v90 = *v11;
        LOWORD(v107) = 257;
        v91 = sub_15A10B0(v90, 1.0);
        v86 = sub_156E040(v108.m128i_i64, v91, (__int64)v86, (__int64)&v105, 0);
      }
      v11 = (__int64 *)sub_170E100(
                         a1->m128i_i64,
                         (__int64)v11,
                         (__int64)v86,
                         a3,
                         *(double *)v12.m128i_i64,
                         *(double *)v13.m128i_i64,
                         a6,
                         v84,
                         v85,
                         a9,
                         a10);
      v111 = v98;
      v110 = v101;
      if ( v108.m128i_i64[0] )
        sub_161E7C0((__int64)&v108, v108.m128i_i64[0]);
      return (unsigned __int8 *)v11;
    }
LABEL_110:
    v79 = (__int64 *)((v78 & 0xFFFFFFFFFFFFFFF8LL) - 72);
    goto LABEL_111;
  }
  a3 = (__m128)0x3FF0000000000000uLL;
  if ( !(unsigned __int8)sub_17802A0(v26, 1.0) )
  {
    if ( *(_BYTE *)(v26 + 16) > 0x10u )
      goto LABEL_34;
    goto LABEL_33;
  }
  return (unsigned __int8 *)sub_170E100(
                              a1->m128i_i64,
                              (__int64)v11,
                              v102,
                              (__m128)0x3FF0000000000000uLL,
                              *(double *)v12.m128i_i64,
                              *(double *)v13.m128i_i64,
                              a6,
                              v87,
                              v88,
                              a9,
                              a10);
}
