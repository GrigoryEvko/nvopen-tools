// Function: sub_17A17A0
// Address: 0x17a17a0
//
__int64 __fastcall sub_17A17A0(
        __m128i *a1,
        __int64 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // r12
  __m128 v12; // xmm0
  __m128i v13; // xmm1
  char v14; // al
  unsigned __int8 *v15; // rax
  __int64 v16; // rbx
  __int64 v17; // r13
  __int64 v18; // r14
  _QWORD *v19; // rax
  double v20; // xmm4_8
  double v21; // xmm5_8
  __int64 v23; // rax
  double v24; // xmm4_8
  double v25; // xmm5_8
  __int64 v26; // r12
  _BYTE *v27; // rax
  __int64 v28; // rcx
  _BYTE *v29; // rbx
  __int64 v30; // r14
  unsigned __int8 v31; // al
  _BYTE *v32; // rax
  __int64 v33; // rcx
  unsigned __int8 v34; // al
  _BYTE *v35; // rdi
  __int64 v36; // rdx
  _BYTE *v37; // rax
  _QWORD *v38; // rax
  __int64 v39; // r12
  __int64 v40; // r14
  __int64 v41; // r12
  __int64 v42; // rax
  unsigned int v43; // r15d
  unsigned __int64 v44; // rdx
  __int64 *v45; // rsi
  __int64 v46; // rdx
  __int64 v47; // rax
  char v48; // al
  __int64 v49; // rdx
  _BYTE *v50; // rax
  __int64 v51; // rax
  __int64 v52; // rdi
  __int64 *v53; // rax
  __int64 v54; // rdx
  int v55; // edx
  unsigned int v56; // r12d
  unsigned __int64 v57; // rdx
  __int64 v58; // rbx
  __int64 v59; // r14
  bool v60; // al
  void *v61; // rax
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rdx
  unsigned int v65; // esi
  int v66; // edx
  __int64 v67; // rsi
  __int64 v68; // rax
  unsigned __int8 *v69; // rax
  __int64 v70; // r14
  _QWORD *v71; // rax
  __int64 v72; // rdx
  __int64 v73; // rbx
  bool v74; // al
  __int64 v75; // rax
  _BYTE *v76; // rdi
  unsigned __int8 v77; // al
  _QWORD *v78; // rax
  unsigned int v79; // eax
  __int64 v80; // rax
  int v81; // eax
  __int64 v82; // rdx
  unsigned int v83; // eax
  __int64 v84; // rdi
  _QWORD *v85; // rax
  __int64 v86; // rdx
  __int64 v87; // rdi
  __int64 **v88; // rdx
  __int64 v89; // rdx
  __int64 v90; // rdi
  __int64 **v91; // rdx
  __int64 v92; // rax
  _QWORD *v93; // rax
  _QWORD *v94; // [rsp+8h] [rbp-98h]
  unsigned int v95; // [rsp+10h] [rbp-90h]
  __int64 **v96; // [rsp+18h] [rbp-88h]
  __int64 *v97; // [rsp+20h] [rbp-80h] BYREF
  _BYTE *v98; // [rsp+28h] [rbp-78h] BYREF
  unsigned __int64 v99; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v100; // [rsp+38h] [rbp-68h]
  __m128 v101; // [rsp+40h] [rbp-60h] BYREF
  __m128i v102; // [rsp+50h] [rbp-50h]
  __int64 v103; // [rsp+60h] [rbp-40h]

  v11 = a2;
  v12 = (__m128)_mm_loadu_si128(a1 + 167);
  v13 = _mm_loadu_si128(a1 + 168);
  v103 = a2;
  v101 = v12;
  v102 = v13;
  v14 = sub_15F23D0(a2);
  v15 = sub_13E1070(*(unsigned __int8 **)(a2 - 48), *(_BYTE **)(a2 - 24), v14, &v101);
  if ( v15 )
  {
    v16 = *(_QWORD *)(a2 + 8);
    if ( v16 )
    {
      v17 = a1->m128i_i64[0];
      v18 = (__int64)v15;
      do
      {
        v19 = sub_1648700(v16);
        sub_170B990(v17, (__int64)v19);
        v16 = *(_QWORD *)(v16 + 8);
      }
      while ( v16 );
      if ( a2 == v18 )
        v18 = sub_1599EF0(*(__int64 ***)a2);
      sub_164D160(a2, v18, v12, *(double *)v13.m128i_i64, a5, a6, v20, v21, a9, a10);
      return v11;
    }
    return 0;
  }
  v23 = (__int64)sub_1707490((__int64)a1, (unsigned __int8 *)a2, *(double *)v12.m128_u64, *(double *)v13.m128i_i64, a5);
  if ( v23 )
    return v23;
  v23 = sub_179FBB0(a1->m128i_i64, a2, v12, *(double *)v13.m128i_i64, a5, a6, v24, v25, a9, a10);
  if ( v23 )
    return v23;
  v29 = *(_BYTE **)(a2 - 24);
  v30 = *(_QWORD *)(a2 - 48);
  v96 = *(__int64 ***)a2;
  v31 = v29[16];
  if ( v31 == 13 )
  {
    v32 = v29 + 24;
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)v29 + 8LL) != 16 )
      goto LABEL_46;
    if ( v31 > 0x10u )
      goto LABEL_46;
    v62 = sub_15A1020(v29, a2, *(_QWORD *)v29, v28);
    if ( !v62 || *(_BYTE *)(v62 + 16) != 13 )
      goto LABEL_46;
    v32 = (_BYTE *)(v62 + 24);
  }
  if ( *((_DWORD *)v32 + 2) <= 0x40u )
    v94 = *(_QWORD **)v32;
  else
    v94 = **(_QWORD ***)v32;
  v95 = sub_16431D0((__int64)v96);
  v34 = *(_BYTE *)(v30 + 16);
  if ( v34 <= 0x17u )
  {
    if ( v34 == 5 )
    {
      if ( *(_WORD *)(v30 + 18) == 23
        && (v72 = *(_DWORD *)(v30 + 20) & 0xFFFFFFF, a2 = 4 * v72, (v33 = *(_QWORD *)(v30 - 24 * v72)) != 0) )
      {
        v97 = *(__int64 **)(v30 - 24LL * (*(_DWORD *)(v30 + 20) & 0xFFFFFFF));
        v33 = 1 - v72;
        v35 = *(_BYTE **)(v30 + 24 * (1 - v72));
        if ( v35[16] == 13 )
          goto LABEL_25;
        v36 = *(_QWORD *)v35;
        if ( *(_BYTE *)(*(_QWORD *)v35 + 8LL) == 16 )
        {
LABEL_116:
          v75 = sub_15A1020(v35, a2, v36, v33);
          if ( v75 && *(_BYTE *)(v75 + 16) == 13 )
          {
            v37 = (_BYTE *)(v75 + 24);
            v98 = v37;
LABEL_26:
            if ( *((_DWORD *)v37 + 2) <= 0x40u )
              v38 = *(_QWORD **)v37;
            else
              v38 = **(_QWORD ***)v37;
            if ( (unsigned int)v94 > (unsigned int)v38 )
            {
              v58 = sub_15A0680((__int64)v96, (unsigned int)((_DWORD)v94 - (_DWORD)v38), 0);
              if ( sub_15F2370(v30) )
              {
                v102.m128i_i16[0] = 257;
                v73 = sub_15FB440(24, v97, v58, (__int64)&v101, 0);
                v74 = sub_15F23D0(v11);
                v11 = v73;
                sub_15F2350(v73, v74);
                return v11;
              }
              v59 = a1->m128i_i64[1];
              v60 = sub_15F23D0(v11);
              v102.m128i_i16[0] = 257;
              v61 = sub_172C310(
                      v59,
                      (__int64)v97,
                      v58,
                      (__int64 *)&v101,
                      v60,
                      *(double *)v12.m128_u64,
                      *(double *)v13.m128i_i64,
                      a5);
            }
            else
            {
              if ( (unsigned int)v94 >= (unsigned int)v38 )
              {
                v100 = v95;
                v56 = v95 - (_DWORD)v94;
                if ( v95 > 0x40 )
                  sub_16A4EF0((__int64)&v99, 0, 0);
                else
                  v99 = 0;
                if ( v56 )
                {
                  if ( v56 > 0x40 )
                  {
                    sub_16A5260(&v99, 0, v56);
                  }
                  else
                  {
                    v57 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v94 - (unsigned __int8)v95 + 64);
                    if ( v100 > 0x40 )
                      *(_QWORD *)v99 |= v57;
                    else
                      v99 |= v57;
                  }
                }
                v102.m128i_i16[0] = 257;
                v45 = v97;
                v46 = sub_15A1070((__int64)v96, (__int64)&v99);
                goto LABEL_42;
              }
              v39 = sub_15A0680((__int64)v96, (unsigned int)((_DWORD)v38 - (_DWORD)v94), 0);
              if ( sub_15F2370(v30) )
              {
                v102.m128i_i16[0] = 257;
                v11 = sub_15FB440(23, v97, v39, (__int64)&v101, 0);
                sub_15F2310(v11, 1);
                return v11;
              }
              v40 = a1->m128i_i64[1];
              v102.m128i_i16[0] = 257;
              if ( *((_BYTE *)v97 + 16) <= 0x10u && *(_BYTE *)(v39 + 16) <= 0x10u )
              {
                v41 = sub_15A2D50(v97, v39, 0, 0, *(double *)v12.m128_u64, *(double *)v13.m128i_i64, a5);
                v42 = sub_14DBA30(v41, *(_QWORD *)(v40 + 96), 0);
                if ( v42 )
                  v41 = v42;
                goto LABEL_35;
              }
              v61 = sub_179D030(v40, v97, v39, (__int64 *)&v101, 0, 0);
            }
            v41 = (__int64)v61;
LABEL_35:
            v100 = v95;
            v43 = v95 - (_DWORD)v94;
            if ( v95 > 0x40 )
              sub_16A4EF0((__int64)&v99, 0, 0);
            else
              v99 = 0;
            if ( v43 )
            {
              if ( v43 > 0x40 )
              {
                sub_16A5260(&v99, 0, v43);
              }
              else
              {
                v44 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v94 - (unsigned __int8)v95 + 64);
                if ( v100 > 0x40 )
                  *(_QWORD *)v99 |= v44;
                else
                  v99 |= v44;
              }
            }
            v102.m128i_i16[0] = 257;
            v45 = (__int64 *)v41;
            v46 = sub_15A1070((__int64)v96, (__int64)&v99);
LABEL_42:
            v11 = sub_15FB440(26, v45, v46, (__int64)&v101, 0);
            if ( v100 > 0x40 && v99 )
              j_j___libc_free_0_0(v99);
            return v11;
          }
          v34 = *(_BYTE *)(v30 + 16);
          goto LABEL_88;
        }
        v54 = *(_QWORD *)(v30 + 8);
        if ( !v54 )
          goto LABEL_59;
      }
      else
      {
        v54 = *(_QWORD *)(v30 + 8);
        if ( !v54 )
          goto LABEL_59;
      }
      if ( *(_QWORD *)(v54 + 8) )
        goto LABEL_59;
      goto LABEL_154;
    }
  }
  else
  {
    if ( v34 == 78 )
    {
      v63 = *(_QWORD *)(v30 - 24);
      if ( *(_BYTE *)(v63 + 16) )
        goto LABEL_88;
      if ( (*(_BYTE *)(v63 + 33) & 0x20) == 0 )
        goto LABEL_88;
      if ( !v95 )
        goto LABEL_88;
      v33 = v95 - 1;
      if ( ((unsigned int)v33 & v95) != 0 )
        goto LABEL_88;
      _BitScanReverse(&v65, v95);
      a2 = v65 ^ 0x1F;
      v33 = (unsigned int)(31 - a2);
      if ( (_DWORD)v94 != (_DWORD)v33 )
        goto LABEL_88;
      v66 = *(_DWORD *)(v63 + 36);
      if ( v66 == 31 )
      {
        v67 = 0;
      }
      else
      {
        v33 = (unsigned int)(v66 - 32);
        if ( (unsigned int)v33 > 1 )
          goto LABEL_88;
        v67 = -(__int64)(v66 == 32);
      }
      v68 = sub_15A0930((__int64)v96, v67);
      v102.m128i_i16[0] = 257;
      v69 = sub_17203D0(
              a1->m128i_i64[1],
              32,
              *(_QWORD *)(v30 - 24LL * (*(_DWORD *)(v30 + 20) & 0xFFFFFFF)),
              v68,
              (__int64 *)&v101);
      v102.m128i_i16[0] = 257;
      v70 = (__int64)v69;
      v71 = sub_1648A60(56, 1u);
      v11 = (__int64)v71;
      if ( !v71 )
        return v11;
LABEL_98:
      sub_15FC690((__int64)v71, v70, (__int64)v96, (__int64)&v101, 0);
      return v11;
    }
    if ( v34 == 47 )
    {
      if ( !*(_QWORD *)(v30 - 48) )
        goto LABEL_101;
      v97 = *(__int64 **)(v30 - 48);
      v35 = *(_BYTE **)(v30 - 24);
      v36 = (unsigned __int8)v35[16];
      if ( (_BYTE)v36 == 13 )
      {
LABEL_25:
        v37 = v35 + 24;
        v98 = v35 + 24;
        goto LABEL_26;
      }
      v33 = *(_QWORD *)v35;
      if ( *(_BYTE *)(*(_QWORD *)v35 + 8LL) != 16 || (unsigned __int8)v36 > 0x10u )
        goto LABEL_101;
      goto LABEL_116;
    }
  }
LABEL_88:
  v64 = *(_QWORD *)(v30 + 8);
  if ( !v64 || *(_QWORD *)(v64 + 8) )
    goto LABEL_89;
  if ( v34 > 0x17u )
  {
    if ( v34 != 61 )
      goto LABEL_101;
    goto LABEL_155;
  }
  if ( v34 != 5 )
    goto LABEL_61;
LABEL_154:
  if ( *(_WORD *)(v30 + 18) == 37 )
  {
LABEL_155:
    if ( (*(_BYTE *)(v30 + 23) & 0x40) != 0 )
      v33 = *(_QWORD *)(v30 - 8);
    else
      v33 = v30 - 24LL * (*(_DWORD *)(v30 + 20) & 0xFFFFFFF);
    a2 = *(_QWORD *)v33;
    if ( *(_QWORD *)v33 )
    {
      v97 = *(__int64 **)v33;
      if ( *((_BYTE *)v96 + 8) != 11 )
      {
LABEL_159:
        v90 = a1->m128i_i64[1];
        v102.m128i_i16[0] = 257;
        v85 = sub_173E800(
                v90,
                a2,
                (unsigned int)v94,
                (__int64 *)&v101,
                0,
                *(double *)v12.m128_u64,
                *(double *)v13.m128i_i64,
                a5);
        goto LABEL_139;
      }
      v91 = *(__int64 ***)a2;
      a2 = (__int64)v96;
      if ( (unsigned __int8)sub_1705440((__int64)a1, (__int64)v96, (__int64)v91) )
      {
        a2 = (__int64)v97;
        goto LABEL_159;
      }
      v34 = *(_BYTE *)(v30 + 16);
    }
LABEL_89:
    if ( v34 <= 0x17u )
    {
      if ( v34 != 5 )
        goto LABEL_61;
      goto LABEL_59;
    }
LABEL_101:
    v55 = v34 - 24;
    goto LABEL_60;
  }
LABEL_59:
  v55 = *(unsigned __int16 *)(v30 + 18);
  v34 = 5;
LABEL_60:
  if ( v55 == 38 )
  {
    v33 = (*(_BYTE *)(v30 + 23) & 0x40) != 0 ? *(_QWORD *)(v30 - 8) : v30 - 24LL * (*(_DWORD *)(v30 + 20) & 0xFFFFFFF);
    a2 = *(_QWORD *)v33;
    if ( *(_QWORD *)v33 )
    {
      v97 = *(__int64 **)v33;
      if ( *((_BYTE *)v96 + 8) == 11 )
      {
        v88 = *(__int64 ***)a2;
        a2 = (__int64)v96;
        if ( !(unsigned __int8)sub_1705440((__int64)a1, (__int64)v96, (__int64)v88) )
        {
LABEL_132:
          v34 = *(_BYTE *)(v30 + 16);
          goto LABEL_61;
        }
        a2 = (__int64)v97;
      }
      v81 = sub_16431D0(*(_QWORD *)a2);
      v33 = v95;
      if ( (_DWORD)v94 == v95 - 1 )
      {
        if ( v81 == 1 )
        {
          v102.m128i_i16[0] = 257;
          v93 = sub_1648A60(56, 1u);
          v11 = (__int64)v93;
          if ( v93 )
            sub_15FC690((__int64)v93, (__int64)v97, (__int64)v96, (__int64)&v101, 0);
          return v11;
        }
        v86 = *(_QWORD *)(v30 + 8);
        if ( v86 && !*(_QWORD *)(v86 + 8) )
        {
          v87 = a1->m128i_i64[1];
          v102.m128i_i16[0] = 257;
          v85 = sub_173E800(
                  v87,
                  a2,
                  (unsigned int)(v81 - 1),
                  (__int64 *)&v101,
                  0,
                  *(double *)v12.m128_u64,
                  *(double *)v13.m128i_i64,
                  a5);
          goto LABEL_139;
        }
      }
      else if ( (_DWORD)v94 == v95 - v81 )
      {
        v82 = *(_QWORD *)(v30 + 8);
        if ( v82 )
        {
          if ( !*(_QWORD *)(v82 + 8) )
          {
            v83 = v81 - 1;
            v84 = a1->m128i_i64[1];
            v102.m128i_i16[0] = 257;
            if ( v83 > (unsigned int)v94 )
              v83 = (unsigned int)v94;
            v85 = sub_173E590(v84, a2, v83, (__int64 *)&v101, 0, *(double *)v12.m128_u64, *(double *)v13.m128i_i64, a5);
LABEL_139:
            v70 = (__int64)v85;
            v102.m128i_i16[0] = 257;
            v71 = sub_1648A60(56, 1u);
            v11 = (__int64)v71;
            if ( !v71 )
              return v11;
            goto LABEL_98;
          }
        }
      }
      goto LABEL_132;
    }
  }
LABEL_61:
  v101.m128_u64[0] = (unsigned __int64)&v97;
  v101.m128_u64[1] = (unsigned __int64)&v98;
  if ( v34 == 48 )
  {
    if ( *(_QWORD *)(v30 - 48) )
    {
      v97 = *(__int64 **)(v30 - 48);
      v76 = *(_BYTE **)(v30 - 24);
      v77 = v76[16];
      if ( v77 == 13 )
      {
        v98 = v76 + 24;
        goto LABEL_122;
      }
      if ( *(_BYTE *)(*(_QWORD *)v76 + 8LL) == 16 && v77 <= 0x10u )
      {
        v92 = sub_15A1020(v76, a2, *(_QWORD *)v76, v33);
        if ( v92 )
        {
          if ( *(_BYTE *)(v92 + 16) == 13 )
          {
            *(_QWORD *)v101.m128_u64[1] = v92 + 24;
            goto LABEL_122;
          }
        }
      }
    }
  }
  else if ( v34 == 5 && *(_WORD *)(v30 + 18) == 24 )
  {
    v89 = *(_DWORD *)(v30 + 20) & 0xFFFFFFF;
    if ( *(_QWORD *)(v30 - 24 * v89) )
    {
      v97 = *(__int64 **)(v30 - 24 * v89);
      if ( (unsigned __int8)sub_13D7780((_QWORD **)&v101.m128_u64[1], *(_BYTE **)(v30 + 24 * (1 - v89))) )
      {
LABEL_122:
        v78 = *(_QWORD **)v98;
        if ( *((_DWORD *)v98 + 2) > 0x40u )
          v78 = (_QWORD *)*v78;
        v79 = (_DWORD)v94 + (_DWORD)v78;
        if ( v95 > v79 )
        {
          v102.m128i_i16[0] = 257;
          v80 = sub_15A0680((__int64)v96, v79, 0);
          return sub_15FB440(24, v97, v80, (__int64)&v101, 0);
        }
      }
    }
  }
  if ( sub_15F23D0(v11) )
    goto LABEL_46;
  sub_13D0120((__int64)&v101, v95, (unsigned int)v94);
  if ( !(unsigned __int8)sub_14C1670(
                           v30,
                           (__int64)&v101,
                           a1[166].m128i_i64[1],
                           0,
                           a1[165].m128i_i64[0],
                           v11,
                           a1[166].m128i_i64[0]) )
  {
    if ( v101.m128_i32[2] > 0x40u && v101.m128_u64[0] )
      j_j___libc_free_0_0(v101.m128_u64[0]);
LABEL_46:
    v47 = *(_QWORD *)(v30 + 8);
    if ( !v47 || *(_QWORD *)(v47 + 8) )
      return 0;
    v48 = *(_BYTE *)(v30 + 16);
    if ( v48 == 47 )
    {
      v26 = *(_QWORD *)(v30 - 48);
      if ( !v26 )
        return 0;
      v27 = *(_BYTE **)(v30 - 24);
      if ( !v27 || v29 != v27 )
        return 0;
    }
    else
    {
      if ( v48 != 5 )
        return 0;
      if ( *(_WORD *)(v30 + 18) != 23 )
        return 0;
      v49 = *(_DWORD *)(v30 + 20) & 0xFFFFFFF;
      v26 = *(_QWORD *)(v30 - 24 * v49);
      if ( !v26 )
        return 0;
      v50 = *(_BYTE **)(v30 + 24 * (1 - v49));
      if ( v29 != v50 || !v50 )
        return 0;
    }
    v51 = sub_15A04A0(v96);
    v52 = a1->m128i_i64[1];
    v102.m128i_i16[0] = 257;
    v53 = sub_172C310(
            v52,
            v51,
            (__int64)v29,
            (__int64 *)&v101,
            0,
            *(double *)v12.m128_u64,
            *(double *)v13.m128i_i64,
            a5);
    v102.m128i_i16[0] = 257;
    return sub_15FB440(26, v53, v26, (__int64)&v101, 0);
  }
  if ( v101.m128_i32[2] > 0x40u && v101.m128_u64[0] )
    j_j___libc_free_0_0(v101.m128_u64[0]);
  sub_15F2350(v11, 1);
  return v11;
}
