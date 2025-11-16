// Function: sub_17A0140
// Address: 0x17a0140
//
__int64 __fastcall sub_17A0140(
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
  char v14; // bl
  char v15; // al
  unsigned __int8 *v16; // rax
  __int64 v17; // rbx
  __int64 v18; // r13
  __int64 v19; // r14
  _QWORD *v20; // rax
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 v24; // rax
  double v25; // xmm4_8
  double v26; // xmm5_8
  __int64 *v27; // r12
  __int64 v28; // rdi
  __int64 v29; // rcx
  _BYTE *v30; // r14
  __int64 v31; // rbx
  unsigned __int8 v32; // al
  _BYTE *v33; // rax
  unsigned int v34; // eax
  __int64 **v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rsi
  __int64 v38; // rcx
  _QWORD *v39; // rax
  char v40; // al
  __int64 v41; // rax
  unsigned __int8 v42; // al
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rax
  int v46; // eax
  __int64 v47; // rax
  unsigned int v48; // r14d
  __int64 v49; // rax
  __int64 *v50; // rax
  unsigned int v51; // eax
  unsigned int v52; // esi
  char v53; // al
  __int64 v54; // rdi
  __int64 v55; // r12
  __int64 *v56; // rax
  __int64 v57; // rax
  _BYTE *v58; // rax
  __int64 v59; // r12
  __int64 v60; // rax
  __int64 v61; // r14
  __int64 v62; // rax
  _QWORD *v63; // rax
  _BYTE *v64; // rdi
  unsigned __int8 v65; // al
  __int64 v66; // rax
  _QWORD *v67; // rax
  unsigned int v68; // eax
  __int64 v69; // rax
  unsigned int v70; // esi
  __int64 v71; // rax
  __int64 v72; // rbx
  bool v73; // al
  __int64 v74; // rdi
  bool v75; // al
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 *v78; // rcx
  __int64 v79; // r10
  __int64 v80; // rax
  __int64 v81; // r15
  __int64 *v82; // r12
  __int64 v83; // rax
  __int64 v84; // r10
  unsigned __int8 *v85; // rax
  __int64 v86; // rdx
  __int64 v87; // rdx
  _BYTE *v88; // rdi
  __int64 v89; // rax
  __int64 v90; // rdx
  _BYTE *v91; // rax
  __int64 **v92; // [rsp-8h] [rbp-B8h]
  _QWORD *v93; // [rsp+0h] [rbp-B0h]
  unsigned __int8 v94; // [rsp+Bh] [rbp-A5h]
  unsigned int v95; // [rsp+Ch] [rbp-A4h]
  __int64 **v96; // [rsp+10h] [rbp-A0h]
  unsigned int v97; // [rsp+18h] [rbp-98h]
  __int64 v98; // [rsp+18h] [rbp-98h]
  __int64 *v99; // [rsp+28h] [rbp-88h] BYREF
  __int64 v100; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v101; // [rsp+38h] [rbp-78h]
  __int16 v102; // [rsp+40h] [rbp-70h]
  __m128 v103; // [rsp+50h] [rbp-60h] BYREF
  __m128i v104; // [rsp+60h] [rbp-50h]
  __int64 v105; // [rsp+70h] [rbp-40h]

  v11 = a2;
  v12 = (__m128)_mm_loadu_si128(a1 + 167);
  v105 = a2;
  v13 = _mm_loadu_si128(a1 + 168);
  v103 = v12;
  v104 = v13;
  v14 = sub_15F2370(a2);
  v15 = sub_15F2380(a2);
  v16 = sub_13E10E0(*(unsigned __int8 **)(a2 - 48), *(_QWORD *)(a2 - 24), v15, v14, &v103);
  if ( v16 )
  {
    v17 = *(_QWORD *)(a2 + 8);
    if ( v17 )
    {
      v18 = a1->m128i_i64[0];
      v19 = (__int64)v16;
      do
      {
        v20 = sub_1648700(v17);
        sub_170B990(v18, (__int64)v20);
        v17 = *(_QWORD *)(v17 + 8);
      }
      while ( v17 );
      if ( a2 == v19 )
        v19 = sub_1599EF0(*(__int64 ***)a2);
      sub_164D160(a2, v19, v12, *(double *)v13.m128i_i64, a5, a6, v21, v22, a9, a10);
      return v11;
    }
    return 0;
  }
  v24 = (__int64)sub_1707490((__int64)a1, (unsigned __int8 *)a2, *(double *)v12.m128_u64, *(double *)v13.m128i_i64, a5);
  if ( v24 )
    return v24;
  v24 = sub_179FBB0(a1->m128i_i64, a2, v12, *(double *)v13.m128i_i64, a5, a6, v25, v26, a9, a10);
  if ( v24 )
    return v24;
  v30 = *(_BYTE **)(a2 - 24);
  v31 = *(_QWORD *)(a2 - 48);
  v96 = *(__int64 ***)a2;
  v32 = v30[16];
  if ( v32 == 13 )
  {
    v33 = v30 + 24;
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)v30 + 8LL) != 16 )
      goto LABEL_31;
    if ( v32 > 0x10u )
      goto LABEL_31;
    v45 = sub_15A1020(v30, a2, *(_QWORD *)v30, v29);
    if ( !v45 || *(_BYTE *)(v45 + 16) != 13 )
      goto LABEL_31;
    v33 = (_BYTE *)(v45 + 24);
  }
  if ( *((_DWORD *)v33 + 2) <= 0x40u )
    v93 = *(_QWORD **)v33;
  else
    v93 = **(_QWORD ***)v33;
  v97 = (unsigned int)v93;
  v34 = sub_16431D0((__int64)v96);
  v36 = *(unsigned __int8 *)(v31 + 16);
  v95 = v34;
  if ( (unsigned __int8)v36 > 0x17u )
  {
    v46 = (unsigned __int8)v36 - 24;
  }
  else
  {
    if ( (_BYTE)v36 != 5 )
      goto LABEL_21;
    v46 = *(unsigned __int16 *)(v31 + 18);
  }
  if ( v46 == 37 )
  {
    v35 = (*(_BYTE *)(v31 + 23) & 0x40) != 0
        ? *(__int64 ***)(v31 - 8)
        : (__int64 **)(v31 - 24LL * (*(_DWORD *)(v31 + 20) & 0xFFFFFFF));
    v50 = *v35;
    if ( *v35 )
    {
      v99 = *v35;
      v94 = v36;
      v51 = sub_16431D0(*v50);
      v36 = v94;
      if ( (unsigned int)v93 < v51 )
      {
        v103.m128_i32[2] = v51;
        if ( v51 > 0x40 )
        {
          sub_16A4EF0((__int64)&v103, 0, 0);
          v51 = v103.m128_u32[2];
        }
        else
        {
          v103.m128_u64[0] = 0;
        }
        v52 = v51 - (_DWORD)v93;
        if ( v51 - (_DWORD)v93 != v51 )
        {
          if ( v52 > 0x3F || v51 > 0x40 )
            sub_16A5260(&v103, v52, v51);
          else
            v103.m128_u64[0] |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v93) << v52;
        }
        v53 = sub_14C1670(
                (__int64)v99,
                (__int64)&v103,
                a1[166].m128i_i64[1],
                0,
                a1[165].m128i_i64[0],
                v11,
                a1[166].m128i_i64[0]);
        v35 = v92;
        if ( v53 )
        {
          if ( v103.m128_i32[2] > 0x40u && v103.m128_u64[0] )
            j_j___libc_free_0_0(v103.m128_u64[0]);
          v59 = a1->m128i_i64[1];
          v102 = 257;
          v60 = sub_15A0680(*v99, (unsigned int)v93, 0);
          if ( *((_BYTE *)v99 + 16) > 0x10u || *(_BYTE *)(v60 + 16) > 0x10u )
          {
            v61 = (__int64)sub_179D030(v59, v99, v60, &v100, 0, 0);
          }
          else
          {
            v61 = sub_15A2D50(v99, v60, 0, 0, *(double *)v12.m128_u64, *(double *)v13.m128i_i64, a5);
            v62 = sub_14DBA30(v61, *(_QWORD *)(v59 + 96), 0);
            if ( v62 )
              v61 = v62;
          }
          v104.m128i_i16[0] = 257;
          v63 = sub_1648A60(56, 1u);
          v11 = (__int64)v63;
          if ( v63 )
            sub_15FC690((__int64)v63, v61, (__int64)v96, (__int64)&v103, 0);
          return v11;
        }
        if ( v103.m128_i32[2] > 0x40u && v103.m128_u64[0] )
        {
          j_j___libc_free_0_0(v103.m128_u64[0]);
          v36 = *(unsigned __int8 *)(v31 + 16);
        }
        else
        {
          v36 = *(unsigned __int8 *)(v31 + 16);
        }
      }
    }
  }
  if ( (unsigned __int8)v36 <= 0x17u )
  {
    if ( (_BYTE)v36 == 5 && (unsigned int)*(unsigned __int16 *)(v31 + 18) - 24 <= 1 )
    {
      v57 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
      v35 = (__int64 **)(4 * v57);
      v36 = *(_QWORD *)(v31 - 24 * v57);
      if ( v36 )
      {
        v99 = *(__int64 **)(v31 - 24LL * (*(_DWORD *)(v31 + 20) & 0xFFFFFFF));
        v36 = 1 - v57;
        v58 = *(_BYTE **)(v31 + 24 * (1 - v57));
        if ( v58 )
        {
          if ( v30 == v58 )
          {
LABEL_54:
            v101 = v95;
            v48 = (_DWORD)v93 - v95;
            if ( v95 > 0x40 )
            {
              sub_16A4EF0((__int64)&v100, 0, 0);
              v95 = v101;
              v97 = v48 + v101;
            }
            else
            {
              v100 = 0;
            }
            if ( v95 != v97 )
            {
              if ( v97 > 0x3F || v95 > 0x40 )
                sub_16A5260(&v100, v97, v95);
              else
                v100 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v48 + 64) << v97;
            }
            v104.m128i_i16[0] = 257;
            v49 = sub_15A1070((__int64)v96, (__int64)&v100);
            v11 = sub_15FB440(26, v99, v49, (__int64)&v103, 0);
            if ( v101 > 0x40 && v100 )
              j_j___libc_free_0_0(v100);
            return v11;
          }
        }
      }
    }
  }
  else
  {
    if ( (unsigned int)(unsigned __int8)v36 - 48 > 1 )
      goto LABEL_21;
    if ( (*(_BYTE *)(v31 + 23) & 0x40) != 0 )
    {
      if ( !**(_QWORD **)(v31 - 8) )
        goto LABEL_21;
      v99 = **(__int64 ***)(v31 - 8);
      v47 = *(_QWORD *)(v31 - 8);
    }
    else
    {
      v47 = v31 - 24LL * (*(_DWORD *)(v31 + 20) & 0xFFFFFFF);
      v36 = *(_QWORD *)v47;
      if ( !*(_QWORD *)v47 )
        goto LABEL_21;
      v99 = *(__int64 **)v47;
    }
    if ( v30 == *(_BYTE **)(v47 + 24) )
      goto LABEL_54;
  }
LABEL_21:
  v37 = v31;
  v103.m128_u64[0] = (unsigned __int64)&v99;
  v103.m128_u64[1] = (unsigned __int64)&v100;
  if ( sub_179DD80(&v103, v31, v36, (__int64)v35) )
  {
    v39 = *(_QWORD **)v100;
    if ( *(_DWORD *)(v100 + 8) > 0x40u )
      v39 = (_QWORD *)*v39;
    v37 = (unsigned int)v39;
    if ( (unsigned int)v93 > (unsigned int)v39 )
    {
      v71 = sub_15A0680((__int64)v96, (unsigned int)((_DWORD)v93 - (_DWORD)v39), 0);
      v104.m128i_i16[0] = 257;
      v72 = sub_15FB440(23, v99, v71, (__int64)&v103, 0);
      v73 = sub_15F2370(v11);
      sub_15F2310(v72, v73);
      v74 = v11;
      v11 = v72;
      v75 = sub_15F2380(v74);
      sub_15F2330(v72, v75);
      return v11;
    }
    if ( (unsigned int)v93 < (unsigned int)v39 )
    {
      v76 = sub_15A0680((__int64)v96, (unsigned int)((_DWORD)v39 - (_DWORD)v93), 0);
      v104.m128i_i16[0] = 257;
      v11 = sub_15FB440((unsigned int)*(unsigned __int8 *)(v31 + 16) - 24, v99, v76, (__int64)&v103, 0);
      sub_15F2350(v11, 1);
      return v11;
    }
  }
  v40 = *(_BYTE *)(v31 + 16);
  if ( v40 == 47 )
  {
    if ( *(_QWORD *)(v31 - 48) )
    {
      v99 = *(__int64 **)(v31 - 48);
      v64 = *(_BYTE **)(v31 - 24);
      v65 = v64[16];
      if ( v65 == 13 )
      {
        v66 = (__int64)(v64 + 24);
        v100 = (__int64)(v64 + 24);
        goto LABEL_108;
      }
      if ( *(_BYTE *)(*(_QWORD *)v64 + 8LL) == 16 && v65 <= 0x10u )
      {
        v77 = sub_15A1020(v64, v37, *(_QWORD *)v64, v38);
        if ( v77 )
        {
          if ( *(_BYTE *)(v77 + 16) == 13 )
          {
            v66 = v77 + 24;
            v100 = v66;
            goto LABEL_108;
          }
        }
      }
    }
  }
  else if ( v40 == 5 && *(_WORD *)(v31 + 18) == 23 )
  {
    v87 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
    if ( *(_QWORD *)(v31 - 24 * v87) )
    {
      v99 = *(__int64 **)(v31 - 24 * v87);
      v88 = *(_BYTE **)(v31 + 24 * (1 - v87));
      if ( v88[16] == 13 )
      {
        v100 = (__int64)(v88 + 24);
LABEL_157:
        v66 = v100;
LABEL_108:
        if ( *(_DWORD *)(v66 + 8) <= 0x40u )
          v67 = *(_QWORD **)v66;
        else
          v67 = **(_QWORD ***)v66;
        v68 = (_DWORD)v93 + (_DWORD)v67;
        if ( v95 > v68 )
        {
          v104.m128i_i16[0] = 257;
          v69 = sub_15A0680((__int64)v96, v68, 0);
          return sub_15FB440(23, v99, v69, (__int64)&v103, 0);
        }
        goto LABEL_29;
      }
      if ( *(_BYTE *)(*(_QWORD *)v88 + 8LL) == 16 )
      {
        v89 = sub_15A1020(v88, v37, v87, v38);
        if ( v89 )
        {
          if ( *(_BYTE *)(v89 + 16) == 13 )
          {
            v100 = v89 + 24;
            goto LABEL_157;
          }
        }
      }
    }
  }
LABEL_29:
  if ( sub_15F2370(v11) )
    goto LABEL_30;
  v103.m128_i32[2] = v95;
  if ( v95 > 0x40 )
  {
    sub_16A4EF0((__int64)&v103, 0, 0);
    v95 = v103.m128_u32[2];
  }
  else
  {
    v103.m128_u64[0] = 0;
  }
  v70 = v95 - (_DWORD)v93;
  if ( v95 - (_DWORD)v93 != v95 )
  {
    if ( v70 > 0x3F || v95 > 0x40 )
      sub_16A5260(&v103, v70, v95);
    else
      v103.m128_u64[0] |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v93) << v70;
  }
  if ( !(unsigned __int8)sub_14C1670(
                           v31,
                           (__int64)&v103,
                           a1[166].m128i_i64[1],
                           0,
                           a1[165].m128i_i64[0],
                           v11,
                           a1[166].m128i_i64[0]) )
  {
    if ( v103.m128_i32[2] > 0x40u && v103.m128_u64[0] )
      j_j___libc_free_0_0(v103.m128_u64[0]);
LABEL_30:
    if ( !sub_15F2380(v11)
      && (unsigned int)v93 < (unsigned int)sub_14C23D0(
                                             v31,
                                             a1[166].m128i_i64[1],
                                             0,
                                             a1[165].m128i_i64[0],
                                             v11,
                                             a1[166].m128i_i64[0]) )
    {
      sub_15F2330(v11, 1);
      return v11;
    }
LABEL_31:
    v41 = *(_QWORD *)(v31 + 8);
    if ( !v41 || *(_QWORD *)(v41 + 8) )
    {
      if ( v30[16] > 0x10u )
        return 0;
      v42 = *(_BYTE *)(v31 + 16);
      goto LABEL_35;
    }
    v42 = *(_BYTE *)(v31 + 16);
    if ( v42 <= 0x17u )
    {
      if ( v42 != 5 )
      {
        if ( v30[16] > 0x10u )
          return 0;
        goto LABEL_35;
      }
      if ( (unsigned int)*(unsigned __int16 *)(v31 + 18) - 24 > 1
        || (v90 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF, (v79 = *(_QWORD *)(v31 - 24 * v90)) == 0)
        || (v91 = *(_BYTE **)(v31 + 24 * (1 - v90)), v30 != v91)
        || !v91 )
      {
        if ( v30[16] > 0x10u )
          return 0;
        if ( *(_WORD *)(v31 + 18) != 23 )
          goto LABEL_37;
        v86 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
        v54 = *(_QWORD *)(v31 - 24 * v86);
        if ( !v54 )
          goto LABEL_37;
        v55 = *(_QWORD *)(v31 + 24 * (1 - v86));
        if ( !v55 )
          goto LABEL_37;
LABEL_86:
        v104.m128i_i16[0] = 257;
        v56 = (__int64 *)sub_15A2D50(
                           (__int64 *)v54,
                           (__int64)v30,
                           0,
                           0,
                           *(double *)v12.m128_u64,
                           *(double *)v13.m128i_i64,
                           a5);
        return sub_15FB440(23, v56, v55, (__int64)&v103, 0);
      }
      goto LABEL_136;
    }
    if ( (unsigned int)v42 - 48 > 1 )
      goto LABEL_82;
    if ( (*(_BYTE *)(v31 + 23) & 0x40) != 0 )
    {
      v78 = *(__int64 **)(v31 - 8);
      v79 = *v78;
      if ( !*v78 )
        goto LABEL_82;
    }
    else
    {
      v78 = (__int64 *)(v31 - 24LL * (*(_DWORD *)(v31 + 20) & 0xFFFFFFF));
      v79 = *v78;
      if ( !*v78 )
        goto LABEL_82;
    }
    if ( v30 == (_BYTE *)v78[3] )
    {
LABEL_136:
      v98 = v79;
      v80 = sub_15A04A0(v96);
      v81 = a1->m128i_i64[1];
      v104.m128i_i16[0] = 257;
      if ( *(_BYTE *)(v80 + 16) > 0x10u || v30[16] > 0x10u )
      {
        v85 = sub_179D030(v81, (__int64 *)v80, (__int64)v30, (__int64 *)&v103, 0, 0);
        v84 = v98;
        v82 = (__int64 *)v85;
      }
      else
      {
        v82 = (__int64 *)sub_15A2D50(
                           (__int64 *)v80,
                           (__int64)v30,
                           0,
                           0,
                           *(double *)v12.m128_u64,
                           *(double *)v13.m128i_i64,
                           a5);
        v83 = sub_14DBA30((__int64)v82, *(_QWORD *)(v81 + 96), 0);
        v84 = v98;
        if ( v83 )
          v82 = (__int64 *)v83;
      }
      v104.m128i_i16[0] = 257;
      return sub_15FB440(26, v82, v84, (__int64)&v103, 0);
    }
LABEL_82:
    if ( v30[16] > 0x10u )
      return 0;
    if ( v42 == 47 )
    {
      v54 = *(_QWORD *)(v31 - 48);
      if ( *(_BYTE *)(v54 + 16) > 0x10u )
        return 0;
      v55 = *(_QWORD *)(v31 - 24);
      if ( !v55 )
        return 0;
      goto LABEL_86;
    }
LABEL_35:
    if ( v42 == 39 )
    {
      v27 = *(__int64 **)(v31 - 48);
      if ( !v27 )
        return 0;
      v28 = *(_QWORD *)(v31 - 24);
      if ( *(_BYTE *)(v28 + 16) > 0x10u )
        return 0;
      goto LABEL_40;
    }
    if ( v42 != 5 )
      return 0;
LABEL_37:
    if ( *(_WORD *)(v31 + 18) != 15 )
      return 0;
    v43 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
    v27 = *(__int64 **)(v31 - 24 * v43);
    if ( !v27 )
      return 0;
    v28 = *(_QWORD *)(v31 + 24 * (1 - v43));
    if ( !v28 )
      return 0;
LABEL_40:
    v104.m128i_i16[0] = 257;
    v44 = sub_15A2D50((__int64 *)v28, (__int64)v30, 0, 0, *(double *)v12.m128_u64, *(double *)v13.m128i_i64, a5);
    return sub_15FB440(15, v27, v44, (__int64)&v103, 0);
  }
  if ( v103.m128_i32[2] > 0x40u && v103.m128_u64[0] )
    j_j___libc_free_0_0(v103.m128_u64[0]);
  sub_15F2310(v11, 1);
  return v11;
}
