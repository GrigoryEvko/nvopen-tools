// Function: sub_1784DF0
// Address: 0x1784df0
//
__int64 __fastcall sub_1784DF0(
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
  _BYTE *v14; // rsi
  _QWORD *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // r14
  __int64 v19; // r13
  _QWORD *v20; // rax
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  double v27; // xmm4_8
  double v28; // xmm5_8
  __int64 v29; // rsi
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r13
  _BYTE *v33; // rbx
  char v34; // al
  __int64 v35; // r14
  _QWORD *v36; // rdx
  _QWORD *v37; // rsi
  unsigned __int16 v38; // di
  _QWORD *v39; // r12
  __int64 v40; // rax
  int v41; // eax
  __int64 v42; // rcx
  int v43; // r8d
  __int64 *v44; // r9
  unsigned __int64 v45; // rdx
  int v46; // eax
  __int64 **v47; // rax
  __int64 *v48; // r10
  unsigned __int8 v49; // dl
  __int64 **v50; // rcx
  int v51; // eax
  __int64 **v52; // rdx
  bool v53; // al
  __int64 *v54; // r10
  _BYTE *v55; // rdi
  unsigned __int8 v56; // al
  unsigned __int8 v57; // al
  int v58; // eax
  __int64 **v59; // rdx
  __int64 *v60; // rax
  __int64 v61; // rdi
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rax
  unsigned __int64 v66; // rcx
  __m128i *v67; // rdi
  __int64 v68; // r15
  __int64 v69; // rax
  __int64 v70; // rbx
  unsigned __int64 v71; // rax
  __int64 v72; // rsi
  __int64 v73; // rdx
  _QWORD *v74; // r10
  unsigned __int8 v75; // al
  __int64 *v76; // rax
  __int64 v77; // rax
  int v78; // eax
  __int64 v79; // rax
  unsigned __int16 v80; // cx
  bool v81; // al
  __int64 v82; // [rsp+8h] [rbp-128h]
  __int64 *v83; // [rsp+18h] [rbp-118h]
  __int64 *v84; // [rsp+20h] [rbp-110h]
  __int64 v85; // [rsp+20h] [rbp-110h]
  __int64 **v86; // [rsp+28h] [rbp-108h]
  __int64 v87; // [rsp+28h] [rbp-108h]
  __int64 v88; // [rsp+28h] [rbp-108h]
  bool v89; // [rsp+37h] [rbp-F9h] BYREF
  __int64 *v90; // [rsp+38h] [rbp-F8h] BYREF
  __int64 *v91; // [rsp+40h] [rbp-F0h] BYREF
  __int64 **v92; // [rsp+48h] [rbp-E8h]
  __int16 v93; // [rsp+50h] [rbp-E0h]
  __m128 v94; // [rsp+60h] [rbp-D0h] BYREF
  __m128i v95; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v96; // [rsp+80h] [rbp-B0h]

  v11 = a2;
  v96 = a2;
  v12 = (__m128)_mm_loadu_si128(a1 + 167);
  v13 = _mm_loadu_si128(a1 + 168);
  v14 = *(_BYTE **)(a2 - 24);
  v15 = *(_QWORD **)(v11 - 48);
  v94 = v12;
  v95 = v13;
  v16 = sub_13E11B0(v15, v14, (__int64 *)&v94);
  if ( v16 )
  {
    v17 = *(_QWORD *)(v11 + 8);
    if ( !v17 )
      return 0;
    v18 = a1->m128i_i64[0];
    v19 = v16;
    do
    {
      v20 = sub_1648700(v17);
      sub_170B990(v18, (__int64)v20);
      v17 = *(_QWORD *)(v17 + 8);
    }
    while ( v17 );
    if ( v11 == v19 )
      v19 = sub_1599EF0(*(__int64 ***)v11);
    sub_164D160(v11, v19, v12, *(double *)v13.m128i_i64, a5, a6, v21, v22, a9, a10);
    return v11;
  }
  v24 = (__int64)sub_1707490((__int64)a1, (unsigned __int8 *)v11, *(double *)v12.m128_u64, *(double *)v13.m128i_i64, a5);
  if ( v24 )
    return v24;
  v29 = v11;
  v24 = sub_1782440(a1->m128i_i64, v11, v25, v26, v12, *(double *)v13.m128i_i64, a5, a6, v27, v28, a9, a10);
  if ( v24 )
    return v24;
  v32 = *(_QWORD *)(v11 - 48);
  v33 = *(_BYTE **)(v11 - 24);
  v34 = *(_BYTE *)(v32 + 16);
  if ( v34 == 48 )
  {
    if ( !*(_QWORD *)(v32 - 48) )
      goto LABEL_16;
    v90 = *(__int64 **)(v32 - 48);
    v55 = *(_BYTE **)(v32 - 24);
    v56 = v55[16];
    if ( v56 != 13 )
    {
      v30 = *(_QWORD *)v55;
      if ( *(_BYTE *)(*(_QWORD *)v55 + 8LL) != 16 || v56 > 0x10u )
        goto LABEL_16;
      goto LABEL_79;
    }
LABEL_54:
    v30 = (__int64)(v55 + 24);
    goto LABEL_55;
  }
  if ( v34 != 5 )
    goto LABEL_16;
  if ( *(_WORD *)(v32 + 18) != 24 )
    goto LABEL_16;
  v65 = *(_DWORD *)(v32 + 20) & 0xFFFFFFF;
  v31 = 4 * v65;
  v30 = *(_QWORD *)(v32 - 24 * v65);
  if ( !v30 )
    goto LABEL_16;
  v90 = *(__int64 **)(v32 - 24LL * (*(_DWORD *)(v32 + 20) & 0xFFFFFFF));
  v30 = 1 - v65;
  v55 = *(_BYTE **)(v32 + 24 * (1 - v65));
  if ( v55[16] == 13 )
    goto LABEL_54;
  if ( *(_BYTE *)(*(_QWORD *)v55 + 8LL) != 16 )
    goto LABEL_16;
LABEL_79:
  v64 = sub_15A1020(v55, v11, v30, v31);
  v30 = v64;
  if ( !v64 || *(_BYTE *)(v64 + 16) != 13 )
    goto LABEL_16;
  v30 = v64 + 24;
LABEL_55:
  v57 = v33[16];
  v29 = (__int64)(v33 + 24);
  if ( v57 != 13 )
  {
    v31 = *(_QWORD *)v33;
    if ( *(_BYTE *)(*(_QWORD *)v33 + 8LL) != 16 )
      goto LABEL_16;
    if ( v57 > 0x10u )
      goto LABEL_16;
    v87 = v30;
    v63 = sub_15A1020(v33, v29, v30, v31);
    if ( !v63 || *(_BYTE *)(v63 + 16) != 13 )
      goto LABEL_16;
    v30 = v87;
    v29 = v63 + 24;
  }
  sub_16A7ED0((__int64)&v91, v29, v30, &v89);
  if ( v89 )
  {
    if ( (unsigned int)v92 > 0x40 && v91 )
      j_j___libc_free_0_0(v91);
LABEL_16:
    v86 = *(__int64 ***)v11;
    if ( (unsigned __int8)sub_177F910((unsigned int *)v33, v29, v30, v31) )
    {
      v35 = a1->m128i_i64[1];
      v95.m128i_i16[0] = 257;
      if ( *(_BYTE *)(v32 + 16) <= 0x10u && v33[16] <= 0x10u )
      {
        v36 = v33;
        v37 = (_QWORD *)v32;
        v38 = 35;
        goto LABEL_20;
      }
      v39 = sub_177F2B0(v35, 35, v32, (__int64)v33, (__int64 *)&v94);
      goto LABEL_22;
    }
    v41 = (unsigned __int8)v33[16];
    if ( (unsigned __int8)v41 > 0x17u )
    {
      v58 = v41 - 24;
    }
    else
    {
      if ( (_BYTE)v41 != 5 )
        goto LABEL_25;
      v58 = *((unsigned __int16 *)v33 + 9);
    }
    if ( v58 == 38 )
    {
      v59 = (v33[23] & 0x40) != 0
          ? (__int64 **)*((_QWORD *)v33 - 1)
          : (__int64 **)&v33[-24 * (*((_DWORD *)v33 + 5) & 0xFFFFFFF)];
      v60 = *v59;
      if ( *v59 )
      {
        v90 = *v59;
        v61 = *v60;
        if ( *(_BYTE *)(*v60 + 8) == 16 )
          v61 = **(_QWORD **)(v61 + 16);
        if ( sub_1642F90(v61, 1) )
        {
          v35 = a1->m128i_i64[1];
          v95.m128i_i16[0] = 257;
          v62 = sub_15A04A0(v86);
          v36 = (_QWORD *)v62;
          if ( *(_BYTE *)(v32 + 16) <= 0x10u && *(_BYTE *)(v62 + 16) <= 0x10u )
          {
            v37 = (_QWORD *)v32;
            v38 = 32;
LABEL_20:
            v39 = (_QWORD *)sub_15A37B0(v38, v37, v36, 0);
            v40 = sub_14DBA30((__int64)v39, *(_QWORD *)(v35 + 96), 0);
            if ( v40 )
              v39 = (_QWORD *)v40;
            goto LABEL_22;
          }
          v39 = sub_177F2B0(v35, 32, v32, v62, (__int64 *)&v94);
LABEL_22:
          v95.m128i_i16[0] = 257;
          return sub_15FDE70(v39, (__int64)v86, (__int64)&v94, 0);
        }
      }
    }
LABEL_25:
    v24 = (__int64)sub_1780430(v11, a1->m128i_i64[1], *(double *)v12.m128_u64, *(double *)v13.m128i_i64, a5);
    if ( !v24 )
    {
      v45 = *(unsigned __int8 *)(v32 + 16);
      if ( (unsigned __int8)v45 <= 0x17u )
      {
        if ( (_BYTE)v45 != 5 )
          goto LABEL_88;
        v42 = *(unsigned __int16 *)(v32 + 18);
        if ( (unsigned __int16)v42 > 0x17u )
          goto LABEL_88;
        v46 = (unsigned __int16)v42;
        if ( (((unsigned __int64)&loc_80A800 >> v42) & 1) == 0 )
          goto LABEL_88;
      }
      else
      {
        if ( (unsigned __int8)v45 > 0x2Fu )
          goto LABEL_88;
        v42 = 0x80A800000000LL;
        v46 = (unsigned __int8)v45 - 24;
        if ( !_bittest64(&v42, v45) )
          goto LABEL_88;
      }
      if ( v46 != 15 )
        goto LABEL_88;
      if ( (*(_BYTE *)(v32 + 17) & 2) == 0 )
        goto LABEL_88;
      v47 = (__int64 **)sub_13CF970(v32);
      v48 = *v47;
      if ( !*v47 )
        goto LABEL_88;
      v44 = v47[3];
      if ( !v44 )
        goto LABEL_88;
      v49 = v33[16];
      if ( v49 <= 0x17u )
      {
        if ( v49 != 5 )
          goto LABEL_50;
        v80 = *((_WORD *)v33 + 9);
        if ( v80 > 0x17u )
          goto LABEL_50;
        if ( (((unsigned __int64)&loc_80A800 >> v80) & 1) == 0 || v80 != 15 )
        {
LABEL_133:
          if ( v80 > 0x17u )
            goto LABEL_50;
LABEL_124:
          v51 = v80;
          if ( (((unsigned __int64)&loc_80A800 >> v80) & 1) == 0 )
            goto LABEL_50;
LABEL_44:
          if ( v51 == 15 && (v33[17] & 2) != 0 )
          {
            if ( (v33[23] & 0x40) != 0 )
            {
              if ( **((_QWORD **)v33 - 1) )
              {
                v90 = (__int64 *)**((_QWORD **)v33 - 1);
                v52 = (__int64 **)*((_QWORD *)v33 - 1);
                goto LABEL_49;
              }
            }
            else
            {
              v52 = (__int64 **)&v33[-24 * (*((_DWORD *)v33 + 5) & 0xFFFFFFF)];
              if ( *v52 )
              {
                v90 = *v52;
LABEL_49:
                if ( v48 == v52[3] )
                {
LABEL_99:
                  v95.m128i_i16[0] = 257;
                  return sub_15FB440(17, v44, (__int64)v90, (__int64)&v94, 0);
                }
              }
            }
          }
LABEL_50:
          v83 = v48;
          v91 = v44;
          v84 = v44;
          v92 = &v90;
          v53 = sub_1781C00((__int64)&v91, (__int64)v33);
          v54 = v83;
          if ( v53
            || (v94.m128_u64[0] = (unsigned __int64)&v90,
                v94.m128_u64[1] = (unsigned __int64)v84,
                v81 = sub_1781CC0((__int64)&v94, (__int64)v33),
                v54 = v83,
                v81) )
          {
            v95.m128i_i16[0] = 257;
            return sub_15FB440(17, v54, (__int64)v90, (__int64)&v94, 0);
          }
LABEL_88:
          v94.m128_u64[0] = (unsigned __int64)&v95;
          v94.m128_u64[1] = 0x600000000LL;
          if ( sub_177FA40((__int64)v33, (__int64)&v94, 0, v42, v43, (int)v44) )
          {
            v66 = v94.m128_u64[0];
            v67 = (__m128i *)v94.m128_u64[0];
            if ( v94.m128_i32[2] )
            {
              v85 = v94.m128_u32[2];
              v68 = 0;
              v88 = (unsigned int)(v94.m128_i32[2] - 1);
              while ( 1 )
              {
                v70 = 24 * v68;
                v71 = v66 + 24 * v68;
                v72 = *(_QWORD *)(v71 + 8);
                if ( *(_QWORD *)v71 )
                {
                  v69 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __m128i *))v71)(v32, v72, v11, a1);
                }
                else
                {
                  v73 = *(_QWORD *)(v66 + 24LL * (unsigned int)(v68 - 1) + 16);
                  v74 = *(_QWORD **)(v66 + 24LL * *(_QWORD *)(v71 + 16) + 16);
                  v93 = 257;
                  v69 = sub_14EDD70(*(_QWORD *)(v72 - 72), v74, v73, (__int64)&v91, 0, 0);
                }
                if ( v88 == v68 )
                  break;
                v82 = v69;
                ++v68;
                sub_15F2120(v69, v11);
                v66 = v94.m128_u64[0];
                *(_QWORD *)(v94.m128_u64[0] + v70 + 16) = v82;
                if ( v85 == v68 )
                  goto LABEL_110;
              }
              v67 = (__m128i *)v94.m128_u64[0];
              v11 = v69;
              goto LABEL_112;
            }
          }
          else
          {
LABEL_110:
            v67 = (__m128i *)v94.m128_u64[0];
          }
          v11 = 0;
LABEL_112:
          if ( v67 != &v95 )
            _libc_free((unsigned __int64)v67);
          return v11;
        }
        if ( (v33[17] & 2) == 0 )
          goto LABEL_124;
      }
      else
      {
        if ( v49 > 0x2Fu )
          goto LABEL_50;
        if ( ((0x80A800000000uLL >> v49) & 1) == 0 || v49 != 39 || (v33[17] & 2) == 0 )
          goto LABEL_42;
      }
      if ( (v33[23] & 0x40) != 0 )
        v50 = (__int64 **)*((_QWORD *)v33 - 1);
      else
        v50 = (__int64 **)&v33[-24 * (*((_DWORD *)v33 + 5) & 0xFFFFFFF)];
      if ( v48 == *v50 && v50[3] )
      {
        v90 = v50[3];
        goto LABEL_99;
      }
      if ( v49 > 0x17u )
      {
LABEL_42:
        if ( ((0x80A800000000uLL >> v49) & 1) == 0 )
          goto LABEL_50;
        v51 = v49 - 24;
        goto LABEL_44;
      }
      v80 = *((_WORD *)v33 + 9);
      goto LABEL_133;
    }
    return v24;
  }
  if ( !sub_15F23D0(v11) )
    goto LABEL_117;
  v75 = *(_BYTE *)(v32 + 16);
  if ( v75 > 0x17u )
  {
    if ( (unsigned int)v75 - 41 <= 1 || (unsigned __int8)(v75 - 48) <= 1u )
      goto LABEL_105;
LABEL_117:
    v76 = v90;
    goto LABEL_118;
  }
  if ( v75 != 5 )
    goto LABEL_117;
  v78 = *(unsigned __int16 *)(v32 + 18);
  if ( (unsigned int)(v78 - 17) > 1 && (unsigned __int16)(v78 - 24) > 1u )
    goto LABEL_117;
LABEL_105:
  v76 = v90;
  if ( (*(_BYTE *)(v32 + 17) & 2) == 0 )
  {
LABEL_118:
    v95.m128i_i16[0] = 257;
    v79 = sub_15A1070(*v76, (__int64)&v91);
    v11 = sub_15FB440(17, v90, v79, (__int64)&v94, 0);
    goto LABEL_107;
  }
  v95.m128i_i16[0] = 257;
  v77 = sub_15A1070(*v90, (__int64)&v91);
  v11 = sub_15FB440(17, v90, v77, (__int64)&v94, 0);
  sub_15F2350(v11, 1);
LABEL_107:
  if ( (unsigned int)v92 > 0x40 && v91 )
    j_j___libc_free_0_0(v91);
  return v11;
}
