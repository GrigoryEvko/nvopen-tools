// Function: sub_17A0DD0
// Address: 0x17a0dd0
//
__int64 __fastcall sub_17A0DD0(
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
  __int64 **v26; // rbx
  __int64 v27; // r14
  unsigned __int8 v28; // al
  __int64 v29; // rdx
  unsigned __int64 v30; // rax
  unsigned int v31; // ebx
  __int64 v32; // rcx
  char v33; // bl
  __int64 v34; // rax
  __int64 v35; // rcx
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rax
  int v39; // edx
  int v40; // eax
  __int64 **v41; // rdx
  __int64 v42; // rax
  _BYTE *v43; // rdi
  unsigned __int8 v44; // al
  _QWORD *v45; // rax
  _BYTE *v46; // rdi
  unsigned __int8 v47; // al
  __int64 v48; // rdi
  _QWORD *v49; // rax
  _QWORD **v50; // rax
  unsigned int v51; // eax
  __int64 v52; // rax
  __int64 v53; // rax
  int v54; // eax
  int v55; // eax
  __int64 *v56; // rax
  bool v57; // zf
  __int64 v58; // r14
  int v59; // eax
  __int64 v60; // r12
  unsigned int v61; // eax
  __int64 v62; // rax
  _QWORD *v63; // rax
  __int64 v64; // r14
  _QWORD *v65; // rax
  int v66; // edx
  __int64 **v67; // rax
  __int64 *v68; // rax
  _BYTE *v69; // rdx
  int v70; // eax
  _QWORD *v71; // rax
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // rdi
  int v75; // eax
  int v76; // eax
  __int64 **v77; // rax
  __int64 *v78; // rdi
  _BYTE *v79; // rax
  int v80; // eax
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // rbx
  bool v84; // al
  unsigned __int64 **v85; // [rsp+8h] [rbp-98h]
  unsigned __int8 v86; // [rsp+8h] [rbp-98h]
  int v87; // [rsp+18h] [rbp-88h]
  int v88; // [rsp+18h] [rbp-88h]
  unsigned int v89; // [rsp+18h] [rbp-88h]
  int v90; // [rsp+1Ch] [rbp-84h]
  unsigned int v91; // [rsp+1Ch] [rbp-84h]
  unsigned int v92; // [rsp+20h] [rbp-80h]
  __int64 v93; // [rsp+20h] [rbp-80h]
  _BYTE *v94; // [rsp+28h] [rbp-78h]
  __int64 *v95; // [rsp+30h] [rbp-70h] BYREF
  _BYTE *v96; // [rsp+38h] [rbp-68h] BYREF
  __m128 v97; // [rsp+40h] [rbp-60h] BYREF
  __m128i v98; // [rsp+50h] [rbp-50h]
  __int64 v99; // [rsp+60h] [rbp-40h]

  v11 = a2;
  v12 = (__m128)_mm_loadu_si128(a1 + 167);
  v13 = _mm_loadu_si128(a1 + 168);
  v99 = a2;
  v97 = v12;
  v98 = v13;
  v14 = sub_15F23D0(a2);
  v15 = sub_13E1000(*(unsigned __int8 **)(a2 - 48), *(_QWORD *)(a2 - 24), v14, &v97);
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
  v26 = *(__int64 ***)a2;
  v27 = *(_QWORD *)(a2 - 48);
  v94 = *(_BYTE **)(a2 - 24);
  v92 = sub_16431D0(*(_QWORD *)a2);
  v28 = v94[16];
  v29 = (__int64)(v94 + 24);
  if ( v28 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v94 + 8LL) != 16 )
      goto LABEL_16;
    if ( v28 > 0x10u )
      goto LABEL_16;
    v34 = sub_15A1020(v94, a2, *(_QWORD *)v94, (__int64)v94);
    if ( !v34 || *(_BYTE *)(v34 + 16) != 13 )
      goto LABEL_16;
    v29 = v34 + 24;
  }
  if ( *(_DWORD *)(v29 + 8) > 0x40u )
  {
    v85 = (unsigned __int64 **)v29;
    v90 = *(_DWORD *)(v29 + 8);
    if ( v90 - (unsigned int)sub_16A57B0(v29) > 0x40 )
      goto LABEL_16;
    v30 = **v85;
    if ( v92 <= v30 )
      goto LABEL_16;
  }
  else
  {
    v30 = *(_QWORD *)v29;
    if ( (unsigned __int64)v92 <= *(_QWORD *)v29 )
      goto LABEL_16;
  }
  v35 = *(unsigned __int8 *)(v27 + 16);
  v91 = v30;
  if ( (_BYTE)v35 == 47 )
  {
    v38 = *(_QWORD *)(v27 - 48);
    v39 = *(unsigned __int8 *)(v38 + 16);
    if ( (unsigned __int8)v39 > 0x17u )
    {
      v66 = v39 - 24;
    }
    else
    {
      if ( (_BYTE)v39 != 5 )
        goto LABEL_49;
      v66 = *(unsigned __int16 *)(v38 + 18);
    }
    if ( v66 == 37 )
    {
      v67 = (*(_BYTE *)(v38 + 23) & 0x40) != 0
          ? *(__int64 ***)(v38 - 8)
          : (__int64 **)(v38 - 24LL * (*(_DWORD *)(v38 + 20) & 0xFFFFFFF));
      v68 = *v67;
      if ( v68 )
      {
        v95 = v68;
        v69 = *(_BYTE **)(v27 - 24);
        if ( v94 == v69 )
        {
          if ( v69 )
          {
            v70 = sub_16431D0(*v68);
            v35 = 47;
            if ( v91 == v92 - v70 )
              goto LABEL_104;
            goto LABEL_73;
          }
        }
      }
    }
LABEL_49:
    v40 = (unsigned __int8)v35 - 24;
    goto LABEL_50;
  }
  if ( (_BYTE)v35 != 5 )
  {
    if ( (unsigned __int8)v35 <= 0x17u || (unsigned __int8)v35 > 0x2Fu )
      goto LABEL_37;
LABEL_73:
    if ( ((0x80A800000000uLL >> v35) & 1) == 0 )
      goto LABEL_37;
    goto LABEL_49;
  }
  v36 = *(unsigned __int16 *)(v27 + 18);
  if ( (_WORD)v36 != 23 )
  {
    if ( (unsigned __int16)v36 > 0x17u )
      goto LABEL_37;
    goto LABEL_77;
  }
  a2 = *(_DWORD *)(v27 + 20) & 0xFFFFFFF;
  v74 = *(_QWORD *)(v27 - 24 * a2);
  v75 = *(unsigned __int8 *)(v74 + 16);
  if ( (unsigned __int8)v75 > 0x17u )
  {
    v76 = v75 - 24;
  }
  else
  {
    if ( (_BYTE)v75 != 5 )
      goto LABEL_77;
    v76 = *(unsigned __int16 *)(v74 + 18);
  }
  if ( v76 == 37 )
  {
    v89 = *(unsigned __int16 *)(v27 + 18);
    v77 = (__int64 **)sub_13CF970(v74);
    v35 = 5;
    v36 = v89;
    v78 = *v77;
    if ( *v77 )
    {
      v95 = *v77;
      v79 = *(_BYTE **)(v27 + 24 * (1 - a2));
      if ( v94 == v79 )
      {
        if ( v79 )
        {
          v80 = sub_16431D0(*v78);
          v35 = 5;
          if ( v91 != v92 - v80 )
          {
            v40 = 23;
            goto LABEL_50;
          }
LABEL_104:
          v98.m128i_i16[0] = 257;
          v71 = sub_1648A60(56, 1u);
          v11 = (__int64)v71;
          if ( v71 )
            sub_15FC810((__int64)v71, (__int64)v95, (__int64)v26, (__int64)&v97, 0);
          return v11;
        }
      }
    }
  }
LABEL_77:
  a2 = (__int64)&loc_80A800;
  v40 = (unsigned __int16)v36;
  if ( !_bittest64(&a2, v36) )
    goto LABEL_37;
LABEL_50:
  if ( v40 == 23 && (*(_BYTE *)(v27 + 17) & 4) != 0 )
  {
    v41 = (*(_BYTE *)(v27 + 23) & 0x40) != 0
        ? *(__int64 ***)(v27 - 8)
        : (__int64 **)(v27 - 24LL * (*(_DWORD *)(v27 + 20) & 0xFFFFFFF));
    if ( *v41 )
    {
      v86 = v35;
      v95 = *v41;
      v42 = sub_13CF970(v27);
      v35 = v86;
      v43 = *(_BYTE **)(v42 + 24);
      v44 = v43[16];
      if ( v44 == 13 )
      {
        v96 = v43 + 24;
        goto LABEL_57;
      }
      if ( *(_BYTE *)(*(_QWORD *)v43 + 8LL) == 16 && v44 <= 0x10u )
      {
        v81 = sub_15A1020(v43, a2, *(_QWORD *)v43, v86);
        if ( !v81 || *(_BYTE *)(v81 + 16) != 13 )
          goto LABEL_59;
        v96 = (_BYTE *)(v81 + 24);
LABEL_57:
        if ( *((_DWORD *)v96 + 2) > 0x40u )
        {
          v88 = *((_DWORD *)v96 + 2);
          if ( v88 - (unsigned int)sub_16A57B0((__int64)v96) > 0x40 )
            goto LABEL_59;
          v45 = **(_QWORD ***)v96;
          if ( v92 <= (unsigned __int64)v45 )
            goto LABEL_59;
        }
        else
        {
          v45 = *(_QWORD **)v96;
          if ( (unsigned __int64)v92 <= *(_QWORD *)v96 )
          {
LABEL_59:
            v35 = *(unsigned __int8 *)(v27 + 16);
            goto LABEL_37;
          }
        }
        a2 = (unsigned int)v45;
        if ( v91 > (unsigned int)v45 )
        {
          v82 = sub_15A0680((__int64)v26, v91 - (unsigned int)v45, 0);
          v98.m128i_i16[0] = 257;
          v83 = sub_15FB440(25, v95, v82, (__int64)&v97, 0);
          v84 = sub_15F23D0(v11);
          v11 = v83;
          sub_15F2350(v83, v84);
          return v11;
        }
        if ( v91 < (unsigned int)v45 )
        {
          v53 = sub_15A0680((__int64)v26, (unsigned int)v45 - v91, 0);
          v98.m128i_i16[0] = 257;
          v11 = sub_15FB440(23, v95, v53, (__int64)&v97, 0);
          sub_15F2330(v11, 1);
          return v11;
        }
        goto LABEL_59;
      }
    }
  }
LABEL_37:
  v97.m128_u64[0] = (unsigned __int64)&v95;
  v97.m128_u64[1] = (unsigned __int64)&v96;
  if ( (_BYTE)v35 == 49 )
  {
    if ( !*(_QWORD *)(v27 - 48) )
      goto LABEL_40;
    v95 = *(__int64 **)(v27 - 48);
    v46 = *(_BYTE **)(v27 - 24);
    v47 = v46[16];
    if ( v47 == 13 )
    {
      v48 = (__int64)(v46 + 24);
      v96 = (_BYTE *)v48;
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v46 + 8LL) != 16 )
        goto LABEL_40;
      if ( v47 > 0x10u )
        goto LABEL_40;
      v72 = sub_15A1020(v46, a2, *(_QWORD *)v46, v35);
      if ( !v72 || *(_BYTE *)(v72 + 16) != 13 )
        goto LABEL_40;
      *(_QWORD *)v97.m128_u64[1] = v72 + 24;
      v48 = (__int64)v96;
    }
  }
  else
  {
    if ( (_BYTE)v35 != 5 )
      goto LABEL_40;
    if ( *(_WORD *)(v27 + 18) != 25 )
      goto LABEL_40;
    v73 = *(_DWORD *)(v27 + 20) & 0xFFFFFFF;
    if ( !*(_QWORD *)(v27 - 24 * v73) )
      goto LABEL_40;
    v95 = *(__int64 **)(v27 - 24 * v73);
    if ( !(unsigned __int8)sub_13D7780((_QWORD **)&v97.m128_u64[1], *(_BYTE **)(v27 + 24 * (1 - v73))) )
      goto LABEL_40;
    v48 = (__int64)v96;
  }
  if ( *(_DWORD *)(v48 + 8) > 0x40u )
  {
    v87 = *(_DWORD *)(v48 + 8);
    if ( v87 - (unsigned int)sub_16A57B0(v48) > 0x40 )
      goto LABEL_40;
    v49 = **(_QWORD ***)v48;
  }
  else
  {
    v49 = *(_QWORD **)v48;
  }
  if ( v92 > (unsigned __int64)v49 )
  {
    if ( *(_DWORD *)(v48 + 8) <= 0x40u )
      v50 = *(_QWORD ***)v48;
    else
      v50 = **(_QWORD ****)v48;
    v51 = v91 + (_DWORD)v50;
    v98.m128i_i16[0] = 257;
    if ( v51 > v92 - 1 )
      v51 = v92 - 1;
    v52 = sub_15A0680((__int64)v26, v51, 0);
    return sub_15FB440(25, v95, v52, (__int64)&v97, 0);
  }
LABEL_40:
  v37 = *(_QWORD *)(v27 + 8);
  if ( !v37 || *(_QWORD *)(v37 + 8) )
    goto LABEL_42;
  v54 = *(unsigned __int8 *)(v27 + 16);
  if ( (unsigned __int8)v54 > 0x17u )
  {
    v55 = v54 - 24;
  }
  else
  {
    if ( (_BYTE)v54 != 5 )
      goto LABEL_42;
    v55 = *(unsigned __int16 *)(v27 + 18);
  }
  if ( v55 == 38 )
  {
    v56 = *(__int64 **)sub_13CF970(v27);
    if ( v56 )
    {
      v57 = *((_BYTE *)v26 + 8) == 16;
      v95 = v56;
      if ( v57 )
      {
LABEL_92:
        v58 = *v56;
        v59 = sub_16431D0(*v56);
        v60 = a1->m128i_i64[1];
        v61 = v59 - 1;
        v98.m128i_i16[0] = 257;
        if ( v61 > v91 )
          v61 = v91;
        v62 = sub_15A0680(v58, v61, 0);
        v63 = sub_173DE00(
                v60,
                (__int64)v95,
                v62,
                (__int64 *)&v97,
                0,
                *(double *)v12.m128_u64,
                *(double *)v13.m128i_i64,
                a5);
        v98.m128i_i16[0] = 257;
        v64 = (__int64)v63;
        v65 = sub_1648A60(56, 1u);
        v11 = (__int64)v65;
        if ( v65 )
          sub_15FC810((__int64)v65, v64, (__int64)v26, (__int64)&v97, 0);
        return v11;
      }
      if ( (unsigned __int8)sub_1705440((__int64)a1, (__int64)v26, *v56) )
      {
        v56 = v95;
        goto LABEL_92;
      }
    }
  }
LABEL_42:
  if ( sub_15F23D0(v11) )
    goto LABEL_16;
  sub_13D0120((__int64)&v97, v92, v91);
  if ( !(unsigned __int8)sub_14C1670(
                           v27,
                           (__int64)&v97,
                           a1[166].m128i_i64[1],
                           0,
                           a1[165].m128i_i64[0],
                           v11,
                           a1[166].m128i_i64[0]) )
  {
    if ( v97.m128_i32[2] > 0x40u && v97.m128_u64[0] )
      j_j___libc_free_0_0(v97.m128_u64[0]);
LABEL_16:
    v31 = v92 - 1;
    v97.m128_i32[2] = v92;
    v32 = 1LL << ((unsigned __int8)v92 - 1);
    if ( v92 > 0x40 )
    {
      v93 = 1LL << ((unsigned __int8)v92 - 1);
      sub_16A4EF0((__int64)&v97, 0, 0);
      v32 = v93;
      if ( v97.m128_i32[2] > 0x40u )
      {
        *(_QWORD *)(v97.m128_u64[0] + 8LL * (v31 >> 6)) |= v93;
        goto LABEL_19;
      }
    }
    else
    {
      v97.m128_u64[0] = 0;
    }
    v97.m128_u64[0] |= v32;
LABEL_19:
    v33 = sub_14C1670(v27, (__int64)&v97, a1[166].m128i_i64[1], 0, a1[165].m128i_i64[0], v11, a1[166].m128i_i64[0]);
    if ( v97.m128_i32[2] > 0x40u && v97.m128_u64[0] )
      j_j___libc_free_0_0(v97.m128_u64[0]);
    if ( v33 )
    {
      v98.m128i_i16[0] = 257;
      return sub_15FB440(24, (__int64 *)v27, (__int64)v94, (__int64)&v97, 0);
    }
    return 0;
  }
  if ( v97.m128_i32[2] > 0x40u && v97.m128_u64[0] )
    j_j___libc_free_0_0(v97.m128_u64[0]);
  sub_15F2350(v11, 1);
  return v11;
}
