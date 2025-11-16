// Function: sub_1786350
// Address: 0x1786350
//
__int64 __fastcall sub_1786350(
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
  __int64 v10; // r12
  __m128i v12; // xmm2
  __m128 v13; // xmm0
  _QWORD *v14; // rdi
  _BYTE *v15; // rsi
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rbx
  __int64 v19; // r14
  _QWORD *v20; // rax
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  double v27; // xmm4_8
  double v28; // xmm5_8
  unsigned __int64 v29; // rsi
  _BYTE *v30; // r15
  __int64 v31; // rbx
  unsigned int v32; // eax
  __int64 v33; // rdx
  unsigned int v34; // r14d
  bool v35; // al
  __int64 v36; // rcx
  _BYTE *v37; // r8
  bool v38; // al
  __int64 v39; // r8
  unsigned __int64 v40; // rax
  unsigned int v41; // r14d
  __int64 v42; // rdx
  __int64 v43; // r14
  __int64 v44; // rdx
  __int64 v45; // rax
  unsigned int v46; // r14d
  int v47; // eax
  __int64 *v48; // rax
  __int64 v49; // rdi
  __int64 v50; // r14
  __int64 v51; // rax
  __int64 **v52; // r15
  _QWORD *v53; // rax
  __int64 v54; // rax
  unsigned int v55; // r14d
  unsigned int v56; // eax
  __int64 v57; // rcx
  unsigned __int64 v58; // rdx
  __int64 v59; // rcx
  int v60; // eax
  __int64 *v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rcx
  _BYTE *v64; // r14
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // rcx
  __int64 *v68; // r14
  __int64 v69; // rax
  __int64 v70; // rbx
  bool v71; // al
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // rbx
  bool v75; // al
  __int64 v76; // rax
  unsigned int v77; // r14d
  __int64 v78; // rax
  unsigned int v79; // esi
  bool v80; // al
  unsigned __int16 v81; // cx
  int v82; // eax
  int v83; // eax
  int v84; // eax
  __int64 *v85; // rdx
  __int64 v86; // r14
  unsigned int v87; // edx
  unsigned __int64 v88; // rax
  __int64 v89; // rsi
  __int64 v90; // r12
  __int64 v91; // r14
  __int64 v92; // rax
  __int64 v93; // r15
  _QWORD *v94; // rax
  __int64 v95; // rdi
  _QWORD *v96; // rax
  bool v97; // al
  unsigned int v98; // edx
  __int64 v99; // rax
  char v100; // cl
  unsigned int v101; // edx
  int v102; // eax
  bool v103; // al
  __int64 v104; // [rsp+8h] [rbp-B8h]
  unsigned int v105; // [rsp+8h] [rbp-B8h]
  int v106; // [rsp+8h] [rbp-B8h]
  __int64 v107; // [rsp+10h] [rbp-B0h]
  unsigned int v108; // [rsp+10h] [rbp-B0h]
  __int64 **v109; // [rsp+10h] [rbp-B0h]
  unsigned int v110; // [rsp+10h] [rbp-B0h]
  __int64 v112; // [rsp+20h] [rbp-A0h]
  __int64 v113; // [rsp+20h] [rbp-A0h]
  unsigned int v114; // [rsp+20h] [rbp-A0h]
  int v115; // [rsp+20h] [rbp-A0h]
  unsigned int v116; // [rsp+20h] [rbp-A0h]
  int v117; // [rsp+20h] [rbp-A0h]
  int v118; // [rsp+20h] [rbp-A0h]
  __int64 v119; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v120; // [rsp+38h] [rbp-88h]
  const char *v121; // [rsp+40h] [rbp-80h] BYREF
  __int64 v122; // [rsp+48h] [rbp-78h]
  __int16 v123; // [rsp+50h] [rbp-70h]
  __m128 v124; // [rsp+60h] [rbp-60h] BYREF
  __m128i v125; // [rsp+70h] [rbp-50h]
  __int64 v126; // [rsp+80h] [rbp-40h]

  v10 = a2;
  v12 = _mm_loadu_si128(a1 + 168);
  v13 = (__m128)_mm_loadu_si128(a1 + 167);
  v126 = a2;
  v14 = *(_QWORD **)(a2 - 48);
  v15 = *(_BYTE **)(a2 - 24);
  v124 = v13;
  v125 = v12;
  v16 = sub_13E11D0(v14, v15, (__int64 *)&v124);
  if ( v16 )
  {
    v17 = *(_QWORD *)(v10 + 8);
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
    if ( v10 == v19 )
      v19 = sub_1599EF0(*(__int64 ***)v10);
    sub_164D160(v10, v19, v13, a4, *(double *)v12.m128i_i64, a6, v21, v22, a9, a10);
    return v10;
  }
  v24 = (__int64)sub_1707490((__int64)a1, (unsigned __int8 *)v10, *(double *)v13.m128_u64, a4, *(double *)v12.m128i_i64);
  if ( v24 )
    return v24;
  v29 = v10;
  v24 = sub_1782440(a1->m128i_i64, v10, v25, v26, v13, a4, *(double *)v12.m128i_i64, a6, v27, v28, a9, a10);
  if ( v24 )
    return v24;
  v30 = *(_BYTE **)(v10 - 24);
  v31 = *(_QWORD *)(v10 - 48);
  v32 = (unsigned __int8)v30[16];
  v33 = (unsigned __int8)v30[16];
  if ( (_BYTE)v32 == 13 )
  {
    v34 = *((_DWORD *)v30 + 8);
    if ( v34 <= 0x40 )
    {
      v36 = 64 - v34;
      if ( *((_QWORD *)v30 + 3) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v34) )
        goto LABEL_17;
LABEL_19:
      v33 = v32;
      goto LABEL_20;
    }
    v35 = v34 == (unsigned int)sub_16A58F0((__int64)(v30 + 24));
LABEL_16:
    if ( v35 )
    {
LABEL_17:
      v125.m128i_i16[0] = 257;
      return sub_15FB530((__int64 *)v31, (__int64)&v124, 0, v36);
    }
    v32 = (unsigned __int8)v30[16];
    goto LABEL_19;
  }
  v36 = *(_QWORD *)v30;
  if ( *(_BYTE *)(*(_QWORD *)v30 + 8LL) == 16 && (unsigned __int8)v32 <= 0x10u )
  {
    v45 = sub_15A1020(*(_BYTE **)(v10 - 24), v10, v33, v36);
    if ( v45 && *(_BYTE *)(v45 + 16) == 13 )
    {
      v46 = *(_DWORD *)(v45 + 32);
      if ( v46 <= 0x40 )
      {
        v36 = 64 - v46;
        v35 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v46) == *(_QWORD *)(v45 + 24);
      }
      else
      {
        v35 = v46 == (unsigned int)sub_16A58F0(v45 + 24);
      }
      goto LABEL_16;
    }
    v115 = *(_QWORD *)(*(_QWORD *)v30 + 32LL);
    if ( !v115 )
      goto LABEL_17;
    v77 = 0;
    while ( 1 )
    {
      v29 = v77;
      v78 = sub_15A0A60((__int64)v30, v77);
      if ( !v78 )
        break;
      v36 = *(unsigned __int8 *)(v78 + 16);
      if ( (_BYTE)v36 != 9 )
      {
        if ( (_BYTE)v36 != 13 )
          break;
        v79 = *(_DWORD *)(v78 + 32);
        if ( v79 <= 0x40 )
        {
          v36 = 64 - v79;
          v29 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v79);
          v80 = v29 == *(_QWORD *)(v78 + 24);
        }
        else
        {
          v108 = *(_DWORD *)(v78 + 32);
          v29 = v108;
          v80 = v108 == (unsigned int)sub_16A58F0(v78 + 24);
        }
        if ( !v80 )
          break;
      }
      if ( v115 == ++v77 )
        goto LABEL_17;
    }
    v33 = (unsigned __int8)v30[16];
  }
LABEL_20:
  if ( (unsigned __int8)v33 > 0x17u )
  {
    v47 = (unsigned __int8)v33 - 24;
  }
  else
  {
    if ( (_BYTE)v33 != 5 )
      goto LABEL_22;
    v47 = *((unsigned __int16 *)v30 + 9);
  }
  if ( v47 != 38 )
    goto LABEL_47;
  if ( (v30[23] & 0x40) != 0 )
    v36 = *((_QWORD *)v30 - 1);
  else
    v36 = (__int64)&v30[-24 * (*((_DWORD *)v30 + 5) & 0xFFFFFFF)];
  v48 = *(__int64 **)v36;
  if ( *(_QWORD *)v36 )
  {
    v49 = *v48;
    if ( *(_BYTE *)(*v48 + 8) == 16 )
      v49 = **(_QWORD **)(v49 + 16);
    v29 = 1;
    if ( sub_1642F90(v49, 1) )
      goto LABEL_17;
    v33 = (unsigned __int8)v30[16];
  }
LABEL_22:
  if ( (_BYTE)v33 == 13 )
  {
    v37 = v30 + 24;
    goto LABEL_24;
  }
LABEL_47:
  if ( *(_BYTE *)(*(_QWORD *)v30 + 8LL) != 16 )
    goto LABEL_48;
  if ( (unsigned __int8)v33 > 0x10u )
    goto LABEL_61;
  v76 = sub_15A1020(v30, v29, v33, v36);
  if ( !v76 || *(_BYTE *)(v76 + 16) != 13 )
    goto LABEL_80;
  v37 = (_BYTE *)(v76 + 24);
LABEL_24:
  v112 = (__int64)v37;
  v38 = sub_15F23D0(v10);
  v39 = v112;
  if ( v38 )
  {
    v29 = *(unsigned int *)(v112 + 8);
    v40 = *(_QWORD *)v112;
    v41 = v29 - 1;
    v36 = (unsigned int)(v29 - 1);
    v42 = 1LL << ((unsigned __int8)v29 - 1);
    if ( (unsigned int)v29 > 0x40 )
    {
      v36 = v41 >> 6;
      if ( (*(_QWORD *)(v40 + 8 * v36) & v42) == 0 )
      {
        v82 = sub_16A5940(v112);
        v39 = v112;
        if ( v82 == 1 )
        {
          LODWORD(v40) = sub_16A57B0(v112);
          goto LABEL_29;
        }
      }
    }
    else if ( v40 )
    {
      v36 = v40 - 1;
      if ( (((v40 - 1) | v42) & v40) == 0 )
      {
        _BitScanReverse64(&v40, v40);
        LODWORD(v40) = v29 + (v40 ^ 0x3F) - 64;
LABEL_29:
        v43 = sub_15A0680(*(_QWORD *)v30, (int)(v41 - v40), 0);
        v121 = sub_1649960(v10);
        v122 = v44;
        v125.m128i_i16[0] = 261;
        v124.m128_u64[0] = (unsigned __int64)&v121;
        v10 = sub_15FB440(25, (__int64 *)v31, v43, (__int64)&v124, 0);
        sub_15F2350(v10, 1);
        return v10;
      }
    }
  }
  v72 = *(_QWORD *)(v31 + 8);
  if ( !v72 || *(_QWORD *)(v72 + 8) )
    goto LABEL_80;
  v83 = *(unsigned __int8 *)(v31 + 16);
  if ( (unsigned __int8)v83 <= 0x17u )
  {
    if ( (_BYTE)v83 == 5 )
    {
      v84 = *(unsigned __int16 *)(v31 + 18);
      goto LABEL_114;
    }
LABEL_80:
    v33 = (unsigned __int8)v30[16];
LABEL_48:
    if ( (unsigned __int8)v33 > 0x10u )
      goto LABEL_61;
    if ( sub_15964D0((__int64)v30, v29, v33, v36) )
    {
      v123 = 257;
      if ( *(_BYTE *)(v31 + 16) > 0x10u || v30[16] > 0x10u )
      {
        v50 = (__int64)sub_177F2B0(a1->m128i_i64[1], 32, v31, (__int64)v30, (__int64 *)&v121);
      }
      else
      {
        v113 = a1->m128i_i64[1];
        v50 = sub_15A37B0(0x20u, (_QWORD *)v31, v30, 0);
        v51 = sub_14DBA30(v50, *(_QWORD *)(v113 + 96), 0);
        if ( v51 )
          v50 = v51;
      }
      v52 = *(__int64 ***)v10;
      v125.m128i_i16[0] = 257;
      v53 = sub_1648A60(56, 1u);
      v10 = (__int64)v53;
      if ( v53 )
        sub_15FC690((__int64)v53, v50, (__int64)v52, (__int64)&v124, 0);
      return v10;
    }
    v58 = *(unsigned __int8 *)(v31 + 16);
    if ( (unsigned __int8)v58 <= 0x17u )
    {
      if ( (_BYTE)v58 != 5 )
        goto LABEL_61;
      v81 = *(_WORD *)(v31 + 18);
      if ( v81 > 0x17u )
        goto LABEL_61;
      v60 = v81;
      if ( (((unsigned __int64)&loc_80A800 >> v81) & 1) == 0 )
        goto LABEL_61;
    }
    else
    {
      if ( (unsigned __int8)v58 > 0x2Fu )
        goto LABEL_61;
      v59 = 0x80A800000000LL;
      v60 = (unsigned __int8)v58 - 24;
      if ( !_bittest64(&v59, v58) )
        goto LABEL_61;
    }
    if ( v60 != 13 )
      goto LABEL_61;
    if ( (*(_BYTE *)(v31 + 17) & 4) == 0 )
      goto LABEL_61;
    v61 = (__int64 *)sub_13CF970(v31);
    v64 = (_BYTE *)*v61;
    if ( *(_BYTE *)(*v61 + 16) > 0x10u )
      goto LABEL_61;
    if ( !sub_1593BB0(*v61, v29, v62, v63) )
    {
      if ( v64[16] == 13 )
      {
        if ( *((_DWORD *)v64 + 8) <= 0x40u )
        {
          v97 = *((_QWORD *)v64 + 3) == 0;
        }
        else
        {
          v117 = *((_DWORD *)v64 + 8);
          v97 = v117 == (unsigned int)sub_16A57B0((__int64)(v64 + 24));
        }
        if ( !v97 )
          goto LABEL_61;
      }
      else
      {
        if ( *(_BYTE *)(*(_QWORD *)v64 + 8LL) != 16 )
          goto LABEL_61;
        v54 = sub_15A1020(v64, v29, v65, v66);
        if ( v54 && *(_BYTE *)(v54 + 16) == 13 )
        {
          v55 = *(_DWORD *)(v54 + 32);
          if ( v55 <= 0x40 )
          {
            if ( *(_QWORD *)(v54 + 24) )
              goto LABEL_61;
          }
          else if ( v55 != (unsigned int)sub_16A57B0(v54 + 24) )
          {
            goto LABEL_61;
          }
        }
        else
        {
          v98 = 0;
          v118 = *(_DWORD *)(*(_QWORD *)v64 + 32LL);
          while ( v118 != v98 )
          {
            v110 = v98;
            v99 = sub_15A0A60((__int64)v64, v98);
            if ( !v99 )
              goto LABEL_61;
            v100 = *(_BYTE *)(v99 + 16);
            v101 = v110;
            if ( v100 != 9 )
            {
              if ( v100 != 13 )
                goto LABEL_61;
              if ( *(_DWORD *)(v99 + 32) <= 0x40u )
              {
                v103 = *(_QWORD *)(v99 + 24) == 0;
              }
              else
              {
                v106 = *(_DWORD *)(v99 + 32);
                v102 = sub_16A57B0(v99 + 24);
                v101 = v110;
                v103 = v106 == v102;
              }
              if ( !v103 )
                goto LABEL_61;
            }
            v98 = v101 + 1;
          }
        }
      }
    }
    v68 = *(__int64 **)(sub_13CF970(v31) + 24);
    if ( v68 )
    {
      v125.m128i_i16[0] = 257;
      v69 = sub_15A2B90((__int64 *)v30, 0, 0, v67, *(double *)v13.m128_u64, a4, *(double *)v12.m128i_i64);
      v70 = sub_15FB440(18, v68, v69, (__int64)&v124, 0);
      v71 = sub_15F23D0(v10);
      v10 = v70;
      sub_15F2350(v70, v71);
      return v10;
    }
LABEL_61:
    v56 = sub_16431D0(*(_QWORD *)v10);
    v120 = v56;
    v57 = 1LL << ((unsigned __int8)v56 - 1);
    if ( v56 > 0x40 )
    {
      v107 = 1LL << ((unsigned __int8)v56 - 1);
      v114 = v56 - 1;
      sub_16A4EF0((__int64)&v119, 0, 0);
      v57 = v107;
      if ( v120 > 0x40 )
      {
        *(_QWORD *)(v119 + 8LL * (v114 >> 6)) |= v107;
        goto LABEL_64;
      }
    }
    else
    {
      v119 = 0;
    }
    v119 |= v57;
LABEL_64:
    if ( (unsigned __int8)sub_14C1670(
                            v31,
                            (__int64)&v119,
                            a1[166].m128i_i64[1],
                            0,
                            a1[165].m128i_i64[0],
                            v10,
                            a1[166].m128i_i64[0])
      && ((unsigned __int8)sub_14C1670(
                             (__int64)v30,
                             (__int64)&v119,
                             a1[166].m128i_i64[1],
                             0,
                             a1[165].m128i_i64[0],
                             v10,
                             a1[166].m128i_i64[0])
       || (unsigned __int8)sub_14BDFF0(
                             (__int64)v30,
                             a1[166].m128i_i64[1],
                             1u,
                             0,
                             a1[165].m128i_i64[0],
                             v10,
                             a1[166].m128i_i64[0])) )
    {
      v121 = sub_1649960(v10);
      v122 = v73;
      v125.m128i_i16[0] = 261;
      v124.m128_u64[0] = (unsigned __int64)&v121;
      v74 = sub_15FB440(17, (__int64 *)v31, (__int64)v30, (__int64)&v124, 0);
      v75 = sub_15F23D0(v10);
      v10 = v74;
      sub_15F2350(v74, v75);
    }
    else
    {
      v10 = 0;
    }
    if ( v120 > 0x40 && v119 )
      j_j___libc_free_0_0(v119);
    return v10;
  }
  v84 = v83 - 24;
LABEL_114:
  if ( v84 != 38 )
    goto LABEL_80;
  v85 = (*(_BYTE *)(v31 + 23) & 0x40) != 0
      ? *(__int64 **)(v31 - 8)
      : (__int64 *)(v31 - 24LL * (*(_DWORD *)(v31 + 20) & 0xFFFFFFF));
  v86 = *v85;
  v104 = v39;
  if ( !*v85 )
    goto LABEL_80;
  v109 = *(__int64 ***)v86;
  v116 = sub_16431D0(*(_QWORD *)v86);
  v87 = *(_DWORD *)(v104 + 8);
  v88 = *(_QWORD *)v104;
  v36 = v87 - 1;
  v29 = 1LL << ((unsigned __int8)v87 - 1);
  if ( v87 > 0x40 )
  {
    v29 &= *(_QWORD *)(v88 + 8LL * ((unsigned int)v36 >> 6));
    v95 = v104;
    v105 = *(_DWORD *)(v104 + 8);
    LODWORD(v88) = v29 ? sub_16A5810(v95) : (unsigned int)sub_16A57B0(v95);
    v87 = v105;
  }
  else if ( (v29 & v88) != 0 )
  {
    v36 = 64 - v87;
    v88 = ~(v88 << (64 - (unsigned __int8)v87));
    if ( v88 )
    {
      _BitScanReverse64(&v88, v88);
      LODWORD(v88) = v88 ^ 0x3F;
    }
    else
    {
      LODWORD(v88) = 64;
    }
  }
  else
  {
    if ( v88 )
    {
      _BitScanReverse64(&v88, v88);
      LODWORD(v88) = v88 ^ 0x3F;
    }
    else
    {
      LODWORD(v88) = 64;
    }
    LODWORD(v88) = v87 + v88 - 64;
  }
  if ( v116 < v87 + 1 - (unsigned int)v88 )
    goto LABEL_80;
  v123 = 257;
  v89 = sub_15A43B0((unsigned __int64)v30, v109, 0);
  v90 = a1->m128i_i64[1];
  if ( *(_BYTE *)(v86 + 16) > 0x10u || *(_BYTE *)(v89 + 16) > 0x10u )
  {
    v125.m128i_i16[0] = 257;
    v96 = (_QWORD *)sub_15FB440(18, (__int64 *)v86, v89, (__int64)&v124, 0);
    v91 = (__int64)sub_171D920(v90, v96, (__int64 *)&v121);
  }
  else
  {
    v91 = sub_15A2C90((__int64 *)v86, v89, 0, *(double *)v13.m128_u64, a4, *(double *)v12.m128i_i64);
    v92 = sub_14DBA30(v91, *(_QWORD *)(v90 + 96), 0);
    if ( v92 )
      v91 = v92;
  }
  v93 = *(_QWORD *)v31;
  v125.m128i_i16[0] = 257;
  v94 = sub_1648A60(56, 1u);
  v10 = (__int64)v94;
  if ( v94 )
    sub_15FC810((__int64)v94, v91, v93, (__int64)&v124, 0);
  return v10;
}
