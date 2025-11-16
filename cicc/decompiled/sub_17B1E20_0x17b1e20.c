// Function: sub_17B1E20
// Address: 0x17b1e20
//
__int64 __fastcall sub_17B1E20(
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
  unsigned __int8 *v10; // r15
  __m128 v13; // xmm0
  __m128i v14; // xmm1
  __int64 v15; // rsi
  _BYTE *v16; // rdi
  __int64 v17; // rax
  double v18; // xmm4_8
  double v19; // xmm5_8
  __int64 v20; // rbx
  __int64 v21; // r14
  __int64 v22; // r13
  _QWORD *v23; // rax
  double v24; // xmm4_8
  double v25; // xmm5_8
  _BYTE *v27; // r9
  __int64 v28; // r13
  __int64 v29; // rdx
  __int64 v30; // r8
  unsigned __int64 v31; // r11
  __int64 v32; // rax
  __int32 v33; // edx
  unsigned __int8 v34; // al
  int v35; // edx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdx
  unsigned __int8 *v39; // rax
  __int64 v40; // r14
  const char *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rdx
  unsigned __int8 *v44; // rax
  __int64 v45; // rbx
  __int64 v46; // r14
  _QWORD *v47; // rax
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rcx
  unsigned __int64 v52; // rsi
  __int64 v53; // rcx
  __int64 v54; // rdx
  __int64 v55; // r12
  __int64 v56; // rax
  _QWORD *v57; // rsi
  int v58; // edx
  _QWORD *v59; // r14
  __int64 v60; // rax
  _QWORD *v61; // rax
  __int64 v62; // rax
  __int64 v63; // r12
  unsigned __int8 *v64; // rax
  __int64 v65; // rax
  _QWORD *v66; // rax
  _QWORD *v67; // r13
  __int64 v68; // rcx
  unsigned __int64 v69; // rdx
  __int64 v70; // rcx
  _BYTE *v71; // rdi
  __int64 v72; // rdx
  __int64 v73; // rdi
  unsigned __int8 *v74; // rax
  __int64 v75; // r15
  __int64 **v76; // rdx
  __int64 v77; // rax
  __int64 v78; // rbx
  __int64 v79; // r14
  _QWORD *v80; // rax
  __int64 v81; // r13
  __int64 **v82; // r12
  unsigned __int8 *v83; // rax
  __int64 v84; // rax
  __int64 v85; // rbx
  __int64 v86; // r14
  _QWORD *v87; // rax
  __int64 v88; // rdx
  unsigned __int64 v89; // rcx
  _BYTE *v90; // [rsp+8h] [rbp-A8h]
  __int64 v91; // [rsp+10h] [rbp-A0h]
  __int64 v92; // [rsp+10h] [rbp-A0h]
  __int64 v93; // [rsp+18h] [rbp-98h]
  unsigned int v94; // [rsp+18h] [rbp-98h]
  int v95; // [rsp+20h] [rbp-90h]
  int v96; // [rsp+20h] [rbp-90h]
  int v97; // [rsp+20h] [rbp-90h]
  int v98; // [rsp+20h] [rbp-90h]
  int v99; // [rsp+20h] [rbp-90h]
  __int64 v100; // [rsp+28h] [rbp-88h]
  unsigned __int64 v101; // [rsp+28h] [rbp-88h]
  __int64 v102; // [rsp+28h] [rbp-88h]
  __int64 *v103; // [rsp+28h] [rbp-88h]
  int v104; // [rsp+28h] [rbp-88h]
  __int64 v105; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v106; // [rsp+38h] [rbp-78h]
  unsigned __int64 v107; // [rsp+40h] [rbp-70h] BYREF
  __int64 v108; // [rsp+48h] [rbp-68h]
  __m128 v109; // [rsp+50h] [rbp-60h] BYREF
  __m128i v110; // [rsp+60h] [rbp-50h]
  __int64 v111; // [rsp+70h] [rbp-40h]

  v10 = (unsigned __int8 *)a2;
  v111 = a2;
  v13 = (__m128)_mm_loadu_si128(a1 + 167);
  v14 = _mm_loadu_si128(a1 + 168);
  v15 = *(_QWORD *)(a2 - 24);
  v16 = *(_BYTE **)(a2 - 48);
  v109 = v13;
  v110 = v14;
  v17 = sub_13D1770(v16, v15);
  if ( v17 )
  {
    v20 = *(_QWORD *)(a2 + 8);
    if ( v20 )
    {
      v21 = a1->m128i_i64[0];
      v22 = v17;
      do
      {
        v23 = sub_1648700(v20);
        sub_170B990(v21, (__int64)v23);
        v20 = *(_QWORD *)(v20 + 8);
      }
      while ( v20 );
LABEL_5:
      if ( a2 == v22 )
        v22 = sub_1599EF0(*(__int64 ***)a2);
      sub_164D160(a2, v22, v13, *(double *)v14.m128i_i64, a5, a6, v24, v25, a9, a10);
      return (__int64)v10;
    }
    return 0;
  }
  v27 = *(_BYTE **)(a2 - 48);
  v28 = (__int64)v27;
  if ( v27[16] <= 0x10u )
  {
    v100 = *(_QWORD *)(a2 - 48);
    if ( sub_17AE1C0(v100, 0) )
    {
      v22 = sub_15A0A60(v100, 0);
      if ( v22 )
      {
        v45 = *(_QWORD *)(a2 + 8);
        if ( v45 )
        {
          v46 = a1->m128i_i64[0];
          do
          {
            v47 = sub_1648700(v45);
            sub_170B990(v46, (__int64)v47);
            v45 = *(_QWORD *)(v45 + 8);
          }
          while ( v45 );
          goto LABEL_5;
        }
        return 0;
      }
    }
    v27 = *(_BYTE **)(a2 - 48);
    v28 = (__int64)v27;
  }
  v29 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v29 + 16) != 13 )
  {
    v34 = v27[16];
    goto LABEL_23;
  }
  v30 = *(_QWORD *)(*(_QWORD *)v27 + 32LL);
  v101 = (unsigned int)v30;
  if ( *(_DWORD *)(v29 + 32) > 0x40u )
  {
    v90 = v27;
    v91 = *(_QWORD *)(*(_QWORD *)v27 + 32LL);
    v93 = *(_QWORD *)(a2 - 24);
    v95 = *(_DWORD *)(v29 + 32);
    if ( v95 - (unsigned int)sub_16A57B0(v29 + 24) > 0x40 )
      return 0;
    LODWORD(v30) = v91;
    v27 = v90;
    v31 = **(_QWORD **)(v93 + 24);
    if ( v101 < v31 )
      return 0;
  }
  else
  {
    v31 = *(_QWORD *)(v29 + 24);
    if ( (unsigned int)v30 < v31 )
      return 0;
  }
  v32 = *((_QWORD *)v27 + 1);
  if ( v32 )
  {
    v33 = v30;
    if ( !*(_QWORD *)(v32 + 8) && (_DWORD)v30 != 1 )
    {
      v106 = v30;
      v65 = 1LL << v31;
      if ( (unsigned int)v30 > 0x40 )
      {
        v92 = 1LL << v31;
        v94 = v31;
        v97 = v30;
        sub_16A4EF0((__int64)&v105, 0, 0);
        LODWORD(v108) = v97;
        sub_16A4EF0((__int64)&v107, 0, 0);
        v33 = v108;
        LODWORD(v31) = v94;
        v65 = v92;
        if ( (unsigned int)v108 > 0x40 )
        {
          *(_QWORD *)(v107 + 8LL * (v94 >> 6)) |= v92;
          v109.m128_i32[2] = v108;
          if ( (unsigned int)v108 > 0x40 )
          {
            sub_16A4FD0((__int64)&v109, (const void **)&v107);
            v27 = *(_BYTE **)(a2 - 48);
            LODWORD(v31) = v94;
LABEL_59:
            v96 = v31;
            v66 = sub_17A4D70(a1->m128i_i64, v27, (__int64)&v109, &v105, 0);
            LODWORD(v31) = v96;
            v67 = v66;
            if ( v109.m128_i32[2] > 0x40u && v109.m128_u64[0] )
            {
              j_j___libc_free_0_0(v109.m128_u64[0]);
              LODWORD(v31) = v96;
            }
            if ( v67 )
            {
              if ( *(_QWORD *)(a2 - 48) )
              {
                v68 = *(_QWORD *)(a2 - 40);
                v69 = *(_QWORD *)(a2 - 32) & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v69 = v68;
                if ( v68 )
                  *(_QWORD *)(v68 + 16) = *(_QWORD *)(v68 + 16) & 3LL | v69;
              }
              *(_QWORD *)(a2 - 48) = v67;
              v70 = v67[1];
              *(_QWORD *)(a2 - 40) = v70;
              if ( v70 )
                *(_QWORD *)(v70 + 16) = (a2 - 40) | *(_QWORD *)(v70 + 16) & 3LL;
              *(_QWORD *)(a2 - 32) = *(_QWORD *)(a2 - 32) & 3LL | (unsigned __int64)(v67 + 1);
              v67[1] = a2 - 48;
              if ( (unsigned int)v108 > 0x40 && v107 )
                j_j___libc_free_0_0(v107);
              if ( v106 > 0x40 && v105 )
                j_j___libc_free_0_0(v105);
              return (__int64)v10;
            }
            if ( (unsigned int)v108 > 0x40 && v107 )
            {
              v98 = v31;
              j_j___libc_free_0_0(v107);
              LODWORD(v31) = v98;
            }
            if ( v106 > 0x40 && v105 )
            {
              v99 = v31;
              j_j___libc_free_0_0(v105);
              LODWORD(v31) = v99;
            }
            v28 = *(_QWORD *)(a2 - 48);
            goto LABEL_21;
          }
          v27 = *(_BYTE **)(a2 - 48);
LABEL_58:
          v109.m128_u64[0] = v107;
          goto LABEL_59;
        }
        v27 = *(_BYTE **)(a2 - 48);
      }
      else
      {
        v105 = 0;
        LODWORD(v108) = v30;
        v107 = 0;
      }
      v107 |= v65;
      v109.m128_i32[2] = v33;
      goto LABEL_58;
    }
  }
LABEL_21:
  v34 = *(_BYTE *)(v28 + 16);
  if ( v34 == 71 )
  {
    v71 = *(_BYTE **)(v28 - 24);
    if ( *(_BYTE *)(*(_QWORD *)v71 + 8LL) != 16 )
    {
      v35 = 71;
LABEL_96:
      if ( (unsigned int)(v35 - 60) > 0xC )
        return 0;
      goto LABEL_76;
    }
    if ( *(_QWORD *)(*(_QWORD *)v71 + 32LL) != v101 )
    {
LABEL_76:
      v72 = *(_QWORD *)(v28 + 8);
      if ( !v72 || *(_QWORD *)(v72 + 8) || v34 == 71 )
        return 0;
      v73 = a1->m128i_i64[1];
      v110.m128i_i16[0] = 257;
      v74 = sub_17AF100(v73, *(_QWORD *)(v28 - 24), *(_QWORD *)(a2 - 24), (__int64 *)&v109);
      v75 = (__int64)v74;
      if ( v74[16] > 0x17u )
        sub_170B990(a1->m128i_i64[0], (__int64)v74);
      v76 = *(__int64 ***)a2;
      v110.m128i_i16[0] = 257;
      return sub_15FDBD0((unsigned int)*(unsigned __int8 *)(v28 + 16) - 24, v75, (__int64)v76, (__int64)&v109, 0);
    }
    v81 = sub_14C48A0(v71, (unsigned int)v31);
    if ( v81 )
    {
      v82 = *(__int64 ***)a2;
      v110.m128i_i16[0] = 257;
      v83 = (unsigned __int8 *)sub_1648A60(56, 1u);
      v10 = v83;
      if ( v83 )
        sub_15FD590((__int64)v83, v81, (__int64)v82, (__int64)&v109, 0);
      return (__int64)v10;
    }
    v28 = *(_QWORD *)(a2 - 48);
    v34 = *(_BYTE *)(v28 + 16);
  }
  if ( v34 == 77 )
  {
    v77 = sub_17AF550(a1->m128i_i64, (__int64 *)a2, v28, v13, *(double *)v14.m128i_i64, a5, a6, v18, v19, a9, a10);
    if ( v77 )
      return v77;
    v28 = *(_QWORD *)(a2 - 48);
    v34 = *(_BYTE *)(v28 + 16);
  }
LABEL_23:
  if ( v34 <= 0x17u )
    return 0;
  v35 = v34;
  if ( (unsigned int)v34 - 35 <= 0x11 )
  {
    v36 = *(_QWORD *)(v28 + 8);
    if ( v36 && !*(_QWORD *)(v36 + 8) && sub_17AE1C0(v28, *(_BYTE *)(*(_QWORD *)(a2 - 24) + 16LL) == 13) )
    {
      v102 = a1->m128i_i64[1];
      v107 = (unsigned __int64)sub_1649960(a2);
      v108 = v37;
      v38 = *(_QWORD *)(a2 - 24);
      v109.m128_u64[0] = (unsigned __int64)&v107;
      v110.m128i_i16[0] = 773;
      v109.m128_u64[1] = (unsigned __int64)".lhs";
      v39 = sub_17AF100(v102, *(_QWORD *)(v28 - 48), v38, (__int64 *)&v109);
      v40 = a1->m128i_i64[1];
      v103 = (__int64 *)v39;
      v41 = sub_1649960(a2);
      v109.m128_u64[0] = (unsigned __int64)&v107;
      v107 = (unsigned __int64)v41;
      v108 = v42;
      v43 = *(_QWORD *)(a2 - 24);
      v110.m128i_i16[0] = 773;
      v109.m128_u64[1] = (unsigned __int64)".rhs";
      v44 = sub_17AF100(v40, *(_QWORD *)(v28 - 24), v43, (__int64 *)&v109);
      v110.m128i_i16[0] = 257;
      v10 = (unsigned __int8 *)sub_15FB440(
                                 (unsigned int)*(unsigned __int8 *)(v28 + 16) - 24,
                                 v103,
                                 (__int64)v44,
                                 (__int64)&v109,
                                 0);
      sub_15F2530(v10, v28, 1);
      return (__int64)v10;
    }
    return 0;
  }
  if ( v34 != 84 )
  {
    if ( v34 == 85 )
    {
      v56 = *(_QWORD *)(a2 - 24);
      if ( *(_BYTE *)(v56 + 16) != 13 )
        return 0;
      v57 = *(_QWORD **)(v56 + 24);
      if ( *(_DWORD *)(v56 + 32) > 0x40u )
        v57 = (_QWORD *)*v57;
      v58 = sub_15FA9D0(*(_QWORD *)(v28 - 24), (unsigned int)v57);
      if ( v58 < 0 )
      {
        v84 = sub_1599EF0(*(__int64 ***)a2);
        v85 = *(_QWORD *)(a2 + 8);
        v22 = v84;
        if ( v85 )
        {
          v86 = a1->m128i_i64[0];
          do
          {
            v87 = sub_1648700(v85);
            sub_170B990(v86, (__int64)v87);
            v85 = *(_QWORD *)(v85 + 8);
          }
          while ( v85 );
          goto LABEL_5;
        }
        return 0;
      }
      v59 = *(_QWORD **)(v28 - 72);
      v60 = *(_QWORD *)(*v59 + 32LL);
      if ( v58 >= (int)v60 )
      {
        v59 = *(_QWORD **)(v28 - 48);
        v58 -= v60;
      }
      v104 = v58;
      v61 = (_QWORD *)sub_16498A0(a2);
      v62 = sub_1643350(v61);
      v110.m128i_i16[0] = 257;
      v63 = sub_15A0680(v62, v104, 0);
      v64 = (unsigned __int8 *)sub_1648A60(56, 2u);
      v10 = v64;
      if ( v64 )
        sub_15FA320((__int64)v64, v59, v63, (__int64)&v109, 0);
      return (__int64)v10;
    }
    goto LABEL_96;
  }
  v48 = *(_QWORD *)(v28 - 24);
  v49 = *(_QWORD *)(a2 - 24);
  if ( v49 == v48 )
  {
    v78 = *(_QWORD *)(a2 + 8);
    v22 = *(_QWORD *)(v28 - 48);
    if ( v78 )
    {
      v79 = a1->m128i_i64[0];
      do
      {
        v80 = sub_1648700(v78);
        sub_170B990(v79, (__int64)v80);
        v78 = *(_QWORD *)(v78 + 8);
      }
      while ( v78 );
      goto LABEL_5;
    }
    return 0;
  }
  if ( *(_BYTE *)(v48 + 16) > 0x10u || *(_BYTE *)(v49 + 16) > 0x10u )
    return 0;
  sub_170B990(a1->m128i_i64[0], v28);
  v50 = *(_QWORD *)(v28 - 72);
  if ( v50 )
  {
    if ( *(_QWORD *)(a2 - 48) )
    {
      v51 = *(_QWORD *)(a2 - 40);
      v52 = *(_QWORD *)(a2 - 32) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v52 = v51;
      if ( v51 )
        *(_QWORD *)(v51 + 16) = v52 | *(_QWORD *)(v51 + 16) & 3LL;
    }
    *(_QWORD *)(a2 - 48) = v50;
    v53 = *(_QWORD *)(v50 + 8);
    *(_QWORD *)(a2 - 40) = v53;
    if ( v53 )
      *(_QWORD *)(v53 + 16) = (a2 - 40) | *(_QWORD *)(v53 + 16) & 3LL;
    v54 = *(_QWORD *)(a2 - 32);
    v55 = a2 - 48;
    *(_QWORD *)(v55 + 16) = (v50 + 8) | v54 & 3;
    *(_QWORD *)(v50 + 8) = v55;
  }
  else if ( *(_QWORD *)(a2 - 48) )
  {
    v88 = *(_QWORD *)(a2 - 40);
    v89 = *(_QWORD *)(a2 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v89 = v88;
    if ( v88 )
      *(_QWORD *)(v88 + 16) = v89 | *(_QWORD *)(v88 + 16) & 3LL;
    *(_QWORD *)(a2 - 48) = 0;
  }
  return (__int64)v10;
}
