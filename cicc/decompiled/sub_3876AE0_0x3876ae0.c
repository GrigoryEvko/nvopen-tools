// Function: sub_3876AE0
// Address: 0x3876ae0
//
__int64 __fastcall sub_3876AE0(
        __int64 *a1,
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
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // r14
  __int64 v16; // rax
  int v17; // r9d
  __int64 v18; // r8
  __int64 v19; // r15
  __int64 v20; // rax
  __m128i *v21; // rax
  __int64 v22; // rax
  __m128i *v23; // r12
  __int64 v24; // rax
  __int64 *m128i_i64; // r13
  __int64 v26; // rdx
  __int64 v27; // r14
  __int64 v28; // r12
  double v29; // xmm4_8
  double v30; // xmm5_8
  __m128i *v31; // rdi
  __m128i *v32; // rax
  __int64 v33; // rsi
  __int64 ***v34; // r12
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rax
  __int64 v38; // r8
  int v39; // r9d
  __int64 v40; // rdi
  double v41; // xmm4_8
  double v42; // xmm5_8
  __m128i *v43; // rax
  __int64 v44; // rsi
  __int64 v45; // rdx
  unsigned __int64 v46; // r13
  __int64 v47; // r14
  __int64 ***v48; // rax
  int v49; // r9d
  double v50; // xmm4_8
  double v51; // xmm5_8
  __int64 v52; // r8
  unsigned __int64 v53; // r12
  unsigned __int64 v54; // r15
  __int64 v55; // r13
  __int64 v56; // rax
  __int64 v57; // rsi
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  int v61; // r9d
  __int64 ***v62; // r12
  unsigned __int8 v63; // al
  unsigned int v64; // r13d
  unsigned __int64 *v65; // rdi
  signed __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rdx
  unsigned __int64 v70; // r13
  __int64 ***v71; // rax
  __int64 v72; // r8
  int v73; // r9d
  __int64 v74; // r14
  unsigned __int64 i; // r15
  __int64 v76; // rax
  int v78; // eax
  __int64 v79; // rsi
  unsigned __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // r8
  int v83; // r9d
  int v84; // eax
  __int64 v85; // [rsp+8h] [rbp-F8h]
  unsigned __int64 *v86; // [rsp+8h] [rbp-F8h]
  __int64 **v87; // [rsp+10h] [rbp-F0h]
  __m128i *v88; // [rsp+18h] [rbp-E8h]
  __int64 v89; // [rsp+18h] [rbp-E8h]
  __int64 v90[2]; // [rsp+20h] [rbp-E0h] BYREF
  __int64 *v91; // [rsp+30h] [rbp-D0h]
  __m128i *v92; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v93; // [rsp+48h] [rbp-B8h]
  _BYTE v94[176]; // [rsp+50h] [rbp-B0h] BYREF

  v11 = *a1;
  v12 = sub_1456040(**(_QWORD **)(a2 + 32));
  v13 = sub_1456E10(v11, v12);
  v14 = *(_QWORD *)(a2 + 32);
  v87 = (__int64 **)v13;
  v92 = (__m128i *)v94;
  v93 = 0x800000000LL;
  v15 = v14 + 8LL * *(_QWORD *)(a2 + 40);
  if ( v14 == v15 )
  {
    v23 = (__m128i *)v94;
    v26 = 0;
    m128i_i64 = (__int64 *)v94;
  }
  else
  {
    do
    {
      v16 = sub_3873B70((__int64)a1, *(_QWORD *)(v15 - 8));
      v18 = *(_QWORD *)(v15 - 8);
      v19 = v16;
      v20 = (unsigned int)v93;
      if ( (unsigned int)v93 >= HIDWORD(v93) )
      {
        v89 = *(_QWORD *)(v15 - 8);
        sub_16CD150((__int64)&v92, v94, 0, 16, v18, v17);
        v20 = (unsigned int)v93;
        v18 = v89;
      }
      v21 = &v92[v20];
      v15 -= 8;
      v21->m128i_i64[0] = v19;
      v21->m128i_i64[1] = v18;
      v22 = (unsigned int)(v93 + 1);
      LODWORD(v93) = v93 + 1;
    }
    while ( v14 != v15 );
    v23 = v92;
    v24 = v22;
    m128i_i64 = v92[v24].m128i_i64;
    v26 = (v24 * 16) >> 4;
  }
  v27 = *(_QWORD *)(*a1 + 56);
  sub_3872D20(v90, v23, v26);
  if ( v91 )
    sub_3873790(v23->m128i_i64, m128i_i64, v91, v90[1], v27);
  else
    sub_386FCA0(v23->m128i_i64, m128i_i64, v27);
  v28 = 0;
  j_j___libc_free_0((unsigned __int64)v91);
  v88 = v92;
  v31 = v92;
  v32 = &v92[(unsigned int)v93];
  if ( v92 != v32 )
  {
    while ( 1 )
    {
      v40 = v88->m128i_i64[1];
      if ( !v28 )
      {
        v69 = v88->m128i_i64[0];
        v70 = 0;
        while ( v88->m128i_i64[1] == v40 )
        {
          if ( v70 == 0x7FFFFFFFFFFFFFFFLL )
          {
            v28 = (__int64)sub_38761C0(a1, v40, v87, a3, a4, a5, a6, v29, v30, a9, a10);
            v74 = v28;
LABEL_48:
            for ( i = 2; i <= v70; i *= 2LL )
            {
              v76 = sub_3874770((__int64)a1, 0xFu, v74, v74, v72, v73, *(double *)a3.m128_u64, a4, a5);
              v74 = v76;
              if ( (i & v70) != 0 )
              {
                if ( v28 )
                  v28 = sub_3874770((__int64)a1, 0xFu, v28, v76, v72, v73, *(double *)a3.m128_u64, a4, a5);
                else
                  v28 = v76;
              }
            }
            goto LABEL_11;
          }
          ++v88;
          ++v70;
          if ( v32 == v88 || v69 != v88->m128i_i64[0] )
            break;
        }
        v71 = sub_38761C0(a1, v40, v87, a3, a4, a5, a6, v29, v30, a9, a10);
        if ( (v70 & 1) != 0 )
          v28 = (__int64)v71;
        v74 = (__int64)v71;
        if ( v70 > 1 )
          goto LABEL_48;
        goto LABEL_11;
      }
      if ( !sub_1456170(v40) )
        break;
      v33 = v28;
      v34 = sub_38744E0(a1, v28, v87, a3, a4, a5, a6, v41, v42, a9, a10);
      v37 = sub_15A06D0(v87, v33, v35, v36);
      ++v88;
      v28 = sub_3874770((__int64)a1, 0xDu, v37, (__int64)v34, v38, v39, *(double *)a3.m128_u64, a4, a5);
LABEL_11:
      v31 = v92;
      v32 = &v92[(unsigned int)v93];
      if ( v32 == v88 )
        goto LABEL_54;
    }
    v43 = &v92[(unsigned int)v93];
    v44 = v88->m128i_i64[1];
    if ( v88 == v43 )
    {
      v47 = 0;
      sub_38761C0(a1, v44, v87, a3, a4, a5, a6, v41, v42, a9, a10);
    }
    else
    {
      v45 = v88->m128i_i64[0];
      v46 = 0;
      while ( v88->m128i_i64[1] == v44 )
      {
        if ( v46 == 0x7FFFFFFFFFFFFFFFLL )
        {
          v47 = (__int64)sub_38761C0(a1, v44, v87, a3, a4, a5, a6, v41, v42, a9, a10);
          v52 = v47;
          goto LABEL_20;
        }
        ++v88;
        ++v46;
        if ( v43 == v88 || v88->m128i_i64[0] != v45 )
          break;
      }
      v47 = 0;
      v48 = sub_38761C0(a1, v44, v87, a3, a4, a5, a6, v41, v42, a9, a10);
      if ( (v46 & 1) != 0 )
        v47 = (__int64)v48;
      v52 = (__int64)v48;
      if ( v46 > 1 )
      {
LABEL_20:
        v85 = v28;
        v53 = 2;
        v54 = v46;
        v55 = v52;
        while ( 1 )
        {
          v56 = sub_3874770((__int64)a1, 0xFu, v55, v55, v52, v49, *(double *)a3.m128_u64, a4, a5);
          v55 = v56;
          if ( (v54 & v53) == 0 )
            goto LABEL_22;
          if ( v47 )
          {
            v47 = sub_3874770((__int64)a1, 0xFu, v47, v56, v52, v49, *(double *)a3.m128_u64, a4, a5);
LABEL_22:
            v53 *= 2LL;
            if ( v53 > v54 )
              goto LABEL_26;
          }
          else
          {
            v53 *= 2LL;
            v47 = v56;
            if ( v53 > v54 )
            {
LABEL_26:
              v28 = v85;
              break;
            }
          }
        }
      }
    }
    v57 = v28;
    v62 = sub_38744E0(a1, v28, v87, a3, a4, a5, a6, v50, v51, a9, a10);
    v63 = *((_BYTE *)v62 + 16);
    if ( v63 > 0x10u )
    {
      v58 = v47;
      v63 = *(_BYTE *)(v47 + 16);
      v47 = (__int64)v62;
      v62 = (__int64 ***)v58;
    }
    if ( v63 == 13 )
    {
      v64 = *((_DWORD *)v62 + 8);
      v65 = (unsigned __int64 *)(v62 + 3);
      if ( v64 > 0x40 )
      {
        v78 = sub_16A5940((__int64)v65);
        v65 = (unsigned __int64 *)(v62 + 3);
        if ( v78 == 1 )
          goto LABEL_58;
      }
      else
      {
        v66 = (signed __int64)v62[3];
        if ( v66 )
        {
          v58 = v66 - 1;
          if ( (v66 & (v66 - 1)) == 0 )
          {
LABEL_58:
            if ( v64 > 0x40 )
            {
              v79 = v64 - 1 - (unsigned int)sub_16A57B0((__int64)v65);
            }
            else
            {
              v79 = 0xFFFFFFFFLL;
              if ( *v65 )
              {
                _BitScanReverse64(&v80, *v65);
                v79 = 63 - ((unsigned int)v80 ^ 0x3F);
              }
            }
            v81 = sub_15A0680((__int64)v87, v79, 0);
            v28 = sub_3874770((__int64)a1, 0x17u, v47, v81, v82, v83, *(double *)a3.m128_u64, a4, a5);
            goto LABEL_11;
          }
        }
      }
      if ( *((_BYTE *)*v62 + 8) != 16 )
        goto LABEL_34;
    }
    else
    {
      v58 = (__int64)*v62;
      if ( *((_BYTE *)*v62 + 8) != 16 || v63 > 0x10u )
      {
LABEL_34:
        v28 = sub_3874770((__int64)a1, 0xFu, v47, (__int64)v62, v60, v61, *(double *)a3.m128_u64, a4, a5);
        goto LABEL_11;
      }
    }
    v67 = sub_15A1020(v62, v57, v58, v59);
    if ( v67 && *(_BYTE *)(v67 + 16) == 13 )
    {
      v64 = *(_DWORD *)(v67 + 32);
      v65 = (unsigned __int64 *)(v67 + 24);
      if ( v64 > 0x40 )
      {
        v86 = (unsigned __int64 *)(v67 + 24);
        v84 = sub_16A5940((__int64)v65);
        v65 = v86;
        if ( v84 == 1 )
          goto LABEL_58;
      }
      else
      {
        v68 = *(_QWORD *)(v67 + 24);
        if ( v68 && (v68 & (v68 - 1)) == 0 )
          goto LABEL_58;
      }
    }
    goto LABEL_34;
  }
LABEL_54:
  if ( v31 != (__m128i *)v94 )
    _libc_free((unsigned __int64)v31);
  return v28;
}
