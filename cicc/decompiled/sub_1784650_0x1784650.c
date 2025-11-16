// Function: sub_1784650
// Address: 0x1784650
//
__int64 __fastcall sub_1784650(
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
  __int64 v14; // rsi
  _QWORD *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // r13
  __int64 v19; // r14
  _QWORD *v20; // rax
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  double v27; // xmm4_8
  double v28; // xmm5_8
  __int64 v29; // rsi
  __int64 v30; // rcx
  _BYTE *v31; // r14
  __int64 v32; // rdx
  unsigned int v33; // eax
  __int64 v34; // rbx
  unsigned int v35; // eax
  unsigned int v36; // ebx
  __int64 v37; // rdx
  unsigned int v38; // ebx
  unsigned int v39; // r8d
  __int64 v40; // rsi
  char v41; // cl
  __int64 v42; // rax
  int v43; // r9d
  unsigned __int64 v44; // rdi
  __int64 v45; // rax
  __int64 v46; // r8
  __int64 v47; // rdx
  _QWORD *v48; // rsi
  __int64 v49; // rsi
  unsigned int v50; // eax
  unsigned __int64 v51; // rsi
  __int32 v52; // eax
  __int64 **v53; // rdi
  __int64 v54; // rax
  __int64 v55; // rcx
  unsigned __int64 v56; // rsi
  __int64 v57; // rcx
  __int64 v58; // rdx
  __m128i *p_s; // rdi
  __int64 *v60; // rdi
  __int64 v61; // rbx
  __int64 v62; // r9
  unsigned int v63; // edx
  __int64 v64; // rax
  __int64 v65; // rcx
  __int64 v66; // rax
  __int64 v67; // rbx
  __int64 v68; // rsi
  __int64 v69; // rdx
  __int64 v70; // rdx
  unsigned __int64 v71; // rcx
  __int64 v72; // [rsp+8h] [rbp-F8h]
  __int64 *v73; // [rsp+10h] [rbp-F0h]
  char v74; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v75; // [rsp+20h] [rbp-E0h] BYREF
  unsigned int v76; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v77; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v78; // [rsp+38h] [rbp-C8h]
  __m128 v79; // [rsp+40h] [rbp-C0h] BYREF
  __m128i s; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v81; // [rsp+60h] [rbp-A0h]

  v11 = a2;
  v81 = a2;
  v12 = (__m128)_mm_loadu_si128(a1 + 167);
  v13 = _mm_loadu_si128(a1 + 168);
  v14 = *(_QWORD *)(a2 - 24);
  v15 = *(_QWORD **)(v11 - 48);
  v79 = v12;
  s = v13;
  v16 = sub_13E0AB0(v15, v14, (__int64 *)&v79);
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
  v24 = sub_1783D50(a1->m128i_i64, (_BYTE *)v11, v25, v26, v12, *(double *)v13.m128i_i64, a5, a6, v27, v28, a9, a10);
  if ( v24 )
    return v24;
  v31 = *(_BYTE **)(v11 - 24);
  v32 = (unsigned __int8)v31[16];
  v73 = *(__int64 **)(v11 - 48);
  if ( (_BYTE)v32 != 13 )
    goto LABEL_42;
  v33 = *((_DWORD *)v31 + 8);
  v29 = *((_QWORD *)v31 + 3);
  v30 = v33 - 1;
  if ( v33 > 0x40 )
    v29 = *(_QWORD *)(v29 + 8LL * ((unsigned int)v30 >> 6));
  v34 = (__int64)(v31 + 24);
  if ( (v29 & (1LL << ((unsigned __int8)v33 - 1))) == 0 )
  {
LABEL_42:
    if ( *(_BYTE *)(*(_QWORD *)v31 + 8LL) != 16 )
      goto LABEL_19;
    if ( (unsigned __int8)v32 > 0x10u )
      goto LABEL_19;
    v45 = sub_15A1020(v31, v29, v32, v30);
    v46 = v45;
    if ( !v45 || *(_BYTE *)(v45 + 16) != 13 )
      goto LABEL_19;
    v33 = *(_DWORD *)(v45 + 32);
    LODWORD(v30) = v33 - 1;
    v47 = *(_QWORD *)(v46 + 24);
    if ( v33 > 0x40 )
      v47 = *(_QWORD *)(v47 + 8LL * ((unsigned int)v30 >> 6));
    if ( (v47 & (1LL << ((unsigned __int8)v33 - 1))) == 0 )
      goto LABEL_19;
    v34 = v46 + 24;
  }
  v48 = *(_QWORD **)v34;
  if ( v33 <= 0x40 )
  {
    if ( v48 != (_QWORD *)(1LL << v30) )
      goto LABEL_47;
  }
  else if ( (v48[(unsigned int)v30 >> 6] & (1LL << v30)) == 0 || (unsigned int)sub_16A58A0(v34) != (_DWORD)v30 )
  {
LABEL_47:
    v49 = *(_QWORD *)(v11 - 24);
    if ( *(_BYTE *)(v49 + 16) > 0x17u )
      sub_170B990(a1->m128i_i64[0], v49);
    v50 = *(_DWORD *)(v34 + 8);
    LODWORD(v78) = v50;
    if ( v50 > 0x40 )
    {
      sub_16A4FD0((__int64)&v77, (const void **)v34);
      LOBYTE(v50) = v78;
      if ( (unsigned int)v78 > 0x40 )
      {
        sub_16A8F40((__int64 *)&v77);
        goto LABEL_52;
      }
      v51 = v77;
    }
    else
    {
      v51 = *(_QWORD *)v34;
    }
    v77 = ~v51 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v50);
LABEL_52:
    sub_16A7400((__int64)&v77);
    v52 = v78;
    v53 = *(__int64 ***)v11;
    LODWORD(v78) = 0;
    v79.m128_i32[2] = v52;
    v79.m128_u64[0] = v77;
    v54 = sub_15A1070((__int64)v53, (__int64)&v79);
    if ( *(_QWORD *)(v11 - 24) )
    {
      v55 = *(_QWORD *)(v11 - 16);
      v56 = *(_QWORD *)(v11 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v56 = v55;
      if ( v55 )
        *(_QWORD *)(v55 + 16) = v56 | *(_QWORD *)(v55 + 16) & 3LL;
    }
    *(_QWORD *)(v11 - 24) = v54;
    if ( v54 )
    {
      v57 = *(_QWORD *)(v54 + 8);
      *(_QWORD *)(v11 - 16) = v57;
      if ( v57 )
        *(_QWORD *)(v57 + 16) = (v11 - 16) | *(_QWORD *)(v57 + 16) & 3LL;
      *(_QWORD *)(v11 - 8) = (v54 + 8) | *(_QWORD *)(v11 - 8) & 3LL;
      *(_QWORD *)(v54 + 8) = v11 - 24;
    }
    if ( v79.m128_i32[2] > 0x40u && v79.m128_u64[0] )
      j_j___libc_free_0_0(v79.m128_u64[0]);
    if ( (unsigned int)v78 <= 0x40 )
      return v11;
    v44 = v77;
    if ( !v77 )
      return v11;
LABEL_37:
    j_j___libc_free_0_0(v44);
    return v11;
  }
LABEL_19:
  v35 = sub_16431D0(*(_QWORD *)v11);
  v76 = v35;
  v36 = v35;
  if ( v35 > 0x40 )
  {
    sub_16A4EF0((__int64)&v75, 0, 0);
    v37 = 1LL << ((unsigned __int8)v36 - 1);
    if ( v76 > 0x40 )
    {
      *(_QWORD *)(v75 + 8LL * ((v36 - 1) >> 6)) |= v37;
      goto LABEL_22;
    }
  }
  else
  {
    v75 = 0;
    v37 = 1LL << ((unsigned __int8)v35 - 1);
  }
  v75 |= v37;
LABEL_22:
  if ( (unsigned __int8)sub_14C1670(
                          (__int64)v31,
                          (__int64)&v75,
                          a1[166].m128i_i64[1],
                          0,
                          a1[165].m128i_i64[0],
                          v11,
                          a1[166].m128i_i64[0])
    && (unsigned __int8)sub_14C1670(
                          (__int64)v73,
                          (__int64)&v75,
                          a1[166].m128i_i64[1],
                          0,
                          a1[165].m128i_i64[0],
                          v11,
                          a1[166].m128i_i64[0]) )
  {
    v77 = (unsigned __int64)sub_1649960(v11);
    v78 = v58;
    s.m128i_i16[0] = 261;
    v79.m128_u64[0] = (unsigned __int64)&v77;
    v11 = sub_15FB440(20, v73, (__int64)v31, (__int64)&v79, 0);
  }
  else
  {
    if ( (v31[16] & 0xFB) == 8 )
    {
      v72 = *(_QWORD *)(*(_QWORD *)v31 + 32LL);
      if ( (_DWORD)v72 )
      {
        v74 = 0;
        v38 = 0;
        while ( 1 )
        {
          v42 = sub_15A0A60((__int64)v31, v38);
          if ( !v42 )
            break;
          if ( *(_BYTE *)(v42 + 16) == 13 )
          {
            v39 = *(_DWORD *)(v42 + 32);
            v40 = *(_QWORD *)(v42 + 24);
            if ( v39 > 0x40 )
              v40 = *(_QWORD *)(v40 + 8LL * ((v39 - 1) >> 6));
            v41 = v74;
            if ( (v40 & (1LL << ((unsigned __int8)v39 - 1))) != 0 )
              v41 = 1;
            v74 = v41;
          }
          if ( ++v38 == (_DWORD)v72 )
          {
            if ( !v74 )
              break;
            p_s = &s;
            v79.m128_u64[0] = (unsigned __int64)&s;
            v79.m128_u64[1] = 0x1000000000LL;
            if ( (unsigned int)v72 > 0x10 )
            {
              sub_16CD150((__int64)&v79, &s, (unsigned int)v72, 8, v39, v43);
              p_s = (__m128i *)v79.m128_u64[0];
            }
            v79.m128_i32[2] = v72;
            if ( 8LL * (unsigned int)v72 )
              memset(p_s, 0, 8LL * (unsigned int)v72);
            v60 = (__int64 *)v79.m128_u64[0];
            v61 = 0;
            do
            {
              v60[v61] = sub_15A0A60((__int64)v31, v61);
              v60 = (__int64 *)v79.m128_u64[0];
              v62 = *(_QWORD *)(v79.m128_u64[0] + 8 * v61);
              if ( *(_BYTE *)(v62 + 16) == 13 )
              {
                v63 = *(_DWORD *)(v62 + 32);
                v64 = *(_QWORD *)(v62 + 24);
                v65 = v63 - 1;
                if ( v63 > 0x40 )
                {
                  v65 = (unsigned int)v65 >> 6;
                  v64 = *(_QWORD *)(v64 + 8LL * (unsigned int)v65);
                }
                if ( (v64 & (1LL << ((unsigned __int8)v63 - 1))) != 0 )
                {
                  v66 = sub_15A2B90(
                          *(__int64 **)(v79.m128_u64[0] + 8 * v61),
                          0,
                          0,
                          v65,
                          *(double *)v12.m128_u64,
                          *(double *)v13.m128i_i64,
                          a5);
                  *(_QWORD *)(v79.m128_u64[0] + 8 * v61) = v66;
                  v60 = (__int64 *)v79.m128_u64[0];
                }
              }
              ++v61;
            }
            while ( v61 != (unsigned int)v72 );
            v67 = sub_15A01B0(v60, v79.m128_u32[2]);
            if ( (_BYTE *)v67 != v31 )
            {
              v68 = *(_QWORD *)(v11 - 24);
              if ( *(_BYTE *)(v68 + 16) <= 0x17u || (sub_170B990(a1->m128i_i64[0], v68), *(_QWORD *)(v11 - 24)) )
              {
                v70 = *(_QWORD *)(v11 - 16);
                v71 = *(_QWORD *)(v11 - 8) & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v71 = v70;
                if ( v70 )
                  *(_QWORD *)(v70 + 16) = v71 | *(_QWORD *)(v70 + 16) & 3LL;
              }
              *(_QWORD *)(v11 - 24) = v67;
              if ( v67 )
              {
                v69 = *(_QWORD *)(v67 + 8);
                *(_QWORD *)(v11 - 16) = v69;
                if ( v69 )
                  *(_QWORD *)(v69 + 16) = (v11 - 16) | *(_QWORD *)(v69 + 16) & 3LL;
                *(_QWORD *)(v11 - 8) = (v67 + 8) | *(_QWORD *)(v11 - 8) & 3LL;
                *(_QWORD *)(v67 + 8) = v11 - 24;
              }
              if ( (__m128i *)v79.m128_u64[0] != &s )
                _libc_free(v79.m128_u64[0]);
              goto LABEL_35;
            }
            if ( (__m128i *)v79.m128_u64[0] != &s )
              _libc_free(v79.m128_u64[0]);
            break;
          }
        }
      }
    }
    v11 = 0;
  }
LABEL_35:
  if ( v76 > 0x40 )
  {
    v44 = v75;
    if ( v75 )
      goto LABEL_37;
  }
  return v11;
}
