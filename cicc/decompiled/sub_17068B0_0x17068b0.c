// Function: sub_17068B0
// Address: 0x17068b0
//
__int64 __fastcall sub_17068B0(
        const __m128i *a1,
        __int64 a2,
        unsigned int a3,
        unsigned __int8 *a4,
        unsigned __int8 *a5,
        unsigned __int8 *a6,
        __m128i a7,
        __m128i a8,
        double a9,
        unsigned __int8 *a10)
{
  int v14; // edx
  __int64 v15; // rax
  unsigned int v16; // ebx
  char v17; // si
  __m128i v18; // xmm2
  __m128i v19; // xmm3
  __int64 v20; // rdi
  __int64 v21; // r8
  __int64 v22; // r8
  unsigned __int8 *v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // rcx
  __int64 v27; // rdx
  unsigned __int8 v28; // r14
  unsigned __int64 v29; // rax
  bool v30; // al
  unsigned __int64 v31; // rax
  __int64 v32; // rdx
  unsigned __int64 v33; // rax
  __int64 v34; // rdx
  unsigned __int8 v35; // al
  __int64 v36; // rdi
  unsigned int v37; // eax
  __int64 v38; // rsi
  unsigned int v39; // ebx
  int v40; // eax
  __int64 v41; // rax
  __int64 v42; // rax
  const char *v43; // rax
  __int64 v44; // rdx
  unsigned __int64 v45; // rax
  void *v46; // rdx
  unsigned __int64 v47; // rax
  void *v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rax
  const char *v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rax
  _BYTE *v54; // [rsp+10h] [rbp-A0h]
  __int64 v55; // [rsp+10h] [rbp-A0h]
  __int64 v56; // [rsp+10h] [rbp-A0h]
  __int64 v57; // [rsp+18h] [rbp-98h]
  __int64 v58; // [rsp+20h] [rbp-90h]
  bool v59; // [rsp+28h] [rbp-88h]
  unsigned __int8 *v61; // [rsp+38h] [rbp-78h]
  __int64 v62; // [rsp+38h] [rbp-78h]
  __int64 v63; // [rsp+38h] [rbp-78h]
  __int64 v64; // [rsp+38h] [rbp-78h]
  __int64 v65; // [rsp+38h] [rbp-78h]
  const char *v66; // [rsp+40h] [rbp-70h] BYREF
  __int64 v67; // [rsp+48h] [rbp-68h]
  __m128 v68; // [rsp+50h] [rbp-60h] BYREF
  __m128 v69; // [rsp+60h] [rbp-50h]
  __int64 v70; // [rsp+70h] [rbp-40h]

  v14 = *(unsigned __int8 *)(a2 + 16);
  v61 = a6;
  v58 = *(_QWORD *)(a2 - 48);
  v15 = *(_QWORD *)(a2 - 24);
  v16 = v14 - 24;
  v17 = *(_BYTE *)(a2 + 16);
  v57 = v15;
  v59 = a3 <= 0x1C && ((1LL << a3) & 0x1C019800) != 0;
  if ( a3 == 26 )
  {
    if ( (unsigned int)(v14 - 51) > 1 )
      goto LABEL_6;
  }
  else if ( a3 == 27 )
  {
    if ( v14 != 50 )
      goto LABEL_6;
  }
  else if ( a3 != 15 || ((v17 - 35) & 0xFD) != 0 )
  {
    goto LABEL_6;
  }
  if ( a4 == a6 )
    goto LABEL_20;
  if ( a4 == a10 && v59 )
  {
    v24 = a10;
    a10 = a6;
    v61 = v24;
LABEL_20:
    a7 = _mm_loadu_si128(a1 + 167);
    v70 = a2;
    a8 = _mm_loadu_si128(a1 + 168);
    v68 = (__m128)a7;
    v69 = (__m128)a8;
    v54 = sub_13E1140(v16, a5, a10, &v68);
    if ( v54 )
      goto LABEL_78;
    v41 = *(_QWORD *)(v58 + 8);
    if ( v41 )
    {
      if ( !*(_QWORD *)(v41 + 8) )
      {
        v42 = *(_QWORD *)(v57 + 8);
        if ( v42 )
        {
          if ( !*(_QWORD *)(v42 + 8) )
          {
            v55 = a1->m128i_i64[1];
            v43 = sub_1649960(v57);
            v67 = v44;
            v66 = v43;
            v69.m128_i16[0] = 261;
            v68.m128_u64[0] = (unsigned __int64)&v66;
            v54 = (_BYTE *)sub_17066B0(
                             v55,
                             v16,
                             (__int64)a5,
                             (__int64)a10,
                             (__int64 *)&v68,
                             0,
                             *(double *)a7.m128i_i64,
                             *(double *)a8.m128i_i64,
                             a9);
            if ( v54 )
            {
LABEL_78:
              v25 = a1->m128i_i64[1];
              v69.m128_i16[0] = 257;
              v21 = sub_17066B0(
                      v25,
                      a3,
                      (__int64)a4,
                      (__int64)v54,
                      (__int64 *)&v68,
                      0,
                      *(double *)a7.m128i_i64,
                      *(double *)a8.m128i_i64,
                      a9);
              if ( v21 )
                goto LABEL_22;
            }
          }
        }
      }
    }
  }
LABEL_6:
  if ( !sub_17044F0(v16, a3) )
    return 0;
  if ( a5 == a10 )
    goto LABEL_11;
  if ( a5 != v61 || !v59 )
    return 0;
  v61 = a10;
LABEL_11:
  v18 = _mm_loadu_si128(a1 + 167);
  v70 = a2;
  v19 = _mm_loadu_si128(a1 + 168);
  v68 = (__m128)v18;
  v69 = (__m128)v19;
  v54 = sub_13E1140(v16, a4, v61, &v68);
  if ( !v54 )
  {
    v49 = *(_QWORD *)(v58 + 8);
    if ( !v49 )
      return 0;
    if ( *(_QWORD *)(v49 + 8) )
      return 0;
    v50 = *(_QWORD *)(v57 + 8);
    if ( !v50 )
      return 0;
    if ( *(_QWORD *)(v50 + 8) )
      return 0;
    v56 = a1->m128i_i64[1];
    v51 = sub_1649960(v58);
    v67 = v52;
    v66 = v51;
    v69.m128_i16[0] = 261;
    v68.m128_u64[0] = (unsigned __int64)&v66;
    v54 = (_BYTE *)sub_17066B0(
                     v56,
                     v16,
                     (__int64)a4,
                     (__int64)v61,
                     (__int64 *)&v68,
                     0,
                     *(double *)a7.m128i_i64,
                     *(double *)a8.m128i_i64,
                     *(double *)v18.m128i_i64);
    if ( !v54 )
      return 0;
  }
  v20 = a1->m128i_i64[1];
  v69.m128_i16[0] = 257;
  v21 = sub_17066B0(
          v20,
          a3,
          (__int64)v54,
          (__int64)a5,
          (__int64 *)&v68,
          0,
          *(double *)a7.m128i_i64,
          *(double *)a8.m128i_i64,
          *(double *)v18.m128i_i64);
  if ( !v21 )
    return 0;
LABEL_22:
  v62 = v21;
  sub_164B7C0(v21, a2);
  v22 = v62;
  v26 = *(unsigned __int8 *)(v62 + 16);
  if ( (unsigned __int8)(v26 - 35) > 0x11u )
    return v22;
  if ( (unsigned __int8)v26 > 0x2Fu )
    return v22;
  v27 = 0x80A800000000LL;
  v28 = ((0x80A800000000uLL >> v26) & 1) == 0;
  if ( ((0x80A800000000uLL >> v26) & 1) == 0 )
    return v22;
  v29 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v29 <= 0x2Fu && _bittest64(&v27, v29) )
  {
    v30 = sub_15F2380(a2);
    v22 = v62;
    v28 = v30;
  }
  v31 = *(unsigned __int8 *)(v58 + 16);
  if ( (unsigned __int8)v31 <= 0x17u )
  {
    if ( (_BYTE)v31 == 5 )
    {
      v45 = *(unsigned __int16 *)(v58 + 18);
      if ( (unsigned __int16)v45 <= 0x17u )
      {
        v46 = &loc_80A800;
        if ( _bittest64((const __int64 *)&v46, v45) )
          goto LABEL_31;
      }
    }
  }
  else if ( (unsigned __int8)v31 <= 0x2Fu )
  {
    v32 = 0x80A800000000LL;
    if ( _bittest64(&v32, v31) )
LABEL_31:
      v28 &= (*(_BYTE *)(v58 + 17) & 4) != 0;
  }
  v33 = *(unsigned __int8 *)(v57 + 16);
  if ( (unsigned __int8)v33 <= 0x17u )
  {
    if ( (_BYTE)v33 == 5 )
    {
      v47 = *(unsigned __int16 *)(v57 + 18);
      if ( (unsigned __int16)v47 <= 0x17u )
      {
        v48 = &loc_80A800;
        if ( _bittest64((const __int64 *)&v48, v47) )
          goto LABEL_35;
      }
    }
  }
  else if ( (unsigned __int8)v33 <= 0x2Fu )
  {
    v34 = 0x80A800000000LL;
    if ( _bittest64(&v34, v33) )
LABEL_35:
      v28 &= (*(_BYTE *)(v57 + 17) & 4) != 0;
  }
  if ( v16 == 11 && a3 == 15 )
  {
    v35 = v54[16];
    if ( v35 == 13 )
    {
      v36 = (__int64)(v54 + 24);
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v54 + 8LL) != 16 )
        return v22;
      if ( v35 > 0x10u )
        return v22;
      v65 = v22;
      v53 = sub_15A1020(v54, a2, *(_QWORD *)v54, v26);
      v22 = v65;
      if ( !v53 || *(_BYTE *)(v53 + 16) != 13 )
        return v22;
      v36 = v53 + 24;
    }
    v37 = *(_DWORD *)(v36 + 8);
    v38 = *(_QWORD *)v36;
    v39 = v37 - 1;
    if ( v37 <= 0x40 )
    {
      if ( v38 == 1LL << v39 )
        return v22;
      goto LABEL_43;
    }
    if ( (*(_QWORD *)(v38 + 8LL * (v39 >> 6)) & (1LL << v39)) == 0
      || (v63 = v22, v40 = sub_16A58A0(v36), v22 = v63, v39 != v40) )
    {
LABEL_43:
      v64 = v22;
      sub_15F2330(v22, v28);
      return v64;
    }
  }
  return v22;
}
