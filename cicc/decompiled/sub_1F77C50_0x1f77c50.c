// Function: sub_1F77C50
// Address: 0x1f77c50
//
__int64 *__fastcall sub_1F77C50(__int64 **a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 *v5; // rax
  __int64 v6; // r14
  unsigned int v7; // r15d
  int v8; // ebx
  __int64 *result; // rax
  __m128 v10; // xmm0
  int v11; // eax
  __m128i v12; // xmm1
  int v13; // ecx
  int v14; // r8d
  int v15; // r9d
  const __m128i *v16; // roff
  __int64 v17; // r8
  unsigned __int8 v18; // cl
  unsigned __int8 v19; // r15
  __int64 v20; // r11
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 *v23; // rax
  unsigned int v24; // r10d
  __int64 v25; // r11
  __int64 v26; // rdx
  char v27; // al
  __int64 *v28; // rax
  unsigned __int64 v29; // rdx
  __int64 v30; // rbx
  unsigned __int64 v31; // rsi
  unsigned __int64 *v32; // rax
  __int64 *v33; // r13
  unsigned __int64 v34; // rcx
  unsigned __int64 v35; // r10
  __int16 *v36; // r11
  __int64 v37; // rax
  char v38; // dl
  __int64 v39; // rax
  unsigned __int128 v40; // kr00_16
  unsigned int v41; // esi
  int v42; // eax
  __int64 v43; // rax
  int v44; // ecx
  int v45; // r8d
  int v46; // r9d
  __int64 v47; // rax
  bool v48; // al
  __int64 v49; // rax
  __int64 v50; // rax
  bool v51; // al
  const __m128i *v52; // roff
  __int64 v53; // rsi
  __int64 *v54; // rax
  __int64 v55; // rdx
  bool v56; // al
  int v57; // eax
  int v58; // eax
  char v59; // al
  __int128 v60; // [rsp-20h] [rbp-F0h]
  __int128 v61; // [rsp-10h] [rbp-E0h]
  __int128 v62; // [rsp-10h] [rbp-E0h]
  __int64 v63; // [rsp+8h] [rbp-C8h]
  unsigned int v64; // [rsp+10h] [rbp-C0h]
  __int64 v65; // [rsp+18h] [rbp-B8h]
  unsigned __int32 v66; // [rsp+20h] [rbp-B0h]
  unsigned int v67; // [rsp+24h] [rbp-ACh]
  __int64 v68; // [rsp+28h] [rbp-A8h]
  __int64 v69; // [rsp+30h] [rbp-A0h]
  __int64 v70; // [rsp+30h] [rbp-A0h]
  const void **v71; // [rsp+38h] [rbp-98h]
  __int64 v72; // [rsp+50h] [rbp-80h]
  char v73; // [rsp+60h] [rbp-70h]
  unsigned __int64 v74; // [rsp+60h] [rbp-70h]
  __int16 *v75; // [rsp+68h] [rbp-68h]
  unsigned __int16 v76; // [rsp+70h] [rbp-60h]
  __m128i v77; // [rsp+70h] [rbp-60h]
  __int64 *v78; // [rsp+70h] [rbp-60h]
  int v79; // [rsp+70h] [rbp-60h]
  int v80; // [rsp+70h] [rbp-60h]
  unsigned __int128 v81; // [rsp+70h] [rbp-60h]
  __int64 v82; // [rsp+80h] [rbp-50h] BYREF
  int v83; // [rsp+88h] [rbp-48h]
  _BYTE v84[8]; // [rsp+90h] [rbp-40h] BYREF
  __int64 v85; // [rsp+98h] [rbp-38h]

  v76 = *(_WORD *)(a2 + 24);
  v5 = *(__int64 **)(a2 + 32);
  v6 = *v5;
  if ( *(_WORD *)(*v5 + 24) != 134 )
    goto LABEL_2;
  v7 = *((_DWORD *)v5 + 2);
  v8 = 0;
  if ( !sub_1D18C00(*v5, 1, v7) )
  {
    v5 = *(__int64 **)(a2 + 32);
LABEL_2:
    v6 = v5[5];
    v7 = *((_DWORD *)v5 + 12);
    v8 = 1;
  }
  if ( *(_WORD *)(v6 + 24) != 134 )
    return 0;
  v73 = sub_1D18C00(v6, 1, v7);
  if ( !v73 )
    return 0;
  v10 = (__m128)_mm_loadu_si128((const __m128i *)(*(_QWORD *)(v6 + 32) + 40LL));
  v69 = *(_QWORD *)(*(_QWORD *)(v6 + 32) + 40LL);
  if ( !(unsigned __int8)sub_1F70310(v10.m128_i64[0], v10.m128_u32[2], 1u) )
  {
    v11 = *(unsigned __int16 *)(v69 + 24);
    if ( v11 != 33 && v11 != 11 && !(unsigned __int8)sub_1D16930(v69) )
      return 0;
  }
  v12 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v6 + 32) + 80LL));
  v65 = *(_QWORD *)(*(_QWORD *)(v6 + 32) + 80LL);
  if ( !(unsigned __int8)sub_1F70310(v12.m128i_i64[0], v12.m128i_u32[2], 1u) )
  {
    v42 = *(unsigned __int16 *)(v65 + 24);
    if ( v42 != 33 && v42 != 11 && !(unsigned __int8)sub_1D16930(v65) )
      return 0;
  }
  v67 = v76;
  if ( (unsigned int)v76 - 118 > 1 )
    goto LABEL_15;
  v43 = sub_1D1ADA0(v10.m128_i64[0], v10.m128_u32[2], v10.m128_i64[1], v13, v14, v15);
  if ( !v43
    || ((v47 = *(_QWORD *)(v43 + 88), *(_DWORD *)(v47 + 32) <= 0x40u)
      ? (v48 = *(_QWORD *)(v47 + 24) == 0)
      : (v79 = *(_DWORD *)(v47 + 32), v48 = v79 == (unsigned int)sub_16A57B0(v47 + 24)),
        !v48) )
  {
    if ( !(unsigned __int8)sub_1F709E0(v10.m128_i64[0], v10.m128_u32[2]) )
      goto LABEL_15;
  }
  if ( (v49 = sub_1D1ADA0(v12.m128i_i64[0], v12.m128i_u32[2], v12.m128i_i64[1], v44, v45, v46)) != 0
    && ((v50 = *(_QWORD *)(v49 + 88), *(_DWORD *)(v50 + 32) <= 0x40u)
      ? (v51 = *(_QWORD *)(v50 + 24) == 0)
      : (v80 = *(_DWORD *)(v50 + 32), v51 = v80 == (unsigned int)sub_16A57B0(v50 + 24)),
        v51)
    || (unsigned __int8)sub_1F709E0(v12.m128i_i64[0], v12.m128i_u32[2]) )
  {
    v52 = (const __m128i *)(*(_QWORD *)(a2 + 32) + 40LL * (v8 ^ 1u));
    v68 = v52->m128i_i64[0];
    v66 = v52->m128i_u32[2];
    v77 = _mm_loadu_si128(v52);
  }
  else
  {
LABEL_15:
    v16 = (const __m128i *)(*(_QWORD *)(a2 + 32) + 40LL * (v8 ^ 1u));
    a5 = _mm_loadu_si128(v16);
    v77 = a5;
    v68 = v16->m128i_i64[0];
    v66 = v16->m128i_u32[2];
    v73 = sub_1F70310(a5.m128i_i64[0], a5.m128i_u32[2], 1u);
    if ( v73 )
    {
      v73 = 0;
    }
    else if ( *(_WORD *)(v68 + 24) != 11 && *(_WORD *)(v68 + 24) != 33 && !(unsigned __int8)sub_1D16930(v68) )
    {
      return 0;
    }
  }
  v17 = *(_QWORD *)(v6 + 40) + 16LL * v7;
  v18 = *(_BYTE *)v17;
  v19 = *(_BYTE *)v17;
  v71 = *(const void ***)(v17 + 8);
  if ( v8 )
  {
    v20 = v66;
    v21 = *(_QWORD *)(v68 + 40) + 16LL * v66;
    if ( v18 == *(_BYTE *)v21 && (*(_QWORD *)(v21 + 8) == *(_QWORD *)(v17 + 8) || v18) )
    {
      v22 = *(_QWORD *)(v6 + 72);
      v82 = v22;
      if ( v22 )
      {
        sub_1623A60((__int64)&v82, v22, 2);
        v20 = v66;
      }
      v83 = *(_DWORD *)(v6 + 64);
      *((_QWORD *)&v61 + 1) = v10.m128_u64[1];
      *(_QWORD *)&v61 = v69;
      v63 = v20;
      v23 = sub_1D332F0(
              *a1,
              v67,
              (__int64)&v82,
              v19,
              v71,
              0,
              *(double *)v10.m128_u64,
              *(double *)v12.m128i_i64,
              a5,
              v77.m128i_i64[0],
              v77.m128i_u64[1],
              v61);
      v24 = v19;
      v72 = (__int64)v23;
      v25 = v63;
      v70 = v26;
      if ( v73 || *((_WORD *)v23 + 12) == 48 )
        goto LABEL_28;
      goto LABEL_25;
    }
    return 0;
  }
  v53 = *(_QWORD *)(v6 + 72);
  v82 = v53;
  if ( v53 )
    sub_1623A60((__int64)&v82, v53, 2);
  v83 = *(_DWORD *)(v6 + 64);
  v54 = sub_1D332F0(
          *a1,
          v67,
          (__int64)&v82,
          v19,
          v71,
          0,
          *(double *)v10.m128_u64,
          *(double *)v12.m128i_i64,
          a5,
          v69,
          v10.m128_u64[1],
          *(_OWORD *)&v77);
  v24 = v19;
  v72 = (__int64)v54;
  v70 = v55;
  if ( v73 || *((_WORD *)v54 + 12) == 48 )
  {
LABEL_54:
    LOBYTE(v24) = v19;
    v28 = sub_1D332F0(
            *a1,
            v67,
            (__int64)&v82,
            v24,
            v71,
            0,
            *(double *)v10.m128_u64,
            *(double *)v12.m128i_i64,
            a5,
            v65,
            v12.m128i_u64[1],
            __PAIR128__(v66 | v77.m128i_i64[1] & 0xFFFFFFFF00000000LL, v68));
    goto LABEL_29;
  }
LABEL_25:
  v64 = v24;
  v27 = sub_1F70310(v72, v70, 1u);
  v24 = v64;
  if ( !v27 )
  {
    v58 = *(unsigned __int16 *)(v72 + 24);
    if ( v58 != 11 && v58 != 33 )
    {
      v59 = sub_1D16930(v72);
      v24 = v64;
      if ( !v59 )
        goto LABEL_68;
    }
  }
  if ( !v8 )
    goto LABEL_54;
  v25 = v66;
LABEL_28:
  *((_QWORD *)&v62 + 1) = v12.m128i_i64[1];
  LOBYTE(v24) = v19;
  *(_QWORD *)&v62 = v65;
  v28 = sub_1D332F0(
          *a1,
          v67,
          (__int64)&v82,
          v24,
          v71,
          0,
          *(double *)v10.m128_u64,
          *(double *)v12.m128i_i64,
          a5,
          v68,
          v25 | v77.m128i_i64[1] & 0xFFFFFFFF00000000LL,
          v62);
LABEL_29:
  v30 = (__int64)v28;
  v31 = v29;
  if ( v73
    || *((_WORD *)v28 + 12) == 48
    || (unsigned __int8)sub_1F70310((__int64)v28, v29, 1u)
    || (v57 = *(unsigned __int16 *)(v30 + 24), v57 == 33)
    || v57 == 11
    || (unsigned __int8)sub_1D16930(v30) )
  {
    v32 = *(unsigned __int64 **)(v6 + 32);
    v33 = *a1;
    v34 = v19;
    v35 = *v32;
    v36 = (__int16 *)v32[1];
    v37 = *(_QWORD *)(*v32 + 40) + 16LL * *((unsigned int *)v32 + 2);
    v38 = *(_BYTE *)v37;
    v39 = *(_QWORD *)(v37 + 8);
    v84[0] = v38;
    v85 = v39;
    v40 = __PAIR128__(v31, v30);
    if ( v38 )
    {
      v41 = ((unsigned __int8)(v38 - 14) < 0x60u) + 134;
    }
    else
    {
      v74 = v35;
      v75 = v36;
      *((_QWORD *)&v81 + 1) = v31;
      *(_QWORD *)&v81 = v30;
      v56 = sub_1F58D20((__int64)v84);
      v34 = v19;
      v35 = v74;
      v36 = v75;
      v40 = v81;
      v41 = 134 - (!v56 - 1);
    }
    *((_QWORD *)&v60 + 1) = v70;
    *(_QWORD *)&v60 = v72;
    result = sub_1D3A900(
               v33,
               v41,
               (__int64)&v82,
               v34,
               v71,
               0,
               v10,
               *(double *)v12.m128i_i64,
               a5,
               v35,
               v36,
               v60,
               v40,
               *((__int64 *)&v40 + 1));
    goto LABEL_35;
  }
LABEL_68:
  result = 0;
LABEL_35:
  if ( v82 )
  {
    v78 = result;
    sub_161E7C0((__int64)&v82, v82);
    return v78;
  }
  return result;
}
