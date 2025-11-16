// Function: sub_20AC6D0
// Address: 0x20ac6d0
//
__int64 *__fastcall sub_20AC6D0(
        double a1,
        double a2,
        __m128i a3,
        __int64 a4,
        unsigned int a5,
        const void **a6,
        __int64 a7,
        __int64 a8,
        int a9,
        __int64 a10,
        int a11,
        __int64 a12,
        __int64 a13)
{
  int v15; // eax
  __int64 *result; // rax
  const __m128i *v17; // rax
  __int64 v18; // r13
  int v19; // ecx
  __m128 v20; // xmm0
  __int64 v21; // rsi
  __int64 v22; // rax
  char v23; // cl
  unsigned int v24; // r8d
  __int64 v25; // r15
  int v26; // eax
  unsigned int v27; // esi
  __int64 v28; // rax
  unsigned __int64 v29; // rax
  unsigned int v30; // r9d
  int v31; // ecx
  unsigned __int64 v32; // rax
  int v33; // esi
  __int64 *v34; // r13
  __int64 v35; // rdi
  __int64 (*v36)(); // rax
  int v37; // eax
  __int128 v38; // rax
  __int64 *v39; // rax
  unsigned __int64 v40; // rdx
  __int64 *v41; // rax
  __int16 *v42; // rdx
  int v43; // eax
  int v44; // eax
  unsigned int v45; // eax
  int v46; // eax
  unsigned int v47; // [rsp+8h] [rbp-88h]
  int v48; // [rsp+Ch] [rbp-84h]
  unsigned int v49; // [rsp+10h] [rbp-80h]
  __int128 v50; // [rsp+10h] [rbp-80h]
  unsigned int v51; // [rsp+10h] [rbp-80h]
  unsigned int v52; // [rsp+20h] [rbp-70h]
  unsigned int v53; // [rsp+24h] [rbp-6Ch]
  unsigned int v54; // [rsp+24h] [rbp-6Ch]
  unsigned int v55; // [rsp+24h] [rbp-6Ch]
  unsigned int v56; // [rsp+28h] [rbp-68h]
  int v57; // [rsp+28h] [rbp-68h]
  __int64 *v58; // [rsp+30h] [rbp-60h]
  unsigned int v59; // [rsp+40h] [rbp-50h] BYREF
  const void **v60; // [rsp+48h] [rbp-48h]
  unsigned __int64 v61; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v62; // [rsp+58h] [rbp-38h]

  v15 = *(unsigned __int16 *)(a10 + 24);
  if ( v15 != 32 && v15 != 10 )
    return 0;
  if ( *(_WORD *)(a7 + 24) != 52 )
    return 0;
  v17 = *(const __m128i **)(a7 + 32);
  v18 = v17[2].m128i_i64[1];
  v19 = *(unsigned __int16 *)(v18 + 24);
  if ( v19 != 10 && v19 != 32 )
    return 0;
  v20 = (__m128)_mm_loadu_si128(v17);
  v21 = *(_QWORD *)(a10 + 88);
  v22 = *(_QWORD *)(v17->m128i_i64[0] + 40) + 16LL * v17->m128i_u32[2];
  v23 = *(_BYTE *)v22;
  v60 = *(const void ***)(v22 + 8);
  LODWORD(v22) = *(_DWORD *)(v21 + 32);
  LOBYTE(v59) = v23;
  v62 = v22;
  if ( (unsigned int)v22 > 0x40 )
  {
    v57 = a9;
    sub_16A4FD0((__int64)&v61, (const void **)(v21 + 24));
    a9 = v57;
  }
  else
  {
    v61 = *(_QWORD *)(v21 + 24);
  }
  switch ( a9 )
  {
    case 12:
      v56 = 17;
      break;
    case 13:
      sub_16A7490((__int64)&v61, 1);
      v56 = 17;
      break;
    case 10:
      sub_16A7490((__int64)&v61, 1);
      v56 = 22;
      break;
    case 11:
      v56 = 22;
      break;
    default:
      goto LABEL_14;
  }
  v25 = *(_QWORD *)(v18 + 88);
  v26 = sub_16A9900((__int64)&v61, (unsigned __int64 *)(v25 + 24));
  v24 = v62;
  if ( v26 <= 0 )
    goto LABEL_15;
  if ( v62 > 0x40 )
  {
    v51 = v62;
    v43 = sub_16A5940((__int64)&v61);
    v24 = v51;
    if ( v43 != 1 )
      goto LABEL_15;
  }
  else if ( !v61 || (v61 & (v61 - 1)) != 0 )
  {
    goto LABEL_15;
  }
  v27 = *(_DWORD *)(v25 + 32);
  if ( v27 > 0x40 )
  {
    v54 = v24;
    v44 = sub_16A5940(v25 + 24);
    v24 = v54;
    if ( v44 != 1 )
      goto LABEL_15;
  }
  else
  {
    v28 = *(_QWORD *)(v25 + 24);
    if ( !v28 || (v28 & (v28 - 1)) != 0 )
      goto LABEL_15;
  }
  v49 = v24 - 1;
  if ( v24 > 0x40 )
  {
    v52 = v24;
    v45 = sub_16A57B0((__int64)&v61);
    v24 = v52;
    v30 = v45;
    v31 = v49 - v45;
  }
  else if ( v61 )
  {
    _BitScanReverse64(&v29, v61);
    v30 = (v29 ^ 0x3F) + v24 - 64;
    v31 = -1 - ((v29 ^ 0x3F) - 64);
  }
  else
  {
    v30 = v24;
    v31 = -1;
  }
  if ( v27 > 0x40 )
  {
    v47 = v24;
    v48 = v31;
    v55 = v30;
    v46 = sub_16A57B0(v25 + 24);
    v24 = v47;
    v31 = v48;
    v30 = v55;
    v33 = v27 - v46;
  }
  else
  {
    v32 = *(_QWORD *)(v25 + 24);
    v33 = 0;
    if ( v32 )
    {
      _BitScanReverse64(&v32, v32);
      v33 = 64 - (v32 ^ 0x3F);
    }
  }
  result = 0;
  if ( v31 != v33 )
    goto LABEL_16;
  v34 = *(__int64 **)(a12 + 16);
  v35 = v34[2];
  v36 = *(__int64 (**)())(*(_QWORD *)v35 + 240LL);
  if ( v36 == sub_1F3CA40 )
    goto LABEL_15;
  v53 = v30;
  if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, const void **))v36)(v35, v59, v60) )
  {
    if ( (_BYTE)v59 )
      v37 = sub_1F3E310(&v59);
    else
      v37 = sub_1F58D40((__int64)&v59);
    *(_QWORD *)&v38 = sub_1D38BB0((__int64)v34, v53 - v49 + v37, a13, v59, v60, 0, (__m128i)v20, a2, a3, 0);
    v50 = v38;
    v39 = sub_1D332F0(
            v34,
            122,
            a13,
            v59,
            v60,
            0,
            *(double *)v20.m128_u64,
            a2,
            a3,
            v20.m128_i64[0],
            v20.m128_u64[1],
            v38);
    v41 = sub_1D332F0(v34, 123, a13, v59, v60, 0, *(double *)v20.m128_u64, a2, a3, (__int64)v39, v40, v50);
    result = sub_1F81070(v34, a13, a5, a6, (unsigned __int64)v41, v42, v20, a2, a3, *(_OWORD *)&v20, v56);
    v24 = v62;
    goto LABEL_16;
  }
LABEL_14:
  v24 = v62;
LABEL_15:
  result = 0;
LABEL_16:
  if ( v24 > 0x40 )
  {
    if ( v61 )
    {
      v58 = result;
      j_j___libc_free_0_0(v61);
      return v58;
    }
  }
  return result;
}
