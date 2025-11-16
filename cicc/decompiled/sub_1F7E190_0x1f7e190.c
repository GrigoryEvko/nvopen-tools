// Function: sub_1F7E190
// Address: 0x1f7e190
//
__int64 *__fastcall sub_1F7E190(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        double a6,
        double a7,
        __m128i a8)
{
  unsigned __int8 *v10; // rsi
  __int64 v11; // rax
  const void **v12; // rsi
  __int64 v13; // rdi
  __int64 v15; // rsi
  __int16 v16; // ax
  bool v17; // r11
  __int64 v18; // rt0
  __int64 v19; // rax
  __int64 v20; // rax
  const __m128i *v21; // rdi
  __int64 v22; // r9
  int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // rdx
  int v26; // eax
  __int64 v27; // r9
  _QWORD *v28; // rax
  unsigned __int64 v29; // rdx
  _QWORD *v30; // rax
  __m128i v31; // xmm0
  __int64 v32; // r14
  __int64 v33; // r9
  __int64 v34; // r15
  unsigned int v35; // eax
  __int64 *v36; // r15
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  unsigned __int64 v39; // r15
  __int64 *v40; // r14
  int v41; // eax
  __int64 *v42; // rbx
  __int64 v43; // rax
  const void **v44; // rdx
  __int128 v45; // rax
  __int64 v46; // rax
  __int64 *v47; // rdi
  __int64 v48; // rdx
  __int64 v49; // rdx
  _QWORD *v50; // rax
  __int64 v51; // rax
  __int64 *v52; // rdi
  __int64 v53; // rdx
  __int64 v54; // rdx
  _QWORD *v55; // rax
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rdi
  _QWORD *v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rcx
  _QWORD *v66; // rax
  __int128 v67; // [rsp-B8h] [rbp-B8h]
  int v68; // [rsp-A8h] [rbp-A8h]
  __int64 v69; // [rsp-A0h] [rbp-A0h]
  char v70; // [rsp-A0h] [rbp-A0h]
  char v71; // [rsp-99h] [rbp-99h]
  int v72; // [rsp-98h] [rbp-98h]
  char v73; // [rsp-98h] [rbp-98h]
  __int64 v74; // [rsp-98h] [rbp-98h]
  __int64 v75; // [rsp-90h] [rbp-90h]
  __int64 v76; // [rsp-90h] [rbp-90h]
  unsigned __int64 v77; // [rsp-88h] [rbp-88h]
  unsigned int v78; // [rsp-80h] [rbp-80h]
  unsigned __int32 v79; // [rsp-7Ch] [rbp-7Ch]
  unsigned __int64 v80; // [rsp-70h] [rbp-70h]
  unsigned int v81; // [rsp-58h] [rbp-58h] BYREF
  const void **v82; // [rsp-50h] [rbp-50h]
  __int64 v83; // [rsp-48h] [rbp-48h] BYREF
  int v84; // [rsp-40h] [rbp-40h]

  if ( !*(_BYTE *)(a1 + 24) )
    return 0;
  v10 = *(unsigned __int8 **)(a2 + 40);
  v11 = *v10;
  v12 = (const void **)*((_QWORD *)v10 + 1);
  LOBYTE(v81) = v11;
  v82 = v12;
  if ( (_BYTE)v11 != 6 && (unsigned __int8)(v11 - 4) > 1u )
    return 0;
  v13 = *(_QWORD *)(a1 + 8);
  if ( !*(_QWORD *)(v13 + 8 * v11 + 120) )
    return 0;
  v15 = 129LL * (unsigned __int8)v11;
  if ( (*(_BYTE *)(v13 + 259LL * (unsigned __int8)v11 + 2549) & 0xFB) != 0 )
    return 0;
  v16 = *(_WORD *)(a3 + 24);
  if ( v16 != 118 )
  {
    if ( *(_WORD *)(a4 + 24) != 118 )
    {
      v15 = 0;
      v17 = 0;
      goto LABEL_12;
    }
    goto LABEL_64;
  }
  if ( *(_WORD *)(**(_QWORD **)(a3 + 32) + 24LL) == 124 )
    goto LABEL_66;
  if ( *(_WORD *)(a4 + 24) == 118 )
  {
LABEL_64:
    if ( *(_WORD *)(**(_QWORD **)(a4 + 32) + 24LL) == 122 )
    {
LABEL_70:
      v57 = a3;
      a3 = a4;
      a4 = v57;
      goto LABEL_49;
    }
    v56 = a4;
    a4 = a3;
    a3 = v56;
LABEL_66:
    if ( *(_WORD *)(a4 + 24) != 118 )
    {
      v15 = 0;
      goto LABEL_57;
    }
    goto LABEL_70;
  }
LABEL_49:
  v46 = *(_QWORD *)(a3 + 48);
  if ( !v46 )
    return 0;
  if ( *(_QWORD *)(v46 + 32) )
    return 0;
  v47 = *(__int64 **)(a3 + 32);
  v48 = v47[5];
  LOBYTE(v15) = *(_WORD *)(v48 + 24) == 10 || *(_WORD *)(v48 + 24) == 32;
  if ( !(_BYTE)v15 )
    return 0;
  v49 = *(_QWORD *)(v48 + 88);
  v50 = *(_QWORD **)(v49 + 24);
  if ( *(_DWORD *)(v49 + 32) > 0x40u )
    v50 = (_QWORD *)*v50;
  v17 = v50 != (_QWORD *)65280 && v50 != (_QWORD *)0xFFFF;
  if ( v17 )
    return 0;
  a3 = *v47;
  if ( *(_WORD *)(a4 + 24) != 118 )
  {
    v16 = *(_WORD *)(a3 + 24);
    goto LABEL_12;
  }
  a3 = a4;
  a4 = *v47;
LABEL_57:
  v51 = *(_QWORD *)(a3 + 48);
  if ( !v51 )
    return 0;
  if ( *(_QWORD *)(v51 + 32) )
    return 0;
  v52 = *(__int64 **)(a3 + 32);
  v53 = v52[5];
  v17 = *(_WORD *)(v53 + 24) == 10 || *(_WORD *)(v53 + 24) == 32;
  if ( !v17 )
    return 0;
  v54 = *(_QWORD *)(v53 + 88);
  v55 = *(_QWORD **)(v54 + 24);
  if ( *(_DWORD *)(v54 + 32) > 0x40u )
    v55 = (_QWORD *)*v55;
  if ( v55 != (_QWORD *)255 )
    return 0;
  v16 = *(_WORD *)(a4 + 24);
  a3 = a4;
  a4 = *v52;
LABEL_12:
  if ( v16 == 124 )
  {
    if ( *(_WORD *)(a4 + 24) != 122 )
      return 0;
  }
  else
  {
    if ( v16 != 122 || *(_WORD *)(a4 + 24) != 124 )
      return 0;
    v18 = a4;
    a4 = a3;
    a3 = v18;
  }
  v19 = *(_QWORD *)(a4 + 48);
  if ( !v19 )
    return 0;
  if ( *(_QWORD *)(v19 + 32) )
    return 0;
  v20 = *(_QWORD *)(a3 + 48);
  if ( !v20 )
    return 0;
  if ( *(_QWORD *)(v20 + 32) )
    return 0;
  v21 = *(const __m128i **)(a4 + 32);
  v22 = v21[2].m128i_i64[1];
  v23 = *(unsigned __int16 *)(v22 + 24);
  if ( v23 != 32 && v23 != 10 )
    return 0;
  v24 = *(_QWORD *)(a3 + 32);
  v25 = *(_QWORD *)(v24 + 40);
  v26 = *(unsigned __int16 *)(v25 + 24);
  if ( v26 != 10 && v26 != 32 )
    return 0;
  v27 = *(_QWORD *)(v22 + 88);
  v28 = *(_QWORD **)(v27 + 24);
  if ( *(_DWORD *)(v27 + 32) > 0x40u )
    v28 = (_QWORD *)*v28;
  if ( v28 != (_QWORD *)8 )
    return 0;
  v29 = *(_QWORD *)(v25 + 88);
  v30 = *(_QWORD **)(v29 + 24);
  if ( *(_DWORD *)(v29 + 32) > 0x40u )
    v30 = (_QWORD *)*v30;
  if ( v30 != (_QWORD *)8 )
    return 0;
  v31 = _mm_loadu_si128(v21);
  v32 = v21->m128i_i64[0];
  v79 = v21->m128i_u32[2];
  v80 = v31.m128i_u64[1];
  if ( !(_BYTE)v15 && *(_WORD *)(v32 + 24) == 118 )
  {
    v58 = *(_QWORD *)(v32 + 48);
    if ( !v58 )
      return 0;
    if ( *(_QWORD *)(v58 + 32) )
      return 0;
    v59 = *(_QWORD *)(v32 + 32);
    v60 = *(_QWORD *)(v59 + 40);
    v15 = *(unsigned __int16 *)(v60 + 24);
    LOBYTE(v15) = (_DWORD)v15 == 32 || (_DWORD)v15 == 10;
    if ( !(_BYTE)v15 )
      return 0;
    v61 = *(_QWORD *)(v60 + 88);
    v62 = *(_QWORD **)(v61 + 24);
    if ( *(_DWORD *)(v61 + 32) > 0x40u )
      v62 = (_QWORD *)*v62;
    if ( v62 != (_QWORD *)255 )
      return 0;
    v32 = *(_QWORD *)v59;
    v29 = *(unsigned int *)(v59 + 8) | v31.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v80 = v29;
    v79 = *(_DWORD *)(v59 + 8);
  }
  v33 = *(_QWORD *)v24;
  v34 = *(unsigned int *)(v24 + 8);
  v77 = *(_QWORD *)(v24 + 8);
  if ( !v17 && *(_WORD *)(v33 + 24) == 118 )
  {
    v63 = *(_QWORD *)(v33 + 48);
    if ( !v63 )
      return 0;
    if ( *(_QWORD *)(v63 + 32) )
      return 0;
    v29 = *(_QWORD *)(v33 + 32);
    v64 = *(_QWORD *)(v29 + 40);
    v17 = *(_WORD *)(v64 + 24) == 32 || *(_WORD *)(v64 + 24) == 10;
    if ( !v17 )
      return 0;
    v65 = *(_QWORD *)(v64 + 88);
    v66 = *(_QWORD **)(v65 + 24);
    if ( *(_DWORD *)(v65 + 32) > 0x40u )
      v66 = (_QWORD *)*v66;
    if ( v66 != (_QWORD *)0xFFFF && v66 != (_QWORD *)65280 )
      return 0;
    v24 = *(unsigned int *)(v29 + 8);
    v33 = *(_QWORD *)v29;
    v34 = v24;
    v77 = v24 | v77 & 0xFFFFFFFF00000000LL;
  }
  if ( v32 != v33 || v79 != (_DWORD)v34 )
    return 0;
  LODWORD(v69) = a5;
  HIBYTE(v69) = v15;
  LOBYTE(v72) = v17;
  v35 = sub_1D159A0((char *)&v81, v15, v29, v24, a5, v33, v68, v69, v72, v33);
  v78 = v35;
  if ( v35 <= 0x10 || !v70 )
    goto LABEL_40;
  if ( !v71 )
    return 0;
  if ( !v73 )
  {
    v74 = v75;
    v76 = *(_QWORD *)a1;
    sub_171A350((__int64)&v83, v35, v35 - 16);
    if ( (unsigned __int8)sub_1D1F940(v76, v74, v34 | v77 & 0xFFFFFFFF00000000LL, (__int64)&v83, 0) )
    {
      sub_135E100(&v83);
      goto LABEL_40;
    }
    sub_135E100(&v83);
    return 0;
  }
LABEL_40:
  v36 = *(__int64 **)a1;
  v83 = *(_QWORD *)(a2 + 72);
  if ( v83 )
    sub_1F6CA20(&v83);
  v84 = *(_DWORD *)(a2 + 64);
  *((_QWORD *)&v67 + 1) = v79 | v80 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v67 = v32;
  v37 = sub_1D309E0(v36, 127, (__int64)&v83, v81, v82, 0, *(double *)v31.m128i_i64, a7, *(double *)a8.m128i_i64, v67);
  v39 = v38;
  v40 = (__int64 *)v37;
  sub_17CD270(&v83);
  if ( v78 > 0x10 )
  {
    v83 = *(_QWORD *)(a2 + 72);
    if ( v83 )
      sub_1F6CA20(&v83);
    v41 = *(_DWORD *)(a2 + 64);
    v42 = *(__int64 **)a1;
    v84 = v41;
    v43 = sub_1F6BF40(a1, v81, (__int64)v82);
    *(_QWORD *)&v45 = sub_1D38BB0((__int64)v42, v78 - 16, (__int64)&v83, v43, v44, 0, v31, a7, a8, 0);
    v40 = sub_1D332F0(v42, 124, (__int64)&v83, v81, v82, 0, *(double *)v31.m128i_i64, a7, a8, (__int64)v40, v39, v45);
    sub_17CD270(&v83);
  }
  return v40;
}
