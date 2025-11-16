// Function: sub_2141030
// Address: 0x2141030
//
__int64 *__fastcall sub_2141030(
        _QWORD *a1,
        __int64 a2,
        unsigned int a3,
        __m128 a4,
        double a5,
        __m128i a6,
        __int64 a7,
        __int64 a8,
        int a9)
{
  __int64 v11; // rax
  const __m128i *v12; // r14
  int v13; // esi
  unsigned __int64 v14; // rcx
  const __m128i *v15; // r15
  unsigned __int64 v16; // rdx
  __m128 *v17; // rax
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // rax
  _BYTE *v21; // r8
  int v22; // edx
  __int64 v23; // rdx
  __int64 *v24; // r12
  unsigned __int64 v26; // r8
  unsigned __int64 v27; // r10
  __int64 v28; // r11
  __int64 v29; // rsi
  unsigned __int8 *v30; // rax
  unsigned __int64 v31; // rcx
  unsigned int v32; // r15d
  __int64 v33; // rax
  unsigned int v34; // edx
  unsigned __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int128 v39; // rax
  int v40; // edx
  __int64 *v41; // r15
  int v42; // ecx
  __int64 v43; // rax
  int v44; // edx
  int v45; // esi
  __int64 v46; // [rsp+8h] [rbp-F8h]
  unsigned __int64 v47; // [rsp+10h] [rbp-F0h]
  __int64 v48; // [rsp+10h] [rbp-F0h]
  __int64 v49; // [rsp+18h] [rbp-E8h]
  unsigned int v50; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v51; // [rsp+20h] [rbp-E0h]
  unsigned int v52; // [rsp+20h] [rbp-E0h]
  int v53; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v54; // [rsp+28h] [rbp-D8h]
  __int64 *v55; // [rsp+28h] [rbp-D8h]
  int v56; // [rsp+28h] [rbp-D8h]
  int v57; // [rsp+38h] [rbp-C8h]
  __int64 v58; // [rsp+60h] [rbp-A0h] BYREF
  int v59; // [rsp+68h] [rbp-98h]
  _BYTE *v60; // [rsp+70h] [rbp-90h] BYREF
  __int64 v61; // [rsp+78h] [rbp-88h]
  _BYTE v62[128]; // [rsp+80h] [rbp-80h] BYREF

  v11 = *(unsigned int *)(a2 + 56);
  v12 = *(const __m128i **)(a2 + 32);
  v13 = 0;
  v60 = v62;
  v14 = 40 * v11;
  v61 = 0x500000000LL;
  v15 = (const __m128i *)((char *)v12 + 40 * v11);
  v16 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v11) >> 3);
  v17 = (__m128 *)v62;
  if ( v14 > 0xC8 )
  {
    v50 = a3;
    v53 = v16;
    sub_16CD150((__int64)&v60, v62, v16, 16, a3, a9);
    v13 = v61;
    a3 = v50;
    LODWORD(v16) = v53;
    v17 = (__m128 *)&v60[16 * (unsigned int)v61];
  }
  if ( v12 != v15 )
  {
    do
    {
      if ( v17 )
      {
        a4 = (__m128)_mm_loadu_si128(v12);
        *v17 = a4;
      }
      v12 = (const __m128i *)((char *)v12 + 40);
      ++v17;
    }
    while ( v15 != v12 );
    v13 = v61;
  }
  v18 = *(_QWORD *)(a2 + 32);
  LODWORD(v61) = v13 + v16;
  if ( a3 == 2 )
  {
    v43 = sub_200E230(
            a1,
            *(_QWORD *)(v18 + 80),
            *(_QWORD *)(v18 + 88),
            *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(v18 + 40) + 40LL) + 16LL * *(unsigned int *)(v18 + 48)),
            *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v18 + 40) + 40LL) + 16LL * *(unsigned int *)(v18 + 48) + 8),
            *(double *)a4.m128_u64,
            a5,
            *(double *)a6.m128i_i64);
    v45 = v44;
    v23 = (__int64)v60;
    *((_QWORD *)v60 + 4) = v43;
    *(_DWORD *)(v23 + 40) = v45;
  }
  else if ( a3 == 4 )
  {
    v26 = *(_QWORD *)(v18 + 160);
    v27 = v26;
    v28 = *(_QWORD *)(v18 + 168);
    v29 = *(_QWORD *)(v26 + 72);
    v30 = (unsigned __int8 *)(*(_QWORD *)(v26 + 40) + 16LL * *(unsigned int *)(v18 + 168));
    v31 = *((_QWORD *)v30 + 1);
    v32 = *v30;
    v58 = v29;
    v54 = v31;
    if ( v29 )
    {
      v47 = v26;
      v49 = v28;
      v51 = v26;
      sub_1623A60((__int64)&v58, v29, 2);
      v27 = v47;
      v28 = v49;
      v26 = v51;
    }
    v46 = v28;
    v59 = *(_DWORD *)(v26 + 64);
    v33 = sub_2138AD0((__int64)a1, v27, v28);
    v52 = v34;
    v35 = v54;
    v48 = v33;
    v55 = (__int64 *)a1[1];
    *(_QWORD *)&v39 = sub_1D2EF30(v55, v32, v35, v36, v37, v38);
    v41 = sub_1D332F0(
            v55,
            148,
            (__int64)&v58,
            *(unsigned __int8 *)(*(_QWORD *)(v48 + 40) + 16LL * v52),
            *(const void ***)(*(_QWORD *)(v48 + 40) + 16LL * v52 + 8),
            0,
            *(double *)a4.m128_u64,
            a5,
            a6,
            v48,
            v52 | v46 & 0xFFFFFFFF00000000LL,
            v39);
    v42 = v40;
    if ( v58 )
    {
      v56 = v40;
      sub_161E7C0((__int64)&v58, v58);
      v42 = v56;
    }
    v23 = (__int64)v60;
    *((_QWORD *)v60 + 8) = v41;
    *(_DWORD *)(v23 + 72) = v42;
  }
  else
  {
    v19 = a3;
    v20 = sub_2138AD0((__int64)a1, *(_QWORD *)(v18 + 40LL * a3), *(_QWORD *)(v18 + 40LL * a3 + 8));
    v21 = &v60[16 * v19];
    v57 = v22;
    *(_QWORD *)v21 = v20;
    v23 = (__int64)v60;
    *((_DWORD *)v21 + 2) = v57;
  }
  v24 = sub_1D2E160((_QWORD *)a1[1], (__int64 *)a2, v23, (unsigned int)v61);
  if ( v60 != v62 )
    _libc_free((unsigned __int64)v60);
  return v24;
}
