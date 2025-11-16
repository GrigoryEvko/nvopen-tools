// Function: sub_382B260
// Address: 0x382b260
//
__int64 __fastcall sub_382B260(__int64 a1, unsigned __int64 a2, unsigned __int32 a3, __m128i a4)
{
  unsigned __int64 v4; // rbx
  __int64 v5; // rax
  __int32 v6; // r14d
  __int64 v8; // r12
  const __m128i *v9; // rsi
  unsigned int *v10; // rcx
  __int64 v11; // r15
  int v12; // edx
  __int64 v14; // r11
  __int64 v15; // r9
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 *v19; // rdi
  __m128i *v20; // rdx
  unsigned int v21; // r14d
  __int64 v22; // r12
  __int64 i; // rbx
  __int64 v24; // r11
  __int64 v25; // rsi
  __int16 v26; // ax
  __int64 v27; // rdi
  unsigned int v28; // edx
  int v29; // eax
  unsigned __int64 v30; // rax
  __int64 v31; // rdi
  unsigned __int8 *v32; // r8
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r9
  __int64 v36; // r11
  unsigned __int64 v37; // rdx
  unsigned __int8 **v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // rsi
  __int64 v42; // rcx
  unsigned __int8 *v43; // r8
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r9
  __int64 v47; // r11
  unsigned __int64 v48; // rdx
  unsigned __int8 **v49; // rax
  unsigned __int32 v50; // r14d
  _BYTE **v51; // rdi
  __int64 v52; // r15
  __int64 v53; // rax
  __m128i v54; // xmm0
  unsigned int v55; // r15d
  unsigned __int8 *v56; // r14
  __int64 v57; // rdx
  __int128 v58; // [rsp-10h] [rbp-E0h]
  __m128i v59; // [rsp+0h] [rbp-D0h] BYREF
  __int64 *v60; // [rsp+10h] [rbp-C0h]
  __int64 v61; // [rsp+18h] [rbp-B8h]
  __m128i v62; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v63; // [rsp+30h] [rbp-A0h]
  __int64 v64; // [rsp+38h] [rbp-98h]
  __int64 *v65; // [rsp+40h] [rbp-90h]
  __m128i *v66; // [rsp+48h] [rbp-88h]
  __int64 v67; // [rsp+50h] [rbp-80h] BYREF
  int v68; // [rsp+58h] [rbp-78h]
  __m128i *v69; // [rsp+60h] [rbp-70h] BYREF
  __int64 v70; // [rsp+68h] [rbp-68h]
  _BYTE v71[96]; // [rsp+70h] [rbp-60h] BYREF

  v5 = a3;
  v6 = a3;
  v8 = a1;
  v9 = *(const __m128i **)(a2 + 40);
  v10 = (unsigned int *)v9 + 10 * a3;
  v11 = *(_QWORD *)v10;
  v12 = *(_DWORD *)(*(_QWORD *)v10 + 24LL);
  if ( v12 != 35 && v12 != 11 )
    return 0;
  v14 = v10[2];
  v66 = (__m128i *)v71;
  v69 = (__m128i *)v71;
  v70 = 0x300000000LL;
  if ( v6 )
  {
    v15 = 5 * v5;
    v16 = 40;
    v17 = 0;
    v63 = a1;
    v18 = 8 * v15;
    v62.m128i_i32[0] = v6;
    a4 = _mm_loadu_si128(v9);
    v19 = (__int64 *)&v69;
    v60 = (__int64 *)v4;
    v20 = v66;
    v21 = v14;
    v22 = v18;
    for ( i = 40; ; i += 40 )
    {
      v20[v17] = a4;
      v17 = (unsigned int)(v70 + 1);
      LODWORD(v70) = v70 + 1;
      if ( v22 == i )
        break;
      a4 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + i));
      if ( v17 + 1 > (unsigned __int64)HIDWORD(v70) )
      {
        v65 = v19;
        v59 = a4;
        sub_C8D5F0((__int64)v19, v66, v17 + 1, 0x10u, v16, v18);
        v17 = (unsigned int)v70;
        a4 = _mm_load_si128(&v59);
        v19 = v65;
      }
      v20 = v69;
    }
    v14 = v21;
    v8 = v63;
    v6 = v62.m128i_i32[0];
    v4 = (unsigned __int64)v60;
  }
  v24 = *(_QWORD *)(v11 + 48) + 16 * v14;
  v25 = *(_QWORD *)(a2 + 80);
  v26 = *(_WORD *)v24;
  v67 = v25;
  LOWORD(v63) = v26;
  v62.m128i_i64[0] = *(_QWORD *)(v24 + 8);
  if ( v25 )
    sub_B96E90((__int64)&v67, v25, 1);
  v27 = *(_QWORD *)(v11 + 96);
  v28 = *(_DWORD *)(v27 + 32);
  v68 = *(_DWORD *)(a2 + 72);
  if ( v28 <= 0x40 )
  {
    v30 = *(_QWORD *)(v27 + 24);
    if ( v30 )
    {
      _BitScanReverse64(&v30, v30);
      if ( 64 - ((unsigned int)v30 ^ 0x3F) > 0x3F )
        goto LABEL_15;
    }
LABEL_21:
    v31 = *(_QWORD *)(v8 + 8);
    v60 = &v67;
    v32 = sub_3400BD0(v31, 2, (__int64)&v67, 8, 0, 1u, a4, 0);
    v33 = (unsigned int)v70;
    v35 = v34;
    v36 = (__int64)v60;
    v37 = (unsigned int)v70 + 1LL;
    if ( v37 > HIDWORD(v70) )
    {
      v65 = v60;
      v60 = (__int64 *)v32;
      v61 = v35;
      sub_C8D5F0((__int64)&v69, v66, v37, 0x10u, (__int64)v32, v35);
      v33 = (unsigned int)v70;
      v36 = (__int64)v65;
      v32 = (unsigned __int8 *)v60;
      v35 = v61;
    }
    v38 = (unsigned __int8 **)&v69[v33];
    *v38 = v32;
    v38[1] = (unsigned __int8 *)v35;
    v39 = *(_QWORD *)(v11 + 96);
    LODWORD(v70) = v70 + 1;
    v40 = *(_QWORD *)(v8 + 8);
    if ( *(_DWORD *)(v39 + 32) <= 0x40u )
      v41 = *(_QWORD *)(v39 + 24);
    else
      v41 = **(_QWORD **)(v39 + 24);
    v42 = (unsigned __int16)v63;
    v63 = v36;
    v43 = sub_3400BD0(v40, v41, v36, v42, v62.m128i_i64[0], 1u, a4, 0);
    v44 = (unsigned int)v70;
    v46 = v45;
    v47 = v63;
    v48 = (unsigned int)v70 + 1LL;
    if ( v48 > HIDWORD(v70) )
    {
      v62.m128i_i64[0] = v63;
      v63 = (__int64)v43;
      v64 = v46;
      sub_C8D5F0((__int64)&v69, v66, v48, 0x10u, (__int64)v43, v46);
      v44 = (unsigned int)v70;
      v47 = v62.m128i_i64[0];
      v43 = (unsigned __int8 *)v63;
      v46 = v64;
    }
    v49 = (unsigned __int8 **)&v69[v44];
    v50 = v6 + 1;
    *v49 = v43;
    v51 = (_BYTE **)&v69;
    v52 = v47;
    v49[1] = (unsigned __int8 *)v46;
    v53 = (unsigned int)(v70 + 1);
    LODWORD(v70) = v70 + 1;
    if ( v50 < *(_DWORD *)(a2 + 64) )
    {
      do
      {
        v54 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL * v50));
        if ( v53 + 1 > (unsigned __int64)HIDWORD(v70) )
        {
          v63 = (__int64)v51;
          v62 = v54;
          sub_C8D5F0((__int64)v51, v66, v53 + 1, 0x10u, (__int64)v43, v46);
          v53 = (unsigned int)v70;
          v54 = _mm_load_si128(&v62);
          v51 = (_BYTE **)v63;
        }
        ++v50;
        v69[v53] = v54;
        v53 = (unsigned int)(v70 + 1);
        LODWORD(v70) = v70 + 1;
      }
      while ( v50 < *(_DWORD *)(a2 + 64) );
      v47 = v52;
    }
    *((_QWORD *)&v58 + 1) = (unsigned int)v53;
    v55 = 0;
    *(_QWORD *)&v58 = v69;
    v56 = sub_3411630(
            *(_QWORD **)(v8 + 8),
            *(unsigned int *)(a2 + 24),
            v47,
            *(unsigned int **)(a2 + 48),
            *(unsigned int *)(a2 + 68),
            *(_QWORD *)(v8 + 8),
            v58);
    while ( v55 < *(_DWORD *)(a2 + 68) )
    {
      v57 = v55++;
      v4 = v57 | v4 & 0xFFFFFFFF00000000LL;
      sub_3760E70(v8, a2, v57, (unsigned __int64)v56, v4);
    }
    goto LABEL_15;
  }
  LODWORD(v60) = v28;
  v29 = sub_C444A0(v27 + 24);
  if ( (unsigned int)((_DWORD)v60 - v29) <= 0x3F )
    goto LABEL_21;
LABEL_15:
  if ( v67 )
    sub_B91220((__int64)&v67, v67);
  if ( v69 != v66 )
    _libc_free((unsigned __int64)v69);
  return 0;
}
