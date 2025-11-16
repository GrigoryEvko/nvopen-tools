// Function: sub_36E2B10
// Address: 0x36e2b10
//
__int64 __fastcall sub_36E2B10(__int64 a1, __int64 a2, __m128i a3)
{
  int v5; // eax
  unsigned __int16 v6; // r12
  unsigned __int64 v7; // r13
  unsigned int v8; // eax
  __int64 v9; // rsi
  int v10; // eax
  const __m128i *v11; // rdx
  __m128i v12; // xmm3
  unsigned __int16 v13; // r13
  __int64 v14; // rax
  char v15; // si
  __int64 v16; // rax
  unsigned int v17; // eax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // r13
  int v21; // r12d
  int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdi
  unsigned __int8 *v26; // rax
  __int64 v27; // rdi
  int v28; // edx
  unsigned __int8 *v29; // rax
  __int64 v30; // rdi
  int v31; // edx
  unsigned __int8 *v32; // rax
  __int64 v33; // rdi
  int v34; // edx
  unsigned __int8 *v35; // rax
  __int64 v36; // rdi
  int v37; // edx
  unsigned __int8 *v38; // rax
  __int64 v39; // rdi
  int v40; // edx
  unsigned __int8 *v41; // rax
  __m128i v42; // xmm0
  __m128i v43; // xmm1
  __m128i v44; // xmm2
  int v45; // edx
  unsigned __int64 v46; // rax
  __int64 v47; // r9
  __int64 v48; // r14
  _QWORD *v49; // rdi
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  const __m128i *v54; // rdx
  __int64 v55; // rdx
  __int128 v56; // [rsp+0h] [rbp-170h]
  __int64 v57; // [rsp+18h] [rbp-158h]
  unsigned int v58; // [rsp+20h] [rbp-150h]
  unsigned int v59; // [rsp+24h] [rbp-14Ch]
  unsigned __int64 v60; // [rsp+28h] [rbp-148h]
  __int64 v61; // [rsp+30h] [rbp-140h]
  __int64 v62; // [rsp+38h] [rbp-138h]
  __int64 v63; // [rsp+58h] [rbp-118h] BYREF
  __int64 v64; // [rsp+60h] [rbp-110h] BYREF
  int v65; // [rsp+68h] [rbp-108h]
  __m128i v66; // [rsp+70h] [rbp-100h] BYREF
  __m128i v67; // [rsp+80h] [rbp-F0h] BYREF
  __m128i v68; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v69; // [rsp+A0h] [rbp-D0h] BYREF
  int v70; // [rsp+A8h] [rbp-C8h]
  unsigned __int8 *v71; // [rsp+B0h] [rbp-C0h]
  int v72; // [rsp+B8h] [rbp-B8h]
  unsigned __int8 *v73; // [rsp+C0h] [rbp-B0h]
  int v74; // [rsp+C8h] [rbp-A8h]
  unsigned __int8 *v75; // [rsp+D0h] [rbp-A0h]
  int v76; // [rsp+D8h] [rbp-98h]
  unsigned __int8 *v77; // [rsp+E0h] [rbp-90h]
  int v78; // [rsp+E8h] [rbp-88h]
  unsigned __int8 *v79; // [rsp+F0h] [rbp-80h]
  int v80; // [rsp+F8h] [rbp-78h]
  unsigned __int8 *v81; // [rsp+100h] [rbp-70h]
  int v82; // [rsp+108h] [rbp-68h]
  __m128i v83; // [rsp+110h] [rbp-60h]
  __m128i v84; // [rsp+120h] [rbp-50h]
  __m128i v85; // [rsp+130h] [rbp-40h]

  v5 = *(_DWORD *)(a2 + 24);
  if ( v5 == 299 )
  {
    if ( (*(_WORD *)(a2 + 32) & 0x380) != 0 )
    {
      LODWORD(v7) = 0;
      return (unsigned int)v7;
    }
    v62 = a2;
    goto LABEL_4;
  }
  v61 = a2;
  v62 = 0;
  if ( v5 != 340 )
  {
    if ( (unsigned int)(v5 - 341) > 0x14 && (unsigned int)(v5 - 338) > 1 )
    {
LABEL_4:
      v61 = 0;
      goto LABEL_5;
    }
    v61 = a2;
    v62 = 0;
  }
LABEL_5:
  v6 = *(_WORD *)(a2 + 96);
  LODWORD(v7) = 0;
  if ( !v6 )
    return (unsigned int)v7;
  v8 = sub_36D7800(*(_QWORD *)(a2 + 112));
  v9 = *(_QWORD *)(a2 + 80);
  v59 = v8;
  v64 = v9;
  if ( v9 )
  {
    sub_B96E90((__int64)&v64, v9, 1);
    v10 = *(_DWORD *)(a2 + 72);
    v11 = *(const __m128i **)(a2 + 40);
    v65 = v10;
    v12 = _mm_loadu_si128(v11);
    v69 = v64;
    v66 = v12;
    if ( v64 )
    {
      sub_B96E90((__int64)&v69, v64, 1);
      v10 = v65;
    }
  }
  else
  {
    v10 = *(_DWORD *)(a2 + 72);
    v54 = *(const __m128i **)(a2 + 40);
    v65 = v10;
    v69 = 0;
    v66 = _mm_loadu_si128(v54);
  }
  v70 = v10;
  v60 = sub_36E1BC0(a1, (__int64)&v69, (__int64)&v66, a2);
  if ( v69 )
    sub_B91220((__int64)&v69, v69);
  v13 = v6 - 17;
  if ( (unsigned __int16)(v6 - 17) <= 0xD3u )
    v6 = word_4456580[v6 - 1];
  if ( v6 <= 1u || (unsigned __int16)(v6 - 504) <= 7u )
    BUG();
  v14 = 16LL * (v6 - 1);
  v15 = byte_444C4A0[v14 + 8];
  v16 = *(_QWORD *)&byte_444C4A0[v14];
  LOBYTE(v70) = v15;
  v69 = v16;
  v17 = sub_CA1930(&v69);
  v18 = 32;
  if ( v13 > 0xD3u )
    v18 = v17;
  v57 = v18;
  v58 = sub_36D79E0(v6, 0);
  if ( v62 )
  {
    v19 = *(_QWORD *)(v62 + 40) + 40LL;
  }
  else
  {
    v55 = *(_QWORD *)(v61 + 40);
    v19 = v55 + 80;
    if ( *(_DWORD *)(v61 + 24) == 339 )
      v19 = v55 + 40;
  }
  v20 = *(_QWORD *)v19;
  v21 = *(_DWORD *)(v19 + 8);
  v67.m128i_i64[0] = 0;
  v22 = *(_DWORD *)(a2 + 24);
  v23 = *(_QWORD *)(a2 + 40);
  v67.m128i_i32[2] = 0;
  v68.m128i_i64[0] = 0;
  v68.m128i_i32[2] = 0;
  if ( v22 <= 365 )
  {
    if ( v22 <= 363 )
    {
      if ( v22 != 339 && (v22 & 0xFFFFFFBF) != 0x12B )
        goto LABEL_23;
      goto LABEL_40;
    }
LABEL_38:
    v24 = v23 + 120;
    goto LABEL_24;
  }
  if ( v22 > 467 )
  {
    if ( v22 == 497 )
      goto LABEL_38;
LABEL_23:
    v24 = v23 + 40;
    goto LABEL_24;
  }
  if ( v22 <= 464 )
    goto LABEL_23;
LABEL_40:
  v24 = v23 + 80;
LABEL_24:
  sub_36DF750(a1, *(_QWORD *)v24, *(_QWORD *)(v24 + 8), (__int64)&v68, (__int64)&v67, a3);
  v25 = *(_QWORD *)(a1 + 64);
  v70 = v21;
  v69 = v20;
  v26 = sub_3400BD0(v25, (unsigned int)v60, (__int64)&v64, 7, 0, 1u, a3, 0);
  v27 = *(_QWORD *)(a1 + 64);
  v72 = v28;
  v71 = v26;
  v29 = sub_3400BD0(v27, HIDWORD(v60), (__int64)&v64, 7, 0, 1u, a3, 0);
  v30 = *(_QWORD *)(a1 + 64);
  v74 = v31;
  v73 = v29;
  v32 = sub_3400BD0(v30, v59, (__int64)&v64, 7, 0, 1u, a3, 0);
  v33 = *(_QWORD *)(a1 + 64);
  v76 = v34;
  v75 = v32;
  v35 = sub_3400BD0(v33, 1, (__int64)&v64, 7, 0, 1u, a3, 0);
  v36 = *(_QWORD *)(a1 + 64);
  v78 = v37;
  v77 = v35;
  v38 = sub_3400BD0(v36, v58, (__int64)&v64, 7, 0, 1u, a3, 0);
  v39 = *(_QWORD *)(a1 + 64);
  v80 = v40;
  v79 = v38;
  v41 = sub_3400BD0(v39, v57, (__int64)&v64, 7, 0, 1u, a3, 0);
  v42 = _mm_loadu_si128(&v68);
  v43 = _mm_loadu_si128(&v67);
  v44 = _mm_loadu_si128(&v66);
  v82 = v45;
  v81 = v41;
  v83 = v42;
  v84 = v43;
  v85 = v44;
  LOWORD(v39) = **(_WORD **)(v20 + 48);
  v63 = 0x100000E80LL;
  v46 = sub_36D6650(v39, 3716, 3713, 3714, 0x100000E83LL, 3711, 0x100000E80LL);
  v7 = HIDWORD(v46);
  if ( BYTE4(v46)
    && (*((_QWORD *)&v56 + 1) = 10,
        *(_QWORD *)&v56 = &v69,
        (v48 = sub_33F7800(*(_QWORD **)(a1 + 64), v46, (__int64)&v64, 1u, 0, v47, v56)) != 0) )
  {
    v49 = *(_QWORD **)(a1 + 64);
    v63 = *(_QWORD *)(a2 + 112);
    sub_33E4DA0(v49, v48, &v63, 1);
    sub_34158F0(*(_QWORD *)(a1 + 64), a2, v48, v50, v51, v52);
    sub_3421DB0(v48);
    sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  }
  else
  {
    LODWORD(v7) = 0;
  }
  if ( v64 )
    sub_B91220((__int64)&v64, v64);
  return (unsigned int)v7;
}
