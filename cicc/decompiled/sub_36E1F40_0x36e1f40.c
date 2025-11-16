// Function: sub_36E1F40
// Address: 0x36e1f40
//
__int64 __fastcall sub_36E1F40(__int64 a1, __int64 a2, __m128i a3)
{
  unsigned __int16 v5; // r13
  unsigned __int64 v6; // r15
  __int64 v7; // rsi
  int v8; // eax
  const __m128i *v9; // rdx
  __m128i v10; // xmm3
  unsigned __int16 v11; // dx
  __int64 v12; // r15
  __int64 v13; // rax
  char v14; // si
  __int64 v15; // rax
  unsigned int v16; // eax
  __int64 v17; // r11
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  unsigned __int8 *v20; // rax
  __int64 v21; // rdi
  int v22; // edx
  unsigned __int8 *v23; // rax
  __int64 v24; // rdi
  int v25; // edx
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
  __m128i v36; // xmm0
  __m128i v37; // xmm1
  _WORD *v38; // rax
  __m128i v39; // xmm2
  int v40; // edx
  __int64 v41; // r9
  unsigned __int16 v42; // r11
  __int64 v43; // r14
  _QWORD *v44; // rdi
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  const __m128i *v49; // rdx
  __int128 v50; // [rsp+0h] [rbp-150h]
  unsigned __int16 v51; // [rsp+18h] [rbp-138h]
  unsigned int v52; // [rsp+1Ch] [rbp-134h]
  unsigned __int64 v53; // [rsp+20h] [rbp-130h]
  __int64 v54; // [rsp+28h] [rbp-128h]
  __int64 v55; // [rsp+28h] [rbp-128h]
  unsigned __int64 v56; // [rsp+38h] [rbp-118h]
  __int64 v57; // [rsp+48h] [rbp-108h] BYREF
  __int64 v58; // [rsp+50h] [rbp-100h] BYREF
  int v59; // [rsp+58h] [rbp-F8h]
  __m128i v60; // [rsp+60h] [rbp-F0h] BYREF
  __m128i v61; // [rsp+70h] [rbp-E0h] BYREF
  __m128i v62; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v63; // [rsp+90h] [rbp-C0h] BYREF
  int v64; // [rsp+98h] [rbp-B8h]
  unsigned __int8 *v65; // [rsp+A0h] [rbp-B0h]
  int v66; // [rsp+A8h] [rbp-A8h]
  unsigned __int8 *v67; // [rsp+B0h] [rbp-A0h]
  int v68; // [rsp+B8h] [rbp-98h]
  unsigned __int8 *v69; // [rsp+C0h] [rbp-90h]
  int v70; // [rsp+C8h] [rbp-88h]
  unsigned __int8 *v71; // [rsp+D0h] [rbp-80h]
  int v72; // [rsp+D8h] [rbp-78h]
  unsigned __int8 *v73; // [rsp+E0h] [rbp-70h]
  int v74; // [rsp+E8h] [rbp-68h]
  __m128i v75; // [rsp+F0h] [rbp-60h]
  __m128i v76; // [rsp+100h] [rbp-50h]
  __m128i v77; // [rsp+110h] [rbp-40h]

  if ( *(_DWORD *)(a2 + 24) == 298 )
  {
    if ( (*(_WORD *)(a2 + 32) & 0x380) != 0 )
    {
      LODWORD(v6) = 0;
      return (unsigned int)v6;
    }
    v54 = a2;
  }
  else
  {
    v54 = 0;
  }
  v5 = *(_WORD *)(a2 + 96);
  LODWORD(v6) = 0;
  if ( !v5 )
    return (unsigned int)v6;
  v52 = sub_36D7800(*(_QWORD *)(a2 + 112));
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL) <= 0x1Fu
    || v52 != 1
    || (*(_BYTE *)(a2 + 32) & 0x40) == 0 && !(unsigned __int8)sub_36D7A70(a2, **(_QWORD **)(a1 + 40)) )
  {
    v7 = *(_QWORD *)(a2 + 80);
    v58 = v7;
    if ( v7 )
    {
      sub_B96E90((__int64)&v58, v7, 1);
      v8 = *(_DWORD *)(a2 + 72);
      v9 = *(const __m128i **)(a2 + 40);
      v59 = v8;
      v10 = _mm_loadu_si128(v9);
      v63 = v58;
      v60 = v10;
      if ( v58 )
      {
        sub_B96E90((__int64)&v63, v58, 1);
        v8 = v59;
      }
    }
    else
    {
      v8 = *(_DWORD *)(a2 + 72);
      v49 = *(const __m128i **)(a2 + 40);
      v59 = v8;
      v63 = 0;
      v60 = _mm_loadu_si128(v49);
    }
    v64 = v8;
    v53 = sub_36E1BC0(a1, (__int64)&v63, (__int64)&v60, a2);
    if ( v63 )
      sub_B91220((__int64)&v63, v63);
    v11 = v5 - 17;
    if ( (unsigned __int16)(v5 - 17) <= 0xD3u )
      v5 = word_4456580[v5 - 1];
    if ( v5 <= 1u || (unsigned __int16)(v5 - 504) <= 7u )
      BUG();
    v51 = v11;
    v12 = 32;
    v13 = 16LL * (v5 - 1);
    v14 = byte_444C4A0[v13 + 8];
    v15 = *(_QWORD *)&byte_444C4A0[v13];
    LOBYTE(v64) = v14;
    v63 = v15;
    v16 = sub_CA1930(&v63);
    if ( v51 > 0xD3u )
    {
      if ( v16 < 8 )
        v16 = 8;
      v12 = v16;
    }
    if ( !v54 || (v17 = 1, ((*(_BYTE *)(v54 + 33) >> 2) & 3) != 2) )
      v17 = (unsigned int)sub_36D79E0(v5, 0);
    v18 = *(_QWORD *)(a2 + 40);
    v61.m128i_i32[2] = 0;
    v55 = v17;
    v62.m128i_i32[2] = 0;
    v19 = *(_QWORD *)(v18 + 48);
    v61.m128i_i64[0] = 0;
    v62.m128i_i64[0] = 0;
    sub_36DF750(a1, *(_QWORD *)(v18 + 40), v19, (__int64)&v62, (__int64)&v61, a3);
    v20 = sub_3400BD0(*(_QWORD *)(a1 + 64), (unsigned int)v53, (__int64)&v58, 7, 0, 1u, a3, 0);
    v21 = *(_QWORD *)(a1 + 64);
    v64 = v22;
    v63 = (__int64)v20;
    v23 = sub_3400BD0(v21, HIDWORD(v53), (__int64)&v58, 7, 0, 1u, a3, 0);
    v24 = *(_QWORD *)(a1 + 64);
    v66 = v25;
    v65 = v23;
    v26 = sub_3400BD0(v24, v52, (__int64)&v58, 7, 0, 1u, a3, 0);
    v27 = *(_QWORD *)(a1 + 64);
    v68 = v28;
    v67 = v26;
    v29 = sub_3400BD0(v27, 1, (__int64)&v58, 7, 0, 1u, a3, 0);
    v30 = *(_QWORD *)(a1 + 64);
    v70 = v31;
    v69 = v29;
    v32 = sub_3400BD0(v30, v55, (__int64)&v58, 7, 0, 1u, a3, 0);
    v33 = *(_QWORD *)(a1 + 64);
    v72 = v34;
    v71 = v32;
    v35 = sub_3400BD0(v33, v12, (__int64)&v58, 7, 0, 1u, a3, 0);
    v36 = _mm_loadu_si128(&v62);
    v37 = _mm_loadu_si128(&v61);
    v73 = v35;
    v38 = *(_WORD **)(a2 + 48);
    v39 = _mm_loadu_si128(&v60);
    v74 = v40;
    v75 = v36;
    v76 = v37;
    v77 = v39;
    LOWORD(v33) = *v38;
    v57 = 0x100000A72LL;
    v56 = sub_36D6650(v33, 2678, 2675, 2676, 0x100000A75LL, 2673, 0x100000A72LL);
    v6 = HIDWORD(v56);
    if ( BYTE4(v56)
      && (*((_QWORD *)&v50 + 1) = 9,
          *(_QWORD *)&v50 = &v63,
          (v43 = sub_33E6B00(*(__int64 **)(a1 + 64), v56, (__int64)&v58, v42, 0, v41, 1u, v50)) != 0) )
    {
      v44 = *(_QWORD **)(a1 + 64);
      v57 = *(_QWORD *)(a2 + 112);
      sub_33E4DA0(v44, v43, &v57, 1);
      sub_34158F0(*(_QWORD *)(a1 + 64), a2, v43, v45, v46, v47);
      sub_3421DB0(v43);
      sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
    }
    else
    {
      LODWORD(v6) = 0;
    }
    if ( v58 )
      sub_B91220((__int64)&v58, v58);
    return (unsigned int)v6;
  }
  return sub_36E0E00(a1, a2, a3);
}
