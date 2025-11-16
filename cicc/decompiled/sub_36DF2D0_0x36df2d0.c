// Function: sub_36DF2D0
// Address: 0x36df2d0
//
__int64 __fastcall sub_36DF2D0(__int64 a1, __int64 a2)
{
  unsigned __int16 *v3; // rdx
  int v4; // eax
  __int64 v5; // rdx
  bool v6; // al
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r9
  __int32 v10; // esi
  __int64 v11; // rcx
  __int32 v12; // r10d
  int v13; // eax
  __m128i v14; // rax
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  __m128i v18; // xmm1
  __m128i v19; // xmm2
  __m128i *v20; // rax
  unsigned int v21; // eax
  __int64 v22; // r15
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 result; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __m128i v30; // rax
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rax
  __m128i v34; // xmm2
  __m128i v35; // xmm3
  __m128i *v36; // rax
  __m128i v37; // rax
  __int64 v38; // r8
  __int64 v39; // rax
  __m128i v40; // xmm5
  __m128i v41; // xmm6
  __m128i *v42; // rax
  __int128 v43; // [rsp-10h] [rbp-120h]
  __int64 v44; // [rsp+10h] [rbp-100h]
  __int32 v45; // [rsp+18h] [rbp-F8h]
  __int64 v46; // [rsp+18h] [rbp-F8h]
  bool v47; // [rsp+2Fh] [rbp-E1h] BYREF
  unsigned int v48; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v49; // [rsp+38h] [rbp-D8h]
  __int64 v50; // [rsp+40h] [rbp-D0h] BYREF
  int v51; // [rsp+48h] [rbp-C8h]
  _QWORD v52[4]; // [rsp+50h] [rbp-C0h] BYREF
  __m128i v53; // [rsp+70h] [rbp-A0h] BYREF
  __m128i v54; // [rsp+80h] [rbp-90h] BYREF
  __m128i v55; // [rsp+90h] [rbp-80h] BYREF
  _BYTE *v56; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v57; // [rsp+A8h] [rbp-68h]
  _BYTE v58[96]; // [rsp+B0h] [rbp-60h] BYREF

  v3 = *(unsigned __int16 **)(a2 + 48);
  v4 = *v3;
  v5 = *((_QWORD *)v3 + 1);
  LOWORD(v48) = v4;
  v49 = v5;
  if ( (_WORD)v4 )
  {
    if ( (unsigned __int16)(v4 - 17) <= 0xD3u )
    {
      if ( word_4456580[v4 - 1] != 10 )
        return 0;
    }
    else if ( (_WORD)v4 != 10 )
    {
      return 0;
    }
  }
  else if ( !sub_30070B0((__int64)&v48) || (unsigned __int16)sub_3009970((__int64)&v48, a2, v27, v28, v29) != 10 )
  {
    return 0;
  }
  if ( sub_305B520((_DWORD *)(*(_QWORD *)(a1 + 952) + 1288LL), *(_DWORD *)(a2 + 24)) )
    return 0;
  if ( (_WORD)v48 )
    v6 = (unsigned __int16)(v48 - 17) <= 0xD3u;
  else
    v6 = sub_30070B0((__int64)&v48);
  v7 = *(_QWORD *)(a2 + 80);
  v47 = v6;
  v50 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v50, v7, 1);
  v51 = *(_DWORD *)(a2 + 72);
  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_QWORD *)v8;
  v10 = *(_DWORD *)(v8 + 8);
  v56 = v58;
  v11 = *(_QWORD *)(v8 + 40);
  v12 = *(_DWORD *)(v8 + 48);
  v52[1] = a1;
  v57 = 0x300000000LL;
  v52[0] = &v47;
  v13 = *(_DWORD *)(a2 + 24);
  v52[2] = &v50;
  v52[3] = &v48;
  if ( v13 == 97 )
  {
    v46 = v9;
    v53.m128i_i64[0] = v11;
    v53.m128i_i32[2] = v12;
    v37.m128i_i64[0] = sub_36D8B00((__int64)v52, -1.0);
    LODWORD(v57) = 0;
    v54 = v37;
    v39 = 0;
    v55.m128i_i64[0] = v46;
    v55.m128i_i32[2] = v10;
    if ( HIDWORD(v57) <= 2 )
    {
      sub_C8D5F0((__int64)&v56, v58, 3u, 0x10u, v38, v46);
      v39 = 16LL * (unsigned int)v57;
    }
    v40 = _mm_loadu_si128(&v54);
    v41 = _mm_loadu_si128(&v55);
    v42 = (__m128i *)&v56[v39];
    *v42 = _mm_loadu_si128(&v53);
    v42[1] = v40;
    v42[2] = v41;
    v21 = v57 + 3;
    LODWORD(v57) = v57 + 3;
  }
  else if ( v13 == 98 )
  {
    v53.m128i_i64[0] = v9;
    v53.m128i_i32[2] = v10;
    v54.m128i_i64[0] = v11;
    v54.m128i_i32[2] = v12;
    v14.m128i_i64[0] = sub_36D8B00((__int64)v52, -0.0);
    LODWORD(v57) = 0;
    v55 = v14;
    v17 = 0;
    if ( HIDWORD(v57) <= 2 )
    {
      sub_C8D5F0((__int64)&v56, v58, 3u, 0x10u, v15, v16);
      v17 = 16LL * (unsigned int)v57;
    }
    v18 = _mm_loadu_si128(&v54);
    v19 = _mm_loadu_si128(&v55);
    v20 = (__m128i *)&v56[v17];
    *v20 = _mm_loadu_si128(&v53);
    v20[1] = v18;
    v20[2] = v19;
    v21 = v57 + 3;
    LODWORD(v57) = v57 + 3;
  }
  else
  {
    v44 = v11;
    v45 = v12;
    if ( v13 != 96 )
      BUG();
    v53.m128i_i64[0] = v9;
    v53.m128i_i32[2] = v10;
    v30.m128i_i64[0] = sub_36D8B00((__int64)v52, 1.0);
    LODWORD(v57) = 0;
    v54 = v30;
    v33 = 0;
    v55.m128i_i64[0] = v44;
    v55.m128i_i32[2] = v45;
    if ( HIDWORD(v57) <= 2 )
    {
      sub_C8D5F0((__int64)&v56, v58, 3u, 0x10u, v31, v32);
      v33 = 16LL * (unsigned int)v57;
    }
    v34 = _mm_loadu_si128(&v54);
    v35 = _mm_loadu_si128(&v55);
    v36 = (__m128i *)&v56[v33];
    *v36 = _mm_loadu_si128(&v53);
    v36[1] = v34;
    v36[2] = v35;
    v21 = v57 + 3;
    LODWORD(v57) = v57 + 3;
  }
  *((_QWORD *)&v43 + 1) = v21;
  *(_QWORD *)&v43 = v56;
  v22 = sub_33F7800(*(_QWORD **)(a1 + 64), 371 - ((unsigned int)!v47 - 1), (__int64)&v50, v48, v49, v21, v43);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v22, v23, v24, v25);
  sub_3421DB0(v22);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v56 != v58 )
    _libc_free((unsigned __int64)v56);
  result = 1;
  if ( v50 )
  {
    sub_B91220((__int64)&v50, v50);
    return 1;
  }
  return result;
}
