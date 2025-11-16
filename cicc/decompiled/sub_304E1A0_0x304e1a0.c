// Function: sub_304E1A0
// Address: 0x304e1a0
//
__int64 __fastcall sub_304E1A0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // r13
  __int64 v10; // rax
  _QWORD *v11; // rsi
  unsigned int v12; // r12d
  int v13; // r11d
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r9
  int v18; // r11d
  unsigned __int64 v19; // rdx
  __int64 *v20; // rax
  __int64 v21; // rax
  __m128i v22; // xmm0
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // r12
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r13
  unsigned __int64 v32; // rdx
  __int64 *v33; // rax
  __int64 v34; // rax
  __m128i v35; // xmm0
  __int64 v36; // rcx
  int v37; // r8d
  __int64 v38; // r12
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  __int64 *v42; // rax
  __m128i v43; // xmm0
  __int128 v44; // [rsp-10h] [rbp-E0h]
  __m128i v45; // [rsp+0h] [rbp-D0h] BYREF
  _BYTE **v46; // [rsp+18h] [rbp-B8h]
  int v47; // [rsp+24h] [rbp-ACh]
  _BYTE *v48; // [rsp+28h] [rbp-A8h]
  __int64 v49; // [rsp+30h] [rbp-A0h] BYREF
  int v50; // [rsp+38h] [rbp-98h]
  _BYTE *v51; // [rsp+40h] [rbp-90h] BYREF
  __int64 v52; // [rsp+48h] [rbp-88h]
  _BYTE v53[128]; // [rsp+50h] [rbp-80h] BYREF

  v6 = *(_QWORD *)(a2 + 80);
  v49 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v49, v6, 1);
  v50 = *(_DWORD *)(a2 + 72);
  v7 = *(_QWORD *)(a2 + 40);
  v8 = *(_QWORD *)(*(_QWORD *)(v7 + 40) + 96LL);
  v9 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  v10 = *(_QWORD *)(*(_QWORD *)(v7 + 80) + 96LL);
  v11 = *(_QWORD **)(v10 + 24);
  if ( *(_DWORD *)(v10 + 32) > 0x40u )
    v11 = (_QWORD *)*v11;
  v12 = ((unsigned int)v11 >> 1) & 0xF;
  if ( (((unsigned int)v11 >> 1) & 0xF) == 1 )
  {
    v13 = ((_DWORD)v9 == 8243) + 1621;
  }
  else if ( (_BYTE)v12 == 2 )
  {
    v13 = ((_DWORD)v9 == 8243) + 1623;
  }
  else
  {
    if ( (_BYTE)v12 )
      BUG();
    v13 = ((_DWORD)v9 == 8243) + 1615;
  }
  v46 = &v51;
  v48 = v53;
  v51 = v53;
  v47 = v13;
  v52 = 0x500000000LL;
  v14 = sub_3400BD0(a4, (_DWORD)v11, (unsigned int)&v49, 7, 0, 1, 0);
  v15 = (unsigned int)v52;
  v17 = v16;
  v18 = v47;
  v19 = (unsigned int)v52 + 1LL;
  if ( v19 > HIDWORD(v52) )
  {
    v45.m128i_i64[0] = v14;
    v45.m128i_i64[1] = v17;
    sub_C8D5F0((__int64)v46, v48, v19, 0x10u, v14, v17);
    v15 = (unsigned int)v52;
    v17 = v45.m128i_i64[1];
    v14 = v45.m128i_i64[0];
    v18 = v47;
  }
  v20 = (__int64 *)&v51[16 * v15];
  *v20 = v14;
  v20[1] = v17;
  v22 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 120LL));
  LODWORD(v52) = v52 + 1;
  v21 = (unsigned int)v52;
  if ( (unsigned __int64)(unsigned int)v52 + 1 > HIDWORD(v52) )
  {
    v47 = v18;
    v45 = v22;
    sub_C8D5F0((__int64)v46, v48, (unsigned int)v52 + 1LL, 0x10u, v14, v17);
    v21 = (unsigned int)v52;
    v22 = _mm_load_si128(&v45);
    v18 = v47;
  }
  v23 = 160;
  *(__m128i *)&v51[16 * v21] = v22;
  v24 = (unsigned int)(v52 + 1);
  LODWORD(v52) = v52 + 1;
  if ( (_DWORD)v9 == 8243 )
  {
    v43 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 160LL));
    if ( v24 + 1 > (unsigned __int64)HIDWORD(v52) )
    {
      v47 = v18;
      v45 = v43;
      sub_C8D5F0((__int64)v46, v48, v24 + 1, 0x10u, v14, v17);
      v24 = (unsigned int)v52;
      v43 = _mm_load_si128(&v45);
      v18 = v47;
    }
    v23 = 200;
    *(__m128i *)&v51[16 * v24] = v43;
    LODWORD(v52) = v52 + 1;
  }
  v25 = *(_QWORD *)(a2 + 40);
  v26 = *(_QWORD *)(v25 + v23);
  v27 = *(_QWORD *)(v25 + v23 + 8);
  if ( (_BYTE)v12 == 2 )
  {
    v40 = (unsigned int)v52;
    v41 = (unsigned int)v52 + 1LL;
    if ( v41 > HIDWORD(v52) )
    {
      v45.m128i_i64[0] = v26;
      v45.m128i_i64[1] = v27;
      v47 = v18;
      sub_C8D5F0((__int64)v46, v48, v41, 0x10u, v26, v27);
      v40 = (unsigned int)v52;
      v27 = v45.m128i_i64[1];
      v26 = v45.m128i_i64[0];
      v18 = v47;
    }
    v42 = (__int64 *)&v51[16 * v40];
    *v42 = v26;
    v42[1] = v27;
    v34 = (unsigned int)(v52 + 1);
    LODWORD(v52) = v52 + 1;
  }
  else
  {
    v44 = *(_OWORD *)(v25 + v23);
    v47 = v18;
    v28 = sub_33FAF80(a4, 214, (unsigned int)&v49, 7, 0, v27, v44);
    v29 = (unsigned int)v52;
    v31 = v30;
    v18 = v47;
    v32 = (unsigned int)v52 + 1LL;
    if ( v32 > HIDWORD(v52) )
    {
      sub_C8D5F0((__int64)v46, v48, v32, 0x10u, v26, v27);
      v29 = (unsigned int)v52;
      v18 = v47;
    }
    v33 = (__int64 *)&v51[16 * v29];
    *v33 = v28;
    v33[1] = v31;
    v34 = (unsigned int)(v52 + 1);
    LODWORD(v52) = v52 + 1;
  }
  v35 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  if ( v34 + 1 > (unsigned __int64)HIDWORD(v52) )
  {
    v47 = v18;
    v45 = v35;
    sub_C8D5F0((__int64)v46, v48, v34 + 1, 0x10u, v26, v27);
    v34 = (unsigned int)v52;
    v35 = _mm_load_si128(&v45);
    v18 = v47;
  }
  *(__m128i *)&v51[16 * v34] = v35;
  v36 = *(_QWORD *)(a2 + 48);
  v37 = *(_DWORD *)(a2 + 68);
  LODWORD(v52) = v52 + 1;
  v38 = sub_33E66D0(a4, v18, (unsigned int)&v49, v36, v37, v27, (__int64)v51, (unsigned int)v52);
  if ( v51 != v48 )
    _libc_free((unsigned __int64)v51);
  if ( v49 )
    sub_B91220((__int64)&v49, v49);
  return v38;
}
