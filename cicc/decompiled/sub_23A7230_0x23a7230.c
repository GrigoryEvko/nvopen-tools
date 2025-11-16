// Function: sub_23A7230
// Address: 0x23a7230
//
unsigned __int64 *__fastcall sub_23A7230(unsigned __int64 *a1, __int64 a2, unsigned __int64 a3, int a4)
{
  __int32 v5; // edx
  __int8 v6; // al
  __m128i v7; // xmm0
  __m128i v8; // xmm4
  __m128i v9; // xmm2
  __m128i v10; // xmm1
  __m128i v11; // xmm3
  int v12; // r13d
  __int64 v13; // rax
  __m128i v14; // xmm5
  __m128i v15; // xmm6
  __m128i v16; // xmm7
  __m128i v17; // xmm5
  int v18; // edx
  __m128i v19; // xmm6
  _BOOL8 v20; // rsi
  __int64 v21; // rax
  __m128i v22; // xmm0
  __m128i v23; // xmm7
  __int64 v24; // rdx
  __int64 v25; // rax
  __m128i v26; // xmm1
  _QWORD *v27; // rax
  _QWORD *v29; // rax
  _QWORD *v30; // rax
  char v31; // bl
  unsigned __int64 v32; // [rsp+8h] [rbp-428h]
  char v33; // [rsp+14h] [rbp-41Ch]
  __int64 v35; // [rsp+28h] [rbp-408h] BYREF
  __m128i v36; // [rsp+30h] [rbp-400h] BYREF
  __m128i v37; // [rsp+40h] [rbp-3F0h] BYREF
  __m128i v38; // [rsp+50h] [rbp-3E0h] BYREF
  __m128i v39; // [rsp+60h] [rbp-3D0h] BYREF
  __m128i v40; // [rsp+70h] [rbp-3C0h] BYREF
  int v41; // [rsp+80h] [rbp-3B0h]
  _BYTE v42[24]; // [rsp+90h] [rbp-3A0h] BYREF
  __m128i v43; // [rsp+A8h] [rbp-388h]
  __m128i v44; // [rsp+B8h] [rbp-378h]
  __m128i v45; // [rsp+C8h] [rbp-368h]
  __m128i v46; // [rsp+D8h] [rbp-358h]
  int v47; // [rsp+E8h] [rbp-348h]
  _BYTE v48[24]; // [rsp+100h] [rbp-330h] BYREF
  __m128i v49; // [rsp+118h] [rbp-318h] BYREF
  __m128i v50; // [rsp+128h] [rbp-308h] BYREF
  __m128i v51; // [rsp+138h] [rbp-2F8h] BYREF
  __m128i v52; // [rsp+148h] [rbp-2E8h] BYREF
  int v53; // [rsp+158h] [rbp-2D8h]

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  v32 = HIDWORD(a3);
  sub_30D6B60(&v36, (unsigned int)a3, HIDWORD(a3));
  v5 = v38.m128i_i32[1];
  v6 = v38.m128i_i8[8];
  if ( a4 == 1 && *(_BYTE *)(a2 + 192) && *(_DWORD *)(a2 + 168) == 3 )
  {
    v6 = 1;
    v5 = 0;
  }
  v38.m128i_i8[8] = v6;
  v7 = _mm_loadu_si128(&v36);
  v39.m128i_i16[7] = 256;
  v8 = _mm_loadu_si128(&v37);
  v9 = _mm_loadu_si128(&v39);
  v10 = _mm_loadu_si128(&v40);
  v38.m128i_i32[1] = v5;
  v11 = _mm_loadu_si128(&v38);
  *(_QWORD *)v48 = v7.m128i_i64[0];
  v12 = dword_4FDE108;
  *(__m128i *)&v42[8] = v7;
  v47 = v41;
  v53 = v41;
  v43 = v8;
  v44 = v11;
  v45 = v9;
  v46 = v10;
  *(__m128i *)&v48[8] = v7;
  v49 = v8;
  v50 = v11;
  v51 = v9;
  v52 = v10;
  v13 = sub_22077B0(0x70u);
  if ( v13 )
  {
    v14 = _mm_loadu_si128((const __m128i *)&v48[8]);
    *(_QWORD *)(v13 + 8) = 0;
    v15 = _mm_loadu_si128(&v49);
    v16 = _mm_loadu_si128(&v50);
    *(_DWORD *)(v13 + 100) = v12;
    *(__m128i *)(v13 + 16) = v14;
    v17 = _mm_loadu_si128(&v51);
    *(_QWORD *)v13 = &unk_4A0D978;
    v18 = v53;
    *(__m128i *)(v13 + 32) = v15;
    v19 = _mm_loadu_si128(&v52);
    *(_DWORD *)(v13 + 96) = v18;
    *(_DWORD *)(v13 + 104) = a4;
    *(__m128i *)(v13 + 48) = v16;
    *(__m128i *)(v13 + 64) = v17;
    *(__m128i *)(v13 + 80) = v19;
  }
  v35 = v13;
  sub_23A2230(a1, (unsigned __int64 *)&v35);
  sub_23501E0(&v35);
  if ( qword_502E468[9] && a4 == 2 )
  {
    v29 = (_QWORD *)sub_22077B0(0x10u);
    if ( v29 )
      *v29 = &unk_4A0D3B8;
    *(_QWORD *)v48 = v29;
    sub_23A2230(a1, (unsigned __int64 *)v48);
    if ( *(_QWORD *)v48 )
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)v48 + 8LL))(*(_QWORD *)v48);
    sub_23A0BA0((__int64)v48, 0);
    sub_23A2670(a1, (__int64)v48);
    sub_233AAF0((__int64)v48);
    v30 = (_QWORD *)sub_22077B0(0x10u);
    if ( v30 )
      *v30 = &unk_4A0D0B8;
    *(_QWORD *)v48 = v30;
    sub_23A2230(a1, (unsigned __int64 *)v48);
    sub_23501E0((__int64 *)v48);
    v31 = *(_BYTE *)(a2 + 32);
    sub_23A54A0((unsigned __int64 *)v48, a2, a3, 2);
    sub_234AAB0((__int64)v42, (__int64 *)v48, v31);
    sub_23571D0(a1, (__int64 *)v42);
    sub_233EFE0((__int64 *)v42);
    sub_233F7F0((__int64)v48);
  }
  else
  {
    v33 = *(_BYTE *)(a2 + 32);
    sub_23A54A0((unsigned __int64 *)v48, a2, a3, a4);
    sub_234AAB0((__int64)v42, (__int64 *)v48, v33);
    sub_23571D0(a1, (__int64 *)v42);
    sub_233EFE0((__int64 *)v42);
    sub_233F7F0((__int64)v48);
    if ( a4 == 1 )
      return a1;
  }
  v20 = 1;
  if ( (_DWORD)v32 == HIDWORD(qword_5033F08) )
    v20 = (_DWORD)qword_5033F08 != (_DWORD)a3;
  sub_24E6490(v42, v20);
  v21 = *(_QWORD *)&v42[16];
  v22 = _mm_loadu_si128((const __m128i *)v42);
  v23 = _mm_loadu_si128((const __m128i *)v48);
  v24 = v49.m128i_i64[0];
  *(_QWORD *)&v42[16] = 0;
  *(_QWORD *)&v48[16] = v21;
  *(__m128i *)v42 = v23;
  v49.m128i_i64[0] = v43.m128i_i64[0];
  v43.m128i_i64[0] = v24;
  v49.m128i_i8[8] = v43.m128i_i8[8];
  *(__m128i *)v48 = v22;
  v25 = sub_22077B0(0x30u);
  if ( v25 )
  {
    v26 = _mm_loadu_si128((const __m128i *)v48);
    v35 = v25;
    *(__m128i *)(v25 + 8) = v26;
    *(_QWORD *)v25 = &unk_4A0EC38;
    *(_QWORD *)(v25 + 24) = *(_QWORD *)&v48[16];
    *(_QWORD *)(v25 + 32) = v49.m128i_i64[0];
    *(_BYTE *)(v25 + 40) = v49.m128i_i8[8];
  }
  else
  {
    v35 = 0;
    if ( *(_QWORD *)&v48[16] )
      (*(void (__fastcall **)(_BYTE *, _BYTE *, __int64))&v48[16])(v48, v48, 3);
  }
  sub_2357280(a1, &v35);
  if ( v35 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v35 + 8LL))(v35);
  if ( *(_QWORD *)&v42[16] )
    (*(void (__fastcall **)(_BYTE *, _BYTE *, __int64))&v42[16])(v42, v42, 3);
  v27 = (_QWORD *)sub_22077B0(0x10u);
  if ( v27 )
    *v27 = &unk_4A0EBF8;
  *(_QWORD *)v48 = v27;
  sub_2357280(a1, (__int64 *)v48);
  if ( *(_QWORD *)v48 )
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)v48 + 8LL))(*(_QWORD *)v48);
  return a1;
}
