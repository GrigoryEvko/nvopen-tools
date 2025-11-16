// Function: sub_23A6B00
// Address: 0x23a6b00
//
unsigned __int64 *__fastcall sub_23A6B00(unsigned __int64 *a1, __int64 a2, unsigned __int64 a3, int a4)
{
  int v5; // r9d
  __int8 v6; // al
  char v7; // dl
  _QWORD *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // rax
  _BOOL8 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rdx
  __m128i v15; // xmm0
  __m128i v16; // xmm6
  __int64 v17; // rax
  __int64 v18; // rcx
  __m128i v19; // xmm0
  __m128i v20; // xmm7
  __int64 v21; // rdx
  __int64 v22; // rdx
  _QWORD *v23; // rax
  _QWORD *v24; // rax
  __int64 v26; // rax
  _QWORD *v27; // rax
  _QWORD *v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // rax
  unsigned __int64 v31; // [rsp+60h] [rbp-140h]
  char v32; // [rsp+68h] [rbp-138h]
  __int64 v34; // [rsp+78h] [rbp-128h] BYREF
  __m128i v35; // [rsp+80h] [rbp-120h] BYREF
  void (__fastcall *v36)(__m128i *, __m128i *, __int64); // [rsp+90h] [rbp-110h]
  __int64 v37; // [rsp+98h] [rbp-108h]
  __int8 v38; // [rsp+A0h] [rbp-100h]
  __m128i v39; // [rsp+B0h] [rbp-F0h] BYREF
  __m128i v40; // [rsp+C0h] [rbp-E0h] BYREF
  __m128i v41; // [rsp+D0h] [rbp-D0h] BYREF
  __m128i v42; // [rsp+E0h] [rbp-C0h] BYREF
  __m128i v43; // [rsp+F0h] [rbp-B0h] BYREF
  int v44; // [rsp+100h] [rbp-A0h]
  __m128i v45; // [rsp+110h] [rbp-90h] BYREF
  __m128i v46; // [rsp+120h] [rbp-80h] BYREF
  __m128i v47; // [rsp+130h] [rbp-70h] BYREF
  __m128i v48; // [rsp+140h] [rbp-60h] BYREF
  __m128i v49; // [rsp+150h] [rbp-50h] BYREF
  int v50; // [rsp+160h] [rbp-40h]

  v31 = HIDWORD(a3);
  if ( *(_DWORD *)(a2 + 28) == -1 )
  {
    sub_30D6B60(&v45, (unsigned int)a3, HIDWORD(a3));
    v39 = _mm_loadu_si128(&v45);
    v6 = v48.m128i_i8[15];
    v40 = _mm_loadu_si128(&v46);
    v44 = v50;
    v41 = _mm_loadu_si128(&v47);
    v42 = _mm_loadu_si128(&v48);
    v43 = _mm_loadu_si128(&v49);
  }
  else
  {
    sub_30D6950(&v39);
    v6 = v42.m128i_i8[15];
  }
  v7 = *(_BYTE *)(a2 + 192);
  if ( a4 == 1 )
  {
    if ( !v7 )
      goto LABEL_5;
    if ( *(_DWORD *)(a2 + 168) == 3 )
    {
      v41.m128i_i32[1] = 0;
      v41.m128i_i8[8] = 1;
    }
  }
  else if ( !v7 )
  {
    goto LABEL_5;
  }
  v42.m128i_i8[14] = byte_4FDE028;
  if ( !v6 )
    v6 = 1;
LABEL_5:
  v42.m128i_i8[15] = v6;
  sub_26124A0(
    (_DWORD)a1,
    (unsigned __int8)qword_4FDDE68,
    a4,
    dword_4FDE108,
    unk_502E168,
    v5,
    *(_OWORD *)&_mm_loadu_si128(&v39),
    *(_OWORD *)&_mm_loadu_si128(&v40),
    *(_OWORD *)&_mm_loadu_si128(&v41),
    *(_OWORD *)&_mm_loadu_si128(&v42),
    *(_OWORD *)&_mm_loadu_si128(&v43),
    v44);
  if ( byte_4FDDAE8 )
  {
    v28 = (_QWORD *)sub_22077B0(0x10u);
    if ( v28 )
      *v28 = &unk_4A0CD38;
    v45.m128i_i64[0] = (__int64)v28;
    sub_23A2230(a1 + 18, (unsigned __int64 *)&v45);
    sub_23501E0(v45.m128i_i64);
    v29 = (_QWORD *)sub_22077B0(0x10u);
    if ( v29 )
      *v29 = &unk_4A127F8;
    v45.m128i_i64[0] = (__int64)v29;
    v45.m128i_i8[8] = 0;
    sub_23571D0(a1 + 18, v45.m128i_i64);
    sub_233EFE0(v45.m128i_i64);
  }
  v8 = (_QWORD *)sub_22077B0(0x10u);
  if ( v8 )
    *v8 = &unk_4A0CB38;
  v45.m128i_i64[0] = (__int64)v8;
  sub_23A2230(a1 + 18, (unsigned __int64 *)&v45);
  sub_23501E0(v45.m128i_i64);
  if ( (byte_4FDC708 & 2) != 0 )
  {
    v27 = (_QWORD *)sub_22077B0(0x10u);
    if ( v27 )
      *v27 = &unk_4A0EAB8;
    v45.m128i_i64[0] = (__int64)v27;
    sub_23A32D0(a1 + 13, (unsigned __int64 *)&v45);
    sub_233F000(v45.m128i_i64);
  }
  v9 = sub_22077B0(0x10u);
  if ( v9 )
  {
    *(_BYTE *)(v9 + 8) = 1;
    *(_QWORD *)v9 = &unk_4A0EC78;
  }
  v45.m128i_i64[0] = v9;
  sub_23A32D0(a1 + 13, (unsigned __int64 *)&v45);
  if ( v45.m128i_i64[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v45.m128i_i64[0] + 8LL))(v45.m128i_i64[0]);
  if ( (_DWORD)v31 != dword_5033EF0[1] )
  {
    if ( (_DWORD)v31 != unk_5033EFC || (_DWORD)a3 != unk_5033EF8 )
      goto LABEL_15;
    goto LABEL_45;
  }
  if ( (_DWORD)a3 == dword_5033EF0[0] )
  {
    v30 = sub_22077B0(0x10u);
    if ( v30 )
    {
      *(_DWORD *)(v30 + 8) = 2;
      *(_QWORD *)v30 = &unk_4A0EA78;
    }
    v45.m128i_i64[0] = v30;
    sub_23A32D0(a1 + 13, (unsigned __int64 *)&v45);
    if ( v45.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v45.m128i_i64[0] + 8LL))(v45.m128i_i64[0]);
    if ( (_DWORD)v31 != unk_5033EFC || (_DWORD)a3 != unk_5033EF8 )
    {
      if ( (_DWORD)v31 != dword_5033EF0[1] )
        goto LABEL_15;
      goto LABEL_51;
    }
  }
  else
  {
    if ( (_DWORD)v31 != unk_5033EFC )
      goto LABEL_15;
    if ( (_DWORD)a3 != unk_5033EF8 )
    {
LABEL_51:
      if ( (_DWORD)a3 != dword_5033EF0[0] )
        goto LABEL_15;
    }
  }
LABEL_45:
  v26 = sub_22077B0(0x10u);
  if ( v26 )
  {
    *(_DWORD *)(v26 + 8) = a4;
    *(_QWORD *)v26 = &unk_4A0EBB8;
  }
  v45.m128i_i64[0] = v26;
  sub_23A32D0(a1 + 13, (unsigned __int64 *)&v45);
  sub_233F000(v45.m128i_i64);
LABEL_15:
  sub_23A0F30(a2, (__int64)(a1 + 13), a3);
  v32 = *(_BYTE *)(a2 + 32);
  sub_23A54A0((unsigned __int64 *)&v45, a2, a3, a4);
  sub_234D2B0((__int64)&v35, v45.m128i_i64, v32, 1);
  sub_235A8B0(a1 + 13, v35.m128i_i64);
  sub_233EFE0(v35.m128i_i64);
  sub_233F7F0((__int64)&v45);
  v10 = sub_22077B0(0x10u);
  if ( v10 )
  {
    *(_BYTE *)(v10 + 8) = 0;
    *(_QWORD *)v10 = &unk_4A0EC78;
  }
  v45.m128i_i64[0] = v10;
  sub_23A32D0(a1 + 13, (unsigned __int64 *)&v45);
  if ( v45.m128i_i64[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v45.m128i_i64[0] + 8LL))(v45.m128i_i64[0]);
  v11 = (_QWORD *)sub_22077B0(0x10u);
  if ( v11 )
    *v11 = &unk_4A135B8;
  v45.m128i_i64[0] = (__int64)v11;
  v45.m128i_i16[4] = 0;
  sub_235A8B0(a1 + 13, v45.m128i_i64);
  sub_233EFE0(v45.m128i_i64);
  if ( a4 != 1 )
  {
    v12 = 1;
    if ( (_DWORD)v31 == HIDWORD(qword_5033F08) )
      v12 = (_DWORD)qword_5033F08 != (_DWORD)a3;
    sub_24E6490(&v35, v12);
    v13 = (__int64)v36;
    v14 = v46.m128i_i64[1];
    v15 = _mm_loadu_si128(&v35);
    v16 = _mm_loadu_si128(&v45);
    v36 = 0;
    v46.m128i_i64[0] = v13;
    v35 = v16;
    v46.m128i_i64[1] = v37;
    v37 = v14;
    v47.m128i_i8[0] = v38;
    v45 = v15;
    v17 = sub_22077B0(0x30u);
    if ( v17 )
    {
      v18 = *(_QWORD *)(v17 + 32);
      v19 = _mm_loadu_si128(&v45);
      v20 = _mm_loadu_si128((const __m128i *)(v17 + 8));
      *(_QWORD *)v17 = &unk_4A0EC38;
      v21 = v46.m128i_i64[0];
      v46.m128i_i64[0] = 0;
      *(_QWORD *)(v17 + 24) = v21;
      v22 = v46.m128i_i64[1];
      v46.m128i_i64[1] = v18;
      *(_QWORD *)(v17 + 32) = v22;
      v45 = v20;
      *(_BYTE *)(v17 + 40) = v47.m128i_i8[0];
      *(__m128i *)(v17 + 8) = v19;
    }
    v34 = v17;
    sub_23A32D0(a1 + 13, (unsigned __int64 *)&v34);
    if ( v34 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v34 + 8LL))(v34);
    if ( v46.m128i_i64[0] )
      ((void (__fastcall *)(__m128i *, __m128i *, __int64))v46.m128i_i64[0])(&v45, &v45, 3);
    if ( v36 )
      v36(&v35, &v35, 3);
    v23 = (_QWORD *)sub_22077B0(0x10u);
    if ( v23 )
      *v23 = &unk_4A0EBF8;
    v45.m128i_i64[0] = (__int64)v23;
    sub_23A32D0(a1 + 13, (unsigned __int64 *)&v45);
    if ( v45.m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v45.m128i_i64[0] + 8LL))(v45.m128i_i64[0]);
  }
  v24 = (_QWORD *)sub_22077B0(0x10u);
  if ( v24 )
    *v24 = &unk_4A135F8;
  v45.m128i_i64[0] = (__int64)v24;
  v45.m128i_i8[8] = 0;
  sub_23571D0(a1 + 23, v45.m128i_i64);
  sub_233EFE0(v45.m128i_i64);
  return a1;
}
