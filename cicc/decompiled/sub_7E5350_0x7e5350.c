// Function: sub_7E5350
// Address: 0x7e5350
//
__int64 __fastcall sub_7E5350(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r14
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 i; // r14
  __int64 j; // r13
  __m128i *v11; // rax
  char v12; // al
  char v13; // si
  char v14; // al
  char v15; // al
  char v16; // al
  _QWORD *v17; // rsi
  _QWORD *v18; // rax
  _QWORD *v19; // rax
  _QWORD *v20; // rbx
  const __m128i *v21; // rdx
  __m128i *v22; // rax
  const __m128i *v23; // r13
  __m128i *k; // rax
  _QWORD *m128i_i64; // r14
  _QWORD *v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  __int64 v29; // [rsp+18h] [rbp-48h]

  if ( a3 )
  {
    v6 = *(_QWORD *)(a3 + 104);
    if ( (*(_BYTE *)(a3 + 96) & 2) != 0
      || (a3 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a3 + 112) + 8LL) + 16LL), (*(_BYTE *)(a3 + 96) & 2) != 0) )
    {
      v29 = v6 - *(_QWORD *)(a3 + 104);
      v28 = *(_QWORD *)(a3 + 144);
    }
    else
    {
      v29 = v6;
      v28 = 0;
    }
  }
  else
  {
    v28 = 0;
    v29 = 0;
  }
  v7 = *(_QWORD *)(a1 + 112);
  if ( v7 )
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v7 + 272);
      if ( !v8 )
        break;
      if ( a1 == v8 && *(_QWORD *)(v7 + 288) == a4 && *(_QWORD *)(v7 + 296) == a5 && *(_QWORD *)(v7 + 304) == v29 )
      {
        if ( *(_QWORD *)(v7 + 312) == v28 )
          return v7;
        v7 = *(_QWORD *)(v7 + 112);
        if ( !v7 )
          break;
      }
      else
      {
        v7 = *(_QWORD *)(v7 + 112);
        if ( !v7 )
          break;
      }
    }
  }
  for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  for ( j = *(_QWORD *)(a2 + 152); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v11 = sub_725FD0();
  v11[-1].m128i_i8[8] &= ~8u;
  v7 = (__int64)v11;
  v11[12].m128i_i8[1] |= 0x10u;
  v12 = *(_BYTE *)(a1 + 88) & 4 | v11[5].m128i_i8[8] & 0xFB;
  *(_BYTE *)(v7 + 88) = v12;
  v13 = *(_BYTE *)(a1 + 172);
  if ( !v13 )
    v13 = 1;
  *(_BYTE *)(v7 + 172) = v13;
  *(_BYTE *)(v7 + 88) = *(_BYTE *)(a1 + 88) & 0x70 | v12 & 0x8F;
  v14 = *(_BYTE *)(a1 + 198) & 8 | *(_BYTE *)(v7 + 198) & 0xF7;
  *(_BYTE *)(v7 + 198) = v14;
  v15 = *(_BYTE *)(a1 + 198) & 0x10 | v14 & 0xEF;
  *(_BYTE *)(v7 + 198) = v15;
  *(_BYTE *)(v7 + 198) = *(_BYTE *)(a1 + 198) & 0x20 | v15 & 0xDF;
  if ( (*(_QWORD *)(a1 + 192) & 0x240000000LL) != 0 )
    *(_BYTE *)(v7 + 196) |= 2u;
  v16 = *(_BYTE *)(a1 + 200) & 7 | *(_BYTE *)(v7 + 200) & 0xF8;
  *(_BYTE *)(v7 + 200) = v16;
  *(_BYTE *)(v7 + 200) = *(_BYTE *)(a1 + 200) & 0x20 | v16 & 0xDF;
  v17 = *(_QWORD **)(a1 + 256);
  if ( v17 )
  {
    v18 = *(_QWORD **)(v7 + 256);
    if ( !v18 )
    {
      v27 = *(_QWORD **)(a1 + 256);
      v18 = (_QWORD *)sub_726210(v7);
      v17 = v27;
    }
    *v18 = *v17;
  }
  sub_736C90(v7, *(_BYTE *)(a1 + 192) >> 7);
  *(_QWORD *)(v7 + 288) = a4;
  *(_QWORD *)(v7 + 272) = a1;
  *(_QWORD *)(v7 + 280) = a2;
  *(_QWORD *)(v7 + 296) = a5;
  *(_QWORD *)(v7 + 304) = v29;
  *(_QWORD *)(v7 + 312) = v28;
  v19 = sub_7259C0(7);
  *(_QWORD *)(v7 + 152) = v19;
  v20 = v19;
  *((_BYTE *)v19 - 8) &= ~8u;
  v19[20] = sub_7F8700(j);
  v21 = *(const __m128i **)(i + 168);
  v22 = (__m128i *)v20[21];
  *v22 = _mm_loadu_si128(v21);
  v22[1] = _mm_loadu_si128(v21 + 1);
  v22[2] = _mm_loadu_si128(v21 + 2);
  v22[3] = _mm_loadu_si128(v21 + 3);
  *(_QWORD *)(v20[21] + 8LL) = 0;
  *(_BYTE *)(v20[21] + 17LL) &= ~1u;
  *(_BYTE *)(v20[21] + 17LL) &= ~2u;
  *(_QWORD *)v20[21] = 0;
  v23 = (const __m128i *)sub_7E5340(a1);
  for ( k = 0; v23; v23 = (const __m128i *)v23->m128i_i64[0] )
  {
    while ( 1 )
    {
      m128i_i64 = k->m128i_i64;
      k = (__m128i *)sub_724EF0(v23->m128i_i64[1]);
      k[-1].m128i_i8[8] &= ~8u;
      *k = _mm_loadu_si128(v23);
      k[1] = _mm_loadu_si128(v23 + 1);
      k[2] = _mm_loadu_si128(v23 + 2);
      k[3] = _mm_loadu_si128(v23 + 3);
      k[4] = _mm_loadu_si128(v23 + 4);
      k[5].m128i_i64[0] = v23[5].m128i_i64[0];
      if ( !m128i_i64 )
        break;
      *m128i_i64 = k;
      k->m128i_i64[0] = 0;
      v23 = (const __m128i *)v23->m128i_i64[0];
      if ( !v23 )
        goto LABEL_31;
    }
    *(_QWORD *)v20[21] = k;
    k->m128i_i64[0] = 0;
  }
LABEL_31:
  sub_814B20(v7);
  *(_QWORD *)(v7 + 112) = *(_QWORD *)(a1 + 112);
  *(_QWORD *)(a1 + 112) = v7;
  if ( a1 == *(_QWORD *)(qword_4D03FF0 + 72) )
    *(_QWORD *)(qword_4D03FF0 + 72) = v7;
  return v7;
}
