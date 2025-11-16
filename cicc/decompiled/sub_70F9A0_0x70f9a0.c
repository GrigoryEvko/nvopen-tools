// Function: sub_70F9A0
// Address: 0x70f9a0
//
unsigned __int64 __fastcall sub_70F9A0(const __m128i *a1, __int64 a2, _DWORD *a3, _BYTE *a4, _DWORD *a5)
{
  __int64 i; // r14
  __int64 j; // rbx
  __int64 v9; // rsi
  unsigned __int8 v10; // al
  char v11; // dl
  char v12; // al
  const __m128i *v13; // rax
  __m128i v14; // xmm1
  char v15; // al
  char v16; // al
  char v17; // al
  unsigned __int64 result; // rax
  unsigned __int8 v19; // [rsp+Eh] [rbp-72h]
  unsigned __int8 v20; // [rsp+Fh] [rbp-71h]
  unsigned int v23; // [rsp+2Ch] [rbp-54h] BYREF
  __m128i v24; // [rsp+30h] [rbp-50h] BYREF
  __m128i v25[4]; // [rsp+40h] [rbp-40h] BYREF

  for ( i = a1[8].m128i_i64[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  for ( j = *(_QWORD *)(a2 + 128); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v9 = 4;
  v19 = *(_BYTE *)(i + 160);
  v10 = *(_BYTE *)(j + 160);
  *a3 = 0;
  v20 = v10;
  *a4 = 5;
  v11 = *(_BYTE *)(j + 140);
  if ( v11 != 5 )
    v9 = 2 * (unsigned int)(v11 == 4) + 3;
  sub_724A80(a2, v9);
  v12 = *(_BYTE *)(i + 140);
  if ( v12 == 3 )
  {
    v17 = *(_BYTE *)(j + 140);
    if ( v17 != 3 )
    {
      if ( v17 != 4 )
      {
        if ( v17 != 5 )
          goto LABEL_30;
        sub_709EF0(a1 + 11, v19, *(_OWORD **)(a2 + 176), v20, &v23, a5);
        sub_70B680(v20, 0, (_OWORD *)(*(_QWORD *)(a2 + 176) + 16LL), &v23);
        goto LABEL_22;
      }
LABEL_21:
      sub_70B680(v20, 0, (_OWORD *)(a2 + 176), &v23);
      goto LABEL_22;
    }
LABEL_25:
    sub_709EF0(a1 + 11, v19, (_OWORD *)(a2 + 176), v20, &v23, a5);
    goto LABEL_22;
  }
  if ( v12 == 4 )
  {
    v16 = *(_BYTE *)(j + 140);
    if ( v16 != 4 )
    {
      if ( v16 != 3 )
      {
        if ( v16 != 5 )
          goto LABEL_30;
        sub_70B680(v20, 0, *(_OWORD **)(a2 + 176), &v23);
        sub_709EF0(a1 + 11, v19, (_OWORD *)(*(_QWORD *)(a2 + 176) + 16LL), v20, &v23, a5);
        goto LABEL_22;
      }
      goto LABEL_21;
    }
    goto LABEL_25;
  }
  if ( v12 != 5 )
    goto LABEL_30;
  v13 = (const __m128i *)a1[11].m128i_i64[0];
  if ( a1[10].m128i_i8[13] == 4 )
  {
    v24 = _mm_loadu_si128(v13);
    v25[0] = _mm_loadu_si128(v13 + 1);
  }
  else
  {
    v14 = _mm_loadu_si128((const __m128i *)(v13[7].m128i_i64[1] + 176));
    v24 = _mm_loadu_si128(v13 + 11);
    v25[0] = v14;
  }
  v15 = *(_BYTE *)(j + 140);
  switch ( v15 )
  {
    case 4:
      sub_709EF0(v25, v19, (_OWORD *)(a2 + 176), v20, &v23, a5);
      break;
    case 5:
      sub_709EF0(&v24, v19, *(_OWORD **)(a2 + 176), v20, &v23, a5);
      sub_709EF0(v25, v19, (_OWORD *)(*(_QWORD *)(a2 + 176) + 16LL), v20, &v23, a5);
      break;
    case 3:
      sub_709EF0(&v24, v19, (_OWORD *)(a2 + 176), v20, &v23, a5);
      break;
    default:
LABEL_30:
      sub_721090(a2);
  }
LABEL_22:
  result = v23;
  if ( v23 )
  {
    result = (unsigned __int64)a4;
    *a3 = 221;
    *a4 = 8;
  }
  return result;
}
