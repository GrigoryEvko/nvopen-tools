// Function: sub_70F370
// Address: 0x70f370
//
__int64 __fastcall sub_70F370(
        const __m128i *a1,
        __int64 a2,
        const __m128i *a3,
        __m128i *a4,
        int *a5,
        _DWORD *a6,
        _BYTE *a7)
{
  char v9; // r12
  __m128i *v11; // rax
  __m128i *v12; // rdi
  __int64 i; // r15
  unsigned int v14; // r14d
  unsigned __int64 v15; // r15
  __int8 v16; // al
  int v17; // esi
  __int8 v19; // al
  int v20; // eax
  __int64 k; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 j; // r14
  __int64 v26; // r15
  __int64 v27; // rax
  bool v28; // [rsp+23h] [rbp-5Dh]
  _BOOL4 v32; // [rsp+44h] [rbp-3Ch] BYREF
  __m128i *v33; // [rsp+48h] [rbp-38h] BYREF

  v9 = a2;
  v11 = (__m128i *)sub_724DC0(a1, a2, a3, a4, a5, a6);
  *a5 = 0;
  v33 = v11;
  v12 = v11;
  *a6 = 0;
  *a7 = 5;
  v32 = 0;
  if ( (unsigned __int8)(a2 - 39) <= 1u )
  {
    v14 = 1;
    v15 = 1;
LABEL_7:
    v16 = a1[10].m128i_i8[13];
    if ( v16 != 1 )
      goto LABEL_18;
    goto LABEL_8;
  }
  for ( i = sub_8D46C0(a1[8].m128i_i64[0]); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( (unsigned int)sub_8D4070(i) )
  {
    *a5 = 1;
    goto LABEL_12;
  }
  v14 = dword_4F077C0;
  if ( !dword_4F077C0 )
    goto LABEL_6;
  while ( *(_BYTE *)(i + 140) == 12 )
    i = *(_QWORD *)(i + 160);
  if ( !(unsigned int)sub_8D2600(i) )
  {
    v14 = sub_8D2310(i);
    if ( !v14 )
    {
LABEL_6:
      v15 = *(_QWORD *)(i + 128);
      v12 = v33;
      goto LABEL_7;
    }
  }
  v16 = a1[10].m128i_i8[13];
  v12 = v33;
  v14 = 0;
  v15 = 1;
  if ( v16 != 1 )
  {
LABEL_18:
    if ( v16 != 6 )
      goto LABEL_36;
    sub_72BAF0(v12, a1[12].m128i_i64[0], unk_4F06A60);
    if ( !v14 )
      goto LABEL_20;
LABEL_9:
    v17 = sub_620E90((__int64)a1);
    goto LABEL_10;
  }
LABEL_8:
  *v12 = _mm_loadu_si128(a1);
  v12[1] = _mm_loadu_si128(a1 + 1);
  v12[2] = _mm_loadu_si128(a1 + 2);
  v12[3] = _mm_loadu_si128(a1 + 3);
  v12[4] = _mm_loadu_si128(a1 + 4);
  v12[5] = _mm_loadu_si128(a1 + 5);
  v12[6] = _mm_loadu_si128(a1 + 6);
  v12[7] = _mm_loadu_si128(a1 + 7);
  v12[8] = _mm_loadu_si128(a1 + 8);
  v12[9] = _mm_loadu_si128(a1 + 9);
  v12[10] = _mm_loadu_si128(a1 + 10);
  v12[11] = _mm_loadu_si128(a1 + 11);
  v12[12] = _mm_loadu_si128(a1 + 12);
  if ( v14 )
    goto LABEL_9;
LABEL_20:
  v17 = sub_620E90((__int64)v33);
LABEL_10:
  *a5 = 0;
  v32 = 0;
  v28 = v9 == 40 || v9 == 51;
  if ( a3[10].m128i_i8[13] != 1 )
  {
    *a5 = 1;
LABEL_12:
    sub_72C970(a4);
    return sub_724E30(&v33);
  }
  sub_70D820(v33, v17, v28, a3, v15, v14 & (v17 == 0), &v32);
  if ( v32 )
    goto LABEL_22;
  if ( *a5 )
    goto LABEL_12;
  sub_72A510(a1, a4);
  v19 = a4[10].m128i_i8[13];
  v12 = v33;
  if ( v19 == 1 )
  {
    *a4 = _mm_loadu_si128(v33);
    a4[1] = _mm_loadu_si128(v12 + 1);
    a4[2] = _mm_loadu_si128(v12 + 2);
    a4[3] = _mm_loadu_si128(v12 + 3);
    a4[4] = _mm_loadu_si128(v12 + 4);
    a4[5] = _mm_loadu_si128(v12 + 5);
    a4[6] = _mm_loadu_si128(v12 + 6);
    a4[7] = _mm_loadu_si128(v12 + 7);
    a4[8] = _mm_loadu_si128(v12 + 8);
    a4[9] = _mm_loadu_si128(v12 + 9);
    a4[10] = _mm_loadu_si128(v12 + 10);
    a4[11] = _mm_loadu_si128(v12 + 11);
    a4[12] = _mm_loadu_si128(v12 + 12);
    goto LABEL_28;
  }
  if ( v19 != 6 )
LABEL_36:
    sub_721090(v12);
  a4[12].m128i_i64[0] = sub_620FA0((__int64)v33, &v32);
LABEL_28:
  if ( ((unsigned __int8)v14 & (v17 == 0)) != 0 )
  {
    v32 = 0;
    v20 = *a5;
    goto LABEL_30;
  }
  if ( a4[10].m128i_i8[13] == 6 )
  {
    v12 = (__m128i *)a3;
    v26 = sub_77F710(a4, 1, 0);
    v27 = sub_620FA0((__int64)a3, &v32);
    if ( v28 )
      *(_QWORD *)(v26 + 16) -= v27;
    else
      *(_QWORD *)(v26 + 16) += v27;
  }
  if ( v32 )
  {
LABEL_22:
    *a6 = 61;
    *a7 = 8;
    return sub_724E30(&v33);
  }
  v20 = *a5;
LABEL_30:
  if ( v20 )
    goto LABEL_12;
  if ( !v14 && a4[10].m128i_i8[13] != 1 )
  {
    switch ( a4[11].m128i_i8[0] )
    {
      case 0:
      case 5:
      case 6:
        goto LABEL_37;
      case 1:
        for ( j = *(_QWORD *)(a4[11].m128i_i64[1] + 120); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        if ( (unsigned int)sub_8D23B0(j)
          || (unsigned __int8)(*(_BYTE *)(j + 140) - 9) <= 2u && (*(_BYTE *)(j + 179) & 8) != 0 )
        {
LABEL_37:
          if ( a4[12].m128i_i64[0] >= 0 )
            return sub_724E30(&v33);
        }
        else
        {
          v22 = *(_QWORD *)(j + 128);
LABEL_42:
          v23 = a4[12].m128i_i64[0];
          if ( v23 >= 0 && (!v22 || v22 >= v23) )
            return sub_724E30(&v33);
        }
        if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) && !HIDWORD(qword_4F077B4) )
          *a5 = 1;
        *a6 = 5 * (v9 == 92) + 170;
        *a7 = 5;
        return sub_724E30(&v33);
      case 2:
        v24 = a4[11].m128i_i64[1];
        if ( *(_BYTE *)(v24 + 173) == 2 )
        {
          v22 = *(_QWORD *)(v24 + 176);
        }
        else
        {
          for ( k = *(_QWORD *)(v24 + 128); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
            ;
LABEL_41:
          v22 = *(_QWORD *)(k + 128);
        }
        goto LABEL_42;
      case 3:
        for ( k = *(_QWORD *)(a4[11].m128i_i64[1] + 128); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
          ;
        goto LABEL_41;
      case 4:
        k = sub_8D46C0(a4[8].m128i_i64[0]);
        goto LABEL_41;
      default:
        goto LABEL_36;
    }
  }
  return sub_724E30(&v33);
}
