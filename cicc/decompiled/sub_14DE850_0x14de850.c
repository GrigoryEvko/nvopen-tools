// Function: sub_14DE850
// Address: 0x14de850
//
_BYTE *__fastcall sub_14DE850(_QWORD *a1, __int64 a2)
{
  __m128i *v3; // rdx
  __m128i si128; // xmm0
  __int64 v5; // rdi
  __int64 v6; // rdx
  __m128i v7; // xmm0
  _QWORD *v8; // r13
  _QWORD *i; // r12
  _BYTE *v10; // rax
  __m128i *v11; // rdx
  __m128i v12; // xmm0
  _QWORD *v13; // r13
  _QWORD *j; // r12
  _BYTE *v15; // rax
  __m128i *v16; // rdx
  _BYTE *result; // rax
  __m128i v18; // xmm0
  _QWORD *v19; // r13
  _QWORD *k; // r12

  v3 = *(__m128i **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v3 <= 0x3Du )
  {
    v5 = sub_16E7EE0(a2, "-------------------------------------------------------------\n", 62);
    v6 = *(_QWORD *)(v5 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v5 + 16) - v6) > 0x12 )
      goto LABEL_3;
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F8E660);
    v5 = a2;
    qmemcpy(&v3[3], "-------------\n", 14);
    *v3 = si128;
    v3[1] = si128;
    v3[2] = si128;
    v6 = *(_QWORD *)(a2 + 24) + 62LL;
    *(_QWORD *)(a2 + 24) = v6;
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v6) > 0x12 )
    {
LABEL_3:
      v7 = _mm_load_si128((const __m128i *)&xmmword_4290FC0);
      *(_BYTE *)(v6 + 18) = 10;
      *(_WORD *)(v6 + 16) = 14963;
      *(__m128i *)v6 = v7;
      *(_QWORD *)(v5 + 24) += 19LL;
      goto LABEL_4;
    }
  }
  sub_16E7EE0(v5, "Interval Contents:\n", 19);
LABEL_4:
  v8 = (_QWORD *)a1[2];
  for ( i = (_QWORD *)a1[1]; v8 != i; ++i )
  {
    while ( 1 )
    {
      sub_155C2B0(*i, a2, 0);
      v10 = *(_BYTE **)(a2 + 24);
      if ( *(_BYTE **)(a2 + 16) == v10 )
        break;
      ++i;
      *v10 = 10;
      ++*(_QWORD *)(a2 + 24);
      if ( v8 == i )
        goto LABEL_9;
    }
    sub_16E7EE0(a2, "\n", 1);
  }
LABEL_9:
  v11 = *(__m128i **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v11 <= 0x16u )
  {
    sub_16E7EE0(a2, "Interval Predecessors:\n", 23);
  }
  else
  {
    v12 = _mm_load_si128((const __m128i *)&xmmword_4290FD0);
    v11[1].m128i_i32[0] = 1919906675;
    v11[1].m128i_i16[2] = 14963;
    v11[1].m128i_i8[6] = 10;
    *v11 = v12;
    *(_QWORD *)(a2 + 24) += 23LL;
  }
  v13 = (_QWORD *)a1[8];
  for ( j = (_QWORD *)a1[7]; v13 != j; ++j )
  {
    while ( 1 )
    {
      sub_155C2B0(*j, a2, 0);
      v15 = *(_BYTE **)(a2 + 24);
      if ( *(_BYTE **)(a2 + 16) == v15 )
        break;
      ++j;
      *v15 = 10;
      ++*(_QWORD *)(a2 + 24);
      if ( v13 == j )
        goto LABEL_16;
    }
    sub_16E7EE0(a2, "\n", 1);
  }
LABEL_16:
  v16 = *(__m128i **)(a2 + 24);
  result = (_BYTE *)(*(_QWORD *)(a2 + 16) - (_QWORD)v16);
  if ( (unsigned __int64)result <= 0x14 )
  {
    result = (_BYTE *)sub_16E7EE0(a2, "Interval Successors:\n", 21);
  }
  else
  {
    v18 = _mm_load_si128((const __m128i *)&xmmword_4290FE0);
    v16[1].m128i_i32[0] = 980644463;
    v16[1].m128i_i8[4] = 10;
    *v16 = v18;
    *(_QWORD *)(a2 + 24) += 21LL;
  }
  v19 = (_QWORD *)a1[5];
  for ( k = (_QWORD *)a1[4]; v19 != k; result = (_BYTE *)sub_16E7EE0(a2, "\n", 1) )
  {
    while ( 1 )
    {
      sub_155C2B0(*k, a2, 0);
      result = *(_BYTE **)(a2 + 24);
      if ( *(_BYTE **)(a2 + 16) == result )
        break;
      ++k;
      *result = 10;
      ++*(_QWORD *)(a2 + 24);
      if ( v19 == k )
        return result;
    }
    ++k;
  }
  return result;
}
