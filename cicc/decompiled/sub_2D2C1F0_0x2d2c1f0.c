// Function: sub_2D2C1F0
// Address: 0x2d2c1f0
//
__int64 __fastcall sub_2D2C1F0(_QWORD *a1, const __m128i *a2)
{
  _QWORD *v2; // rcx
  __int64 v5; // rax
  unsigned __int64 v6; // rdx
  __int64 v7; // r12
  __int64 result; // rax
  char v9; // si
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rsi
  __m128i *v12; // rsi
  const __m128i *v13; // [rsp+8h] [rbp-28h] BYREF

  v2 = a1 + 1;
  v5 = a1[2];
  if ( !v5 )
  {
    v7 = (__int64)(a1 + 1);
    goto LABEL_22;
  }
  v6 = a2->m128i_i64[0];
  v7 = (__int64)(a1 + 1);
  do
  {
    while ( 1 )
    {
      if ( *(_QWORD *)(v5 + 32) < v6 )
      {
LABEL_6:
        v5 = *(_QWORD *)(v5 + 24);
        goto LABEL_7;
      }
      if ( *(_QWORD *)(v5 + 32) == v6 )
        break;
LABEL_4:
      v7 = v5;
      v5 = *(_QWORD *)(v5 + 16);
      if ( !v5 )
        goto LABEL_8;
    }
    v9 = *(_BYTE *)(v5 + 56);
    if ( a2[1].m128i_i8[8] )
    {
      if ( !v9 )
        goto LABEL_6;
      v10 = *(_QWORD *)(v5 + 40);
      v11 = a2->m128i_u64[1];
      if ( v10 < v11 || v10 == v11 && *(_QWORD *)(v5 + 48) < a2[1].m128i_i64[0] )
        goto LABEL_6;
      if ( v10 > v11 || a2[1].m128i_i64[0] < *(_QWORD *)(v5 + 48) || *(_QWORD *)(v5 + 64) >= a2[2].m128i_i64[0] )
        goto LABEL_4;
    }
    else if ( v9 || *(_QWORD *)(v5 + 64) >= a2[2].m128i_i64[0] )
    {
      goto LABEL_4;
    }
    v5 = *(_QWORD *)(v5 + 24);
LABEL_7:
    ;
  }
  while ( v5 );
LABEL_8:
  if ( v2 == (_QWORD *)v7 || sub_2A4D650((__int64)a2, v7 + 32) )
  {
LABEL_22:
    v13 = a2;
    v7 = sub_2D2C110(a1, v7, &v13);
    result = *(unsigned int *)(v7 + 72);
    if ( !(_DWORD)result )
      goto LABEL_23;
    return result;
  }
  result = *(unsigned int *)(v7 + 72);
  if ( (_DWORD)result )
    return result;
LABEL_23:
  *(_DWORD *)(v7 + 72) = -858993459 * ((__int64)(a1[7] - a1[6]) >> 3) + 1;
  v12 = (__m128i *)a1[7];
  if ( v12 == (__m128i *)a1[8] )
  {
    sub_2D294F0(a1 + 6, v12, a2);
  }
  else
  {
    if ( v12 )
    {
      *v12 = _mm_loadu_si128(a2);
      v12[1] = _mm_loadu_si128(a2 + 1);
      v12[2].m128i_i64[0] = a2[2].m128i_i64[0];
      v12 = (__m128i *)a1[7];
    }
    a1[7] = (char *)v12 + 40;
  }
  return *(unsigned int *)(v7 + 72);
}
