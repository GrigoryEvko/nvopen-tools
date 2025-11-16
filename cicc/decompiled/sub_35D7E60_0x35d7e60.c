// Function: sub_35D7E60
// Address: 0x35d7e60
//
__int64 __fastcall sub_35D7E60(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v10; // r11
  __int64 i; // rcx
  bool v13; // r10
  __int64 v14; // rcx
  __m128i *v15; // rdx
  __m128i *v16; // rcx
  __int64 result; // rax
  __int64 v18; // r8
  unsigned __int32 v19; // r10d
  bool v20; // cf
  unsigned __int32 v21; // r14d
  unsigned __int64 v22; // r11
  __int64 v23; // rbx
  __int32 v24; // r10d
  __int32 v25; // r8d
  __int64 v26; // rcx
  const __m128i *v27; // rsi
  bool v28; // cf
  bool v29; // zf
  unsigned __int64 v30; // rdx
  const __m128i *v31; // rcx

  v10 = (a3 - 1) / 2;
  if ( a2 >= v10 )
  {
    result = 3 * a2;
    v15 = (__m128i *)(a1 + 24 * a2);
    if ( (a3 & 1) != 0 )
    {
      v22 = a7;
      v23 = a8;
      v24 = a9;
      v25 = HIDWORD(a9);
      goto LABEL_23;
    }
    result = a2;
    goto LABEL_26;
  }
  for ( i = a2; ; i = result )
  {
    result = 2 * (i + 1);
    v15 = (__m128i *)(a1 + 48 * (i + 1));
    v18 = a1 + 24 * (result - 1);
    v19 = v15[1].m128i_u32[1];
    v20 = *(_DWORD *)(v18 + 20) < v19;
    if ( *(_DWORD *)(v18 + 20) != v19 )
      goto LABEL_3;
    v21 = *(_DWORD *)(v18 + 16);
    if ( v15[1].m128i_i32[0] == v21 )
    {
      v20 = v15->m128i_i64[0] < *(_QWORD *)v18;
LABEL_3:
      v13 = v20;
      goto LABEL_4;
    }
    v13 = v15[1].m128i_i32[0] > v21;
LABEL_4:
    v14 = 3 * i;
    if ( v13 )
      v15 = (__m128i *)(a1 + 24 * (result - 1));
    v16 = (__m128i *)(a1 + 8 * v14);
    if ( v13 )
      --result;
    *v16 = _mm_loadu_si128(v15);
    v16[1].m128i_i64[0] = v15[1].m128i_i64[0];
    if ( result >= v10 )
      break;
  }
  if ( (a3 & 1) == 0 )
  {
LABEL_26:
    if ( (a3 - 2) / 2 == result )
    {
      result = 2 * result + 1;
      v31 = (const __m128i *)(a1 + 24 * result);
      *v15 = _mm_loadu_si128(v31);
      v15[1].m128i_i64[0] = v31[1].m128i_i64[0];
      v15 = (__m128i *)v31;
    }
  }
  v22 = a7;
  v23 = a8;
  v24 = a9;
  v25 = HIDWORD(a9);
  v26 = (result - 1) / 2;
  if ( result <= a2 )
    goto LABEL_23;
  while ( 2 )
  {
    v27 = (const __m128i *)(a1 + 24 * v26);
    v28 = v27[1].m128i_i32[1] < HIDWORD(a9);
    v29 = v27[1].m128i_i32[1] == HIDWORD(a9);
    if ( v27[1].m128i_i32[1] != HIDWORD(a9) )
      goto LABEL_17;
    v30 = v27->m128i_i64[0];
    if ( (_DWORD)a9 == v27[1].m128i_i32[0] )
    {
      v28 = a7 < v30;
      v29 = a7 == v30;
LABEL_17:
      result *= 3;
      v15 = (__m128i *)(a1 + 8 * result);
      if ( v28 || v29 )
        goto LABEL_23;
      goto LABEL_18;
    }
    result *= 3;
    v15 = (__m128i *)(a1 + 8 * result);
    if ( (unsigned int)a9 >= v27[1].m128i_i32[0] )
      goto LABEL_23;
LABEL_18:
    *v15 = _mm_loadu_si128(v27);
    v15[1].m128i_i64[0] = v27[1].m128i_i64[0];
    result = v26;
    if ( a2 < v26 )
    {
      v26 = (v26 - 1) / 2;
      continue;
    }
    break;
  }
  v15 = (__m128i *)(a1 + 24 * v26);
LABEL_23:
  v15->m128i_i64[0] = v22;
  v15->m128i_i64[1] = v23;
  v15[1].m128i_i32[0] = v24;
  v15[1].m128i_i32[1] = v25;
  return result;
}
