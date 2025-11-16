// Function: sub_981E60
// Address: 0x981e60
//
__int64 __fastcall sub_981E60(
        __int64 a1,
        __m128i *a2,
        __m128i *a3,
        __m128i *a4,
        __int64 (__fastcall *a5)(__m128i *, __m128i *))
{
  char v10; // al
  __int64 v11; // r11
  __int64 v12; // r10
  bool v13; // zf
  __int64 v14; // r9
  __int64 v15; // r8
  __int32 v16; // edi
  __int8 v17; // si
  __int8 v18; // cl
  __int64 v19; // rdx
  __int64 result; // rax
  char v21; // al

  if ( (unsigned __int8)a5(a2, a3) )
  {
    if ( !(unsigned __int8)a5(a3, a4) )
    {
      v10 = a5(a2, a4);
      v11 = *(_QWORD *)a1;
      v12 = *(_QWORD *)(a1 + 8);
      v13 = v10 == 0;
      v14 = *(_QWORD *)(a1 + 16);
      v15 = *(_QWORD *)(a1 + 24);
      v16 = *(_DWORD *)(a1 + 32);
      v17 = *(_BYTE *)(a1 + 36);
      v18 = *(_BYTE *)(a1 + 40);
      v19 = *(_QWORD *)(a1 + 48);
      result = *(_QWORD *)(a1 + 56);
      if ( !v13 )
      {
LABEL_4:
        *(__m128i *)a1 = _mm_loadu_si128(a4);
        *(__m128i *)(a1 + 16) = _mm_loadu_si128(a4 + 1);
        *(__m128i *)(a1 + 32) = _mm_loadu_si128(a4 + 2);
        *(__m128i *)(a1 + 48) = _mm_loadu_si128(a4 + 3);
        a4->m128i_i64[1] = v12;
        a4[1].m128i_i64[0] = v14;
        a4[1].m128i_i64[1] = v15;
        a4[2].m128i_i32[0] = v16;
        a4[2].m128i_i8[4] = v17;
        a4[2].m128i_i8[8] = v18;
        a4[3].m128i_i64[0] = v19;
        a4[3].m128i_i64[1] = result;
        a4->m128i_i64[0] = v11;
        return result;
      }
      goto LABEL_10;
    }
    v11 = *(_QWORD *)a1;
    v12 = *(_QWORD *)(a1 + 8);
    v14 = *(_QWORD *)(a1 + 16);
    *(__m128i *)a1 = _mm_loadu_si128(a3);
    v15 = *(_QWORD *)(a1 + 24);
    v16 = *(_DWORD *)(a1 + 32);
    v17 = *(_BYTE *)(a1 + 36);
    *(__m128i *)(a1 + 16) = _mm_loadu_si128(a3 + 1);
    v18 = *(_BYTE *)(a1 + 40);
    v19 = *(_QWORD *)(a1 + 48);
    result = *(_QWORD *)(a1 + 56);
    *(__m128i *)(a1 + 32) = _mm_loadu_si128(a3 + 2);
    *(__m128i *)(a1 + 48) = _mm_loadu_si128(a3 + 3);
LABEL_8:
    a3->m128i_i64[1] = v12;
    a3[1].m128i_i64[0] = v14;
    a3[1].m128i_i64[1] = v15;
    a3[2].m128i_i32[0] = v16;
    a3[2].m128i_i8[4] = v17;
    a3[2].m128i_i8[8] = v18;
    a3[3].m128i_i64[0] = v19;
    a3[3].m128i_i64[1] = result;
    a3->m128i_i64[0] = v11;
    return result;
  }
  if ( !(unsigned __int8)a5(a2, a4) )
  {
    v21 = a5(a3, a4);
    v11 = *(_QWORD *)a1;
    v12 = *(_QWORD *)(a1 + 8);
    v13 = v21 == 0;
    v14 = *(_QWORD *)(a1 + 16);
    v15 = *(_QWORD *)(a1 + 24);
    v16 = *(_DWORD *)(a1 + 32);
    v17 = *(_BYTE *)(a1 + 36);
    v18 = *(_BYTE *)(a1 + 40);
    v19 = *(_QWORD *)(a1 + 48);
    result = *(_QWORD *)(a1 + 56);
    if ( !v13 )
      goto LABEL_4;
    *(__m128i *)a1 = _mm_loadu_si128(a3);
    *(__m128i *)(a1 + 16) = _mm_loadu_si128(a3 + 1);
    *(__m128i *)(a1 + 32) = _mm_loadu_si128(a3 + 2);
    *(__m128i *)(a1 + 48) = _mm_loadu_si128(a3 + 3);
    goto LABEL_8;
  }
  v11 = *(_QWORD *)a1;
  v12 = *(_QWORD *)(a1 + 8);
  v14 = *(_QWORD *)(a1 + 16);
  v15 = *(_QWORD *)(a1 + 24);
  v16 = *(_DWORD *)(a1 + 32);
  v17 = *(_BYTE *)(a1 + 36);
  v18 = *(_BYTE *)(a1 + 40);
  v19 = *(_QWORD *)(a1 + 48);
  result = *(_QWORD *)(a1 + 56);
LABEL_10:
  *(__m128i *)a1 = _mm_loadu_si128(a2);
  *(__m128i *)(a1 + 16) = _mm_loadu_si128(a2 + 1);
  *(__m128i *)(a1 + 32) = _mm_loadu_si128(a2 + 2);
  *(__m128i *)(a1 + 48) = _mm_loadu_si128(a2 + 3);
  a2->m128i_i64[1] = v12;
  a2[1].m128i_i64[0] = v14;
  a2[1].m128i_i64[1] = v15;
  a2[2].m128i_i32[0] = v16;
  a2[2].m128i_i8[4] = v17;
  a2[2].m128i_i8[8] = v18;
  a2[3].m128i_i64[0] = v19;
  a2[3].m128i_i64[1] = result;
  a2->m128i_i64[0] = v11;
  return result;
}
