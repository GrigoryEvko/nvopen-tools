// Function: sub_E9CD20
// Address: 0xe9cd20
//
__int64 __fastcall sub_E9CD20(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __m128i v4; // xmm0
  __int64 v5; // rcx
  __int64 v6; // rcx
  __int64 v7; // rcx
  __int64 v8; // rcx
  __int64 v9; // rsi
  __int64 v10; // rsi

  result = a1[1];
  if ( result == a1[2] )
  {
    sub_E9C930(a1, (const __m128i *)a1[1], a2);
    return a1[1] - 104;
  }
  else
  {
    if ( result )
    {
      v4 = _mm_loadu_si128((const __m128i *)(a2 + 8));
      *(_QWORD *)result = *(_QWORD *)a2;
      v5 = *(_QWORD *)(a2 + 24);
      *(__m128i *)(result + 8) = v4;
      *(_QWORD *)(result + 24) = v5;
      *(_BYTE *)(result + 32) = *(_BYTE *)(a2 + 32);
      *(_QWORD *)(result + 40) = *(_QWORD *)(a2 + 40);
      v6 = *(_QWORD *)(a2 + 48);
      *(_QWORD *)(a2 + 48) = 0;
      *(_QWORD *)(result + 48) = v6;
      v7 = *(_QWORD *)(a2 + 56);
      *(_QWORD *)(a2 + 56) = 0;
      *(_QWORD *)(result + 56) = v7;
      v8 = *(_QWORD *)(a2 + 64);
      *(_QWORD *)(a2 + 64) = 0;
      *(_QWORD *)(result + 64) = v8;
      *(_QWORD *)(result + 72) = result + 88;
      v9 = *(_QWORD *)(a2 + 72);
      if ( v9 == a2 + 88 )
      {
        *(__m128i *)(result + 88) = _mm_loadu_si128((const __m128i *)(a2 + 88));
      }
      else
      {
        *(_QWORD *)(result + 72) = v9;
        *(_QWORD *)(result + 88) = *(_QWORD *)(a2 + 88);
      }
      v10 = *(_QWORD *)(a2 + 80);
      *(_QWORD *)(a2 + 72) = a2 + 88;
      *(_QWORD *)(a2 + 80) = 0;
      *(_QWORD *)(result + 80) = v10;
      *(_BYTE *)(a2 + 88) = 0;
      result = a1[1];
    }
    a1[1] = result + 104;
  }
  return result;
}
