// Function: sub_35CFE10
// Address: 0x35cfe10
//
__int64 __fastcall sub_35CFE10(int *a1)
{
  int v1; // r9d
  __int64 v2; // rsi
  bool v3; // zf
  unsigned __int8 v4; // dl
  unsigned __int8 v5; // al
  __int64 v6; // rax
  int v7; // eax
  __m128i v8; // xmm0
  __m128i v9; // xmm1
  __m128i v10; // xmm2
  __m128i v11; // xmm3
  __int64 result; // rax
  __m128i v13; // [rsp+0h] [rbp-30h] BYREF
  __m128i v14; // [rsp+10h] [rbp-20h] BYREF
  __int64 v15; // [rsp+20h] [rbp-10h]

  v1 = *a1;
  v2 = *((_QWORD *)a1 + 2) + *((_QWORD *)a1 + 3);
  v15 = *((_QWORD *)a1 + 4);
  v3 = a1[8] == 2;
  v13 = _mm_loadu_si128((const __m128i *)a1);
  v14 = _mm_loadu_si128((const __m128i *)a1 + 1);
  v4 = !v3;
  while ( 1 )
  {
    v5 = *(a1 - 2) != 2;
    if ( v5 >= v4 )
    {
      if ( v5 != v4 )
        break;
      v6 = *((_QWORD *)a1 - 2) + *((_QWORD *)a1 - 3);
      if ( v6 >= v2 && (v6 != v2 || *(a1 - 10) >= v1) )
        break;
    }
    v7 = *(a1 - 2);
    v8 = _mm_loadu_si128((const __m128i *)(a1 - 10));
    a1 -= 10;
    v9 = _mm_loadu_si128((const __m128i *)a1 + 1);
    a1[18] = v7;
    LOBYTE(v7) = *((_BYTE *)a1 + 36);
    *(__m128i *)(a1 + 10) = v8;
    *(__m128i *)(a1 + 14) = v9;
    *((_BYTE *)a1 + 76) = v7;
  }
  v10 = _mm_loadu_si128(&v13);
  v11 = _mm_loadu_si128(&v14);
  a1[8] = v15;
  result = BYTE4(v15);
  *((__m128i *)a1 + 1) = v11;
  *((_BYTE *)a1 + 36) = result;
  *(__m128i *)a1 = v10;
  return result;
}
