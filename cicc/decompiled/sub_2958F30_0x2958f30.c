// Function: sub_2958F30
// Address: 0x2958f30
//
unsigned __int64 __fastcall sub_2958F30(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v6; // rax
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rsi
  int v10; // ecx
  unsigned __int64 result; // rax
  __int64 v12; // rdx
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  __m128i v15; // xmm2
  __int64 v16; // rdx
  unsigned __int64 v17; // rbx

  v6 = *(unsigned int *)(a1 + 8);
  v8 = *(_QWORD *)a1;
  v9 = v6 + 1;
  v10 = v6;
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    if ( v8 > a2 || a2 >= v8 + 80 * v6 )
    {
      sub_2958E00(a1, v9, v8, v6, a5, a6);
      v6 = *(unsigned int *)(a1 + 8);
      v8 = *(_QWORD *)a1;
      v10 = *(_DWORD *)(a1 + 8);
    }
    else
    {
      v17 = a2 - v8;
      sub_2958E00(a1, v9, v8, v6, a5, a6);
      v8 = *(_QWORD *)a1;
      v6 = *(unsigned int *)(a1 + 8);
      a2 = *(_QWORD *)a1 + v17;
      v10 = *(_DWORD *)(a1 + 8);
    }
  }
  result = v8 + 80 * v6;
  if ( result )
  {
    *(_QWORD *)result = *(_QWORD *)a2;
    v12 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(a2 + 8) = 0;
    *(_QWORD *)(result + 8) = v12;
    v13 = _mm_loadu_si128((const __m128i *)(a2 + 16));
    v14 = _mm_loadu_si128((const __m128i *)(a2 + 40));
    *(_QWORD *)(result + 32) = *(_QWORD *)(a2 + 32);
    v15 = _mm_loadu_si128((const __m128i *)(a2 + 56));
    v16 = *(_QWORD *)(a2 + 72);
    *(__m128i *)(result + 16) = v13;
    *(__m128i *)(result + 40) = v14;
    *(_QWORD *)(result + 72) = v16;
    *(__m128i *)(result + 56) = v15;
    v10 = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = v10 + 1;
  return result;
}
