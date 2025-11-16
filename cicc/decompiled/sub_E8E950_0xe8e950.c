// Function: sub_E8E950
// Address: 0xe8e950
//
__int64 __fastcall sub_E8E950(
        _QWORD *a1,
        int a2,
        int a3,
        __int64 a4,
        __int64 a5,
        int a6,
        int a7,
        __m128i a8,
        __m128i si128)
{
  int v9; // r15d
  int v12; // ebx
  __int64 v13; // rdx
  unsigned int *v14; // rsi
  __m128i v15; // xmm0
  __m128i v18; // [rsp+10h] [rbp-50h] BYREF
  __m128i v19[4]; // [rsp+20h] [rbp-40h] BYREF

  v9 = a4;
  v12 = a5;
  v13 = a1[36];
  v14 = *(unsigned int **)(v13 + 8);
  v18 = _mm_loadu_si128(&a8);
  v19[0] = _mm_loadu_si128(&si128);
  sub_E7BC40(a1, v14, v13, a4, a5);
  v15 = _mm_load_si128(&v18);
  si128 = _mm_load_si128(v19);
  a8 = v15;
  return sub_E97590((_DWORD)a1, a2, a3, v9, v12, a6, a7);
}
