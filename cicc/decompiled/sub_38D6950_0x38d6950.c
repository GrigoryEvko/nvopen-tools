// Function: sub_38D6950
// Address: 0x38d6950
//
__int64 __fastcall sub_38D6950(__int64 a1, int a2, int a3, int a4, int a5, int a6, int a7, __m128i a8)
{
  unsigned int *v11; // rsi
  __int64 v13; // rax
  __m128i v14; // xmm0
  __m128i v16; // [rsp+0h] [rbp-50h] BYREF
  int v17; // [rsp+18h] [rbp-38h]
  int v18; // [rsp+1Ch] [rbp-34h]

  v11 = 0;
  v13 = *(unsigned int *)(a1 + 120);
  v14 = _mm_loadu_si128(&a8);
  if ( (_DWORD)v13 )
    v11 = *(unsigned int **)(*(_QWORD *)(a1 + 112) + 32 * v13 - 32);
  v17 = a7;
  v18 = a6;
  v16 = v14;
  sub_38CB070((_QWORD *)a1, v11);
  a8 = _mm_load_si128(&v16);
  return sub_38DBB60(a1, a2, a3, a4, a5, v18, a7);
}
