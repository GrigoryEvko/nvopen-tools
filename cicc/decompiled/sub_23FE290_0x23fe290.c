// Function: sub_23FE290
// Address: 0x23fe290
//
__int64 __fastcall sub_23FE290(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i v6; // xmm0
  __m128i v7; // xmm1
  __m128i v8; // xmm2
  __int64 v9; // rax
  __int64 result; // rax

  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
  *(_BYTE *)(a1 + 12) = *(_BYTE *)(a2 + 12);
  v6 = _mm_loadu_si128((const __m128i *)(a2 + 24));
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
  *(__m128i *)(a1 + 24) = v6;
  v7 = _mm_loadu_si128((const __m128i *)(a2 + 48));
  v8 = _mm_loadu_si128((const __m128i *)(a2 + 64));
  *(_QWORD *)a1 = &unk_49D9D40;
  v9 = *(_QWORD *)(a2 + 40);
  *(__m128i *)(a1 + 48) = v7;
  *(_QWORD *)(a1 + 40) = v9;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x400000000LL;
  *(__m128i *)(a1 + 64) = v8;
  if ( *(_DWORD *)(a2 + 88) )
    sub_23FE010(a1 + 80, a2 + 80, a3, a4, a5, a6);
  *(_BYTE *)(a1 + 416) = *(_BYTE *)(a2 + 416);
  result = *(unsigned int *)(a2 + 420);
  *(_DWORD *)(a1 + 420) = result;
  return result;
}
