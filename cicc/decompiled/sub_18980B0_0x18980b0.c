// Function: sub_18980B0
// Address: 0x18980b0
//
__int64 __fastcall sub_18980B0(__int64 a1, __int64 a2)
{
  __m128i v2; // xmm0
  __int64 v3; // rax
  __m128i v4; // xmm1
  __int64 v5; // rax
  __int64 result; // rax

  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
  *(_BYTE *)(a1 + 12) = *(_BYTE *)(a2 + 12);
  v2 = _mm_loadu_si128((const __m128i *)(a2 + 24));
  *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
  v3 = *(_QWORD *)(a2 + 40);
  *(__m128i *)(a1 + 24) = v2;
  *(_QWORD *)(a1 + 40) = v3;
  v4 = _mm_loadu_si128((const __m128i *)(a2 + 56));
  *(_QWORD *)a1 = &unk_49ECF68;
  v5 = *(_QWORD *)(a2 + 48);
  *(__m128i *)(a1 + 56) = v4;
  *(_QWORD *)(a1 + 48) = v5;
  LOBYTE(v5) = *(_BYTE *)(a2 + 80);
  *(_BYTE *)(a1 + 80) = v5;
  if ( (_BYTE)v5 )
    *(_QWORD *)(a1 + 72) = *(_QWORD *)(a2 + 72);
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_QWORD *)(a1 + 96) = 0x400000000LL;
  if ( *(_DWORD *)(a2 + 96) )
    sub_1897E20(a1 + 88, a2 + 88);
  *(_BYTE *)(a1 + 456) = *(_BYTE *)(a2 + 456);
  result = *(unsigned int *)(a2 + 460);
  *(_DWORD *)(a1 + 460) = result;
  return result;
}
