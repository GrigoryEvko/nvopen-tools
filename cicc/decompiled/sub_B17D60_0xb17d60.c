// Function: sub_B17D60
// Address: 0xb17d60
//
void *__fastcall sub_B17D60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __m128i v5; // xmm0
  __int64 v6; // rax
  __m128i v8[3]; // [rsp+0h] [rbp-30h] BYREF

  sub_B157E0((__int64)v8, (_QWORD *)(a2 + 48));
  v4 = *(_QWORD *)(a2 + 40);
  v5 = _mm_loadu_si128(v8);
  *(_QWORD *)(a1 + 40) = a3;
  v6 = *(_QWORD *)(v4 + 72);
  *(_DWORD *)(a1 + 8) = 27;
  *(_BYTE *)(a1 + 12) = 1;
  *(_QWORD *)(a1 + 16) = v6;
  *(__m128i *)(a1 + 24) = v5;
  *(_QWORD *)a1 = &unk_49D9F18;
  return &unk_49D9F18;
}
