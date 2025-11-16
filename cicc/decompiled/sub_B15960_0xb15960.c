// Function: sub_B15960
// Address: 0xb15960
//
void *__fastcall sub_B15960(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char a6, int a7)
{
  __int64 v10; // rax
  __m128i v11; // xmm0
  __m128i v14[4]; // [rsp+10h] [rbp-40h] BYREF

  v10 = sub_B92180(a2);
  sub_B15890(v14, v10);
  *(_QWORD *)(a1 + 16) = a2;
  v11 = _mm_loadu_si128(v14);
  *(_QWORD *)(a1 + 40) = a2;
  *(_DWORD *)(a1 + 8) = a7;
  *(_QWORD *)(a1 + 48) = a3;
  *(_QWORD *)(a1 + 56) = a4;
  *(_QWORD *)(a1 + 64) = a5;
  *(_BYTE *)(a1 + 12) = a6;
  *(_QWORD *)a1 = &unk_49D9BE8;
  *(__m128i *)(a1 + 24) = v11;
  return &unk_49D9BE8;
}
