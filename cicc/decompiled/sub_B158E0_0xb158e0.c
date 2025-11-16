// Function: sub_B158E0
// Address: 0xb158e0
//
void *__fastcall sub_B158E0(__int64 a1, __int64 a2, __int64 a3, const __m128i *a4, char a5)
{
  __m128i v7; // xmm0
  __int64 v9; // rax
  __m128i v10[3]; // [rsp+0h] [rbp-30h] BYREF

  if ( a4->m128i_i64[0] )
  {
    v10[0] = _mm_loadu_si128(a4);
  }
  else
  {
    v9 = sub_B92180(a3);
    sub_B15890(v10, v9);
  }
  v7 = _mm_loadu_si128(v10);
  *(_BYTE *)(a1 + 12) = a5;
  *(_QWORD *)(a1 + 16) = a3;
  *(_QWORD *)(a1 + 40) = a2;
  *(_DWORD *)(a1 + 8) = 3;
  *(_QWORD *)a1 = &unk_49D9BB8;
  *(__m128i *)(a1 + 24) = v7;
  return &unk_49D9BB8;
}
