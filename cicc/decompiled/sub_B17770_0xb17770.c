// Function: sub_B17770
// Address: 0xb17770
//
void *__fastcall sub_B17770(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r13
  __int64 v9; // rax
  __m128i v10; // xmm0
  __m128i v12[4]; // [rsp+10h] [rbp-40h] BYREF

  if ( a5 + 72 == (*(_QWORD *)(a5 + 72) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v8 = 0;
  }
  else
  {
    v8 = *(_QWORD *)(a5 + 80);
    if ( v8 )
      v8 -= 24;
  }
  v9 = sub_B92180(a5);
  sub_B15890(v12, v9);
  v10 = _mm_loadu_si128(v12);
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x400000000LL;
  *(_QWORD *)(a1 + 16) = a5;
  *(_QWORD *)(a1 + 48) = a3;
  *(_QWORD *)(a1 + 56) = a4;
  *(_QWORD *)(a1 + 424) = v8;
  *(_DWORD *)(a1 + 8) = 14;
  *(_BYTE *)(a1 + 12) = 2;
  *(_QWORD *)(a1 + 40) = a2;
  *(_BYTE *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 416) = 0;
  *(_DWORD *)(a1 + 420) = -1;
  *(_QWORD *)a1 = &unk_49D9DB0;
  *(__m128i *)(a1 + 24) = v10;
  return &unk_49D9DB0;
}
