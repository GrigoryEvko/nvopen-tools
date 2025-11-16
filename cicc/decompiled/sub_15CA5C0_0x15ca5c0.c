// Function: sub_15CA5C0
// Address: 0x15ca5c0
//
void *__fastcall sub_15CA5C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __m128i v8; // xmm0
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v12; // [rsp+8h] [rbp-58h]
  __m128i v13; // [rsp+10h] [rbp-50h] BYREF
  __int64 v14; // [rsp+20h] [rbp-40h]

  v12 = *(_QWORD *)(a5 + 40);
  sub_15C9090((__int64)&v13, (_QWORD *)(a5 + 48));
  v8 = _mm_loadu_si128(&v13);
  v9 = *(_QWORD *)(*(_QWORD *)(a5 + 40) + 56LL);
  *(_QWORD *)(a1 + 96) = 0x400000000LL;
  *(_QWORD *)(a1 + 48) = a2;
  *(_QWORD *)(a1 + 16) = v9;
  v10 = v14;
  *(_QWORD *)(a1 + 464) = v12;
  *(_QWORD *)(a1 + 40) = v10;
  *(_QWORD *)(a1 + 56) = a3;
  *(_QWORD *)(a1 + 64) = a4;
  *(_DWORD *)(a1 + 8) = 9;
  *(_BYTE *)(a1 + 12) = 2;
  *(_BYTE *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_BYTE *)(a1 + 456) = 0;
  *(_DWORD *)(a1 + 460) = -1;
  *(_QWORD *)a1 = &unk_49ECFC8;
  *(__m128i *)(a1 + 24) = v8;
  return &unk_49ECFC8;
}
