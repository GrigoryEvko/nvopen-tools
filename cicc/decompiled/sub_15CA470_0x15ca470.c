// Function: sub_15CA470
// Address: 0x15ca470
//
void *__fastcall sub_15CA470(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rax
  __m128i v11; // xmm0
  __m128i v13; // [rsp+10h] [rbp-50h] BYREF
  __int64 v14; // [rsp+20h] [rbp-40h]

  v8 = *(_QWORD *)(a5 + 80);
  if ( v8 )
    v8 -= 24;
  v9 = sub_1626D20(a5);
  sub_15C9150((const char **)&v13, v9);
  v10 = v14;
  v11 = _mm_loadu_si128(&v13);
  *(_QWORD *)(a1 + 16) = a5;
  *(_QWORD *)(a1 + 56) = a3;
  *(_QWORD *)(a1 + 40) = v10;
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_QWORD *)(a1 + 96) = 0x400000000LL;
  *(_QWORD *)(a1 + 64) = a4;
  *(_QWORD *)(a1 + 464) = v8;
  *(_DWORD *)(a1 + 8) = 8;
  *(_BYTE *)(a1 + 12) = 2;
  *(_QWORD *)(a1 + 48) = a2;
  *(_BYTE *)(a1 + 80) = 0;
  *(_BYTE *)(a1 + 456) = 0;
  *(_DWORD *)(a1 + 460) = -1;
  *(_QWORD *)a1 = &unk_49ECF98;
  *(__m128i *)(a1 + 24) = v11;
  return &unk_49ECF98;
}
