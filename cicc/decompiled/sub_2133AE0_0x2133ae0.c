// Function: sub_2133AE0
// Address: 0x2133ae0
//
void __fastcall sub_2133AE0(
        __int64 a1,
        unsigned __int64 a2,
        __m128i a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // r14
  __int128 v18; // rax
  __int64 *v19; // r14
  const __m128i *v20; // r9
  const __m128i *v21; // r9
  unsigned int v22; // [rsp+10h] [rbp-50h]
  const void **v23; // [rsp+18h] [rbp-48h]
  __int64 v24; // [rsp+20h] [rbp-40h] BYREF
  int v25; // [rsp+28h] [rbp-38h]

  v11 = *(_QWORD *)(a2 + 72);
  v24 = v11;
  if ( v11 )
    sub_1623A60((__int64)&v24, v11, 2);
  v12 = *(_QWORD *)(a1 + 8);
  v23 = *(const void ***)(a2 + 96);
  v13 = *(unsigned __int8 *)(a2 + 88);
  v22 = *(unsigned __int8 *)(a2 + 88);
  v25 = *(_DWORD *)(a2 + 64);
  v14 = sub_1D25E70(v12, v13, (__int64)v23, 2, 0, a9, 1, 0);
  v16 = v15;
  v17 = v14;
  *(_QWORD *)&v18 = sub_1D38BB0(*(_QWORD *)(a1 + 8), 0, (__int64)&v24, v22, v23, 0, a3, a4, a5, 0);
  v19 = sub_1D24690(
          *(_QWORD **)(a1 + 8),
          0xDEu,
          (__int64)&v24,
          *(_BYTE *)(a2 + 88),
          *(_QWORD *)(a2 + 96),
          *(_QWORD *)(a2 + 104),
          v17,
          v16,
          *(_OWORD *)*(_QWORD *)(a2 + 32),
          *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
          v18,
          v18);
  sub_2013400(a1, a2, 0, (__int64)v19, 0, v20);
  sub_2013400(a1, a2, 1, (__int64)v19, (__m128i *)2, v21);
  if ( v24 )
    sub_161E7C0((__int64)&v24, v24);
}
