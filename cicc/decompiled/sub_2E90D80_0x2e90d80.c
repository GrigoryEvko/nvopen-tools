// Function: sub_2E90D80
// Address: 0x2e90d80
//
_QWORD *__fastcall sub_2E90D80(
        __int64 a1,
        unsigned __int64 *a2,
        unsigned __int8 **a3,
        _WORD *a4,
        char a5,
        __int64 a6,
        const __m128i *a7,
        __int64 a8,
        __int64 a9)
{
  _QWORD *v9; // r14
  __int64 v10; // rdx
  __int64 v11; // r12
  unsigned __int64 v12; // rdx
  __int64 v13; // rax

  v9 = *(_QWORD **)(a1 + 32);
  sub_2E908B0(v9, a3, a4, a5, a7, a8, a6, a9);
  v11 = v10;
  sub_2E31040((__int64 *)(a1 + 40), v10);
  v12 = *a2;
  v13 = *(_QWORD *)v11;
  *(_QWORD *)(v11 + 8) = a2;
  v12 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v11 = v12 | v13 & 7;
  *(_QWORD *)(v12 + 8) = v11;
  *a2 = v11 | *a2 & 7;
  return v9;
}
