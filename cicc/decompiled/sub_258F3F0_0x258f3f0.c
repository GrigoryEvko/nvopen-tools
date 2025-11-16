// Function: sub_258F3F0
// Address: 0x258f3f0
//
__int64 __fastcall sub_258F3F0(__int64 a1, unsigned __int64 a2)
{
  _BYTE *v2; // r12
  __m128i v3; // rax
  __int64 v4; // rsi
  _QWORD *v5; // rdi
  __m128i v7; // [rsp+0h] [rbp-20h] BYREF

  v2 = *(_BYTE **)(a1 + 16);
  v3.m128i_i64[0] = sub_250D2C0(a2, 0);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_QWORD **)a1;
  v7 = v3;
  return sub_258F340(v5, v4, &v7, 1, v2, 0, 0);
}
