// Function: sub_258FC10
// Address: 0x258fc10
//
__int64 __fastcall sub_258FC10(__int64 a1, unsigned __int64 a2)
{
  __m128i v2; // rax
  __int64 v3; // rsi
  _QWORD *v4; // rdi
  char v6; // [rsp+Fh] [rbp-21h] BYREF
  __m128i v7; // [rsp+10h] [rbp-20h] BYREF

  v2.m128i_i64[0] = sub_250D2C0(a2, **(_QWORD **)a1);
  v3 = *(_QWORD *)(a1 + 16);
  v4 = *(_QWORD **)(a1 + 8);
  v7 = v2;
  return sub_258F340(v4, v3, &v7, 0, &v6, 0, 0);
}
