// Function: sub_30F4F20
// Address: 0x30f4f20
//
bool __fastcall sub_30F4F20(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v4; // rsi
  __m128i v5; // xmm4
  __m128i v6; // xmm5
  _OWORD v8[3]; // [rsp+0h] [rbp-C0h] BYREF
  _OWORD v9[3]; // [rsp+30h] [rbp-90h] BYREF
  __m128i v10; // [rsp+60h] [rbp-60h] BYREF
  __m128i v11; // [rsp+70h] [rbp-50h] BYREF
  __m128i v12[4]; // [rsp+80h] [rbp-40h] BYREF

  sub_D66840(&v10, *(_BYTE **)(a1 + 8));
  v4 = *(_BYTE **)(a2 + 8);
  v8[0] = _mm_loadu_si128(&v10);
  v8[1] = _mm_loadu_si128(&v11);
  v8[2] = _mm_loadu_si128(v12);
  sub_D66840(&v10, v4);
  v5 = _mm_loadu_si128(&v11);
  v6 = _mm_loadu_si128(v12);
  v9[0] = _mm_loadu_si128(&v10);
  v9[1] = v5;
  v9[2] = v6;
  return (unsigned __int8)sub_CF4E00(a3, (__int64)v8, (__int64)v9) == 3;
}
