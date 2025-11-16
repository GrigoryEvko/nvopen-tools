// Function: sub_3805620
// Address: 0x3805620
//
__m128i *__fastcall sub_3805620(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r14
  __int64 v5; // rsi
  __int64 *v6; // r15
  __int16 v7; // r8
  unsigned __int8 v8; // cl
  unsigned __int8 v9; // dl
  unsigned __int16 v10; // si
  __m128i *v11; // r14
  unsigned __int8 v13; // [rsp+3h] [rbp-6Dh]
  __int16 v14; // [rsp+4h] [rbp-6Ch]
  __int64 v15; // [rsp+10h] [rbp-60h] BYREF
  int v16; // [rsp+18h] [rbp-58h]
  _OWORD v17[5]; // [rsp+20h] [rbp-50h] BYREF

  v3 = *(_QWORD *)(a2 + 112);
  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = *(__int64 **)(a1 + 8);
  v15 = v5;
  v17[1] = _mm_loadu_si128((const __m128i *)(v3 + 56));
  v7 = *(_WORD *)(v3 + 32);
  v8 = *(_BYTE *)(v3 + 34);
  v17[0] = _mm_loadu_si128((const __m128i *)(v3 + 40));
  if ( v5 )
  {
    v13 = v8;
    v14 = v7;
    sub_B96E90((__int64)&v15, v5, 1);
    v8 = v13;
    v7 = v14;
  }
  v9 = *(_BYTE *)(a2 + 33);
  v10 = *(_WORD *)(a2 + 32);
  v16 = *(_DWORD *)(a2 + 72);
  v11 = sub_33EA290(
          v6,
          (v10 >> 7) & 7,
          (v9 >> 2) & 3,
          6u,
          0,
          (__int64)&v15,
          *(_OWORD *)v4,
          *(_QWORD *)(v4 + 40),
          *(_QWORD *)(v4 + 48),
          *(_OWORD *)(v4 + 80),
          *(_OWORD *)v3,
          *(_QWORD *)(v3 + 16),
          6,
          0,
          v8,
          v7,
          (__int64)v17,
          0);
  if ( v15 )
    sub_B91220((__int64)&v15, v15);
  sub_3760E70(a1, a2, 1, (unsigned __int64)v11, 1);
  return v11;
}
