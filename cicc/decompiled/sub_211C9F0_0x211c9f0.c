// Function: sub_211C9F0
// Address: 0x211c9f0
//
unsigned __int64 __fastcall sub_211C9F0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        double a5,
        double a6,
        __m128i a7)
{
  __int64 v10; // rax
  __int64 v11; // rsi
  __m128i *v12; // r10
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  unsigned __int8 *v15; // rsi
  int v16; // r11d
  char v17; // cl
  __int64 v18; // r9
  unsigned __int64 v19; // rbx
  __m128i *v21; // [rsp+8h] [rbp-88h]
  __int64 v22; // [rsp+8h] [rbp-88h]
  __int64 v23; // [rsp+10h] [rbp-80h] BYREF
  int v24; // [rsp+18h] [rbp-78h]
  _OWORD v25[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v26[10]; // [rsp+40h] [rbp-50h] BYREF

  v10 = *(_QWORD *)(a2 + 32);
  v11 = *(_QWORD *)(a2 + 72);
  v12 = (__m128i *)*a1;
  v13 = _mm_loadu_si128((const __m128i *)v10);
  v14 = _mm_loadu_si128((const __m128i *)(v10 + 40));
  v23 = v11;
  v25[0] = v13;
  v25[1] = v14;
  if ( v11 )
  {
    v21 = v12;
    sub_1623A60((__int64)&v23, v11, 2);
    v12 = v21;
  }
  v15 = *(unsigned __int8 **)(a2 + 40);
  v16 = 62;
  v24 = *(_DWORD *)(a2 + 64);
  v17 = *v15;
  if ( *v15 != 9 )
  {
    v16 = 63;
    if ( v17 != 10 )
    {
      v16 = 64;
      if ( v17 != 11 )
      {
        v16 = 65;
        if ( v17 != 12 )
        {
          v16 = 462;
          if ( v17 == 13 )
            v16 = 66;
        }
      }
    }
  }
  sub_20BE530(
    (__int64)v26,
    v12,
    a1[1],
    v16,
    *v15,
    *((_QWORD *)v15 + 1),
    v13,
    v14,
    a7,
    (__int64)v25,
    2u,
    0,
    (__int64)&v23,
    0,
    1);
  v18 = v26[0];
  v19 = v26[1];
  if ( v23 )
  {
    v22 = v26[0];
    sub_161E7C0((__int64)&v23, v23);
    v18 = v22;
  }
  return sub_200D960(a1, v18, v19, a3, a4, v13, *(double *)v14.m128i_i64, a7);
}
