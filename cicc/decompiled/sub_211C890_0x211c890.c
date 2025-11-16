// Function: sub_211C890
// Address: 0x211c890
//
unsigned __int64 __fastcall sub_211C890(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rsi
  __m128i *v9; // r10
  __m128i v10; // xmm0
  __m128i v11; // xmm1
  __m128i v12; // xmm2
  unsigned __int8 *v13; // rsi
  int v14; // r11d
  char v15; // cl
  __int64 v16; // r9
  unsigned __int64 v17; // rbx
  __m128i *v19; // [rsp+8h] [rbp-98h]
  __int64 v20; // [rsp+8h] [rbp-98h]
  __int64 v21; // [rsp+10h] [rbp-90h] BYREF
  int v22; // [rsp+18h] [rbp-88h]
  _QWORD v23[4]; // [rsp+20h] [rbp-80h] BYREF
  _OWORD v24[6]; // [rsp+40h] [rbp-60h] BYREF

  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(_QWORD *)(a2 + 72);
  v9 = (__m128i *)*a1;
  v10 = _mm_loadu_si128((const __m128i *)v7);
  v11 = _mm_loadu_si128((const __m128i *)(v7 + 40));
  v21 = v8;
  v12 = _mm_loadu_si128((const __m128i *)(v7 + 80));
  v24[0] = v10;
  v24[1] = v11;
  v24[2] = v12;
  if ( v8 )
  {
    v19 = v9;
    sub_1623A60((__int64)&v21, v8, 2);
    v9 = v19;
  }
  v13 = *(unsigned __int8 **)(a2 + 40);
  v14 = 77;
  v22 = *(_DWORD *)(a2 + 64);
  v15 = *v13;
  if ( *v13 != 9 )
  {
    v14 = 78;
    if ( v15 != 10 )
    {
      v14 = 79;
      if ( v15 != 11 )
      {
        v14 = 80;
        if ( v15 != 12 )
        {
          v14 = 462;
          if ( v15 == 13 )
            v14 = 81;
        }
      }
    }
  }
  sub_20BE530(
    (__int64)v23,
    v9,
    a1[1],
    v14,
    *v13,
    *((_QWORD *)v13 + 1),
    v10,
    v11,
    v12,
    (__int64)v24,
    3u,
    0,
    (__int64)&v21,
    0,
    1);
  v16 = v23[0];
  v17 = v23[1];
  if ( v21 )
  {
    v20 = v23[0];
    sub_161E7C0((__int64)&v21, v21);
    v16 = v20;
  }
  return sub_200D960(a1, v16, v17, a3, a4, v10, *(double *)v11.m128i_i64, v12);
}
