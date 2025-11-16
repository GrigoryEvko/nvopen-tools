// Function: sub_1317C00
// Address: 0x1317c00
//
__int64 __fastcall sub_1317C00(__int64 a1)
{
  __int64 v1; // r12
  __int64 v3; // rsi
  __m128i si128; // xmm0

  v1 = qword_50579C0[dword_4F96B5C];
  if ( !v1 )
  {
    v1 = qword_50579C0[dword_4F96B5C];
    v3 = (unsigned int)dword_4F96B5C;
    if ( v1 || (v1 = sub_1300B80(a1, dword_4F96B5C, (__int64)&off_49E8000)) != 0 )
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42875F0);
      *(_QWORD *)(v1 + 78968) = 0;
      *(_DWORD *)(v1 + 78976) = 0;
      *(_WORD *)(v1 + 78980) = 0;
      *(_BYTE *)(v1 + 78982) = 0;
      *(_BYTE *)(v1 + 78983) = 0;
      *(__m128i *)(v1 + 78952) = si128;
      if ( sub_1316FF0(a1, v3) > 0 )
      {
        v3 = v1;
        sub_1315120(a1, v1, 1, 0);
      }
      if ( sub_1317030(a1, v3) > 0 )
        sub_1315120(a1, v1, 2, 0);
    }
  }
  return v1;
}
