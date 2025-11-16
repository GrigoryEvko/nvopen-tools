// Function: sub_8470D0
// Address: 0x8470d0
//
__int64 __fastcall sub_8470D0(
        __int64 a1,
        const __m128i *a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        _BOOL4 *a6,
        __int64 *a7)
{
  unsigned int v7; // r10d
  char v12; // al
  unsigned int v13; // edx
  __int64 v14; // rdi
  int v16; // eax
  bool v17; // bl
  __int64 v18; // r13
  bool v19; // zf
  unsigned int v21; // [rsp+10h] [rbp-200h]
  __int64 v22[6]; // [rsp+20h] [rbp-1F0h] BYREF
  __m128i v23[3]; // [rsp+50h] [rbp-1C0h] BYREF
  _OWORD v24[9]; // [rsp+80h] [rbp-190h] BYREF
  __m128i v25; // [rsp+110h] [rbp-100h]
  __m128i v26; // [rsp+120h] [rbp-F0h]
  __m128i v27; // [rsp+130h] [rbp-E0h]
  __m128i v28; // [rsp+140h] [rbp-D0h]
  __m128i v29; // [rsp+150h] [rbp-C0h]
  __m128i v30; // [rsp+160h] [rbp-B0h]
  __m128i v31; // [rsp+170h] [rbp-A0h]
  __m128i v32; // [rsp+180h] [rbp-90h]
  __m128i v33; // [rsp+190h] [rbp-80h]
  __m128i v34; // [rsp+1A0h] [rbp-70h]
  __m128i v35; // [rsp+1B0h] [rbp-60h]
  __m128i v36; // [rsp+1C0h] [rbp-50h]
  __m128i v37; // [rsp+1D0h] [rbp-40h]

  v7 = a5;
  v12 = *(_BYTE *)(a1 + 16);
  v13 = (a4 & 0x400) == 0;
  v24[0] = _mm_loadu_si128((const __m128i *)a1);
  v24[1] = _mm_loadu_si128((const __m128i *)(a1 + 16));
  v24[2] = _mm_loadu_si128((const __m128i *)(a1 + 32));
  v24[3] = _mm_loadu_si128((const __m128i *)(a1 + 48));
  v24[4] = _mm_loadu_si128((const __m128i *)(a1 + 64));
  v24[5] = _mm_loadu_si128((const __m128i *)(a1 + 80));
  v24[6] = _mm_loadu_si128((const __m128i *)(a1 + 96));
  v24[7] = _mm_loadu_si128((const __m128i *)(a1 + 112));
  v24[8] = _mm_loadu_si128((const __m128i *)(a1 + 128));
  if ( v12 == 2 )
  {
    v25 = _mm_loadu_si128((const __m128i *)(a1 + 144));
    v26 = _mm_loadu_si128((const __m128i *)(a1 + 160));
    v27 = _mm_loadu_si128((const __m128i *)(a1 + 176));
    v28 = _mm_loadu_si128((const __m128i *)(a1 + 192));
    v29 = _mm_loadu_si128((const __m128i *)(a1 + 208));
    v30 = _mm_loadu_si128((const __m128i *)(a1 + 224));
    v31 = _mm_loadu_si128((const __m128i *)(a1 + 240));
    v32 = _mm_loadu_si128((const __m128i *)(a1 + 256));
    v33 = _mm_loadu_si128((const __m128i *)(a1 + 272));
    v34 = _mm_loadu_si128((const __m128i *)(a1 + 288));
    v35 = _mm_loadu_si128((const __m128i *)(a1 + 304));
    v36 = _mm_loadu_si128((const __m128i *)(a1 + 320));
    v37 = _mm_loadu_si128((const __m128i *)(a1 + 336));
  }
  else if ( v12 == 5 || v12 == 1 )
  {
    v25.m128i_i64[0] = *(_QWORD *)(a1 + 144);
  }
  if ( a6 )
    *a6 = 0;
  *a7 = 0;
  if ( (a4 & 0x80u) != 0
    && dword_4D04474
    && *(_BYTE *)(a1 + 16) == 1
    && (*(_BYTE *)(a1 + 20) & 0x10) == 0
    && (v21 = (a4 & 0x400) == 0, v16 = sub_837960((__m128i *)a1, a2, v13, v21, a4, v22, v23), v13 = v21, v7 = a5, v16)
    || (unsigned int)sub_840D60(
                       (__m128i *)a1,
                       a2,
                       0,
                       (__int64)a2,
                       v13,
                       v13,
                       0,
                       a4,
                       v7,
                       (FILE *)(a1 + 68),
                       (__int64)v22,
                       v23) )
  {
    sub_846560((_QWORD *)a1, (__int64)a2, (__int64)v22, v23, a3, (a4 & 0x20000) == 0, a6, a7, 0);
    v14 = *a7;
    if ( *a7 )
    {
      if ( (*(_BYTE *)(v14 + 51) & 0x40) != 0 && (a4 & 1) != 0 )
      {
        v17 = (a4 & 0x200) != 0;
        v18 = sub_730770(v14, 0);
        v19 = *(_QWORD *)(v18 + 24) == 0;
        *(_BYTE *)(v18 + 49) = v17 | *(_BYTE *)(v18 + 49) & 0xFE;
        if ( !v19 )
        {
          sub_733B20((_QWORD *)v18);
          sub_7340D0(v18, v17, 1);
        }
      }
    }
  }
  return sub_6E4BC0(a1, (__int64)v24);
}
