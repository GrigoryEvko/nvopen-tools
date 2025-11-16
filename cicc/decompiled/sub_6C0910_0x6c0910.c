// Function: sub_6C0910
// Address: 0x6c0910
//
__int64 __fastcall sub_6C0910(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        _QWORD *a4,
        int a5,
        int a6,
        unsigned int a7,
        int a8,
        __int64 a9,
        unsigned int a10,
        __int64 a11,
        _QWORD *a12,
        __m128i *a13,
        _DWORD *a14,
        _QWORD *a15)
{
  char v17; // bl
  __int16 v18; // r13
  __int16 v19; // ax
  _QWORD *v20; // rdx
  int v21; // eax
  __int64 result; // rax
  __int64 v23; // rcx
  char v24; // dl
  int v28; // [rsp+1Ch] [rbp-94h]
  int v29; // [rsp+1Ch] [rbp-94h]
  _BYTE v30[32]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v31; // [rsp+40h] [rbp-70h]
  __int64 v32; // [rsp+70h] [rbp-40h]

  v17 = a9 == 0 && a10 == 0;
  if ( v17 && qword_4D03C50 )
    v17 = ((*(_BYTE *)(qword_4D03C50 + 21LL) >> 3) ^ 1) & 1;
  v18 = *(_WORD *)(qword_4D03C50 + 20LL);
  LOBYTE(v19) = v18 & 0xF7;
  HIBYTE(v19) = ((unsigned __int16)(v18 & 0xFBF7) >> 8) | 4;
  *(_WORD *)(qword_4D03C50 + 20LL) = v19;
  if ( a12 )
    *a12 = 0;
  if ( a14 )
    *a14 = 0;
  if ( a5 )
  {
    a2 = 0;
    a1 = 0;
  }
  sub_831320(a1, a2, v30);
  if ( a6 )
  {
    v30[16] = 1;
  }
  else
  {
    a1 = a7;
    if ( a7 )
      v30[17] = 1;
  }
  if ( a10 )
  {
    v32 = *(_QWORD *)&dword_4F063F8;
    if ( !a15 || a9 )
    {
      v21 = 0;
    }
    else
    {
      *a15 = *(_QWORD *)&dword_4F063F8;
      v21 = 0;
    }
  }
  else if ( a9 )
  {
    a11 = sub_690A60(*(_QWORD *)(a9 + 16), a9, v20);
    v21 = 1;
  }
  else
  {
    if ( !a3 )
      sub_7B8B50(a1, a10, a3, a9 == 0);
    a11 = sub_6BDC10(0x1Cu, qword_4D0495C != 0, 0, 0);
    v32 = *(_QWORD *)&dword_4F063F8;
    *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
    v21 = 1;
    if ( a15 )
      *a15 = *(_QWORD *)&dword_4F063F8;
  }
  if ( !a13 || !a11 || *(_QWORD *)a11 || *(_BYTE *)(a11 + 8) )
  {
    if ( a5 )
    {
      *a12 = a11;
      *a4 = 0;
    }
    else
    {
      v28 = v21;
      sub_849040(a11, v30);
      *a4 = v31;
      if ( v28 )
        sub_6E1990(a11);
    }
  }
  else
  {
    v23 = *(_QWORD *)(a11 + 24);
    *a13 = _mm_loadu_si128((const __m128i *)(v23 + 8));
    a13[1] = _mm_loadu_si128((const __m128i *)(v23 + 24));
    a13[2] = _mm_loadu_si128((const __m128i *)(v23 + 40));
    a13[3] = _mm_loadu_si128((const __m128i *)(v23 + 56));
    a13[4] = _mm_loadu_si128((const __m128i *)(v23 + 72));
    a13[5] = _mm_loadu_si128((const __m128i *)(v23 + 88));
    a13[6] = _mm_loadu_si128((const __m128i *)(v23 + 104));
    a13[7] = _mm_loadu_si128((const __m128i *)(v23 + 120));
    a13[8] = _mm_loadu_si128((const __m128i *)(v23 + 136));
    v24 = *(_BYTE *)(v23 + 24);
    switch ( v24 )
    {
      case 2:
        a13[9] = _mm_loadu_si128((const __m128i *)(v23 + 152));
        a13[10] = _mm_loadu_si128((const __m128i *)(v23 + 168));
        a13[11] = _mm_loadu_si128((const __m128i *)(v23 + 184));
        a13[12] = _mm_loadu_si128((const __m128i *)(v23 + 200));
        a13[13] = _mm_loadu_si128((const __m128i *)(v23 + 216));
        a13[14] = _mm_loadu_si128((const __m128i *)(v23 + 232));
        a13[15] = _mm_loadu_si128((const __m128i *)(v23 + 248));
        a13[16] = _mm_loadu_si128((const __m128i *)(v23 + 264));
        a13[17] = _mm_loadu_si128((const __m128i *)(v23 + 280));
        a13[18] = _mm_loadu_si128((const __m128i *)(v23 + 296));
        a13[19] = _mm_loadu_si128((const __m128i *)(v23 + 312));
        a13[20] = _mm_loadu_si128((const __m128i *)(v23 + 328));
        a13[21] = _mm_loadu_si128((const __m128i *)(v23 + 344));
        break;
      case 5:
        a13[9].m128i_i64[0] = *(_QWORD *)(v23 + 152);
        break;
      case 1:
        a13[9].m128i_i64[0] = *(_QWORD *)(v23 + 152);
        break;
    }
    v29 = v21;
    sub_6E45A0(a13);
    if ( v29 )
      sub_6E1990(a11);
    *a14 = 1;
  }
  if ( v17 )
  {
    unk_4F061D8 = qword_4F063F0;
    if ( !a8 )
      sub_7BE280(28, 18, 0, 0);
  }
  result = *(_WORD *)(qword_4D03C50 + 20LL) & 0xFBF7;
  *(_WORD *)(qword_4D03C50 + 20LL) = result | v18 & 0x408;
  return result;
}
