// Function: sub_6F82C0
// Address: 0x6f82c0
//
__int64 __fastcall sub_6F82C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v7; // al
  __int64 v8; // r13
  __int64 *v9; // rdi
  __m128i v11; // xmm2
  __m128i v12; // xmm3
  __m128i v13; // xmm4
  __m128i v14; // xmm5
  __m128i v15; // xmm6
  __m128i v16; // xmm7
  __m128i v17; // xmm1
  __m128i v18; // xmm2
  __m128i v19; // xmm3
  __m128i v20; // xmm4
  __m128i v21; // xmm5
  __m128i v22; // xmm6
  _BOOL4 v23; // eax
  __int64 v24; // r8
  __int64 v25; // r9
  _BOOL4 *v26; // rsi
  _BYTE *v27; // r14
  _BOOL4 v28; // [rsp+14h] [rbp-18Ch] BYREF
  __int64 v29; // [rsp+18h] [rbp-188h] BYREF
  _OWORD v30[9]; // [rsp+20h] [rbp-180h] BYREF
  __m128i v31; // [rsp+B0h] [rbp-F0h]
  __m128i v32; // [rsp+C0h] [rbp-E0h]
  __m128i v33; // [rsp+D0h] [rbp-D0h]
  __m128i v34; // [rsp+E0h] [rbp-C0h]
  __m128i v35; // [rsp+F0h] [rbp-B0h]
  __m128i v36; // [rsp+100h] [rbp-A0h]
  __m128i v37; // [rsp+110h] [rbp-90h]
  __m128i v38; // [rsp+120h] [rbp-80h]
  __m128i v39; // [rsp+130h] [rbp-70h]
  __m128i v40; // [rsp+140h] [rbp-60h]
  __m128i v41; // [rsp+150h] [rbp-50h]
  __m128i v42; // [rsp+160h] [rbp-40h]
  __m128i v43; // [rsp+170h] [rbp-30h]

  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 && !(unsigned int)sub_693A90() )
  {
    sub_6E68E0(0x1Cu, a1);
    return sub_6E26D0(2, a1);
  }
  v30[0] = _mm_loadu_si128((const __m128i *)a1);
  v7 = *(_BYTE *)(a1 + 16);
  v30[1] = _mm_loadu_si128((const __m128i *)(a1 + 16));
  v30[2] = _mm_loadu_si128((const __m128i *)(a1 + 32));
  v30[3] = _mm_loadu_si128((const __m128i *)(a1 + 48));
  v30[4] = _mm_loadu_si128((const __m128i *)(a1 + 64));
  v30[5] = _mm_loadu_si128((const __m128i *)(a1 + 80));
  v30[6] = _mm_loadu_si128((const __m128i *)(a1 + 96));
  v30[7] = _mm_loadu_si128((const __m128i *)(a1 + 112));
  v30[8] = _mm_loadu_si128((const __m128i *)(a1 + 128));
  if ( v7 != 2 )
  {
    if ( v7 == 5 || v7 == 1 )
      v31.m128i_i64[0] = *(_QWORD *)(a1 + 144);
    goto LABEL_6;
  }
  v11 = _mm_loadu_si128((const __m128i *)(a1 + 160));
  v12 = _mm_loadu_si128((const __m128i *)(a1 + 176));
  v13 = _mm_loadu_si128((const __m128i *)(a1 + 192));
  v14 = _mm_loadu_si128((const __m128i *)(a1 + 208));
  v31 = _mm_loadu_si128((const __m128i *)(a1 + 144));
  v15 = _mm_loadu_si128((const __m128i *)(a1 + 224));
  v16 = _mm_loadu_si128((const __m128i *)(a1 + 240));
  v32 = v11;
  v17 = _mm_loadu_si128((const __m128i *)(a1 + 256));
  v18 = _mm_loadu_si128((const __m128i *)(a1 + 272));
  v33 = v12;
  v19 = _mm_loadu_si128((const __m128i *)(a1 + 288));
  v34 = v13;
  v20 = _mm_loadu_si128((const __m128i *)(a1 + 304));
  v35 = v14;
  v21 = _mm_loadu_si128((const __m128i *)(a1 + 320));
  v36 = v15;
  v22 = _mm_loadu_si128((const __m128i *)(a1 + 336));
  v37 = v16;
  v38 = v17;
  v39 = v18;
  v40 = v19;
  v41 = v20;
  v42 = v21;
  v43 = v22;
  if ( !(unsigned int)sub_72EA80(a1 + 144, &v29, 0) )
  {
LABEL_6:
    v8 = sub_6F6F40((const __m128i *)a1, 0, a3, a4, a5, a6);
    if ( *(_BYTE *)(a1 + 17) == 1 )
    {
      v23 = sub_6ED0A0(a1);
      if ( !v23 )
      {
        if ( *(_BYTE *)(v8 + 24) != 3 )
          goto LABEL_16;
        v27 = *(_BYTE **)(v8 + 56);
        if ( (v27[176] & 1) != 0 && (v27[170] & 0x10) != 0 )
        {
          sub_5EB3F0(*(_QWORD **)(v8 + 56));
          v23 = 0;
        }
        if ( v27[177] == 1 && (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0 )
        {
LABEL_16:
          v26 = &v28;
        }
        else
        {
          if ( (v27[176] & 8) == 0 )
            v23 = v27[136] <= 2u;
          v28 = v23;
          v26 = 0;
        }
        v8 = (__int64)sub_6ED3D0(v8, (unsigned __int64)v26, 0, a1 + 68, v24, v25);
        sub_6E5A30(*(_QWORD *)(a1 + 88), 4, 8);
      }
    }
    v9 = (__int64 *)sub_73DDB0(v8);
    *(__int64 *)((char *)v9 + 28) = *(_QWORD *)(a1 + 68);
    goto LABEL_8;
  }
  v9 = (__int64 *)sub_731250(v29);
LABEL_8:
  sub_6E7150(v9, a1);
  sub_6E4BC0(a1, (__int64)v30);
  *(_QWORD *)(a1 + 88) = 0;
  return sub_6E26D0(2, a1);
}
