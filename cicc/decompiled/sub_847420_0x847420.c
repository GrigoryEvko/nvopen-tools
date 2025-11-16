// Function: sub_847420
// Address: 0x847420
//
__int64 __fastcall sub_847420(__int64 a1, const __m128i *a2, _BYTE *a3, unsigned int a4)
{
  _BYTE *v4; // r13
  __m128i v5; // xmm1
  __m128i v6; // xmm2
  __m128i v7; // xmm3
  __m128i v8; // xmm4
  __m128i v9; // xmm5
  __m128i v10; // xmm6
  char v11; // al
  __m128i v12; // xmm7
  __m128i v13; // xmm0
  __int64 result; // rax
  __m128i v15; // xmm2
  __m128i v16; // xmm3
  __m128i v17; // xmm4
  __m128i v18; // xmm5
  __m128i v19; // xmm6
  __m128i v20; // xmm7
  __m128i v21; // xmm1
  __m128i v22; // xmm2
  __m128i v23; // xmm3
  __m128i v24; // xmm4
  __m128i v25; // xmm5
  __m128i v26; // xmm6
  __int64 *v27; // [rsp+0h] [rbp-1C0h] BYREF
  __int64 v28; // [rsp+8h] [rbp-1B8h] BYREF
  _BYTE v29[48]; // [rsp+10h] [rbp-1B0h] BYREF
  _OWORD v30[9]; // [rsp+40h] [rbp-180h] BYREF
  __m128i v31; // [rsp+D0h] [rbp-F0h]
  __m128i v32; // [rsp+E0h] [rbp-E0h]
  __m128i v33; // [rsp+F0h] [rbp-D0h]
  __m128i v34; // [rsp+100h] [rbp-C0h]
  __m128i v35; // [rsp+110h] [rbp-B0h]
  __m128i v36; // [rsp+120h] [rbp-A0h]
  __m128i v37; // [rsp+130h] [rbp-90h]
  __m128i v38; // [rsp+140h] [rbp-80h]
  __m128i v39; // [rsp+150h] [rbp-70h]
  __m128i v40; // [rsp+160h] [rbp-60h]
  __m128i v41; // [rsp+170h] [rbp-50h]
  __m128i v42; // [rsp+180h] [rbp-40h]
  __m128i v43; // [rsp+190h] [rbp-30h]

  if ( !a3 || (v4 = a3, (a3[16] & 0x88) != 0) )
  {
    v4 = v29;
    result = sub_840D60((__m128i *)a1, a2, 0, (__int64)a2, 1u, 1u, 0, 0x10000u, a4, (FILE *)(a1 + 68), (__int64)v29, 0);
    if ( !(_DWORD)result )
      return result;
  }
  else if ( !*(_QWORD *)a3 )
  {
    sub_6F69D0((_QWORD *)a1, 8u);
  }
  v5 = _mm_loadu_si128((const __m128i *)(a1 + 16));
  v6 = _mm_loadu_si128((const __m128i *)(a1 + 32));
  v7 = _mm_loadu_si128((const __m128i *)(a1 + 48));
  v8 = _mm_loadu_si128((const __m128i *)(a1 + 64));
  v9 = _mm_loadu_si128((const __m128i *)(a1 + 80));
  v30[0] = _mm_loadu_si128((const __m128i *)a1);
  v10 = _mm_loadu_si128((const __m128i *)(a1 + 96));
  v11 = *(_BYTE *)(a1 + 16);
  v30[1] = v5;
  v12 = _mm_loadu_si128((const __m128i *)(a1 + 112));
  v30[2] = v6;
  v13 = _mm_loadu_si128((const __m128i *)(a1 + 128));
  v30[3] = v7;
  v30[4] = v8;
  v30[5] = v9;
  v30[6] = v10;
  v30[7] = v12;
  v30[8] = v13;
  if ( v11 == 2 )
  {
    v15 = _mm_loadu_si128((const __m128i *)(a1 + 160));
    v16 = _mm_loadu_si128((const __m128i *)(a1 + 176));
    v17 = _mm_loadu_si128((const __m128i *)(a1 + 192));
    v18 = _mm_loadu_si128((const __m128i *)(a1 + 208));
    v31 = _mm_loadu_si128((const __m128i *)(a1 + 144));
    v19 = _mm_loadu_si128((const __m128i *)(a1 + 224));
    v20 = _mm_loadu_si128((const __m128i *)(a1 + 240));
    v32 = v15;
    v21 = _mm_loadu_si128((const __m128i *)(a1 + 256));
    v22 = _mm_loadu_si128((const __m128i *)(a1 + 272));
    v33 = v16;
    v23 = _mm_loadu_si128((const __m128i *)(a1 + 288));
    v34 = v17;
    v24 = _mm_loadu_si128((const __m128i *)(a1 + 304));
    v35 = v18;
    v25 = _mm_loadu_si128((const __m128i *)(a1 + 320));
    v36 = v19;
    v26 = _mm_loadu_si128((const __m128i *)(a1 + 336));
    v37 = v20;
    v38 = v21;
    v39 = v22;
    v40 = v23;
    v41 = v24;
    v42 = v25;
    v43 = v26;
  }
  else if ( v11 == 5 || v11 == 1 )
  {
    v31.m128i_i64[0] = *(_QWORD *)(a1 + 144);
  }
  if ( dword_4D041AC || !(unsigned int)sub_8D5830(a2) )
  {
    sub_846560((_QWORD *)a1, (__int64)a2, (__int64)v4, 0, 1, 1, 0, &v28, (__int64 *)&v27);
    sub_6E7150(v27, a1);
  }
  else
  {
    if ( (unsigned int)sub_6E5430() )
      sub_5EB950(8u, 603, (__int64)a2, a1 + 68);
    sub_6E6840(a1);
  }
  sub_6E4BC0(a1, (__int64)v30);
  return sub_6E26D0(2, a1);
}
