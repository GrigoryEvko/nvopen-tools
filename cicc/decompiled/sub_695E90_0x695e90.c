// Function: sub_695E90
// Address: 0x695e90
//
__int64 __fastcall sub_695E90(__int64 a1, __int64 a2)
{
  __m128i v2; // xmm1
  __m128i v3; // xmm2
  __m128i v4; // xmm3
  __m128i v5; // xmm4
  char v6; // al
  __m128i v7; // xmm5
  __m128i v8; // xmm6
  __m128i v9; // xmm7
  __m128i v10; // xmm0
  unsigned int v11; // r12d
  __m128i v13; // xmm2
  __m128i v14; // xmm3
  __m128i v15; // xmm4
  __m128i v16; // xmm5
  __m128i v17; // xmm6
  __m128i v18; // xmm7
  __m128i v19; // xmm1
  __m128i v20; // xmm2
  __m128i v21; // xmm3
  __m128i v22; // xmm4
  __m128i v23; // xmm5
  __m128i v24; // xmm6
  _BYTE v25[18]; // [rsp+0h] [rbp-220h] BYREF
  char v26; // [rsp+12h] [rbp-20Eh]
  _OWORD v27[5]; // [rsp+A0h] [rbp-180h] BYREF
  __m128i v28; // [rsp+F0h] [rbp-130h]
  __m128i v29; // [rsp+100h] [rbp-120h]
  __m128i v30; // [rsp+110h] [rbp-110h]
  __m128i v31; // [rsp+120h] [rbp-100h]
  __m128i v32; // [rsp+130h] [rbp-F0h]
  __m128i v33; // [rsp+140h] [rbp-E0h]
  __m128i v34; // [rsp+150h] [rbp-D0h]
  __m128i v35; // [rsp+160h] [rbp-C0h]
  __m128i v36; // [rsp+170h] [rbp-B0h]
  __m128i v37; // [rsp+180h] [rbp-A0h]
  __m128i v38; // [rsp+190h] [rbp-90h]
  __m128i v39; // [rsp+1A0h] [rbp-80h]
  __m128i v40; // [rsp+1B0h] [rbp-70h]
  __m128i v41; // [rsp+1C0h] [rbp-60h]
  __m128i v42; // [rsp+1D0h] [rbp-50h]
  __m128i v43; // [rsp+1E0h] [rbp-40h]
  __m128i v44; // [rsp+1F0h] [rbp-30h]

  sub_6E1E00(2, v25, 0, 0);
  v2 = _mm_loadu_si128((const __m128i *)(a1 + 24));
  v3 = _mm_loadu_si128((const __m128i *)(a1 + 40));
  v4 = _mm_loadu_si128((const __m128i *)(a1 + 56));
  v5 = _mm_loadu_si128((const __m128i *)(a1 + 72));
  v27[0] = _mm_loadu_si128((const __m128i *)(a1 + 8));
  v6 = *(_BYTE *)(a1 + 24);
  v7 = _mm_loadu_si128((const __m128i *)(a1 + 88));
  v8 = _mm_loadu_si128((const __m128i *)(a1 + 104));
  v27[1] = v2;
  v9 = _mm_loadu_si128((const __m128i *)(a1 + 120));
  v10 = _mm_loadu_si128((const __m128i *)(a1 + 136));
  v27[2] = v3;
  v26 |= 1u;
  v27[3] = v4;
  v27[4] = v5;
  v28 = v7;
  v29 = v8;
  v30 = v9;
  v31 = v10;
  if ( v6 == 2 )
  {
    v13 = _mm_loadu_si128((const __m128i *)(a1 + 168));
    v14 = _mm_loadu_si128((const __m128i *)(a1 + 184));
    v15 = _mm_loadu_si128((const __m128i *)(a1 + 200));
    v16 = _mm_loadu_si128((const __m128i *)(a1 + 216));
    v17 = _mm_loadu_si128((const __m128i *)(a1 + 232));
    v32 = _mm_loadu_si128((const __m128i *)(a1 + 152));
    v18 = _mm_loadu_si128((const __m128i *)(a1 + 248));
    v19 = _mm_loadu_si128((const __m128i *)(a1 + 264));
    v33 = v13;
    v34 = v14;
    v20 = _mm_loadu_si128((const __m128i *)(a1 + 280));
    v21 = _mm_loadu_si128((const __m128i *)(a1 + 296));
    v35 = v15;
    v22 = _mm_loadu_si128((const __m128i *)(a1 + 312));
    v36 = v16;
    v23 = _mm_loadu_si128((const __m128i *)(a1 + 328));
    v37 = v17;
    v24 = _mm_loadu_si128((const __m128i *)(a1 + 344));
    v38 = v18;
    v39 = v19;
    v40 = v20;
    v41 = v21;
    v42 = v22;
    v43 = v23;
    v44 = v24;
  }
  else if ( v6 == 5 || v6 == 1 )
  {
    v32.m128i_i64[0] = *(_QWORD *)(a1 + 152);
  }
  v28.m128i_i64[1] = 0;
  if ( !(unsigned int)sub_8D2FB0(a2) && (unsigned int)sub_8D2FB0(*(_QWORD *)&v27[0]) )
    sub_6F82C0(v27);
  sub_6E6B60(v27, 1);
  v11 = sub_84D700(v27, a2);
  sub_6E2B30(v27, a2);
  return v11;
}
