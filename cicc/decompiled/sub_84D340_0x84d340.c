// Function: sub_84D340
// Address: 0x84d340
//
_BOOL8 __fastcall sub_84D340(__int64 a1, const __m128i *a2, __m128i *a3)
{
  __m128i *v5; // r10
  _BOOL4 v6; // r13d
  _QWORD *v8; // r13
  int v9; // r14d
  __int8 v10; // dl
  const __m128i *v11; // rax
  __m128i v12; // xmm1
  __m128i v13; // xmm2
  __m128i v14; // xmm3
  __m128i v15; // xmm4
  char v16; // al
  __m128i v17; // xmm5
  __m128i v18; // xmm6
  __m128i v19; // xmm7
  __m128i v20; // xmm0
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rsi
  __m128i *v26; // rdi
  __m128i v27; // xmm2
  __m128i v28; // xmm3
  __m128i v29; // xmm4
  __m128i v30; // xmm5
  __m128i v31; // xmm6
  __m128i v32; // xmm7
  __m128i v33; // xmm1
  __m128i v34; // xmm2
  __m128i v35; // xmm3
  __m128i v36; // xmm4
  __m128i v37; // xmm5
  __m128i v38; // xmm6
  __int64 v39; // [rsp-10h] [rbp-2E0h]
  __m128i *v40; // [rsp-8h] [rbp-2D8h]
  __m128i *v41; // [rsp+8h] [rbp-2C8h] BYREF
  __int64 v42; // [rsp+10h] [rbp-2C0h] BYREF
  __int64 v43; // [rsp+18h] [rbp-2B8h] BYREF
  __m128i v44; // [rsp+20h] [rbp-2B0h] BYREF
  __m128i v45; // [rsp+30h] [rbp-2A0h] BYREF
  _BYTE v46[160]; // [rsp+A0h] [rbp-230h] BYREF
  __m128i v47; // [rsp+140h] [rbp-190h] BYREF
  __m128i v48; // [rsp+150h] [rbp-180h]
  __m128i v49; // [rsp+160h] [rbp-170h]
  __m128i v50; // [rsp+170h] [rbp-160h]
  __m128i v51; // [rsp+180h] [rbp-150h]
  __m128i v52; // [rsp+190h] [rbp-140h]
  __m128i v53; // [rsp+1A0h] [rbp-130h]
  __m128i v54; // [rsp+1B0h] [rbp-120h]
  __m128i v55; // [rsp+1C0h] [rbp-110h]
  __m128i v56; // [rsp+1D0h] [rbp-100h]
  __m128i v57; // [rsp+1E0h] [rbp-F0h]
  __m128i v58; // [rsp+1F0h] [rbp-E0h]
  __m128i v59; // [rsp+200h] [rbp-D0h]
  __m128i v60; // [rsp+210h] [rbp-C0h]
  __m128i v61; // [rsp+220h] [rbp-B0h]
  __m128i v62; // [rsp+230h] [rbp-A0h]
  __m128i v63; // [rsp+240h] [rbp-90h]
  __m128i v64; // [rsp+250h] [rbp-80h]
  __m128i v65; // [rsp+260h] [rbp-70h]
  __m128i v66; // [rsp+270h] [rbp-60h]
  __m128i v67; // [rsp+280h] [rbp-50h]
  __m128i v68; // [rsp+290h] [rbp-40h]

  v41 = (__m128i *)a2;
  if ( (unsigned int)sub_8D3F60(a2) )
  {
    v8 = (_QWORD *)sub_6E3060((const __m128i *)a1);
    v47.m128i_i32[0] = 0;
    v9 = sub_84CF20((__int64)v41, 0, 0, 0, (__int64)v8, (FILE *)(a1 + 68), (const __m128i **)&v41, &v47);
    sub_6E1990(v8);
    if ( !v9 )
      return 0;
    v5 = v41;
    v10 = v41[8].m128i_i8[12];
    if ( v10 == 12 )
    {
      v11 = v41;
      do
      {
        v11 = (const __m128i *)v11[10].m128i_i64[0];
        v10 = v11[8].m128i_i8[12];
      }
      while ( v10 == 12 );
    }
    if ( !v10 )
      return 0;
  }
  else
  {
    v5 = v41;
  }
  v6 = 0;
  sub_838020(a1, 0, v5, 0, word_4D04898, 0, &v45);
  if ( v45.m128i_i32[2] != 7 )
  {
    v12 = _mm_loadu_si128((const __m128i *)(a1 + 16));
    v13 = _mm_loadu_si128((const __m128i *)(a1 + 32));
    v14 = _mm_loadu_si128((const __m128i *)(a1 + 48));
    v15 = _mm_loadu_si128((const __m128i *)(a1 + 64));
    v47 = _mm_loadu_si128((const __m128i *)a1);
    v16 = *(_BYTE *)(a1 + 16);
    v17 = _mm_loadu_si128((const __m128i *)(a1 + 80));
    v18 = _mm_loadu_si128((const __m128i *)(a1 + 96));
    v48 = v12;
    v19 = _mm_loadu_si128((const __m128i *)(a1 + 112));
    v20 = _mm_loadu_si128((const __m128i *)(a1 + 128));
    v49 = v13;
    v50 = v14;
    v51 = v15;
    v52 = v17;
    v53 = v18;
    v54 = v19;
    v55 = v20;
    switch ( v16 )
    {
      case 2:
        v27 = _mm_loadu_si128((const __m128i *)(a1 + 160));
        v28 = _mm_loadu_si128((const __m128i *)(a1 + 176));
        v29 = _mm_loadu_si128((const __m128i *)(a1 + 192));
        v30 = _mm_loadu_si128((const __m128i *)(a1 + 208));
        v31 = _mm_loadu_si128((const __m128i *)(a1 + 224));
        v56 = _mm_loadu_si128((const __m128i *)(a1 + 144));
        v32 = _mm_loadu_si128((const __m128i *)(a1 + 240));
        v33 = _mm_loadu_si128((const __m128i *)(a1 + 256));
        v57 = v27;
        v58 = v28;
        v34 = _mm_loadu_si128((const __m128i *)(a1 + 272));
        v35 = _mm_loadu_si128((const __m128i *)(a1 + 288));
        v59 = v29;
        v36 = _mm_loadu_si128((const __m128i *)(a1 + 304));
        v60 = v30;
        v37 = _mm_loadu_si128((const __m128i *)(a1 + 320));
        v61 = v31;
        v38 = _mm_loadu_si128((const __m128i *)(a1 + 336));
        v62 = v32;
        v63 = v33;
        v64 = v34;
        v65 = v35;
        v66 = v36;
        v67 = v37;
        v68 = v38;
        break;
      case 5:
        v56.m128i_i64[0] = *(_QWORD *)(a1 + 144);
        break;
      case 1:
        v56.m128i_i64[0] = *(_QWORD *)(a1 + 144);
        v56.m128i_i64[0] = (__int64)sub_73B8B0((const __m128i *)v56.m128i_i64[0], 128);
        break;
    }
    sub_6E1DD0(&v42);
    sub_6E1E00(5u, (__int64)v46, 0, 1);
    *(_BYTE *)(qword_4D03C50 + 18LL) |= 0x80u;
    sub_8470D0((__int64)&v47, v41, 1u, 4u, 0, 0, &v43);
    v25 = v39;
    v6 = 0;
    v26 = v40;
    if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 1) == 0 )
    {
      v25 = a1 + 68;
      v44 = 0u;
      v26 = &v44;
      v6 = sub_7A1C60(v43, (FILE *)(a1 + 68), (unsigned __int64)v41, 1, a3, &v44, 0) != 0;
      sub_67E3D0(&v44);
    }
    if ( v48.m128i_i8[0] == 1 )
    {
      v26 = &v47;
      sub_6E4710((__int64)&v47, v25, v21, v22, v23, v24);
    }
    sub_6E2B30((__int64)v26, v25);
    sub_6E1DF0(v42);
  }
  return v6;
}
