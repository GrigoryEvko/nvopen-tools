// Function: sub_696090
// Address: 0x696090
//
__int64 __fastcall sub_696090(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v4; // rsi
  char v6; // cl
  __int64 v7; // rax
  char i; // dl
  char v9; // dl
  __int64 v10; // rax
  __m128i v11; // xmm1
  __m128i v12; // xmm2
  __m128i v13; // xmm3
  __m128i v14; // xmm4
  __m128i v15; // xmm5
  __m128i v16; // xmm6
  __m128i v17; // xmm7
  __m128i v18; // xmm0
  __m128i v20; // xmm2
  __m128i v21; // xmm3
  __m128i v22; // xmm4
  __m128i v23; // xmm5
  __m128i v24; // xmm6
  __m128i v25; // xmm7
  __m128i v26; // xmm1
  __m128i v27; // xmm2
  __m128i v28; // xmm3
  __m128i v29; // xmm4
  __m128i v30; // xmm5
  __m128i v31; // xmm6
  unsigned int v32; // [rsp+1Ch] [rbp-224h] BYREF
  _BYTE v33[18]; // [rsp+20h] [rbp-220h] BYREF
  char v34; // [rsp+32h] [rbp-20Eh]
  __m128i v35; // [rsp+C0h] [rbp-180h] BYREF
  __m128i v36; // [rsp+D0h] [rbp-170h]
  __m128i v37; // [rsp+E0h] [rbp-160h]
  __m128i v38; // [rsp+F0h] [rbp-150h]
  __m128i v39; // [rsp+100h] [rbp-140h]
  __m128i v40; // [rsp+110h] [rbp-130h]
  __m128i v41; // [rsp+120h] [rbp-120h]
  __m128i v42; // [rsp+130h] [rbp-110h]
  __m128i v43; // [rsp+140h] [rbp-100h]
  __m128i v44; // [rsp+150h] [rbp-F0h]
  __m128i v45; // [rsp+160h] [rbp-E0h]
  __m128i v46; // [rsp+170h] [rbp-D0h]
  __m128i v47; // [rsp+180h] [rbp-C0h]
  __m128i v48; // [rsp+190h] [rbp-B0h]
  __m128i v49; // [rsp+1A0h] [rbp-A0h]
  __m128i v50; // [rsp+1B0h] [rbp-90h]
  __m128i v51; // [rsp+1C0h] [rbp-80h]
  __m128i v52; // [rsp+1D0h] [rbp-70h]
  __m128i v53; // [rsp+1E0h] [rbp-60h]
  __m128i v54; // [rsp+1F0h] [rbp-50h]
  __m128i v55; // [rsp+200h] [rbp-40h]
  __m128i v56; // [rsp+210h] [rbp-30h]

  v4 = v33;
  sub_6E1E00(2, v33, 0, 0);
  v34 |= 1u;
  sub_7296C0(&v32);
  v6 = *(_BYTE *)(a1 + 24);
  if ( !v6 )
    goto LABEL_18;
  v7 = *(_QWORD *)(a1 + 8);
  for ( i = *(_BYTE *)(v7 + 140); i == 12; i = *(_BYTE *)(v7 + 140) )
    v7 = *(_QWORD *)(v7 + 160);
  if ( !i )
    goto LABEL_18;
  if ( !a2 )
    goto LABEL_10;
  v9 = *(_BYTE *)(a2 + 140);
  if ( v9 == 12 )
  {
    v10 = a2;
    do
    {
      v10 = *(_QWORD *)(v10 + 160);
      v9 = *(_BYTE *)(v10 + 140);
    }
    while ( v9 == 12 );
  }
  if ( v9 )
  {
LABEL_10:
    v11 = _mm_loadu_si128((const __m128i *)(a1 + 24));
    v12 = _mm_loadu_si128((const __m128i *)(a1 + 40));
    v13 = _mm_loadu_si128((const __m128i *)(a1 + 56));
    v14 = _mm_loadu_si128((const __m128i *)(a1 + 72));
    v15 = _mm_loadu_si128((const __m128i *)(a1 + 88));
    v35 = _mm_loadu_si128((const __m128i *)(a1 + 8));
    v16 = _mm_loadu_si128((const __m128i *)(a1 + 104));
    v17 = _mm_loadu_si128((const __m128i *)(a1 + 120));
    v36 = v11;
    v18 = _mm_loadu_si128((const __m128i *)(a1 + 136));
    v37 = v12;
    v38 = v13;
    v39 = v14;
    v40 = v15;
    v41 = v16;
    v42 = v17;
    v43 = v18;
    if ( v6 == 2 )
    {
      v20 = _mm_loadu_si128((const __m128i *)(a1 + 168));
      v21 = _mm_loadu_si128((const __m128i *)(a1 + 184));
      v22 = _mm_loadu_si128((const __m128i *)(a1 + 200));
      v23 = _mm_loadu_si128((const __m128i *)(a1 + 216));
      v24 = _mm_loadu_si128((const __m128i *)(a1 + 232));
      v44 = _mm_loadu_si128((const __m128i *)(a1 + 152));
      v25 = _mm_loadu_si128((const __m128i *)(a1 + 248));
      v26 = _mm_loadu_si128((const __m128i *)(a1 + 264));
      v45 = v20;
      v46 = v21;
      v27 = _mm_loadu_si128((const __m128i *)(a1 + 280));
      v28 = _mm_loadu_si128((const __m128i *)(a1 + 296));
      v47 = v22;
      v29 = _mm_loadu_si128((const __m128i *)(a1 + 312));
      v48 = v23;
      v30 = _mm_loadu_si128((const __m128i *)(a1 + 328));
      v49 = v24;
      v31 = _mm_loadu_si128((const __m128i *)(a1 + 344));
      v50 = v25;
      v51 = v26;
      v52 = v27;
      v53 = v28;
      v54 = v29;
      v55 = v30;
      v56 = v31;
    }
    else if ( v6 == 5 || v6 == 1 )
    {
      v44.m128i_i64[0] = *(_QWORD *)(a1 + 152);
    }
    if ( v36.m128i_i8[0] == 1 )
      v44.m128i_i64[0] = sub_73B8B0(v44.m128i_i64[0], 0);
    v40.m128i_i64[1] = sub_6E15C0(*(_QWORD *)(a1 + 96));
    if ( a2 )
    {
      if ( !(unsigned int)sub_8D2FB0(a2) && (unsigned int)sub_8D2FB0(v35.m128i_i64[0]) )
        sub_6F82C0(&v35);
    }
    else
    {
      a2 = v35.m128i_i64[0];
    }
    v4 = (_BYTE *)a2;
    sub_695B00(&v35, a2, a3);
  }
  else
  {
LABEL_18:
    sub_72C970(a3);
  }
  sub_6E2AC0(a3);
  sub_6E2B30(a3, v4);
  return sub_729730(v32);
}
