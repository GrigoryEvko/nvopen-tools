// Function: sub_6FC070
// Address: 0x6fc070
//
__int64 __fastcall sub_6FC070(__int64 a1, __int64 a2, int a3, int a4, int a5)
{
  int v7; // r12d
  __int64 v8; // rax
  char v9; // dl
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  __m128i v12; // xmm3
  __m128i v13; // xmm4
  __m128i v14; // xmm5
  __m128i v15; // xmm6
  __m128i v16; // xmm7
  __m128i v17; // xmm0
  __int64 result; // rax
  __m128i v19; // xmm2
  __m128i v20; // xmm3
  __m128i v21; // xmm4
  __m128i v22; // xmm5
  __m128i v23; // xmm6
  __m128i v24; // xmm7
  __m128i v25; // xmm1
  __m128i v26; // xmm2
  __m128i v27; // xmm3
  __m128i v28; // xmm4
  __m128i v29; // xmm5
  __m128i v30; // xmm6
  __int64 v31; // [rsp+0h] [rbp-1E0h]
  _BYTE v34[4]; // [rsp+1Ch] [rbp-1C4h] BYREF
  _BYTE v35[4]; // [rsp+20h] [rbp-1C0h] BYREF
  int v36; // [rsp+24h] [rbp-1BCh] BYREF
  int v37; // [rsp+28h] [rbp-1B8h] BYREF
  _BYTE v38[4]; // [rsp+2Ch] [rbp-1B4h] BYREF
  _BYTE v39[32]; // [rsp+30h] [rbp-1B0h] BYREF
  _OWORD v40[9]; // [rsp+50h] [rbp-190h] BYREF
  __m128i v41; // [rsp+E0h] [rbp-100h]
  __m128i v42; // [rsp+F0h] [rbp-F0h]
  __m128i v43; // [rsp+100h] [rbp-E0h]
  __m128i v44; // [rsp+110h] [rbp-D0h]
  __m128i v45; // [rsp+120h] [rbp-C0h]
  __m128i v46; // [rsp+130h] [rbp-B0h]
  __m128i v47; // [rsp+140h] [rbp-A0h]
  __m128i v48; // [rsp+150h] [rbp-90h]
  __m128i v49; // [rsp+160h] [rbp-80h]
  __m128i v50; // [rsp+170h] [rbp-70h]
  __m128i v51; // [rsp+180h] [rbp-60h]
  __m128i v52; // [rsp+190h] [rbp-50h]
  __m128i v53; // [rsp+1A0h] [rbp-40h]

  v7 = sub_8D2FB0(a1);
  v31 = *(_QWORD *)(a2 + 136);
  v8 = sub_82C9F0(
         v31,
         (*(_BYTE *)(a2 + 19) & 8) != 0,
         *(_QWORD *)(a2 + 104),
         *(_BYTE *)(a2 + 17) == 3,
         a1,
         a3,
         a4,
         (__int64)v34,
         (__int64)v39,
         (__int64)&v37,
         (__int64)&v36,
         (__int64)v35);
  if ( v8 )
  {
    v9 = *(_BYTE *)(a2 + 16);
    v10 = _mm_loadu_si128((const __m128i *)(a2 + 16));
    v11 = _mm_loadu_si128((const __m128i *)(a2 + 32));
    v12 = _mm_loadu_si128((const __m128i *)(a2 + 48));
    v13 = _mm_loadu_si128((const __m128i *)(a2 + 64));
    v40[0] = _mm_loadu_si128((const __m128i *)a2);
    v14 = _mm_loadu_si128((const __m128i *)(a2 + 80));
    v15 = _mm_loadu_si128((const __m128i *)(a2 + 96));
    v40[1] = v10;
    v16 = _mm_loadu_si128((const __m128i *)(a2 + 112));
    v40[2] = v11;
    v17 = _mm_loadu_si128((const __m128i *)(a2 + 128));
    v40[3] = v12;
    v40[4] = v13;
    v40[5] = v14;
    v40[6] = v15;
    v40[7] = v16;
    v40[8] = v17;
    if ( v9 == 2 )
    {
      v19 = _mm_loadu_si128((const __m128i *)(a2 + 160));
      v20 = _mm_loadu_si128((const __m128i *)(a2 + 176));
      v21 = _mm_loadu_si128((const __m128i *)(a2 + 192));
      v22 = _mm_loadu_si128((const __m128i *)(a2 + 208));
      v41 = _mm_loadu_si128((const __m128i *)(a2 + 144));
      v23 = _mm_loadu_si128((const __m128i *)(a2 + 224));
      v24 = _mm_loadu_si128((const __m128i *)(a2 + 240));
      v42 = v19;
      v25 = _mm_loadu_si128((const __m128i *)(a2 + 256));
      v26 = _mm_loadu_si128((const __m128i *)(a2 + 272));
      v43 = v20;
      v27 = _mm_loadu_si128((const __m128i *)(a2 + 288));
      v44 = v21;
      v28 = _mm_loadu_si128((const __m128i *)(a2 + 304));
      v45 = v22;
      v29 = _mm_loadu_si128((const __m128i *)(a2 + 320));
      v46 = v23;
      v30 = _mm_loadu_si128((const __m128i *)(a2 + 336));
      v47 = v24;
      v48 = v25;
      v49 = v26;
      v50 = v27;
      v51 = v28;
      v52 = v29;
      v53 = v30;
    }
    else if ( v9 == 5 || v9 == 1 )
    {
      v41.m128i_i64[0] = *(_QWORD *)(a2 + 144);
    }
    sub_82F430(v8, v31, (unsigned int)v40, 0, 0, 0, v7, v7 == 0, a2, (__int64)v38, 0);
    result = sub_6E4EE0(a2, (__int64)v40);
    if ( !a3 )
    {
      sub_6E5010((_BYTE *)a2, v40);
      result = sub_6E5070(a2, (__int64)v40);
    }
  }
  else if ( v36 )
  {
    result = sub_6F3BA0((__m128i *)a2, v7 == 0);
  }
  else
  {
    if ( (unsigned int)sub_6E5430() )
      sub_6854C0(0x12Bu, (FILE *)(a2 + 68), *(_QWORD *)(a2 + 136));
    result = sub_6E6840(a2);
  }
  if ( !a5 )
  {
    if ( v7 )
      return sub_6FAB30((const __m128i *)a2, a1, 0, 0, 0);
    else
      return sub_6FB850(a1, (__m128i *)a2, 0, (a3 == 0) | (unsigned __int8)(a4 != 0), 1, a3 == 0, 0, v37);
  }
  return result;
}
