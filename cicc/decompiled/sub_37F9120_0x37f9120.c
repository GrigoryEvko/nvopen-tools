// Function: sub_37F9120
// Address: 0x37f9120
//
__int64 *__fastcall sub_37F9120(__int64 *a1, __int64 a2, int a3)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rdx
  __m128i v5; // xmm0
  __m128i v6; // xmm0
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  __int64 *result; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __m128i si128; // xmm0
  __m128i v13; // xmm0
  unsigned __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __m128i v18; // xmm0
  __m128i v19; // xmm0
  __m128i v20; // xmm0
  unsigned __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __m128i v25; // xmm0
  __m128i v26; // xmm0
  unsigned __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __m128i v31; // xmm0
  __m128i v32; // xmm0
  unsigned __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  __m128i v37; // xmm0
  unsigned __int64 v38; // rax
  __int64 v39; // rdx
  unsigned __int64 v40[2]; // [rsp+8h] [rbp-18h] BYREF

  switch ( a3 )
  {
    case 1:
      v40[0] = 39;
      *a1 = (__int64)(a1 + 2);
      v10 = sub_22409D0((__int64)a1, v40, 0);
      v11 = v40[0];
      si128 = _mm_load_si128((const __m128i *)&xmmword_4527880);
      *a1 = v10;
      a1[2] = v11;
      *(__m128i *)v10 = si128;
      v13 = _mm_load_si128((const __m128i *)&xmmword_4527890);
      *(_DWORD *)(v10 + 32) = 1920103779;
      *(_WORD *)(v10 + 36) = 25701;
      *(_BYTE *)(v10 + 38) = 46;
      *(__m128i *)(v10 + 16) = v13;
      v14 = v40[0];
      v15 = *a1;
      a1[1] = v40[0];
      *(_BYTE *)(v15 + v14) = 0;
      result = a1;
      break;
    case 2:
      v40[0] = 69;
      *a1 = (__int64)(a1 + 2);
      v16 = sub_22409D0((__int64)a1, v40, 0);
      v17 = v40[0];
      v18 = _mm_load_si128((const __m128i *)&xmmword_45278A0);
      *a1 = v16;
      a1[2] = v17;
      *(__m128i *)v16 = v18;
      v19 = _mm_load_si128((const __m128i *)&xmmword_45278B0);
      *(_DWORD *)(v16 + 64) = 1936028793;
      *(__m128i *)(v16 + 16) = v19;
      v20 = _mm_load_si128((const __m128i *)&xmmword_45278C0);
      *(_BYTE *)(v16 + 68) = 46;
      *(__m128i *)(v16 + 32) = v20;
      *(__m128i *)(v16 + 48) = _mm_load_si128((const __m128i *)&xmmword_45278D0);
      v21 = v40[0];
      v22 = *a1;
      a1[1] = v40[0];
      *(_BYTE *)(v22 + v21) = 0;
      result = a1;
      break;
    case 3:
      v40[0] = 41;
      *a1 = (__int64)(a1 + 2);
      v23 = sub_22409D0((__int64)a1, v40, 0);
      v24 = v40[0];
      v25 = _mm_load_si128((const __m128i *)&xmmword_4527910);
      *a1 = v23;
      a1[2] = v24;
      *(__m128i *)v23 = v25;
      v26 = _mm_load_si128((const __m128i *)&xmmword_4527920);
      *(_QWORD *)(v23 + 32) = 0x646574726F707075LL;
      *(_BYTE *)(v23 + 40) = 46;
      *(__m128i *)(v23 + 16) = v26;
      v27 = v40[0];
      v28 = *a1;
      a1[1] = v40[0];
      *(_BYTE *)(v28 + v27) = 0;
      result = a1;
      break;
    case 4:
      v40[0] = 33;
      *a1 = (__int64)(a1 + 2);
      v29 = sub_22409D0((__int64)a1, v40, 0);
      v30 = v40[0];
      v31 = _mm_load_si128((const __m128i *)&xmmword_45278E0);
      *a1 = v29;
      a1[2] = v30;
      *(__m128i *)v29 = v31;
      v32 = _mm_load_si128((const __m128i *)&xmmword_45278F0);
      *(_BYTE *)(v29 + 32) = 46;
      *(__m128i *)(v29 + 16) = v32;
      v33 = v40[0];
      v34 = *a1;
      a1[1] = v40[0];
      *(_BYTE *)(v34 + v33) = 0;
      result = a1;
      break;
    case 5:
      v40[0] = 21;
      *a1 = (__int64)(a1 + 2);
      v35 = sub_22409D0((__int64)a1, v40, 0);
      v36 = v40[0];
      v37 = _mm_load_si128((const __m128i *)&xmmword_4527900);
      *a1 = v35;
      a1[2] = v36;
      *(_DWORD *)(v35 + 16) = 1935962735;
      *(_BYTE *)(v35 + 20) = 46;
      *(__m128i *)v35 = v37;
      v38 = v40[0];
      v39 = *a1;
      a1[1] = v40[0];
      *(_BYTE *)(v39 + v38) = 0;
      result = a1;
      break;
    case 6:
      v40[0] = 40;
      *a1 = (__int64)(a1 + 2);
      v3 = sub_22409D0((__int64)a1, v40, 0);
      v4 = v40[0];
      v5 = _mm_load_si128((const __m128i *)&xmmword_4527930);
      *a1 = v3;
      a1[2] = v4;
      *(__m128i *)v3 = v5;
      v6 = _mm_load_si128((const __m128i *)&xmmword_4527940);
      *(_QWORD *)(v3 + 32) = 0x2E65707974206E77LL;
      *(__m128i *)(v3 + 16) = v6;
      v7 = v40[0];
      v8 = *a1;
      a1[1] = v40[0];
      *(_BYTE *)(v8 + v7) = 0;
      result = a1;
      break;
    default:
      BUG();
  }
  return result;
}
