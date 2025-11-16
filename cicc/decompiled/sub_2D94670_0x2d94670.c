// Function: sub_2D94670
// Address: 0x2d94670
//
_QWORD *__fastcall sub_2D94670(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int64 a9,
        __int128 a10,
        __int128 a11,
        __int64 a12,
        __int128 a13,
        __int128 a14,
        __int64 a15,
        __int128 a16,
        __int128 a17,
        __int64 a18,
        __int128 a19,
        __int128 a20,
        __int64 a21,
        __int128 a22,
        __int128 a23,
        __int64 a24)
{
  __int64 v24; // rax
  __m128i v25; // xmm2
  __m128i v26; // xmm3
  __m128i v27; // xmm5
  __m128i v28; // xmm6
  __m128i v29; // xmm7
  __m128i v30; // xmm0
  __m128i v31; // xmm1
  __m128i v32; // xmm2
  __int64 v33; // rax
  __m128i v34; // xmm3
  __m128i v35; // xmm4
  __m128i v36; // xmm5
  __m128i v37; // xmm6
  __m128i v38; // xmm7
  __m128i v39; // xmm0
  __m128i v40; // xmm1
  __m128i v41; // xmm2
  __m128i v43; // [rsp+0h] [rbp-100h] BYREF
  __m128i v44; // [rsp+10h] [rbp-F0h] BYREF
  _BYTE v45[24]; // [rsp+20h] [rbp-E0h] BYREF
  __m128i v46; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v47; // [rsp+48h] [rbp-B8h]
  __m128i v48; // [rsp+50h] [rbp-B0h] BYREF
  __m128i v49; // [rsp+60h] [rbp-A0h] BYREF
  _BYTE v50[24]; // [rsp+70h] [rbp-90h] BYREF
  __m128i v51; // [rsp+88h] [rbp-78h] BYREF
  __int64 v52; // [rsp+98h] [rbp-68h]
  __m128i v53; // [rsp+A0h] [rbp-60h] BYREF
  __m128i v54; // [rsp+B0h] [rbp-50h] BYREF
  _BYTE v55[24]; // [rsp+C0h] [rbp-40h] BYREF
  __m128i v56; // [rsp+D8h] [rbp-28h] BYREF
  __int64 v57; // [rsp+E8h] [rbp-18h]

  *(_QWORD *)v45 = a9;
  v43 = _mm_loadu_si128((const __m128i *)&a7);
  v47 = a12;
  v44 = _mm_loadu_si128((const __m128i *)&a8);
  *(_QWORD *)v50 = a15;
  v48 = _mm_loadu_si128((const __m128i *)&a13);
  v52 = a18;
  v49 = _mm_loadu_si128((const __m128i *)&a14);
  *(__m128i *)&v50[8] = _mm_loadu_si128((const __m128i *)&a16);
  v51 = _mm_loadu_si128((const __m128i *)&a17);
  v53 = _mm_loadu_si128((const __m128i *)&a19);
  v54 = _mm_loadu_si128((const __m128i *)&a20);
  *(__m128i *)&v45[8] = _mm_loadu_si128((const __m128i *)&a10);
  v46 = _mm_loadu_si128((const __m128i *)&a11);
  *(_QWORD *)v55 = a21;
  v24 = a24;
  *a1 = a1 + 2;
  v25 = _mm_loadu_si128((const __m128i *)&a22);
  v57 = v24;
  v26 = _mm_loadu_si128((const __m128i *)&a23);
  a1[1] = 0x400000000LL;
  *(__m128i *)&v55[8] = v25;
  v56 = v26;
  sub_C8D5F0((__int64)a1, a1 + 2, 6u, 0x28u, a5, a6);
  v27 = _mm_loadu_si128(&v44);
  v28 = _mm_loadu_si128((const __m128i *)v45);
  v29 = _mm_loadu_si128((const __m128i *)&v45[16]);
  v30 = _mm_loadu_si128((const __m128i *)&v46.m128i_u64[1]);
  v31 = _mm_loadu_si128(&v48);
  v32 = _mm_loadu_si128(&v49);
  v33 = *a1 + 40LL * *((unsigned int *)a1 + 2);
  v34 = _mm_loadu_si128((const __m128i *)v50);
  *(__m128i *)v33 = _mm_loadu_si128(&v43);
  v35 = _mm_loadu_si128((const __m128i *)&v50[16]);
  *(__m128i *)(v33 + 16) = v27;
  v36 = _mm_loadu_si128((const __m128i *)&v51.m128i_u64[1]);
  *(__m128i *)(v33 + 32) = v28;
  v37 = _mm_loadu_si128(&v53);
  *(__m128i *)(v33 + 48) = v29;
  v38 = _mm_loadu_si128(&v54);
  *(__m128i *)(v33 + 64) = v30;
  v39 = _mm_loadu_si128((const __m128i *)v55);
  *(__m128i *)(v33 + 80) = v31;
  v40 = _mm_loadu_si128((const __m128i *)&v55[16]);
  *(__m128i *)(v33 + 96) = v32;
  v41 = _mm_loadu_si128((const __m128i *)&v56.m128i_u64[1]);
  *(__m128i *)(v33 + 112) = v34;
  *(__m128i *)(v33 + 128) = v35;
  *(__m128i *)(v33 + 144) = v36;
  *(__m128i *)(v33 + 160) = v37;
  *(__m128i *)(v33 + 176) = v38;
  *(__m128i *)(v33 + 192) = v39;
  *(__m128i *)(v33 + 208) = v40;
  *(__m128i *)(v33 + 224) = v41;
  *((_DWORD *)a1 + 2) += 6;
  return a1;
}
