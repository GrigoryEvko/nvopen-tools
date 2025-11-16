// Function: sub_2D94850
// Address: 0x2d94850
//
_QWORD *__fastcall sub_2D94850(
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
        __int64 a21)
{
  __m128i v21; // xmm3
  __m128i v22; // xmm4
  __m128i v23; // xmm5
  __m128i v24; // xmm6
  __m128i v25; // xmm7
  __int64 v26; // rax
  __m128i v27; // xmm0
  __m128i v28; // xmm1
  __int64 v29; // rdx
  __m128i v30; // xmm2
  __m128i v31; // xmm3
  __m128i v32; // xmm4
  __m128i v33; // xmm5
  __m128i v35; // [rsp+0h] [rbp-E0h] BYREF
  __m128i v36; // [rsp+10h] [rbp-D0h] BYREF
  _BYTE v37[24]; // [rsp+20h] [rbp-C0h] BYREF
  __m128i v38; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v39; // [rsp+48h] [rbp-98h]
  __m128i v40; // [rsp+50h] [rbp-90h] BYREF
  __m128i v41; // [rsp+60h] [rbp-80h] BYREF
  _BYTE v42[24]; // [rsp+70h] [rbp-70h] BYREF
  __m128i v43; // [rsp+88h] [rbp-58h] BYREF
  __int64 v44; // [rsp+98h] [rbp-48h]
  __m128i v45; // [rsp+A0h] [rbp-40h] BYREF
  __m128i v46; // [rsp+B0h] [rbp-30h] BYREF
  __int64 v47; // [rsp+C0h] [rbp-20h]

  *(_QWORD *)v37 = a9;
  v35 = _mm_loadu_si128((const __m128i *)&a7);
  v39 = a12;
  v36 = _mm_loadu_si128((const __m128i *)&a8);
  *(_QWORD *)v42 = a15;
  *(__m128i *)&v37[8] = _mm_loadu_si128((const __m128i *)&a10);
  v44 = a18;
  v38 = _mm_loadu_si128((const __m128i *)&a11);
  v40 = _mm_loadu_si128((const __m128i *)&a13);
  v41 = _mm_loadu_si128((const __m128i *)&a14);
  *(__m128i *)&v42[8] = _mm_loadu_si128((const __m128i *)&a16);
  v43 = _mm_loadu_si128((const __m128i *)&a17);
  v45 = _mm_loadu_si128((const __m128i *)&a19);
  v46 = _mm_loadu_si128((const __m128i *)&a20);
  v47 = a21;
  a1[1] = 0x400000000LL;
  *a1 = a1 + 2;
  sub_C8D5F0((__int64)a1, a1 + 2, 5u, 0x28u, a5, a6);
  v21 = _mm_loadu_si128(&v36);
  v22 = _mm_loadu_si128((const __m128i *)v37);
  v23 = _mm_loadu_si128((const __m128i *)&v37[16]);
  v24 = _mm_loadu_si128((const __m128i *)&v38.m128i_u64[1]);
  v25 = _mm_loadu_si128(&v40);
  v26 = *a1 + 40LL * *((unsigned int *)a1 + 2);
  v27 = _mm_loadu_si128(&v41);
  v28 = _mm_loadu_si128((const __m128i *)v42);
  *(__m128i *)v26 = _mm_loadu_si128(&v35);
  v29 = v47;
  v30 = _mm_loadu_si128((const __m128i *)&v42[16]);
  *(__m128i *)(v26 + 16) = v21;
  v31 = _mm_loadu_si128((const __m128i *)&v43.m128i_u64[1]);
  *(__m128i *)(v26 + 32) = v22;
  v32 = _mm_loadu_si128(&v45);
  *(__m128i *)(v26 + 48) = v23;
  v33 = _mm_loadu_si128(&v46);
  *(_QWORD *)(v26 + 192) = v29;
  *(__m128i *)(v26 + 64) = v24;
  *(__m128i *)(v26 + 80) = v25;
  *(__m128i *)(v26 + 96) = v27;
  *(__m128i *)(v26 + 112) = v28;
  *(__m128i *)(v26 + 128) = v30;
  *(__m128i *)(v26 + 144) = v31;
  *(__m128i *)(v26 + 160) = v32;
  *(__m128i *)(v26 + 176) = v33;
  *((_DWORD *)a1 + 2) += 5;
  return a1;
}
