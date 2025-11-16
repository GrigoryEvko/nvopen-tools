// Function: sub_3290460
// Address: 0x3290460
//
__int64 __fastcall sub_3290460(
        const __m128i *a1,
        unsigned int a2,
        int a3,
        int a4,
        int a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9)
{
  int v12; // eax
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  __m128i v15; // xmm3
  __m128i v16; // xmm4
  __int64 v17; // rdi
  int v18; // r9d
  __int128 v20; // [rsp-10h] [rbp-80h]
  _OWORD v21[7]; // [rsp+0h] [rbp-70h] BYREF

  v12 = sub_33CB7C0(a2);
  v13 = _mm_loadu_si128((const __m128i *)&a8);
  *((_QWORD *)&v20 + 1) = 5;
  v14 = _mm_loadu_si128((const __m128i *)&a9);
  *(_QWORD *)&v20 = v21;
  v15 = _mm_loadu_si128(a1 + 1);
  v16 = _mm_loadu_si128(a1 + 2);
  v17 = a1->m128i_i64[0];
  v21[0] = _mm_loadu_si128((const __m128i *)&a7);
  v21[1] = v13;
  v21[2] = v14;
  v21[3] = v15;
  v21[4] = v16;
  return sub_33FC220(v17, v12, a3, a4, a5, v18, v20);
}
