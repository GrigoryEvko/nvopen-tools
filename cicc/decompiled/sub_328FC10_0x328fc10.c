// Function: sub_328FC10
// Address: 0x328fc10
//
__int64 __fastcall sub_328FC10(
        const __m128i *a1,
        unsigned int a2,
        int a3,
        int a4,
        int a5,
        __int64 a6,
        __int128 a7,
        __int128 a8)
{
  int v11; // eax
  __m128i v12; // xmm1
  __m128i v13; // xmm2
  __m128i v14; // xmm3
  __int64 v15; // rdi
  int v16; // r9d
  __int128 v18; // [rsp-10h] [rbp-70h]
  _OWORD v19[6]; // [rsp+0h] [rbp-60h] BYREF

  v11 = sub_33CB7C0(a2);
  v12 = _mm_loadu_si128((const __m128i *)&a8);
  *((_QWORD *)&v18 + 1) = 4;
  v13 = _mm_loadu_si128(a1 + 1);
  *(_QWORD *)&v18 = v19;
  v14 = _mm_loadu_si128(a1 + 2);
  v15 = a1->m128i_i64[0];
  v19[0] = _mm_loadu_si128((const __m128i *)&a7);
  v19[1] = v12;
  v19[2] = v13;
  v19[3] = v14;
  return sub_33FC220(v15, v11, a3, a4, a5, v16, v18);
}
