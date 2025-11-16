// Function: sub_328FBA0
// Address: 0x328fba0
//
__int64 __fastcall sub_328FBA0(const __m128i *a1, unsigned int a2, int a3, int a4, int a5, __int64 a6, __int128 a7)
{
  int v11; // eax
  __int64 v12; // rdi
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  int v15; // r9d
  __int128 v17; // [rsp-10h] [rbp-60h]
  _OWORD v18[5]; // [rsp+0h] [rbp-50h] BYREF

  v11 = sub_33CB7C0(a2);
  v12 = a1->m128i_i64[0];
  *((_QWORD *)&v17 + 1) = 3;
  *(_QWORD *)&v17 = v18;
  v13 = _mm_loadu_si128(a1 + 1);
  v14 = _mm_loadu_si128(a1 + 2);
  v18[0] = _mm_loadu_si128((const __m128i *)&a7);
  v18[1] = v13;
  v18[2] = v14;
  return sub_33FC220(v12, v11, a3, a4, a5, v15, v17);
}
