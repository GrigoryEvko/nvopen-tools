// Function: sub_149E2E0
// Address: 0x149e2e0
//
void __fastcall sub_149E2E0(__int64 a1, const __m128i *a2, __int64 a3)
{
  const __m128i *v3; // r14
  __int64 v4; // r8
  __int64 v5; // r9
  __m128i *v6; // r13
  __m128i *v7; // r15
  unsigned __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r8
  __int64 v13; // r9
  __m128i *v14; // r12
  __m128i *v15; // r13
  unsigned __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9

  v3 = (const __m128i *)((char *)a2 + 40 * a3);
  sub_149AF50((const __m128i **)(a1 + 152), *(__m128i **)(a1 + 160), a2, v3);
  v6 = *(__m128i **)(a1 + 152);
  v7 = *(__m128i **)(a1 + 160);
  if ( v7 != v6 )
  {
    _BitScanReverse64(&v8, 0xCCCCCCCCCCCCCCCDLL * (((char *)v7 - (char *)v6) >> 3));
    sub_149E0E0(
      *(_QWORD *)(a1 + 152),
      *(__m128i **)(a1 + 160),
      2LL * (int)(63 - (v8 ^ 0x3F)),
      (unsigned __int8 (__fastcall *)(__m128i *, __m128i *))sub_149B2B0,
      v4,
      v5);
    sub_149DCE0(v6, v7, (unsigned __int8 (__fastcall *)(__m128i *, __int8 *))sub_149B2B0, v9, v10, v11);
  }
  sub_149AF50((const __m128i **)(a1 + 176), *(__m128i **)(a1 + 184), a2, v3);
  v14 = *(__m128i **)(a1 + 176);
  v15 = *(__m128i **)(a1 + 184);
  if ( v15 != v14 )
  {
    _BitScanReverse64(&v16, 0xCCCCCCCCCCCCCCCDLL * (((char *)v15 - (char *)v14) >> 3));
    sub_149E0E0(
      *(_QWORD *)(a1 + 176),
      *(__m128i **)(a1 + 184),
      2LL * (int)(63 - (v16 ^ 0x3F)),
      (unsigned __int8 (__fastcall *)(__m128i *, __m128i *))sub_149B320,
      v12,
      v13);
    sub_149DCE0(v14, v15, (unsigned __int8 (__fastcall *)(__m128i *, __int8 *))sub_149B320, v17, v18, v19);
  }
}
