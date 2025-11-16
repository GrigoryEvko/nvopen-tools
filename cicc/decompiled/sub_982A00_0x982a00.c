// Function: sub_982A00
// Address: 0x982a00
//
void __fastcall sub_982A00(__m128i **a1, const __m128i *a2, __int64 a3)
{
  const __m128i *v3; // r12
  char *v4; // r14
  char *v5; // r15
  unsigned __int64 v6; // rax
  char *v7; // r12
  char *v8; // r13
  unsigned __int64 v9; // rax

  v3 = &a2[4 * a3];
  sub_97EC20(a1 + 22, a1[23], a2, v3);
  v4 = (char *)a1[22];
  v5 = (char *)a1[23];
  if ( v5 != v4 )
  {
    _BitScanReverse64(&v6, (v5 - v4) >> 6);
    sub_982780(
      a1[22],
      (unsigned __int64)a1[23],
      2LL * (int)(63 - (v6 ^ 0x3F)),
      (__int64 (__fastcall *)(__m128i *, __m128i *))sub_97E7F0);
    sub_9822B0(v4, v5, (__int64 (__fastcall *)(__m128i *, const __m128i *))sub_97E7F0);
  }
  sub_97EC20(a1 + 25, a1[26], a2, v3);
  v7 = (char *)a1[25];
  v8 = (char *)a1[26];
  if ( v8 != v7 )
  {
    _BitScanReverse64(&v9, (v8 - v7) >> 6);
    sub_982780(
      a1[25],
      (unsigned __int64)a1[26],
      2LL * (int)(63 - (v9 ^ 0x3F)),
      (__int64 (__fastcall *)(__m128i *, __m128i *))sub_97E1B0);
    sub_9822B0(v7, v8, (__int64 (__fastcall *)(__m128i *, const __m128i *))sub_97E1B0);
  }
}
