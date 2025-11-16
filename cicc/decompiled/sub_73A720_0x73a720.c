// Function: sub_73A720
// Address: 0x73a720
//
_QWORD *__fastcall sub_73A720(const __m128i *a1, __int64 a2)
{
  _QWORD *v2; // r12
  __int64 v3; // rdx
  __int64 v4; // rcx
  _UNKNOWN *__ptr32 *v5; // r8

  v2 = sub_726700(2);
  v2[7] = sub_73A460(a1, a2, v3, v4, v5);
  *v2 = a1[8].m128i_i64[0];
  v2[10] = a1[9].m128i_i64[1];
  return v2;
}
