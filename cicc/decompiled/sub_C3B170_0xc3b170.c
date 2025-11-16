// Function: sub_C3B170
// Address: 0xc3b170
//
unsigned int *__fastcall sub_C3B170(__int64 a1, __m128i a2)
{
  unsigned int *result; // rax
  __int64 v3; // [rsp-10h] [rbp-18h] BYREF
  unsigned int v4; // [rsp-8h] [rbp-10h]

  v4 = 32;
  v3 = (unsigned int)_mm_cvtsi128_si32(a2);
  result = sub_C3AF00(a1, dword_3F657C0, &v3);
  if ( v4 > 0x40 )
  {
    if ( v3 )
      return (unsigned int *)j_j___libc_free_0_0(v3);
  }
  return result;
}
