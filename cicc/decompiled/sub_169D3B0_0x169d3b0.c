// Function: sub_169D3B0
// Address: 0x169d3b0
//
char __fastcall sub_169D3B0(__int64 a1, __m128i a2)
{
  char result; // al
  __int64 v3; // [rsp-10h] [rbp-18h] BYREF
  unsigned int v4; // [rsp-8h] [rbp-10h]

  v4 = 32;
  v3 = (unsigned int)_mm_cvtsi128_si32(a2);
  result = sub_169CFC0(a1, &unk_42AE9E0, &v3);
  if ( v4 > 0x40 )
  {
    if ( v3 )
      return j_j___libc_free_0_0(v3);
  }
  return result;
}
