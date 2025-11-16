// Function: sub_8895C0
// Address: 0x8895c0
//
_BOOL8 __fastcall sub_8895C0(char *src)
{
  size_t v1; // rax
  _BOOL8 result; // rax
  __int64 v3; // rcx
  __int64 v4[2]; // [rsp+0h] [rbp-50h] BYREF
  __m128i v5; // [rsp+10h] [rbp-40h]
  __m128i v6; // [rsp+20h] [rbp-30h]
  __m128i v7; // [rsp+30h] [rbp-20h]

  v4[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v5 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v6 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v7 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v4[1] = *(_QWORD *)&dword_4F077C8;
  v1 = strlen(src);
  sub_878540(src, v1, v4);
  result = 0;
  if ( v4[0] )
  {
    result = 1;
    if ( (*(_BYTE *)(v4[0] + 73) & 0x20) == 0 )
    {
      v3 = *(_QWORD *)(v4[0] + 24);
      if ( !v3 || *(_BYTE *)(v3 + 80) )
      {
        result = 0;
        if ( (*(_BYTE *)(v4[0] + 73) & 2) != 0 )
          return sub_887690(v4[0]);
      }
    }
  }
  return result;
}
