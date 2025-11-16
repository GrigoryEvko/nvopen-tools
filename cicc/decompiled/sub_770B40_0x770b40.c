// Function: sub_770B40
// Address: 0x770b40
//
__int64 __fastcall sub_770B40(__int16 *a1, int a2, int a3, __int64 a4)
{
  __int64 result; // rax
  int v5; // [rsp+Ch] [rbp-34h] BYREF
  _OWORD v6[3]; // [rsp+10h] [rbp-30h] BYREF

  if ( (*(_BYTE *)(a4 + 162) & 4) != 0 )
  {
    result = sub_621000(a1, 0, (__int16 *)&xmmword_4F08290, 0);
    if ( (_DWORD)result )
      *(__m128i *)a1 = _mm_load_si128((const __m128i *)&xmmword_4F08280);
    else
      *(__m128i *)a1 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
  }
  else if ( a3 )
  {
    return sub_6215A0(a1, a2);
  }
  else
  {
    v6[0] = _mm_load_si128((const __m128i *)&xmmword_4F08280);
    sub_621410((__int64)v6, a2, &v5);
    sub_6215F0((unsigned __int16 *)v6, (__int16 *)&xmmword_4F08280, 0, (_BOOL4 *)&v5);
    return sub_6213D0((__int64)a1, (__int64)v6);
  }
  return result;
}
