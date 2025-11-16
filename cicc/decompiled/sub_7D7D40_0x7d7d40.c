// Function: sub_7D7D40
// Address: 0x7d7d40
//
__int64 __fastcall sub_7D7D40(__int64 *a1)
{
  __int64 i; // rdx
  __int64 v3; // rcx
  char **v4; // rdi
  const __m128i *v5; // rax

  for ( i = *a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v3 = *(unsigned __int8 *)(i + 160);
  v4 = &off_4B7B300;
  if ( (_BYTE)v3 != 10 )
  {
    v4 = off_4B7B310;
    if ( (_BYTE)v3 != 11 )
    {
      v4 = off_4B7B320;
      if ( (_BYTE)v3 != 12 )
      {
        v4 = off_4B7B340;
        if ( (_BYTE)v3 != 13 )
          v4 = &off_4B7B340[(unsigned __int8)v3 - 8];
      }
    }
  }
  if ( *((_QWORD *)&xmmword_4F18760 + *(unsigned __int8 *)(i + 160)) )
    v5 = (const __m128i *)sub_7F88E0(*((_QWORD *)&xmmword_4F18760 + *(unsigned __int8 *)(i + 160)), a1[9]);
  else
    v5 = (const __m128i *)sub_7F8B20(*v4, (char *)&xmmword_4F18760 + 8 * v3, i, i, 0, a1[9]);
  return sub_730620((__int64)a1, v5);
}
