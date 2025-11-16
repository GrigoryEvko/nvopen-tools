// Function: sub_7D8240
// Address: 0x7d8240
//
__int64 __fastcall sub_7D8240(__int64 *a1)
{
  __int64 i; // rdx
  __int64 *v3; // rsi
  __int64 j; // rcx
  __int64 v5; // rdi
  char **v6; // r10
  __int64 v7; // rax
  __int64 v8; // rdi
  const __m128i *v9; // rax

  for ( i = *a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v3 = (__int64 *)a1[9];
  for ( j = *v3; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v5 = *(unsigned __int8 *)(j + 160);
  v6 = &off_4B7B0C0;
  v7 = v5;
  if ( (_BYTE)v5 != 10 )
  {
    v6 = off_4B7B0D0;
    if ( (_BYTE)v5 != 11 )
    {
      v6 = off_4B7B0E0;
      if ( (_BYTE)v5 != 12 )
      {
        v6 = off_4B7B100;
        if ( (_BYTE)v5 != 13 )
          v6 = &off_4B7B100[(unsigned __int8)v5 - 8];
      }
    }
  }
  v8 = *((_QWORD *)&xmmword_4F18520 + *(unsigned __int8 *)(j + 160));
  if ( v8 )
    v9 = (const __m128i *)sub_7F88E0(v8, v3);
  else
    v9 = (const __m128i *)sub_7F8B20(*v6, (char *)&xmmword_4F18520 + 8 * v7, i, j, j, a1[9]);
  return sub_730620((__int64)a1, v9);
}
