// Function: sub_300AC80
// Address: 0x300ac80
//
void *__fastcall sub_300AC80(unsigned __int16 *a1, __int64 a2)
{
  int v2; // ebx
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  unsigned __int16 v7[9]; // [rsp+Eh] [rbp-12h] BYREF

  v2 = *a1;
  if ( (_WORD)v2 )
  {
    if ( (unsigned __int16)(v2 - 17) <= 0xD3u )
      LOWORD(v2) = word_4456580[v2 - 1];
  }
  else if ( sub_30070B0((__int64)a1) )
  {
    LOWORD(v2) = sub_3009970((__int64)a1, a2, v4, v5, v6);
  }
  v7[0] = v2;
  return sub_300AC00(v7);
}
