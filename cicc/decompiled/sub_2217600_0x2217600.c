// Function: sub_2217600
// Address: 0x2217600
//
__int64 __fastcall sub_2217600(__int64 a1)
{
  __int64 v1; // rbx
  int v2; // eax
  __int64 i; // rbx
  unsigned __int64 j; // rcx
  unsigned __int16 v5; // si
  unsigned __int64 v6; // r13
  int v7; // esi
  int v8; // esi

  v1 = 0;
  __uselocale();
  do
  {
    v2 = wctob(v1);
    if ( v2 == -1 )
    {
      *(_BYTE *)(a1 + 24) = 0;
      goto LABEL_5;
    }
    *(_BYTE *)(a1 + v1++ + 25) = v2;
  }
  while ( v1 != 128 );
  *(_BYTE *)(a1 + 24) = 1;
LABEL_5:
  for ( i = 0; i != 256; ++i )
    *(_DWORD *)(a1 + 4 * i + 156) = btowc(i);
  for ( j = 0; ; j = v6 )
  {
    v6 = j + 1;
    v7 = 1 << j;
    if ( j <= 7 )
    {
      v5 = (_WORD)v7 << 8;
      *(_WORD *)(a1 + 2 * v6 + 1178) = v5;
      *(_QWORD *)(a1 + 8 * v6 + 1208) = sub_2217070(a1, v5);
      continue;
    }
    v8 = v7 >> 8;
    *(_WORD *)(a1 + 2 * v6 + 1178) = v8;
    *(_QWORD *)(a1 + 8 * v6 + 1208) = sub_2217070(a1, v8);
    if ( v6 == 12 )
      break;
  }
  return __uselocale();
}
