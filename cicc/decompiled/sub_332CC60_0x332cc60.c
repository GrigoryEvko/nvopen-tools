// Function: sub_332CC60
// Address: 0x332cc60
//
__int64 __fastcall sub_332CC60(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // rax
  __int64 v5; // rdx
  unsigned __int16 v6; // dx
  unsigned __int64 v7; // r10
  unsigned __int16 v8; // dx

  v3 = *(_QWORD *)(a1 + 525256);
  if ( v3 )
  {
    v5 = a1 + 525248;
    do
    {
      while ( a2 <= *(_DWORD *)(v3 + 32) && (a2 != *(_DWORD *)(v3 + 32) || (unsigned __int16)a3 <= *(_WORD *)(v3 + 36)) )
      {
        v5 = v3;
        v3 = *(_QWORD *)(v3 + 16);
        if ( !v3 )
          goto LABEL_8;
      }
      v3 = *(_QWORD *)(v3 + 24);
    }
    while ( v3 );
LABEL_8:
    if ( v5 != a1 + 525248
      && a2 >= *(_DWORD *)(v5 + 32)
      && (a2 != *(_DWORD *)(v5 + 32) || (unsigned __int16)a3 >= *(_WORD *)(v5 + 36)) )
    {
      return *(unsigned __int16 *)(v5 + 40);
    }
  }
  v6 = a3;
  if ( (unsigned __int16)(a3 - 17) <= 0xD3u )
    v6 = word_4456580[(unsigned __int16)a3 - 1];
  if ( v6 <= 1u || (unsigned __int16)(v6 - 504) <= 7u )
LABEL_30:
    BUG();
  v7 = *(_QWORD *)&byte_444C4A0[16 * v6 - 16];
  do
  {
    ++a3;
    while ( 1 )
    {
      v8 = a3;
      if ( (unsigned __int16)(a3 - 17) <= 0xD3u )
        v8 = word_4456580[(unsigned __int16)a3 - 1];
      if ( v8 <= 1u || (unsigned __int16)(v8 - 504) <= 7u )
        goto LABEL_30;
      if ( v7 < *(_QWORD *)&byte_444C4A0[16 * v8 - 16] )
        break;
      ++a3;
    }
  }
  while ( !(_WORD)a3
       || !*(_QWORD *)(a1 + 8LL * (unsigned __int16)a3 + 112)
       || a2 <= 0x1F3 && *(_BYTE *)(a1 + 500LL * (unsigned __int16)a3 + a2 + 6414) == 1 );
  return a3;
}
