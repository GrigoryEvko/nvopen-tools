// Function: sub_2E33380
// Address: 0x2e33380
//
__int64 __fastcall sub_2E33380(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // rax
  __int64 v7; // rsi
  int v9; // edi
  __int16 v10; // dx

  v5 = *(_QWORD *)(a1 + 56);
  v7 = a1 + 48;
  if ( v5 != a1 + 48 )
  {
    while ( 1 )
    {
      LOBYTE(a5) = (unsigned __int16)(*(_WORD *)(v5 + 68) - 14) > 4u && *(_WORD *)(v5 + 68) != 24;
      if ( (_BYTE)a5 )
        break;
      if ( (*(_BYTE *)v5 & 4) != 0 )
      {
        v5 = *(_QWORD *)(v5 + 8);
        if ( v7 == v5 )
          return 0;
      }
      else
      {
        while ( (*(_BYTE *)(v5 + 44) & 8) != 0 )
          v5 = *(_QWORD *)(v5 + 8);
        v5 = *(_QWORD *)(v5 + 8);
        if ( v7 == v5 )
          return 0;
      }
    }
    if ( v7 != v5 )
    {
      v9 = 1;
      if ( !a2 )
        return a5;
LABEL_14:
      if ( (*(_BYTE *)v5 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v5 + 44) & 8) != 0 )
          v5 = *(_QWORD *)(v5 + 8);
      }
      while ( 1 )
      {
        v5 = *(_QWORD *)(v5 + 8);
        if ( v7 == v5 )
          break;
        v10 = *(_WORD *)(v5 + 68);
        if ( (unsigned __int16)(v10 - 14) > 4u && v10 != 24 )
        {
          if ( v7 == v5 )
            return 0;
          if ( ++v9 > a2 )
            return a5;
          goto LABEL_14;
        }
        if ( (*(_BYTE *)v5 & 4) == 0 && (*(_BYTE *)(v5 + 44) & 8) != 0 )
        {
          do
            v5 = *(_QWORD *)(v5 + 8);
          while ( (*(_BYTE *)(v5 + 44) & 8) != 0 );
        }
      }
    }
  }
  return 0;
}
