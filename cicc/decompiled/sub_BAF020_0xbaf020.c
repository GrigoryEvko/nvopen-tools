// Function: sub_BAF020
// Address: 0xbaf020
//
char __fastcall sub_BAF020(__int64 a1, __int64 a2, char a3, bool *a4)
{
  __int64 v4; // r9
  char v5; // si
  char v6; // si
  char result; // al

  v4 = a2;
  if ( !*(_DWORD *)(a2 + 8) )
    v4 = *(_QWORD *)(a2 + 64);
  v5 = *(_BYTE *)(a2 + 12);
  switch ( v5 & 0xF )
  {
    case 0:
    case 1:
    case 3:
    case 5:
    case 6:
    case 7:
    case 8:
      v6 = v5 & 0x40;
      *a4 = v6 == 0;
      if ( v6 )
        return 0;
      result = 1;
      if ( !a3 )
        return result;
      result = qword_4F81E68;
      if ( (_BYTE)qword_4F81E68 )
      {
        if ( (*(_BYTE *)(v4 + 64) & 4) != 0 )
          return result;
      }
      if ( !*(_BYTE *)(a1 + 337) )
        return *(_DWORD *)(v4 + 48) == 0;
      result = *(_BYTE *)(v4 + 64) & 1;
      if ( !result )
      {
        if ( (*(_BYTE *)(v4 + 64) & 2) != 0 )
          return *(_BYTE *)(a1 + 337);
        else
          return *(_DWORD *)(v4 + 48) == 0;
      }
      return result;
    case 2:
    case 4:
    case 9:
    case 0xA:
      *a4 = 0;
      return 0;
    default:
      BUG();
  }
}
