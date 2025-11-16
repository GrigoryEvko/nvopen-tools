// Function: sub_BDBD20
// Address: 0xbdbd20
//
bool __fastcall sub_BDBD20(__int64 a1)
{
  bool result; // al
  unsigned __int8 v2; // cl
  __int64 v3; // rdx

  result = 0;
  if ( *(_BYTE *)a1 > 0x1Cu )
  {
    v2 = *(_BYTE *)a1 - 34;
    if ( v2 <= 0x33u )
    {
      result = ((0x8000000000041uLL >> v2) & 1) == 0;
      if ( ((0x8000000000041uLL >> v2) & 1) != 0 )
      {
        v3 = *(_QWORD *)(a1 - 32);
        if ( v3 && !*(_BYTE *)v3 && *(_QWORD *)(v3 + 24) == *(_QWORD *)(a1 + 80) )
          return *(_DWORD *)(v3 + 36) == 151;
      }
      else
      {
        return 0;
      }
    }
  }
  return result;
}
