// Function: sub_2753FC0
// Address: 0x2753fc0
//
bool __fastcall sub_2753FC0(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdx
  unsigned int v3; // ecx

  result = 0;
  if ( *(_BYTE *)a1 == 85 )
  {
    v2 = *(_QWORD *)(a1 - 32);
    if ( v2 )
    {
      if ( !*(_BYTE *)v2 && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a1 + 80) && (*(_BYTE *)(v2 + 33) & 0x20) != 0 )
      {
        v3 = *(_DWORD *)(v2 + 36);
        if ( v3 <= 0xD3 )
        {
          if ( v3 > 0xCB )
          {
            return ((1LL << ((unsigned __int8)v3 + 52)) & 0xD1) != 0;
          }
          else if ( v3 == 11 )
          {
            return 1;
          }
          else if ( v3 - 69 <= 2 )
          {
            BUG();
          }
        }
      }
    }
  }
  return result;
}
