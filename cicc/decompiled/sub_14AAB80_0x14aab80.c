// Function: sub_14AAB80
// Address: 0x14aab80
//
char __fastcall sub_14AAB80(__int64 a1)
{
  char result; // al
  __int64 v2; // rdx
  unsigned int v3; // ecx

  result = 0;
  if ( *(_BYTE *)(a1 + 16) == 78 )
  {
    v2 = *(_QWORD *)(a1 - 24);
    if ( !*(_BYTE *)(v2 + 16) )
    {
      v3 = *(_DWORD *)(v2 + 36);
      if ( v3 > 0x95 )
      {
        if ( v3 != 191 )
          return v3 == 215;
      }
      else
      {
        if ( v3 > 0x70 )
          return ((1LL << ((unsigned __int8)v3 - 113)) & 0x108000001BLL) != 0;
        if ( v3 != 4 )
          return v3 - 36 <= 2;
      }
      return 1;
    }
  }
  return result;
}
