// Function: sub_892B20
// Address: 0x892b20
//
__int64 __fastcall sub_892B20(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 result; // rax
  char v3; // dl

  v1 = *(_BYTE *)(a1 + 8);
  if ( v1 == 2 )
  {
    result = *(_QWORD *)(a1 + 32);
    if ( !result )
      return result;
    if ( *(_BYTE *)(result + 120) == 8 )
    {
      result += 128;
      return result;
    }
    return 0;
  }
  if ( v1 > 2u )
  {
    if ( v1 != 3 )
      sub_721090();
    return 0;
  }
  if ( v1 )
  {
    result = *(_QWORD *)(a1 + 32);
    if ( !result )
      return result;
    if ( *(_BYTE *)(result + 173) == 12 && !*(_BYTE *)(result + 176) )
    {
      result += 184;
      return result;
    }
    return 0;
  }
  result = *(_QWORD *)(a1 + 32);
  if ( result )
  {
    while ( 1 )
    {
      v3 = *(_BYTE *)(result + 140);
      if ( v3 != 12 )
        break;
      result = *(_QWORD *)(result + 160);
    }
    if ( v3 == 14 && !*(_BYTE *)(result + 160) )
      return *(_QWORD *)(result + 168) + 24LL;
    return 0;
  }
  return result;
}
