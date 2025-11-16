// Function: sub_39A1AB0
// Address: 0x39a1ab0
//
bool __fastcall sub_39A1AB0(__int64 a1)
{
  __int64 v1; // rcx
  bool result; // al
  unsigned __int64 v3; // rdx
  __int64 v4; // rcx

  v1 = 0x8000000010003LL;
  while ( 1 )
  {
    if ( *(_BYTE *)a1 == 13 )
      return *(_WORD *)(a1 + 2) != 4;
    if ( *(_BYTE *)a1 != 12 )
      break;
    if ( (unsigned __int16)(*(_WORD *)(a1 + 2) - 15) <= 0x33u
      && _bittest64(&v1, (unsigned int)*(unsigned __int16 *)(a1 + 2) - 15) )
    {
      return 1;
    }
    a1 = *(_QWORD *)(a1 + 8 * (3LL - *(unsigned int *)(a1 + 8)));
  }
  v3 = *(unsigned int *)(a1 + 52);
  if ( (unsigned int)v3 > 0x10 )
    return *(_WORD *)(a1 + 2) == 59;
  v4 = 65924;
  result = 1;
  if ( !_bittest64(&v4, v3) )
    return *(_WORD *)(a1 + 2) == 59;
  return result;
}
