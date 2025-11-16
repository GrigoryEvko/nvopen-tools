// Function: sub_B49220
// Address: 0xb49220
//
bool __fastcall sub_B49220(__int64 a1)
{
  bool result; // al

  result = 0;
  if ( *(_BYTE *)a1 == 85 )
    return (*(_WORD *)(a1 + 2) & 3u) - 1 <= 1;
  return result;
}
