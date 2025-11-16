// Function: sub_228A550
// Address: 0x228a550
//
bool __fastcall sub_228A550(__int64 a1)
{
  _BYTE *v1; // rdx
  __int64 v2; // rsi

  if ( !*(_WORD *)(a1 + 40) )
    return 0;
  v1 = *(_BYTE **)(a1 + 48);
  v2 = (__int64)&v1[16 * *(unsigned __int16 *)(a1 + 40)];
  while ( (*v1 & 7) == 2 )
  {
    v1 += 16;
    if ( v1 == (_BYTE *)v2 )
      return 0;
  }
  return (*v1 & 5) == 4;
}
