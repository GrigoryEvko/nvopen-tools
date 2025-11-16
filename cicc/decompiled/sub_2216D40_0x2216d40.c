// Function: sub_2216D40
// Address: 0x2216d40
//
_BYTE *__fastcall sub_2216D40(__int64 a1, _BYTE *a2, _BYTE *a3, void *a4)
{
  if ( a3 != a2 )
    memcpy(a4, a2, a3 - a2);
  return a3;
}
