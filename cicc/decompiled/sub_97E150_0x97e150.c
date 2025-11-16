// Function: sub_97E150
// Address: 0x97e150
//
_BYTE *__fastcall sub_97E150(_BYTE *a1, size_t a2)
{
  _BYTE *v3; // r12

  if ( !a2 )
    return 0;
  v3 = a1;
  if ( memchr(a1, 0, a2) )
    return 0;
  if ( *a1 == 1 )
    return a1 + 1;
  return v3;
}
