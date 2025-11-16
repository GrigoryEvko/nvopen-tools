// Function: sub_321C590
// Address: 0x321c590
//
_BYTE *__fastcall sub_321C590(_BYTE *a1)
{
  char v1; // al
  unsigned __int8 v2; // al

  v1 = *a1;
  if ( *a1 != 26 && v1 != 27 && v1 != 29 )
    BUG();
  v2 = *(a1 - 16);
  if ( (v2 & 2) != 0 )
    return sub_AF3520(**((_BYTE ***)a1 - 4));
  else
    return sub_AF3520(*(_BYTE **)&a1[-8 * ((v2 >> 2) & 0xF) - 16]);
}
