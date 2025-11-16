// Function: sub_AF15A0
// Address: 0xaf15a0
//
_BYTE *__fastcall sub_AF15A0(_BYTE *a1)
{
  if ( (*a1 & 2) != 0 )
    return (_BYTE *)*((_QWORD *)a1 - 2);
  else
    return &a1[-8 * ((*a1 >> 2) & 0xF)];
}
