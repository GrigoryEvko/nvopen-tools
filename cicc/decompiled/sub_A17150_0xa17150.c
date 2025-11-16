// Function: sub_A17150
// Address: 0xa17150
//
_BYTE *__fastcall sub_A17150(_BYTE *a1)
{
  if ( (*a1 & 2) != 0 )
    return (_BYTE *)*((_QWORD *)a1 - 2);
  else
    return &a1[-8 * ((*a1 >> 2) & 0xF)];
}
