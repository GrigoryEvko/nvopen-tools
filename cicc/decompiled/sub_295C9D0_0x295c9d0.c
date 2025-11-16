// Function: sub_295C9D0
// Address: 0x295c9d0
//
_QWORD *__fastcall sub_295C9D0(_QWORD *a1)
{
  if ( (*a1 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return 0;
  if ( (*a1 & 4) != 0 )
    return *(_QWORD **)(*a1 & 0xFFFFFFFFFFFFFFF8LL);
  return a1;
}
