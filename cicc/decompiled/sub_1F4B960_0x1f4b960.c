// Function: sub_1F4B960
// Address: 0x1f4b960
//
_BOOL8 __fastcall sub_1F4B960(__int64 a1, __int64 a2, _WORD *a3)
{
  if ( !sub_1F4B670(a1) )
    return 0;
  if ( !a3 )
    a3 = sub_1F4B8B0(a1, a2);
  return (*a3 & 0x3FFF) != 0x3FFF && (*((_BYTE *)a3 + 1) & 0x40) != 0;
}
