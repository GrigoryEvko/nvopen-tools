// Function: sub_2FF7E60
// Address: 0x2ff7e60
//
_BOOL8 __fastcall sub_2FF7E60(__int64 a1, __int64 a2, _WORD *a3)
{
  if ( !sub_2FF7B70(a1) )
    return 0;
  if ( !a3 )
    a3 = sub_2FF7DB0(a1, a2);
  return (*a3 & 0x1FFF) != 0x1FFF && (*((_BYTE *)a3 + 1) & 0x20) != 0;
}
