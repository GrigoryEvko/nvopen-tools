// Function: sub_2FF7ED0
// Address: 0x2ff7ed0
//
_BOOL8 __fastcall sub_2FF7ED0(__int64 a1, __int64 a2, _WORD *a3)
{
  if ( !sub_2FF7B70(a1) )
    return 0;
  if ( !a3 )
    a3 = sub_2FF7DB0(a1, a2);
  return (*a3 & 0x1FFF) != 0x1FFF && (*((_BYTE *)a3 + 1) & 0x40) != 0;
}
