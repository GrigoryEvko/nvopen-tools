// Function: sub_135CDE0
// Address: 0x135cde0
//
__int64 __fastcall sub_135CDE0(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rdx
  char v5; // cl

  v2 = *(_BYTE *)(a2 + 16);
  switch ( v2 )
  {
    case '6':
      return sub_135C4E0(a1, a2);
    case '7':
      return sub_135C800(a1, a2);
    case 'R':
      return sub_135CB20(a1, a2);
  }
  if ( v2 != 78 )
    return sub_135AA40(a1, a2);
  v4 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v4 + 16) )
    return sub_135AA40(a1, a2);
  v5 = *(_BYTE *)(v4 + 33);
  if ( (v5 & 0x20) == 0 )
    return sub_135AA40(a1, a2);
  if ( (unsigned int)(*(_DWORD *)(v4 + 36) - 137) > 1 )
  {
    if ( (v5 & 0x20) != 0 && (unsigned int)(*(_DWORD *)(v4 + 36) - 133) <= 3 )
      return sub_135CC90(a1, a2);
    return sub_135AA40(a1, a2);
  }
  return sub_135CB90(a1, a2);
}
