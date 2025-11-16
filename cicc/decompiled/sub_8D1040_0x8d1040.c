// Function: sub_8D1040
// Address: 0x8d1040
//
_BOOL8 __fastcall sub_8D1040(__int64 a1, _DWORD *a2)
{
  char v2; // al
  char v3; // al

  v2 = *(_BYTE *)(a1 + 140);
  if ( (unsigned __int8)(v2 - 9) <= 2u )
  {
    v3 = *(_BYTE *)(*(_QWORD *)(a1 + 168) + 111LL);
    if ( (v3 & 0x40) != 0 )
    {
      *a2 = 1;
      return (v3 & 0x20) != 0;
    }
    if ( (*(_BYTE *)(a1 + 89) & 1) == 0 )
      return 0;
LABEL_5:
    *a2 = 1;
    return 1;
  }
  if ( (*(_BYTE *)(a1 + 89) & 1) != 0 || dword_4D047EC && v2 == 8 && (*(_BYTE *)(a1 + 169) & 2) != 0 )
    goto LABEL_5;
  return 0;
}
