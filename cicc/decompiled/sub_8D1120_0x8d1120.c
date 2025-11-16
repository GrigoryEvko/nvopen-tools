// Function: sub_8D1120
// Address: 0x8d1120
//
_BOOL8 __fastcall sub_8D1120(__int64 a1, _DWORD *a2)
{
  char v2; // dl
  char v3; // al

  v2 = *(_BYTE *)(a1 + 140);
  if ( (unsigned __int8)(v2 - 9) <= 2u && (v3 = *(_BYTE *)(*(_QWORD *)(a1 + 168) + 111LL), (v3 & 0x10) != 0) )
  {
    *a2 = 1;
    return (v3 & 8) != 0;
  }
  else
  {
    *a2 = v2 == 0;
    return v2 == 0;
  }
}
