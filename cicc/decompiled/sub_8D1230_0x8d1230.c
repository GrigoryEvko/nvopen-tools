// Function: sub_8D1230
// Address: 0x8d1230
//
__int64 __fastcall sub_8D1230(_BYTE *a1, _DWORD *a2)
{
  char v2; // al

  v2 = a1[140];
  if ( v2 != 8 )
  {
    if ( v2 != 12 || (a1[186] & 1) == 0 )
      return 0;
    goto LABEL_5;
  }
  if ( (a1[169] & 2) != 0 )
  {
LABEL_5:
    *a2 = 1;
    return 1;
  }
  return 0;
}
