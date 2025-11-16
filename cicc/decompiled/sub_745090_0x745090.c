// Function: sub_745090
// Address: 0x745090
//
__int64 __fastcall sub_745090(_BYTE *a1, _DWORD *a2)
{
  char v2; // al
  __int64 result; // rax
  bool v4; // zf

  v2 = a1[140];
  if ( v2 == 14 )
    goto LABEL_4;
  if ( (unsigned __int8)(v2 - 9) <= 2u )
  {
    if ( (a1[177] & 0x20) != 0 )
    {
LABEL_4:
      *a2 = 1;
      return 1;
    }
    return 0;
  }
  if ( v2 == 12 )
  {
    if ( (a1[186] & 0x30) != 0 )
      goto LABEL_4;
    return 0;
  }
  v4 = v2 == 2;
  result = 0;
  if ( v4 && (a1[162] & 0x40) != 0 )
    goto LABEL_4;
  return result;
}
