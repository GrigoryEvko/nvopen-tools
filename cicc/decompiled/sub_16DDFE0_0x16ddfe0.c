// Function: sub_16DDFE0
// Address: 0x16ddfe0
//
__int64 __fastcall sub_16DDFE0(__int64 a1, unsigned __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rax
  __int64 v4; // rax

  if ( a2 <= 3 )
  {
    if ( a2 != 3 )
      return 0;
    goto LABEL_6;
  }
  result = 1;
  if ( *(_DWORD *)(a1 + a2 - 4) != 1717989219 )
  {
LABEL_6:
    v3 = a1 + a2 - 3;
    if ( *(_WORD *)v3 == 27749 && *(_BYTE *)(v3 + 2) == 102 )
      return 2;
    if ( a2 > 4 )
    {
      v4 = a1 + a2 - 5;
      if ( *(_DWORD *)v4 == 1751343469 && *(_BYTE *)(v4 + 4) == 111 )
        return 3;
      return 4 * (unsigned int)(*(_DWORD *)(a1 + a2 - 4) == 1836278135);
    }
    if ( a2 > 3 )
      return 4 * (unsigned int)(*(_DWORD *)(a1 + a2 - 4) == 1836278135);
    return 0;
  }
  return result;
}
