// Function: sub_259EBE0
// Address: 0x259ebe0
//
char __fastcall sub_259EBE0(__int64 a1, __int64 a2)
{
  char result; // al
  _BYTE *v3; // rax
  _BYTE *v4; // r13

  if ( (unsigned int)*(unsigned __int8 *)sub_250D070((_QWORD *)(a1 + 72)) - 12 <= 1 )
  {
LABEL_2:
    result = *(_BYTE *)(a1 + 96);
    *(_BYTE *)(a1 + 97) = result;
    return result;
  }
  v3 = (_BYTE *)sub_250D070((_QWORD *)(a1 + 72));
  v4 = v3;
  if ( *v3 <= 0x1Cu )
  {
    result = sub_259E650(a1, a2, 0);
    if ( result )
      return result;
    goto LABEL_2;
  }
  result = sub_259E650(a1, a2, (unsigned __int64)v3);
  if ( !result )
  {
    if ( ((*v4 - 62) & 0xFD) == 0 )
    {
      result = *(_BYTE *)(a1 + 96) | *(_BYTE *)(a1 + 97) & 0xFE;
      *(_BYTE *)(a1 + 97) = result;
      return result;
    }
    goto LABEL_2;
  }
  return result;
}
