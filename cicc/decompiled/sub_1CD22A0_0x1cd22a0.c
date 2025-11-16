// Function: sub_1CD22A0
// Address: 0x1cd22a0
//
char *__fastcall sub_1CD22A0(__int64 a1, _DWORD *a2)
{
  _BYTE *v3; // rsi
  char *result; // rax

  v3 = *(_BYTE **)(a1 + 8);
  if ( v3 == *(_BYTE **)(a1 + 16) )
    return sub_B8BBF0(a1, v3, a2);
  if ( v3 )
  {
    result = (char *)(unsigned int)*a2;
    *(_DWORD *)v3 = (_DWORD)result;
    v3 = *(_BYTE **)(a1 + 8);
  }
  *(_QWORD *)(a1 + 8) = v3 + 4;
  return result;
}
