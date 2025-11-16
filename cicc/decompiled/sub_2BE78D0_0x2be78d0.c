// Function: sub_2BE78D0
// Address: 0x2be78d0
//
char *__fastcall sub_2BE78D0(__int64 a1, char *a2)
{
  _BYTE *v3; // rsi
  char *result; // rax

  v3 = *(_BYTE **)(a1 + 8);
  if ( v3 == *(_BYTE **)(a1 + 16) )
    return sub_17EB120(a1, v3, a2);
  if ( v3 )
  {
    result = (char *)(unsigned __int8)*a2;
    *v3 = (_BYTE)result;
    v3 = *(_BYTE **)(a1 + 8);
  }
  *(_QWORD *)(a1 + 8) = v3 + 1;
  return result;
}
