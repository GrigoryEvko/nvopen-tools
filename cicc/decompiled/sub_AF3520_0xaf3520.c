// Function: sub_AF3520
// Address: 0xaf3520
//
_BYTE *__fastcall sub_AF3520(_BYTE *a1)
{
  _BYTE *result; // rax
  unsigned __int8 v2; // dl
  __int64 v3; // rax

  for ( result = a1; *result == 20; result = *(_BYTE **)(v3 + 8) )
  {
    v2 = *(result - 16);
    if ( (v2 & 2) != 0 )
      v3 = *((_QWORD *)result - 4);
    else
      v3 = (__int64)&result[-8 * ((v2 >> 2) & 0xF) - 16];
  }
  return result;
}
