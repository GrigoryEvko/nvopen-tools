// Function: sub_BDA280
// Address: 0xbda280
//
unsigned __int8 *__fastcall sub_BDA280(unsigned __int8 *a1)
{
  unsigned __int8 *result; // rax
  int v2; // edx
  unsigned __int8 v3; // dl
  unsigned __int8 *v4; // rax

  result = a1;
  if ( !a1 )
    return 0;
  while ( 1 )
  {
    v2 = *result;
    if ( (_BYTE)v2 == 18 )
      break;
    if ( (unsigned int)(v2 - 19) <= 1 )
    {
      v3 = *(result - 16);
      v4 = (v3 & 2) != 0 ? (unsigned __int8 *)*((_QWORD *)result - 4) : &result[-8 * ((v3 >> 2) & 0xF) - 16];
      result = (unsigned __int8 *)*((_QWORD *)v4 + 1);
      if ( result )
        continue;
    }
    return 0;
  }
  return result;
}
