// Function: sub_987880
// Address: 0x987880
//
unsigned __int64 __fastcall sub_987880(unsigned __int8 *a1)
{
  unsigned __int8 v1; // cl
  unsigned __int64 result; // rax
  int v3; // eax
  int v4; // edx

  v1 = *a1;
  result = 0;
  if ( *a1 <= 0x1Cu )
  {
    if ( v1 == 5 )
    {
      v3 = *((_WORD *)a1 + 1) & 0xFFFD;
      LOBYTE(v3) = (_WORD)v3 == 13;
      v4 = *((_WORD *)a1 + 1) & 0xFFF7;
      LOBYTE(v4) = (*((_WORD *)a1 + 1) & 0xFFF7) == 17;
      return v4 | (unsigned int)v3;
    }
  }
  else if ( v1 <= 0x36u )
  {
    return (0x40540000000000uLL >> v1) & 1;
  }
  return result;
}
