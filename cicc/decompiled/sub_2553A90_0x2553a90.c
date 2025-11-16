// Function: sub_2553A90
// Address: 0x2553a90
//
bool __fastcall sub_2553A90(unsigned __int8 *a1)
{
  bool result; // al
  int v2; // edx
  unsigned __int16 v3; // cx
  unsigned int v4; // edx
  int v5; // eax

  result = sub_B46500(a1);
  if ( result )
  {
    v2 = *a1;
    if ( (_BYTE)v2 == 64 )
      return a1[72] != 0;
    if ( (_BYTE)v2 == 65 )
    {
      v3 = *((_WORD *)a1 + 1);
      if ( ((v3 >> 2) & 7) == 2 )
        return (unsigned __int8)v3 >> 5 != 2;
      return result;
    }
    v4 = v2 - 29;
    if ( v4 > 0x21 )
    {
      if ( v4 == 37 )
      {
        v5 = (*((_WORD *)a1 + 1) >> 1) & 7;
        return (unsigned int)(v5 - 1) > 1;
      }
    }
    else if ( v4 > 0x1F )
    {
      v5 = (*((_WORD *)a1 + 1) >> 7) & 7;
      return (unsigned int)(v5 - 1) > 1;
    }
    BUG();
  }
  return result;
}
