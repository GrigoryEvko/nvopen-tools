// Function: sub_987F40
// Address: 0x987f40
//
__int64 __fastcall sub_987F40(unsigned int *a1, unsigned int *a2, char a3)
{
  __int64 result; // rax
  bool v4; // zf

  result = *a2;
  if ( (result & 3) != 0 )
  {
    if ( (result & 1) == 0 )
    {
      result = *a1;
      *a1 &= 0x3FEu;
      if ( (result & 2) == 0 && !*((_BYTE *)a1 + 5) )
      {
        if ( (result & 0x3C) != 0 )
        {
          if ( (result & 0x3C0) == 0 )
          {
            *((_WORD *)a1 + 2) = 257;
            return 257;
          }
        }
        else
        {
          *((_WORD *)a1 + 2) = 256;
        }
      }
    }
  }
  else
  {
    result = *a1;
    v4 = *((_BYTE *)a1 + 5) == 0;
    *a1 &= 0x3FCu;
    if ( v4 )
    {
      if ( (result & 0x3C) != 0 )
      {
        if ( (result & 0x3C0) == 0 )
          *((_WORD *)a1 + 2) = 257;
      }
      else
      {
        *((_WORD *)a1 + 2) = 256;
      }
    }
    if ( a3 )
    {
      result = *((unsigned __int16 *)a2 + 2);
      *((_WORD *)a1 + 2) = result;
    }
  }
  return result;
}
