// Function: sub_989340
// Address: 0x989340
//
__int64 __fastcall sub_989340(int *a1, int *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  int v5; // eax
  bool v6; // zf

  sub_989280(a1, a2, a3, a4);
  result = (unsigned int)*a2;
  if ( (result & 3) != 0 )
  {
    if ( (result & 1) == 0 )
    {
      result = (unsigned int)*a1;
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
    v5 = *a1;
    v6 = *((_BYTE *)a1 + 5) == 0;
    *a1 &= 0x3FCu;
    if ( v6 )
    {
      if ( (v5 & 0x3C) != 0 )
      {
        if ( (v5 & 0x3C0) == 0 )
          *((_WORD *)a1 + 2) = 257;
      }
      else
      {
        *((_WORD *)a1 + 2) = 256;
      }
    }
    result = *((unsigned __int16 *)a2 + 2);
    *((_WORD *)a1 + 2) = result;
  }
  return result;
}
