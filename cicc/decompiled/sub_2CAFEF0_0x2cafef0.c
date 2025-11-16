// Function: sub_2CAFEF0
// Address: 0x2cafef0
//
__int64 __fastcall sub_2CAFEF0(unsigned __int64 a1, unsigned __int8 *a2, int *a3, __int64 a4, unsigned int a5)
{
  int v6; // eax
  unsigned int v7; // eax
  int v8; // eax

  v6 = *a2;
  if ( (unsigned __int8)v6 <= 0x1Cu )
  {
    v8 = sub_2CAFE10(a1, (__int64)a2);
    a5 = 0;
    if ( !v8 )
      return 0;
  }
  else
  {
    v7 = v6 - 29;
    if ( v7 != 39 )
    {
      if ( v7 > 0x27 )
      {
        if ( v7 == 40 )
          goto LABEL_6;
        return 0;
      }
      if ( v7 != 26 )
      {
        if ( v7 == 27 )
        {
LABEL_6:
          v8 = 1;
          goto LABEL_7;
        }
        return 0;
      }
    }
    v8 = 2;
  }
LABEL_7:
  if ( *a3 )
  {
    LOBYTE(a5) = *a3 == v8;
    return a5;
  }
  else
  {
    *a3 = v8;
    return 1;
  }
}
