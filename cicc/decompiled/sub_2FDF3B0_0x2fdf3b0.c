// Function: sub_2FDF3B0
// Address: 0x2fdf3b0
//
bool __fastcall sub_2FDF3B0(int *a1, int *a2, int a3, int a4)
{
  int v4; // r8d
  int v5; // r9d
  bool result; // al

  v4 = *a1;
  v5 = *a2;
  if ( *a1 == -1 )
  {
    if ( v5 == -1 )
    {
      *a1 = a3;
      *a2 = a4;
      return 1;
    }
    else if ( v5 == a3 )
    {
      *a1 = a4;
      return 1;
    }
    else
    {
      result = 0;
      if ( v5 == a4 )
      {
        *a1 = a3;
        return 1;
      }
    }
  }
  else if ( v5 == -1 )
  {
    if ( v4 == a3 )
    {
      *a2 = a4;
      return 1;
    }
    else
    {
      result = 0;
      if ( v4 == a4 )
      {
        *a2 = a3;
        return 1;
      }
    }
  }
  else
  {
    result = v5 == a4 && v4 == a3;
    if ( !result )
      return v4 == a4 && v5 == a3;
  }
  return result;
}
