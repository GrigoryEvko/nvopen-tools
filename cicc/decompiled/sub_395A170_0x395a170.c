// Function: sub_395A170
// Address: 0x395a170
//
bool __fastcall sub_395A170(__int64 a1, __int64 a2, int *a3)
{
  int v4; // eax
  unsigned int v5; // eax
  int v6; // eax

  v4 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v4 <= 0x17u )
  {
    if ( (unsigned int)sub_14C23D0(a2, a1, 0, 0, 0, 0) )
    {
      v6 = sub_3959780(a1, (__int64 *)a2);
      if ( v6 )
        goto LABEL_7;
    }
    return 0;
  }
  v5 = v4 - 24;
  if ( v5 != 37 )
  {
    if ( v5 > 0x25 )
    {
      if ( v5 == 38 )
        goto LABEL_6;
      return 0;
    }
    if ( v5 != 24 )
    {
      if ( v5 == 25 )
      {
LABEL_6:
        v6 = 1;
        goto LABEL_7;
      }
      return 0;
    }
  }
  v6 = 2;
LABEL_7:
  if ( *a3 )
    return *a3 == v6;
  *a3 = v6;
  return 1;
}
