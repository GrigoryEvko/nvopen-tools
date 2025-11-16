// Function: sub_16E2460
// Address: 0x16e2460
//
bool __fastcall sub_16E2460(__int64 a1, int *a2, _DWORD *a3, _DWORD *a4)
{
  unsigned int v6; // eax
  int v7; // eax
  bool result; // al

  sub_16E2390(a1, a2, a3, a4);
  v6 = *(_DWORD *)(a1 + 44);
  if ( v6 == 7 )
    goto LABEL_11;
  if ( v6 <= 7 )
  {
    if ( *a2 )
    {
      result = 0;
      if ( (unsigned int)*a2 <= 3 )
        return result;
    }
    else
    {
      *a2 = 8;
    }
    *a4 = 0;
    *a3 = *a2 - 4;
    *a2 = 10;
    return 1;
  }
  if ( v6 != 11 )
  {
LABEL_11:
    *a2 = 10;
    *a3 = 4;
    *a4 = 0;
    return 1;
  }
  v7 = *a2;
  if ( !*a2 )
  {
    *a2 = 10;
    *a3 = 4;
    v7 = *a2;
  }
  return v7 == 10;
}
