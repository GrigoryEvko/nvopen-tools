// Function: sub_1369030
// Address: 0x1369030
//
bool __fastcall sub_1369030(_DWORD *a1, _DWORD *a2, _DWORD *a3)
{
  __int64 i; // rax
  _DWORD *v4; // rcx
  bool result; // al

  for ( i = a2 - a1; i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v4 = &a1[i >> 1];
      if ( *v4 >= *a3 )
        break;
      a1 = v4 + 1;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        goto LABEL_5;
    }
  }
LABEL_5:
  result = 0;
  if ( a2 != a1 )
    return *a1 <= *a3;
  return result;
}
