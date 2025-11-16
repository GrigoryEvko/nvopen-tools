// Function: sub_2A4BE80
// Address: 0x2a4be80
//
_DWORD *__fastcall sub_2A4BE80(__int64 a1, _DWORD *a2)
{
  _DWORD *v2; // r8
  __int64 i; // rax
  _DWORD *v4; // rdx

  v2 = *(_DWORD **)a1;
  for ( i = *(unsigned int *)(a1 + 8); i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v4 = &v2[4 * (i >> 1)];
      if ( *v4 >= *a2 )
        break;
      v2 = v4 + 4;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        return v2;
    }
  }
  return v2;
}
