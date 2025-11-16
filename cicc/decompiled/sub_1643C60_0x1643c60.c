// Function: sub_1643C60
// Address: 0x1643c60
//
__int64 __fastcall sub_1643C60(__int64 a1, __int64 a2)
{
  unsigned int v2; // r8d
  __int64 v3; // rax
  unsigned int v4; // r8d

  if ( a1 == a2 )
  {
    return 1;
  }
  else
  {
    v2 = 0;
    if ( ((*(_DWORD *)(a2 + 8) & 0x200) != 0) == ((*(_DWORD *)(a1 + 8) & 0x200) != 0) )
    {
      v3 = *(unsigned int *)(a1 + 12);
      if ( *(_DWORD *)(a2 + 12) == v3 )
      {
        v2 = 1;
        if ( 8 * v3 )
        {
          LOBYTE(v4) = memcmp(*(const void **)(a1 + 16), *(const void **)(a2 + 16), 8 * v3) == 0;
          return v4;
        }
      }
    }
  }
  return v2;
}
