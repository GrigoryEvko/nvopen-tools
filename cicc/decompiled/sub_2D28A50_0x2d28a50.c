// Function: sub_2D28A50
// Address: 0x2d28a50
//
__int64 __fastcall sub_2D28A50(__int64 a1, unsigned int *a2, unsigned int a3, int a4, int a5, int a6)
{
  unsigned int v8; // esi
  __int64 v9; // rdx
  __int64 result; // rax
  unsigned int v11; // eax
  __int64 v12; // rdx
  int v13; // eax
  unsigned int v14; // esi
  __int64 v15; // rdx
  __int64 v16; // rax

  v8 = *a2;
  if ( v8 )
  {
    v9 = v8 - 1;
    if ( *(_DWORD *)(a1 + 4 * v9 + 128) == a6 && *(_DWORD *)(a1 + 8 * v9 + 4) == a4 )
    {
      *a2 = v9;
      if ( v8 != a3 && *(_DWORD *)(a1 + 4LL * v8 + 128) == a6 && *(_DWORD *)(a1 + 8LL * v8) == a5 )
      {
        v13 = *(_DWORD *)(a1 + 8LL * v8 + 4);
        v14 = v8 + 1;
        for ( *(_DWORD *)(a1 + 8 * v9 + 4) = v13;
              a3 != v14;
              *(_DWORD *)(a1 + 4 * v16 + 128) = *(_DWORD *)(a1 + 4 * v15 + 128) )
        {
          v15 = v14;
          v16 = v14++ - 1;
          *(_DWORD *)(a1 + 8 * v16) = *(_DWORD *)(a1 + 8 * v15);
          *(_DWORD *)(a1 + 8 * v16 + 4) = *(_DWORD *)(a1 + 8 * v15 + 4);
        }
        return a3 - 1;
      }
      else
      {
        *(_DWORD *)(a1 + 8 * v9 + 4) = a5;
        return a3;
      }
    }
    result = 17;
    if ( v8 == 16 )
      return result;
  }
  if ( v8 == a3 )
  {
    *(_DWORD *)(a1 + 8LL * v8 + 4) = a5;
    *(_DWORD *)(a1 + 8LL * v8) = a4;
    *(_DWORD *)(a1 + 4LL * v8 + 128) = a6;
    return v8 + 1;
  }
  else if ( *(_DWORD *)(a1 + 4LL * v8 + 128) == a6 && *(_DWORD *)(a1 + 8LL * v8) == a5 )
  {
    *(_DWORD *)(a1 + 8LL * v8) = a4;
    return a3;
  }
  else
  {
    result = 17;
    if ( a3 != 16 )
    {
      v11 = a3 - 1;
      do
      {
        v12 = v11 + 1;
        *(_DWORD *)(a1 + 8 * v12) = *(_DWORD *)(a1 + 8LL * v11);
        *(_DWORD *)(a1 + 8 * v12 + 4) = *(_DWORD *)(a1 + 8LL * v11 + 4);
        *(_DWORD *)(a1 + 4 * v12 + 128) = *(_DWORD *)(a1 + 4LL * v11 + 128);
        LODWORD(v12) = v11--;
      }
      while ( v8 != (_DWORD)v12 );
      *(_DWORD *)(a1 + 8LL * v8) = a4;
      result = a3 + 1;
      *(_DWORD *)(a1 + 8LL * v8 + 4) = a5;
      *(_DWORD *)(a1 + 4LL * v8 + 128) = a6;
    }
  }
  return result;
}
