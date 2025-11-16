// Function: sub_8EED40
// Address: 0x8eed40
//
__int64 __fastcall sub_8EED40(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned int v7; // eax
  int v8; // ecx
  __int64 result; // rax
  _DWORD *v10; // rdx

  if ( *(_DWORD *)(a2 + 2088) )
  {
    v3 = 0;
    v4 = 0;
    do
    {
      v5 = v4 + *(unsigned int *)(a1 + 4 * v3 + 8);
      v4 = 0;
      v6 = v5 - *(unsigned int *)(a2 + 4 * v3 + 8);
      if ( v6 < 0 )
        v4 = -1;
      *(_DWORD *)(a1 + 4 * v3 + 8) = v6;
      v7 = ++v3;
    }
    while ( *(_DWORD *)(a2 + 2088) > (unsigned int)v3 );
    if ( v4 )
    {
      while ( 1 )
      {
        v8 = *(_DWORD *)(a1 + 4LL * v7 + 8);
        if ( v8 )
          break;
        *(_DWORD *)(a1 + 4LL * v7++ + 8) = -1;
      }
      *(_DWORD *)(a1 + 4LL * v7 + 8) = v8 - 1;
    }
  }
  result = *(unsigned int *)(a1 + 2088);
  if ( (_DWORD)result )
  {
    result = (unsigned int)(result - 1);
    v10 = (_DWORD *)(a1 + 4 * result + 8);
    while ( !*v10 )
    {
      *(_DWORD *)(a1 + 2088) = result;
      --v10;
      if ( !(_DWORD)result )
        break;
      result = (unsigned int)(result - 1);
    }
  }
  return result;
}
