// Function: sub_8EEB70
// Address: 0x8eeb70
//
__int64 __fastcall sub_8EEB70(__int64 a1, unsigned __int8 *a2, int a3)
{
  int v3; // r10d
  __int64 result; // rax
  int v6; // r10d
  __int64 v7; // rdi
  unsigned __int8 *v8; // rsi
  int v9; // r8d
  char v10; // cl
  __int64 v11; // rdx

  v3 = a3 + 14;
  result = (unsigned int)(a3 + 7);
  if ( a3 + 7 >= 0 )
    v3 = a3 + 7;
  v6 = v3 >> 3;
  if ( a3 > 0 )
  {
    v7 = *a2;
    result = 1;
    v8 = a2 + 1;
    v9 = 0;
    while ( v6 > (int)result )
    {
      v10 = result;
      result = (unsigned int)(result + 1);
      v7 += (unsigned __int64)*v8 << (8 * (v10 & 3u));
      if ( (result & 3) == 0 )
      {
        v11 = v9++;
        *(_DWORD *)(a1 + 4 * v11 + 8) = v7;
        if ( v7 )
        {
          *(_DWORD *)(a1 + 2088) = v9;
          v7 = 0;
        }
      }
      ++v8;
    }
    if ( v7 )
    {
      *(_DWORD *)(a1 + 4LL * v9 + 8) = v7;
      *(_DWORD *)(a1 + 2088) = v9 + 1;
      return v9;
    }
  }
  return result;
}
