// Function: sub_127F7A0
// Address: 0x127f7a0
//
__int64 __fastcall sub_127F7A0(__int64 a1, __int64 a2, _DWORD *a3)
{
  __int64 result; // rax
  const char *v4; // r8

  result = 0;
  if ( (*(_BYTE *)(a1 + 376) & 1) != 0
    && (*(_BYTE *)(a2 + 89) & 1) == 0
    && *(char *)(a2 + 169) >= 0
    && *(_BYTE *)(a2 + 136) == 1 )
  {
    v4 = *(const char **)(a2 + 8);
    if ( v4 )
    {
      if ( !strcmp(*(const char **)(a2 + 8), "threadIdx") )
      {
        *a3 = 0;
        return 1;
      }
      else if ( !strcmp(v4, "blockIdx") )
      {
        *a3 = 2;
        return 1;
      }
      else if ( !strcmp(v4, "blockDim") )
      {
        *a3 = 1;
        return 1;
      }
      else if ( !strcmp(v4, "gridDim") )
      {
        *a3 = 3;
        return 1;
      }
      else if ( !strcmp(v4, "warpSize") )
      {
        *a3 = 4;
        return 1;
      }
    }
  }
  return result;
}
